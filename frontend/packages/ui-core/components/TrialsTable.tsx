'use client'

import { Fragment, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import {
  DataTable,
  Table,
  TableContainer,
  TableToolbar,
  TableToolbarContent,
  TableToolbarSearch,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
  TableSelectAll,
  TableSelectRow,
  TableExpandHeader,
  TableExpandRow,
  TableExpandedRow,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  CodeSnippet,
  Button,
  InlineNotification,
  InlineLoading,
} from '@carbon/react'
import { ArrowLeft, Compare } from '@carbon/icons-react'
import { RadarChart } from '@carbon/charts-react'
import { useQuery } from '@tanstack/react-query'
import { useChartsTheme } from '../hooks/useTheme'
import { getJobTrials } from '../api/autotunex'
import { listSpaces } from '../api/gbserver'
import { TrialLogViewer } from './TrialLogViewer'
import { TrialCompare } from './TrialCompare'
import { TrialProgressSummary } from './TrialProgressSummary'
import { TrialMetricsCharts } from './TrialMetricsCharts'
import { TrialMetricsPanel } from './TrialMetricsPanel'
import { EMPHASIS_THRESHOLD, METRIC_DE_EMPHASIS, bestTrialId, trialColorScale } from './trialMetrics'
import { formatCell } from './trialsTableFormat'
import styles from './TrialsTable.module.scss'
import type { JobDetail, Trial } from '../types'

const HEADERS = [
  { key: 'created_at', header: 'Created on' },
  { key: 'id', header: 'Trial id' },
  { key: 'status', header: 'Status' },
  { key: 'loss', header: 'Loss' },
  { key: 'total_time', header: 'Total time' },
]

// Poll while the run can still produce trials, at the same 15s cadence
// TuningDetailPageClient polls the job itself, so the table and the page header
// advance together.
const ACTIVE_STATUSES = new Set(['running', 'pending'])

// At most this many trials in one comparison. The ceiling is the palette's:
// EMPHASIS_THRESHOLD marks where METRIC_PALETTE runs out of hues, so a wider
// selection would have to draw two curves in the same colour — and ten
// overlapping curves is already about the limit for reading a line chart.
//
// This caps the *selection*, which is a different question from the one
// EMPHASIS_THRESHOLD answers. That one counts every trial in the job, because
// colour follows the run and not its rank among the ticked ones, so a job with
// more trials than this still de-emphasises however few are ticked. Capping the
// selection does not make those runs distinctly coloured.
const MAX_SELECTED = EMPHASIS_THRESHOLD

function toFeatureLabel(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())
}

// Carbon's RadarChart requires a complete grid: every group (trial) must carry a
// value for every axis (feature). If any (group, feature) pair is missing — e.g.
// one trial reports `loss` and another doesn't — the chart rejects with the name
// of the offending axis (that was the "Uncaught (in promise) Loss" error).
//
// So we take the *union* of metric names across all trials, then emit one entry
// per trial per axis, defaulting a missing metric to 0.
//
// Each axis runs 0..1 over `boundsFrom` rather than over the plotted trials. One
// trial has no range of its own — min === max on every axis — so scaling it
// against itself would pin the whole blob to the centre point. Run-wide bounds
// also hold a blob's shape still as the selection grows, instead of reshaping
// every ticked trial each time another row is ticked.
function toRadarData(
  trials: Trial[],
  boundsFrom: Trial[] = trials
): { product: string; feature: string; score: number }[] {
  const withMetrics = trials.filter((t) => t.metrics && Object.keys(t.metrics).length > 0)
  if (withMetrics.length === 0) return []

  const metricNames = Array.from(
    new Set(withMetrics.flatMap((t) => Object.keys(t.metrics)))
  )

  // Plotted trials join the bounds set (deduped by id) so a plotted value can
  // never land outside its own axis, however the caller picks `boundsFrom`.
  const scaleTrials = Array.from(
    new Map(
      [...boundsFrom, ...withMetrics]
        .filter((t) => t.metrics && Object.keys(t.metrics).length > 0)
        .map((t) => [t.id, t] as const)
    ).values()
  )

  const bounds: Record<string, { min: number; max: number }> = {}
  for (const name of metricNames) {
    const values = scaleTrials
      .map((t) => t.metrics[name])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
    bounds[name] = {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 0,
    }
  }

  const data: { product: string; feature: string; score: number }[] = []
  for (const trial of withMetrics) {
    for (const name of metricNames) {
      const raw = trial.metrics[name]
      const value = typeof raw === 'number' && Number.isFinite(raw) ? raw : bounds[name].min
      const { min, max } = bounds[name]
      // Every trial in the bounds set reporting one value leaves nothing to
      // rank on that axis. Mid-radius reads as "no spread here"; 0 would read as
      // worst-in-run, which is a claim the data does not make.
      const normalized = min === max ? 0.5 : (value - min) / (max - min)
      data.push({
        product: trial.id,
        feature: toFeatureLabel(name),
        score: normalized,
      })
    }
  }
  return data
}

interface Props {
  job: JobDetail
}

export function TrialsTable({ job }: Props) {
  const jobId = job.id
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [showCompare, setShowCompare] = useState(false)
  const theme = useChartsTheme()

  // Same cached `['spaces']` query the rest of the detail view uses to pick a
  // scope — admins read `scope=all` so they can see trials for jobs they don't
  // own. No extra fetch: React Query dedupes on the shared key.
  const { data: spaces = [] } = useQuery({ queryKey: ['spaces'], queryFn: listSpaces })
  const scope = spaces.some((s) => s.is_admin) ? 'all' : 'own'

  // Trials come from GET /jobs/{id}/trials — the job detail no longer nests them.
  // No `enabled` gate: a non-HPO job just returns an empty page, and Carbon
  // mounts every tab panel regardless of which tab is active.
  const {
    data: trials = [],
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ['autotunex-job-trials', jobId, scope],
    queryFn: () => getJobTrials(jobId, scope),
    refetchInterval: ACTIVE_STATUSES.has(job.status) ? 15_000 : false,
  })

  // One colour per run, assigned by creation order and keyed by id, so the
  // charts above and a row's own Metrics tab agree — and so hiding a series in
  // the chart legend never repaints the others. Built here rather than in either
  // consumer because both need the identical map.
  // Computed once and shared: the palette accents this run, and the compare view
  // tags it. Reading it twice would let the tag crown a trial the charts colour
  // as an also-ran if the two ever fell out of step.
  const bestId = bestTrialId(trials)
  const colorScale = useMemo(() => {
    const ordered = [...trials]
      .sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at))
      .map((t) => t.id)
    return trialColorScale(ordered, bestId, theme)
  }, [trials, bestId, theme])

  if (isLoading) {
    return <InlineLoading description="Loading trials…" />
  }

  if (isError) {
    return (
      <InlineNotification
        kind="error"
        title="Failed to load trials"
        subtitle={String(error)}
        hideCloseButton
      />
    )
  }

  if (trials.length === 0) {
    // Still show progress here when the job declared a planned total: "12 queued"
    // before any trial row exists is exactly what users were missing.
    //
    // The charts still render: a run with no search trials (a plain tuning job,
    // or an HPO run whose trials have not been written yet) can already be
    // logging steps, and `derivePhases` attributes those rows to the single
    // training run — which is exactly what they are.
    return (
      <div>
        <TrialProgressSummary job={job} trials={trials} />
        <InlineNotification kind="info" title="No trial data available" hideCloseButton />
        {/* No rows here means no Cancel button, so a selection carried in from a
            page that did have rows would hide the final run with nothing on
            screen able to clear it. There is nothing here to tick, so nothing
            counts as ticked. The mirror itself is left alone rather than reset,
            so a selection survives a query that comes back empty in passing. */}
        <TrialMetricsCharts
          job={job}
          trials={trials}
          trialsLoaded={!isLoading && !isError}
          colorScale={colorScale}
          selectedIds={[]}
          scope={scope}
        />
      </div>
    )
  }

  // Full-screen compare view: replaces the table + radar, mirroring AutoTuneX.
  const selectedForCompare = trials.filter(
    (t) => selectedIds.includes(t.id) && t.status === 'completed' && Object.keys(t.metrics ?? {}).length > 0
  )
  if (showCompare && selectedForCompare.length > 0) {
    return (
      <div>
        <Button
          kind="ghost"
          size="sm"
          renderIcon={ArrowLeft}
          // Keep the selection on the way back. A reader who narrows a
          // comparison and returns to adjust it should not have to re-tick every
          // row. This used to clear the mirror because Carbon's DataTable keeps
          // its own checkbox state and a fresh one mounts unticked — which would
          // strand the Compare button and the radar on screen above visibly
          // empty checkboxes. The rows now carry `isSelected`, so the remounted
          // table comes up matching the mirror instead. Cancel in the toolbar is
          // how a reader clears it.
          onClick={() => setShowCompare(false)}
          style={{ marginBottom: '1rem' }}
        >
          Back to Hyperparameters
        </Button>
        <TrialCompare
          trials={selectedForCompare}
          // Removal drops the trial from the shared selection, so the compare
          // view and the table's checkboxes cannot disagree. Carbon's DataTable
          // is unmounted while this view is up, so there is no internal
          // checkbox state to keep in step — and Back clears the mirror anyway.
          // TrialCompare stops offering removal at two, so this can never empty
          // the view out from under the reader.
          onRemove={(id) => setSelectedIds((prev) => prev.filter((selected) => selected !== id))}
          // The job's winner, not the leading column — a comparison that leaves
          // it out shows no tag rather than promoting the better of two.
          bestTrialId={bestId}
        />
      </div>
    )
  }

  // Default order: lowest loss first — trials without a loss sink to the end.
  //
  // `isSelected` seeds Carbon's own checkbox state. Carbon reads it off the row
  // only when it has no prior state to preserve — on mount, since `normalize`
  // otherwise takes the value from `prevRowsByIds` — so this restores a
  // selection across a remount without contesting ownership of it afterwards.
  // That is what lets Back keep the selection: the compare view unmounts this
  // table, so returning mounts a fresh one that would come up unticked.
  const rows = trials
    .map((t) => ({
      id: t.id,
      created_at: t.created_at,
      status: t.status,
      loss: (t.metric ? t.metrics?.[t.metric] : undefined) ?? undefined,
      total_time: t.metrics?.total_time,
      isSelected: selectedIds.includes(t.id),
    }))
    .sort((a, b) => {
      if (a.loss === undefined && b.loss === undefined) return 0
      if (a.loss === undefined) return 1
      if (b.loss === undefined) return -1
      return a.loss - b.loss
    })

  // Only completed trials with metrics can be plotted — the radar needs a full
  // metric grid, and running/errored trials have no (or partial) metrics.
  const plottableTrials = trials.filter(
    (t) => t.status === 'completed' && Object.keys(t.metrics ?? {}).length > 0
  )
  const comparableTrials = plottableTrials.filter((t) => selectedIds.includes(t.id))
  // Ticked trials draw; the whole run sets the scale they are drawn against.
  const radarData = toRadarData(comparableTrials, plottableTrials)
  // One ticked trial is enough to draw, since the scale spans the run and a lone
  // blob still sits where that trial landed within it. Two axes is Carbon's own
  // floor: a single-axis radar makes RadarChart reject.
  const axisCount = new Set(radarData.map((d) => d.feature)).size
  const canShowRadar = comparableTrials.length >= 1 && axisCount >= 2
  // The diff-table only needs 2+ completed trials with a score — no axis constraint.
  const canOpenCompare = comparableTrials.length >= 2
  const atSelectionCap = selectedIds.length >= MAX_SELECTED

  const trialsById = new Map(trials.map((t) => [t.id, t]))

  return (
    <div>
      <TrialProgressSummary job={job} trials={trials} />
      {/* Table left, radar right once a plottable trial is ticked. `flexWrap` drops
          the radar under the table when the viewport can't seat both, and
          `minWidth: 0` lets the table column actually shrink — without it a flex
          item refuses to go below its content width and overflows the row. With
          no radar the table is the only child and takes the full width. */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 32rem', minWidth: 0, overflowX: 'auto' }}>
      <DataTable
        rows={rows}
        headers={HEADERS}
        isSortable
        // Carbon's default filter matches String(cell.value), but these cells
        // render formatted text — so "5m 20" would miss the row showing
        // "5m 20s" over a raw 320. Same shape as Carbon's defaultFilterRows
        // with formatCell in place of String(). Typed contextually from the
        // prop, so no annotation is needed here.
        filterRows={({ rowIds, headers, cellsById, inputValue, getCellId }) => {
          const query = inputValue.trim().toLowerCase()
          if (!query) return rowIds
          return rowIds.filter((rowId) =>
            headers.some(({ key }) =>
              formatCell(key, cellsById[getCellId(rowId, key)].value).toLowerCase().includes(query)
            )
          )
        }}
      >
        {({ rows: tableRows, headers, getTableProps, getHeaderProps, getRowProps, getExpandedRowProps, getSelectionProps, onInputChange, selectRow, selectedRows }) => (
          <TableContainer>
            <TableToolbar>
              <TableToolbarContent>
                <TableToolbarSearch
                  persistent
                  placeholder="Search trials…"
                  onChange={onInputChange}
                  aria-label="Search trials"
                />
                {trials.length > MAX_SELECTED && (
                  // Explains both disabled states below — the capped row
                  // checkboxes and the withheld select-all — neither of which
                  // Carbon can annotate itself.
                  <span
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      padding: '0 1rem',
                      fontSize: '0.75rem',
                      color: 'var(--cds-text-secondary)',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {selectedIds.length} of {MAX_SELECTED} selected
                    {atSelectionCap && ' — clear one to pick another'}
                  </span>
                )}
                {selectedIds.length > 0 && (
                  <Button
                    kind="ghost"
                    // Clear both copies of the selection from Carbon's own list,
                    // not from `selectedIds`. selectRow toggles a row, so driving
                    // it from the mirror would invert the selection if the two
                    // ever drifted. Rows now carry `isSelected`, which re-seeds
                    // Carbon from the mirror on every remount — the isLoading /
                    // isError / no-trials returns above unmount the DataTable
                    // while this component and the mirror stay alive — so that
                    // known drift is closed; reading Carbon's own list remains
                    // the right thing to do regardless of who drifted.
                    // selectedRows spans every row, not just the filtered ones
                    // (DataTable.js:256), so Cancel still ignores the search.
                    onClick={() => {
                      selectedRows.forEach(({ id }) => selectRow(id))
                      setSelectedIds([])
                    }}
                  >
                    Cancel
                  </Button>
                )}
                {canOpenCompare && (
                  <Button renderIcon={Compare} onClick={() => setShowCompare(true)}>
                    Compare
                  </Button>
                )}
              </TableToolbarContent>
            </TableToolbar>
            <Table {...getTableProps()} size="sm">
              <TableHead>
                <TableRow>
                  <TableExpandHeader aria-label="Expand row" />
                  <TableSelectAll
                    {...getSelectionProps()}
                    // Carbon's handleSelectAll ticks every filtered row in one
                    // go, which would sail past MAX_SELECTED and leave Carbon's
                    // own list holding rows the mirror refused — exactly the
                    // desync the comment below is written to avoid. So it is
                    // offered only when it cannot overshoot. Clearing stays
                    // available: Carbon deselects whenever anything is already
                    // selected (DataTable.js:303), so a non-empty selection
                    // makes this click a clear rather than an add.
                    disabled={selectedIds.length === 0 && tableRows.length > MAX_SELECTED}
                    onSelect={(e) => {
                      getSelectionProps().onSelect(e)
                      // Mirror Carbon's own scope, which is the *filtered* rows:
                      // getUpdatedSelectionState (DataTable.js:276-279) only touches
                      // rows matching an active search, so select-all must add just
                      // the visible ids and deselect-all must remove just those.
                      // Replacing the whole mirror instead would strand a row that
                      // the search has hidden: Carbon keeps it selected while the
                      // mirror forgets it, leaving a ticked checkbox that Compare
                      // and the radar ignore and that Cancel is not shown to clear.
                      // Carbon's handleSelectAll deselects whenever anything is
                      // already selected (DataTable.js:303), so with a selected
                      // row hidden by the search this click is a no-op the user
                      // can repeat — Carbon's own semantics, not a desync.
                      const { checked } = e.target as HTMLInputElement
                      setSelectedIds((prev) =>
                        checked
                          ? [...new Set([...prev, ...tableRows.map((r) => r.id)])]
                          : prev.filter((id) => !tableRows.some((r) => r.id === id))
                      )
                    }}
                  />
                  {(() => {
                    const headerProps = headers.map((h) => getHeaderProps({ header: h }))
                    // Carbon only marks a header as the active sort column once the user
                    // clicks it — it has no notion that `rows` already arrived pre-sorted
                    // by loss. Until the user actually sorts something, show the Loss
                    // header as the (ascending) active sort column so the arrow matches
                    // the real row order.
                    const userHasSorted = headerProps.some((hp) => hp.isSortHeader)
                    return headers.map((h, i) => {
                      const { key: _k, ...hProps } = headerProps[i]
                      const isDefaultLossSort = !userHasSorted && h.key === 'loss'
                      return (
                        <TableHeader
                          key={h.key}
                          {...hProps}
                          isSortHeader={isDefaultLossSort ? true : hProps.isSortHeader}
                          sortDirection={isDefaultLossSort ? 'ASC' : hProps.sortDirection}
                        >
                          {h.header}
                        </TableHeader>
                      )
                    })
                  })()}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableRows.map((row) => {
                  const { key: _k, ...rowProps } = getRowProps({ row })
                  const selectionProps = getSelectionProps({ row })
                  const trial = trialsById.get(row.id)
                  return (
                    <Fragment key={row.id}>
                      <TableExpandRow
                        {...rowProps}
                        // Tint the ticked checkbox to match this trial's line in
                        // the charts below. It rides on the <tr> because
                        // TableSelectRow renders the checkbox cell itself and
                        // forwards neither style nor arbitrary props — only a
                        // single className, which cannot carry a per-trial value.
                        // Nothing else in a row reads --cds-icon-primary: the
                        // expand chevron is filled from --cds-layer-selected-inverse
                        // and the data cells are text, and the expanded panel is a
                        // sibling <tr>, not a child. The checkmark stays
                        // --cds-icon-inverse, which clears 3:1 against all ten
                        // palette hues in both themes — worst 3.33:1, on the
                        // light theme's #b28600.
                        //
                        // Skipped for a de-emphasised run, which is every run but
                        // the best one past EMPHASIS_THRESHOLD trials. The scale
                        // hands them all the same grey, so a tint drawn from it
                        // tells two ticked rows apart no better than Carbon's own
                        // default does — and white on #a8a8a8 is 2.38:1, under the
                        // 3:1 non-text minimum and far under the near-black
                        // default's 19:1. The best run still carries its hue,
                        // which is the one the charts still colour too.
                        //
                        // Only while selected: Carbon draws the *unchecked* box's
                        // border from the same token, so applying this
                        // unconditionally would tint every empty checkbox too.
                        // Keyed off the mirror rather than Carbon's own checked
                        // state because the mirror is what the charts draw, and
                        // matching the charts is the whole point.
                        style={
                          selectedIds.includes(row.id) && colorScale[row.id] !== METRIC_DE_EMPHASIS[theme]
                            ? ({ '--cds-icon-primary': colorScale[row.id] } as CSSProperties)
                            : undefined
                        }
                      >
                        <TableSelectRow
                          {...selectionProps}
                          // Nothing left to give: at the cap an unticked row
                          // cannot be added, so the checkbox says so rather
                          // than swallowing the click. Ticked rows stay live or
                          // the reader would be stuck at ten with no way down.
                          // Carbon puts no tooltip on a disabled checkbox, so
                          // the toolbar carries the reason.
                          disabled={atSelectionCap && !selectedIds.includes(row.id)}
                          onSelect={(e) => {
                            selectionProps.onSelect(e)
                            const checked = (e.target as HTMLInputElement).checked
                            setSelectedIds((prev) => (checked ? [...prev, row.id] : prev.filter((id) => id !== row.id)))
                          }}
                        />
                        {row.cells.map((cell) => (
                          <TableCell key={cell.id}>{formatCell(cell.info.header, cell.value)}</TableCell>
                        ))}
                      </TableExpandRow>
                      {row.isExpanded && trial && (
                        <TableExpandedRow {...getExpandedRowProps({ row })} colSpan={headers.length + 2}>
                          <Tabs>
                            {/* Logs first: when a reader expands a trial they are
                                usually chasing what it did or why it stopped, and
                                the log is the only tab that answers that. */}
                            <TabList aria-label="Trial detail tabs" contained>
                              <Tab>Logs</Tab>
                              <Tab>Configuration</Tab>
                              <Tab>Metrics</Tab>
                            </TabList>
                            <TabPanels>
                              <TabPanel>
                                <TrialLogViewer jobId={jobId} trialId={trial.id} status={trial.status} scope={scope} />
                              </TabPanel>
                              <TabPanel>
                                <CodeSnippet type="multi" wrapText>
                                  {JSON.stringify(trial.config, null, 2)}
                                </CodeSnippet>
                              </TabPanel>
                              <TabPanel>
                                <TrialMetricsPanel
                                  jobId={jobId}
                                  trialId={trial.id}
                                  status={trial.status}
                                  color={colorScale[trial.id]}
                                  scope={scope}
                                />
                              </TabPanel>
                            </TabPanels>
                          </Tabs>
                        </TableExpandedRow>
                      )}
                    </Fragment>
                  )
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </DataTable>
        </div>

        {canShowRadar && (
          <div className={styles.radar} style={{ flex: '0 0 26rem', maxWidth: '100%', height: '420px' }}>
            <RadarChart
              data={radarData}
              options={{
                title: comparableTrials.length > 1 ? 'Trial comparison' : 'Trial metrics',
                radar: { axes: { angle: 'feature', value: 'score' } },
                data: { groupMapsTo: 'product' },
                // The same map the line charts and the row checkboxes use, so a
                // trial reads as one colour across all three. Carbon resolves a
                // radar blob's fill through model.getFillColor, which is what
                // reads this scale; without it the radar picks its own hues by
                // group order, so a trial changes colour whenever the selection
                // does.
                color: { scale: colorScale },
                theme,
                height: '420px',
              }}
            />
          </div>
        )}
      </div>

      <TrialMetricsCharts
        job={job}
        trials={trials}
        trialsLoaded={!isLoading && !isError}
        colorScale={colorScale}
        selectedIds={selectedIds}
        scope={scope}
      />
    </div>
  )
}
