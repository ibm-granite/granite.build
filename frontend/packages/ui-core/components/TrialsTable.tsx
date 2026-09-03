'use client'

import { Fragment, useState } from 'react'
import {
  DataTable,
  Table,
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
import { ArrowLeft } from '@carbon/icons-react'
import { RadarChart } from '@carbon/charts-react'
import { useQuery } from '@tanstack/react-query'
import { useChartsTheme } from '../hooks/useTheme'
import { getJobTrials } from '../api/autotunex'
import { listSpaces } from '../api/gbserver'
import { TrialLogViewer } from './TrialLogViewer'
import { TrialCompare } from './TrialCompare'
import { TrialProgressSummary } from './TrialProgressSummary'
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

function formatTime(seconds: number): string {
  if (seconds <= 0) return '0 s'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
}

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
function toRadarData(trials: Trial[]): { product: string; feature: string; score: number }[] {
  const withMetrics = trials.filter((t) => t.metrics && Object.keys(t.metrics).length > 0)
  if (withMetrics.length === 0) return []

  const metricNames = Array.from(
    new Set(withMetrics.flatMap((t) => Object.keys(t.metrics)))
  )

  const bounds: Record<string, { min: number; max: number }> = {}
  for (const name of metricNames) {
    const values = withMetrics
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
      const normalized = min === max ? 0 : (value - min) / (max - min)
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
    return (
      <div>
        <TrialProgressSummary job={job} trials={trials} />
        <InlineNotification kind="info" title="No trial data available" hideCloseButton />
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
          onClick={() => setShowCompare(false)}
          style={{ marginBottom: '1rem' }}
        >
          Back to Hyperparameters
        </Button>
        <TrialCompare trials={selectedForCompare} />
      </div>
    )
  }

  // Default order: lowest loss first — trials without a loss sink to the end.
  const rows = trials
    .map((t) => ({
      id: t.id,
      created_at: t.created_at,
      status: t.status,
      loss: (t.metric ? t.metrics?.[t.metric] : undefined) ?? undefined,
      total_time: t.metrics?.total_time,
    }))
    .sort((a, b) => {
      if (a.loss === undefined && b.loss === undefined) return 0
      if (a.loss === undefined) return 1
      if (b.loss === undefined) return -1
      return a.loss - b.loss
    })

  const selectedTrials = trials.filter((t) => selectedIds.includes(t.id))
  // Only completed trials with metrics can be plotted — the radar needs a full
  // metric grid, and running/errored trials have no (or partial) metrics.
  const comparableTrials = selectedTrials.filter(
    (t) => t.status === 'completed' && Object.keys(t.metrics ?? {}).length > 0
  )
  const radarData = toRadarData(comparableTrials)
  // A radar needs at least 2 axes (distinct metrics) to render; a single axis
  // makes Carbon's RadarChart reject.
  const axisCount = new Set(radarData.map((d) => d.feature)).size
  const canShowRadar = comparableTrials.length >= 2 && axisCount >= 2
  // The diff-table only needs 2+ completed trials with a score — no axis constraint.
  const canOpenCompare = comparableTrials.length >= 2

  const trialsById = new Map(trials.map((t) => [t.id, t]))

  return (
    <div>
      <TrialProgressSummary job={job} trials={trials} />
      {canOpenCompare && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '0.5rem' }}>
          <Button size="sm" onClick={() => setShowCompare(true)}>
            Compare {comparableTrials.length} trials
          </Button>
        </div>
      )}
      <DataTable rows={rows} headers={HEADERS} isSortable>
        {({ rows: tableRows, headers, getTableProps, getHeaderProps, getRowProps, getExpandedRowProps, getSelectionProps }) => (
          <Table {...getTableProps()} size="sm">
            <TableHead>
              <TableRow>
                <TableExpandHeader aria-label="Expand row" />
                <TableSelectAll
                  {...getSelectionProps()}
                  onSelect={(e) => {
                    getSelectionProps().onSelect(e)
                    setSelectedIds((e.target as HTMLInputElement).checked ? tableRows.map((r) => r.id) : [])
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
                    <TableExpandRow {...rowProps}>
                      <TableSelectRow
                        {...selectionProps}
                        onSelect={(e) => {
                          selectionProps.onSelect(e)
                          const checked = (e.target as HTMLInputElement).checked
                          setSelectedIds((prev) => (checked ? [...prev, row.id] : prev.filter((id) => id !== row.id)))
                        }}
                      />
                      {row.cells.map((cell) => (
                        <TableCell key={cell.id}>
                          {cell.info.header === 'created_at'
                            ? new Date(cell.value as string).toLocaleString()
                            : cell.info.header === 'loss' && typeof cell.value === 'number'
                            ? cell.value.toFixed(4)
                            : cell.info.header === 'total_time' && typeof cell.value === 'number'
                            ? formatTime(cell.value)
                            : (cell.value as React.ReactNode) ?? '—'}
                        </TableCell>
                      ))}
                    </TableExpandRow>
                    {row.isExpanded && trial && (
                      <TableExpandedRow {...getExpandedRowProps({ row })} colSpan={headers.length + 2}>
                        <Tabs>
                          <TabList aria-label="Trial detail tabs" contained>
                            <Tab>Logs</Tab>
                            <Tab>Configuration</Tab>
                          </TabList>
                          <TabPanels>
                            <TabPanel>
                              <TrialLogViewer jobId={jobId} trialId={trial.id} status={trial.status} />
                            </TabPanel>
                            <TabPanel>
                              <CodeSnippet type="multi" wrapText>
                                {JSON.stringify(trial.config, null, 2)}
                              </CodeSnippet>
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
        )}
      </DataTable>

      {canShowRadar && (
        <div style={{ height: '420px', marginTop: '1.5rem' }}>
          <RadarChart
            data={radarData}
            options={{
              title: 'Trial comparison',
              radar: { axes: { angle: 'feature', value: 'score' } },
              data: { groupMapsTo: 'product' },
              theme,
              height: '420px',
            }}
          />
        </div>
      )}
    </div>
  )
}
