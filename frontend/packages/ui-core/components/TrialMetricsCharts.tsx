'use client'

import { useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import {
  Accordion,
  AccordionItem,
  ContentSwitcher,
  Switch,
  FormLabel,
  InlineLoading,
  InlineNotification,
} from '@carbon/react'
import { LineChart } from '@carbon/charts-react'
import { useChartsTheme } from '../hooks/useTheme'
import { useMetricStream } from '../hooks/useMetricStream'
import { getJobMetrics } from '../api/autotunex'
import { metricChartOptions } from './metricChartOptions'
import styles from './TrialMetricsCharts.module.scss'
import {
  METRIC_PALETTE,
  derivePhases,
  rowsForTrials,
  runOrigins,
  splitMetricRows,
  toChartRows,
} from './trialMetrics'
import type { MetricXKey } from './trialMetrics'
import type { JobDetail, Trial } from '../types'

type Scope = 'own' | 'all'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

// The x axes this section offers, in switcher order.
const X_KEYS: readonly MetricXKey[] = ['epoch', 'elapsed', 'global_step']

const X_TITLES: Record<MetricXKey, string> = {
  epoch: 'Epoch',
  elapsed: 'Minutes elapsed',
  global_step: 'Global step',
}

function formatSeconds(seconds: number): string {
  const total = Math.round(seconds)
  const hours = Math.floor(total / 3600)
  const mins = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (hours > 0) return `${hours}h ${mins}m`
  if (mins > 0) return `${mins}m ${String(secs).padStart(2, '0')}s`
  return `${secs}s`
}

function Tile({ label, value, detail, last }: { label: string; value: string; detail?: string; last?: boolean }) {
  return (
    <div
      style={{
        // flex:1 so the tiles share the row's full width — otherwise the
        // container's border draws a phantom empty cell past the last tile.
        flex: 1,
        padding: '0.75rem 1rem',
        borderRight: last ? undefined : '1px solid var(--cds-border-subtle)',
        minWidth: '9rem',
      }}
    >
      <FormLabel style={{ marginBottom: '0.375rem' }}>{label}</FormLabel>
      <div style={{ fontSize: '1.5rem', fontWeight: 600, lineHeight: 1.1 }}>{value}</div>
      {detail && (
        <div style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--cds-text-secondary)', marginTop: '0.25rem' }}>
          {detail}
        </div>
      )}
    </div>
  )
}

function Chart({
  title,
  rows,
  options,
  style,
}: {
  title: string
  rows: unknown[]
  options: object
  style?: CSSProperties
}) {
  if (rows.length === 0) return null
  return (
    <div style={{ marginTop: '1rem', ...style }}>
      <h6 style={{ marginBottom: '0.5rem' }}>{title}</h6>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      <LineChart data={rows as any} options={options as any} />
    </div>
  )
}

interface Props {
  job: JobDetail
  trials: Trial[]
  /** Whether GET /trials resolved — see `derivePhases` for why this matters. */
  trialsLoaded: boolean
  /** Run id → colour, built once by the caller so both views agree. */
  colorScale: Record<string, string>
  /** Trials ticked in the table. Empty means "draw the final run instead". */
  selectedIds: string[]
  scope: Scope
}

/**
 * Per-step training curves for a tuning job, below the trials table.
 *
 * One phase is on screen at a time, and the trials table's selection picks which:
 * nothing ticked draws the final run, ticking trials draws those trials' search
 * curves. The two never share a chart, a y-scale, or the screen.
 * The HPO search trials each saw a fraction of the data for a few epochs; the
 * final run trained the winning config once over everything. Side by side on one
 * axis the final run's long descent next to the trials' short flat stubs reads as
 * a dramatic win, when the real difference is how much data each one saw. See
 * `derivePhases`.
 */
export function TrialMetricsCharts({ job, trials, trialsLoaded, colorScale, selectedIds, scope }: Props) {
  const theme = useChartsTheme()
  const [xKey, setXKey] = useState<MetricXKey>('epoch')

  const isActive = ACTIVE_STATUSES.has(job.status)
  const {
    data: rows = [],
    isLoading,
    isError,
    error,
  } = useMetricStream(
    ['autotunex-job-metrics', job.id, scope],
    (afterId) => getJobMetrics(job.id, { afterId, scope }),
    { isActive }
  )

  const trialIds = useMemo(() => trials.map((t) => t.id), [trials])
  const phases = useMemo(() => derivePhases(rows, trialIds, trialsLoaded), [rows, trialIds, trialsLoaded])

  // A selection narrows the search phase to the ticked trials. With nothing
  // ticked these charts are off screen entirely (see `showSearch`), so the
  // unfiltered rows are only ever what the no-final-run fallback draws.
  const searchRows = useMemo(
    () => (selectedIds.length > 0 ? rowsForTrials(phases.search, selectedIds) : phases.search),
    [phases.search, selectedIds]
  )
  const search = useMemo(() => splitMetricRows(searchRows), [searchRows])
  const final = useMemo(() => splitMetricRows(phases.final), [phases.final])

  const xTitle = X_TITLES[xKey]

  // Origins for the `elapsed` axis, taken from each phase's whole row set rather
  // than from the split series a chart happens to draw — see `runOrigins`. A set
  // narrowed to a selection is still a whole row set: `rowsForTrials` drops whole
  // runs and leaves the survivors' rows intact, so their origins do not move.
  const searchOrigins = useMemo(() => runOrigins(searchRows), [searchRows])
  const finalOrigins = useMemo(() => runOrigins(phases.final), [phases.final])

  const searchLoss = useMemo(
    () => toChartRows(search.trainSteps, xKey, (r) => r.loss, searchOrigins),
    [search.trainSteps, xKey, searchOrigins]
  )
  const searchEval = useMemo(
    () => toChartRows(search.evals, xKey, (r) => r.extra?.eval_loss, searchOrigins),
    [search.evals, xKey, searchOrigins]
  )
  const searchLr = useMemo(
    () => toChartRows(search.trainSteps, xKey, (r) => r.learning_rate, searchOrigins),
    [search.trainSteps, xKey, searchOrigins]
  )
  const searchGrad = useMemo(
    () => toChartRows(search.trainSteps, xKey, (r) => r.grad_norm, searchOrigins),
    [search.trainSteps, xKey, searchOrigins]
  )

  // The final run is one run, so it gets one hue family rather than a slot from
  // the per-trial scale (whose ids it isn't in — it never appears in /trials).
  const palette = METRIC_PALETTE[theme]
  const FINAL_TRAIN = 'Training loss'
  const FINAL_EVAL = 'Eval loss'
  const finalScale = { [FINAL_TRAIN]: palette[0], [FINAL_EVAL]: palette[2] }
  const finalRows = useMemo(() => {
    const train = toChartRows(final.trainSteps, xKey, (r) => r.loss, finalOrigins).map((r) => ({
      ...r,
      group: FINAL_TRAIN,
    }))
    const evals = toChartRows(final.evals, xKey, (r) => r.extra?.eval_loss, finalOrigins).map((r) => ({
      ...r,
      group: FINAL_EVAL,
    }))
    return [...train, ...evals]
  }, [final.trainSteps, final.evals, xKey, finalOrigins])

  const finalSummary = final.summaries[0]?.extra
  const bestFinalEval = useMemo(() => {
    const values = final.evals
      .map((r) => r.extra?.eval_loss)
      .filter((v): v is number => typeof v === 'number')
    return values.length ? Math.min(...values) : undefined
  }, [final.evals])

  // How the two phases differ, straight off the winning trial's own config —
  // the caption that stops a reader comparing them as peers.
  const trainingConfig = trials[0]?.config?.training_config as Record<string, unknown> | undefined
  const searchCaption = useMemo(() => {
    if (!trainingConfig) return null
    const pct = trainingConfig.hpo_dataset_percentage
    const epochs = trainingConfig.hpo_num_epochs
    const parts: string[] = []
    if (typeof pct === 'number') parts.push(`${Math.round(pct * 100)}% of the data set`)
    if (typeof epochs === 'number') parts.push(`${epochs} epochs`)
    return parts.length ? parts.join(' · ') : null
  }, [trainingConfig])
  const finalCaption = useMemo(() => {
    if (!trainingConfig) return null
    const epochs = trainingConfig.num_train_epochs
    return typeof epochs === 'number'
      ? `full data set · ${epochs} ${epochs === 1 ? 'epoch' : 'epochs'}`
      : 'full data set'
  }, [trainingConfig])

  if (isLoading) return <InlineLoading description="Loading metrics…" />

  if (isError) {
    return (
      <InlineNotification
        kind="error"
        title="Couldn't load step metrics"
        subtitle={String(error)}
        lowContrast
        hideCloseButton
      />
    )
  }

  if (rows.length === 0) {
    return (
      <InlineNotification
        kind="info"
        title={isActive ? 'Waiting for the first logged step' : 'This run logged no step metrics'}
        subtitle={
          isActive
            ? 'Curves appear here as the run reports them.'
            : 'Older runs finished before per-step metrics were recorded.'
        }
        lowContrast
        hideCloseButton
      />
    )
  }

  const sharedSpec = { theme, colorScale, xTitle, height: '260px' } as const

  // One phase at a time. With nothing ticked the final run answers the question a
  // reader arrives with — how did the winning config do — and the search trials,
  // which are not comparable with it, stay out of the way. Ticking trials swaps
  // the final run out for those trials' curves rather than adding to it, so the
  // two are never on screen together to be read as peers.
  //
  // The gates are exact complements, which is what makes the second one a
  // fallback as well as a gate: a job with no final run — still searching, or a
  // plain tuning job — keeps drawing its search curves instead of going blank.
  const hasSelection = selectedIds.length > 0
  const showFinalRun = !hasSelection && finalRows.length > 0
  const showSearch = hasSelection || finalRows.length === 0

  return (
    <div style={{ marginTop: '2rem' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-end', marginBottom: '0.5rem' }}>
        {/* Carbon forces `inline-size: 100%` on the switcher *and* on every
            button inside it, and ellipsises the label, so a switch never widens
            to fit its own text — the labels only get whatever room this wrapper
            declares. Three buttons split it evenly and each spends ~2rem on
            padding, so "Realtime" needs 20rem here to render in full; at 16rem
            it came out as "Realti...". */}
        <div style={{ minWidth: '20rem' }}>
          <FormLabel style={{ marginBottom: '0.375rem' }}>X axis</FormLabel>
          {/* Epoch by default: trials differ in batch size, so the same work takes
              a different number of steps and a step axis crushes the shorter runs
              into the left of the plot. Realtime answers a different question —
              how much wall-clock time a config costs — and reads per run, so the
              curves still start together instead of spreading across the clock. */}
          <ContentSwitcher
            size="sm"
            selectedIndex={X_KEYS.indexOf(xKey)}
            onChange={({ index }) => setXKey(X_KEYS[index ?? 0])}
          >
            <Switch name="epoch" text="Epoch" />
            <Switch name="elapsed" text="Realtime" />
            <Switch name="step" text="Step" />
          </ContentSwitcher>
        </div>
      </div>

      {showFinalRun && (
        <section>
          <h5 style={{ marginTop: '1rem' }}>Final run</h5>
          <p style={{ color: 'var(--cds-text-secondary)', fontSize: '0.75rem', margin: '0.25rem 0 0.75rem' }}>
            The winning configuration, trained once{finalCaption ? ` on the ${finalCaption}` : ''}.
            {/* The search curves are only reachable through the table's
                checkboxes now, so say so — otherwise nothing on screen suggests
                they exist. Gated on there being any, so a job without search
                trials does not point at a table that has no rows to tick. */}
            {searchLoss.length > 0 &&
              ' Select trials in the table above to see their search curves instead.'}
          </p>
          {finalSummary && (
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                border: '1px solid var(--cds-border-subtle)',
                background: 'var(--cds-layer)',
              }}
            >
              {(() => {
                const tiles: { label: string; value: string; detail?: string }[] = []
                if (typeof finalSummary.train_loss === 'number')
                  tiles.push({
                    label: 'Final train loss',
                    value: finalSummary.train_loss.toFixed(3),
                    detail: 'extra.train_loss',
                  })
                if (typeof bestFinalEval === 'number')
                  tiles.push({
                    label: 'Best eval loss',
                    value: bestFinalEval.toFixed(3),
                    detail: `${final.evals.length} evals`,
                  })
                if (typeof finalSummary.train_runtime === 'number')
                  tiles.push({
                    label: 'Train runtime',
                    value: formatSeconds(finalSummary.train_runtime),
                    detail: `${Math.round(finalSummary.train_runtime)} s`,
                  })
                if (typeof finalSummary.train_samples_per_second === 'number')
                  tiles.push({
                    label: 'Throughput',
                    value: `${finalSummary.train_samples_per_second.toFixed(2)} smp/s`,
                    detail:
                      typeof finalSummary.train_steps_per_second === 'number'
                        ? `${finalSummary.train_steps_per_second.toFixed(3)} steps/s`
                        : undefined,
                  })
                if (typeof finalSummary.total_flos === 'number')
                  tiles.push({
                    label: 'Total FLOPs',
                    value: finalSummary.total_flos.toExponential(2),
                    detail: 'extra.total_flos',
                  })
                return tiles.map((t, i) => (
                  <Tile key={t.label} {...t} last={i === tiles.length - 1} />
                ))
              })()}
            </div>
          )}
          <Chart
            title="Loss"
            rows={finalRows}
            options={metricChartOptions({
              theme,
              colorScale: finalScale,
              xTitle,
              yTitle: 'Loss',
              height: '260px',
              points: true,
            })}
          />
        </section>
      )}

      {showSearch && searchLoss.length > 0 && (
        <section>
          <h5 style={{ marginTop: '2rem' }}>Search trials</h5>
          {searchCaption && (
            <p style={{ color: 'var(--cds-text-secondary)', fontSize: '0.75rem', margin: '0.25rem 0 0' }}>
              {searchCaption} — not comparable with the final run.
            </p>
          )}
          {/* Train and eval loss side by side. They stay two charts on two
              y-scales rather than one plot: eval loss is sampled once an epoch
              and sits on a different scale, and overlaying them on a shared
              axis is the dual-axis reading we avoid elsewhere in this file.
              `flexWrap` stacks them again when the viewport can't seat both,
              and `minWidth: 0` lets each one actually shrink — without it a flex
              item refuses to go below its content width and overflows the row.
              The charts are the flex items themselves, so a phase with no eval
              rows (Chart renders nothing) leaves training loss spanning the full
              width instead of half of it beside an empty column. */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-start' }}>
            <Chart
              title="Training loss"
              rows={searchLoss}
              options={metricChartOptions({ ...sharedSpec, yTitle: 'Loss' })}
              style={{ flex: '1 1 24rem', minWidth: 0 }}
            />
            <Chart
              title="Eval loss"
              rows={searchEval}
              // Same height as the chart beside it, so the pair's plot areas line up.
              options={metricChartOptions({ ...sharedSpec, yTitle: 'Eval loss', points: true })}
              style={{ flex: '1 1 24rem', minWidth: 0 }}
            />
          </div>
          <div className={styles.diagnostics} style={{ marginTop: '1rem' }}>
            <Accordion>
              <AccordionItem title="Diagnostics — learning rate and gradient norm">
                {/* Paired like the loss charts above — see that comment for why
                    `flexWrap` and `minWidth: 0` are both load-bearing, and why
                    the Charts are the flex items themselves. */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-start' }}>
                  <Chart
                    title="Learning rate"
                    rows={searchLr}
                    options={metricChartOptions({
                      ...sharedSpec,
                      yTitle: 'Learning rate',
                      height: '220px',
                      logY: true,
                    })}
                    style={{ flex: '1 1 24rem', minWidth: 0 }}
                  />
                  <Chart
                    title="Gradient norm"
                    rows={searchGrad}
                    options={metricChartOptions({ ...sharedSpec, yTitle: 'Grad norm', height: '220px' })}
                    style={{ flex: '1 1 24rem', minWidth: 0 }}
                  />
                </div>
              </AccordionItem>
            </Accordion>
          </div>
        </section>
      )}

      {/* Mirrors the search section's own gate exactly, so a selection with
          nothing to draw says so instead of leaving the page blank — the final
          run is off screen for as long as any trial is ticked. */}
      {hasSelection && searchLoss.length === 0 && (
        <InlineNotification
          kind="info"
          title="No step metrics for this selection yet"
          subtitle="Curves appear here as the selected trials report them. Clear the selection to see the final run."
          lowContrast
          hideCloseButton
          style={{ marginTop: '2rem' }}
        />
      )}
    </div>
  )
}
