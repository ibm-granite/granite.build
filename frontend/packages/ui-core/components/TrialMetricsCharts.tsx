'use client'

import { useMemo, useState } from 'react'
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
import {
  METRIC_PALETTE,
  derivePhases,
  emaChartRows,
  splitMetricRows,
  toChartRows,
} from './trialMetrics'
import type { MetricXKey } from './trialMetrics'
import type { JobDetail, Trial } from '../types'

type Scope = 'own' | 'all'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

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

function Chart({ title, rows, options }: { title: string; rows: unknown[]; options: object }) {
  if (rows.length === 0) return null
  return (
    <div style={{ marginTop: '1rem' }}>
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
  scope: Scope
}

/**
 * Per-step training curves for a tuning job, above the trials table.
 *
 * The job's two phases are drawn as separate blocks and never share a y-scale.
 * The HPO search trials each saw a fraction of the data for a few epochs; the
 * final run trained the winning config once over everything. Side by side on one
 * axis the final run's long descent next to the trials' short flat stubs reads as
 * a dramatic win, when the real difference is how much data each one saw. See
 * `derivePhases`.
 */
export function TrialMetricsCharts({ job, trials, trialsLoaded, colorScale, scope }: Props) {
  const theme = useChartsTheme()
  const [xKey, setXKey] = useState<MetricXKey>('epoch')
  const [smooth, setSmooth] = useState(true)

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
  const search = useMemo(() => splitMetricRows(phases.search), [phases.search])
  const final = useMemo(() => splitMetricRows(phases.final), [phases.final])

  const xTitle = xKey === 'epoch' ? 'Epoch' : 'Global step'
  const searchLoss = useMemo(
    () => {
      const raw = toChartRows(search.trainSteps, xKey, (r) => r.loss)
      return smooth ? emaChartRows(raw) : raw
    },
    [search.trainSteps, xKey, smooth]
  )
  const searchEval = useMemo(
    () => toChartRows(search.evals, xKey, (r) => r.extra?.eval_loss),
    [search.evals, xKey]
  )
  const searchLr = useMemo(
    () => toChartRows(search.trainSteps, xKey, (r) => r.learning_rate),
    [search.trainSteps, xKey]
  )
  const searchGrad = useMemo(
    () => {
      const raw = toChartRows(search.trainSteps, xKey, (r) => r.grad_norm)
      return smooth ? emaChartRows(raw) : raw
    },
    [search.trainSteps, xKey, smooth]
  )

  // The final run is one run, so it gets one hue family rather than a slot from
  // the per-trial scale (whose ids it isn't in — it never appears in /trials).
  const palette = METRIC_PALETTE[theme]
  const FINAL_TRAIN = 'Training loss'
  const FINAL_EVAL = 'Eval loss'
  const finalScale = { [FINAL_TRAIN]: palette[0], [FINAL_EVAL]: palette[2] }
  const finalRows = useMemo(() => {
    const rawTrain = toChartRows(final.trainSteps, xKey, (r) => r.loss)
    const train = (smooth ? emaChartRows(rawTrain) : rawTrain).map((r) => ({
      ...r,
      group: FINAL_TRAIN,
    }))
    const evals = toChartRows(final.evals, xKey, (r) => r.extra?.eval_loss).map((r) => ({
      ...r,
      group: FINAL_EVAL,
    }))
    return [...train, ...evals]
  }, [final.trainSteps, final.evals, xKey, smooth])

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

  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-end', marginBottom: '0.5rem' }}>
        <div style={{ minWidth: '11rem' }}>
          <FormLabel style={{ marginBottom: '0.375rem' }}>X axis</FormLabel>
          {/* Epoch by default: trials differ in batch size, so the same work takes
              a different number of steps and a step axis crushes the shorter runs
              into the left of the plot. */}
          <ContentSwitcher
            size="sm"
            selectedIndex={xKey === 'epoch' ? 0 : 1}
            onChange={({ index }) => setXKey(index === 0 ? 'epoch' : 'global_step')}
          >
            <Switch name="epoch" text="Epoch" />
            <Switch name="step" text="Step" />
          </ContentSwitcher>
        </div>
        <div style={{ minWidth: '11rem' }}>
          <FormLabel style={{ marginBottom: '0.375rem' }}>Smoothing</FormLabel>
          <ContentSwitcher
            size="sm"
            selectedIndex={smooth ? 0 : 1}
            onChange={({ index }) => setSmooth(index === 0)}
          >
            <Switch name="ema" text="EMA" />
            <Switch name="raw" text="Raw" />
          </ContentSwitcher>
        </div>
      </div>

      {searchLoss.length > 0 && (
        <section>
          <h5 style={{ marginTop: '1rem' }}>Search trials</h5>
          {searchCaption && (
            <p style={{ color: 'var(--cds-text-secondary)', fontSize: '0.75rem', margin: '0.25rem 0 0' }}>
              {searchCaption} — not comparable with the final run below.
            </p>
          )}
          <Chart
            title={`Training loss${smooth ? ' (smoothed)' : ''}`}
            rows={searchLoss}
            options={metricChartOptions({ ...sharedSpec, yTitle: 'Loss' })}
          />
          <Chart
            title="Eval loss"
            rows={searchEval}
            options={metricChartOptions({ ...sharedSpec, yTitle: 'Eval loss', height: '220px', points: true })}
          />
          <div style={{ marginTop: '1rem' }}>
            <Accordion>
              <AccordionItem title="Diagnostics — learning rate and gradient norm">
              <Chart
                title="Learning rate"
                rows={searchLr}
                options={metricChartOptions({
                  ...sharedSpec,
                  yTitle: 'Learning rate',
                  height: '220px',
                  logY: true,
                })}
              />
              <Chart
                title={`Gradient norm${smooth ? ' (smoothed)' : ''}`}
                rows={searchGrad}
                options={metricChartOptions({ ...sharedSpec, yTitle: 'Grad norm', height: '220px' })}
              />
              </AccordionItem>
            </Accordion>
          </div>
        </section>
      )}

      {finalRows.length > 0 && (
        <section>
          <h5 style={{ marginTop: '2rem' }}>Final run</h5>
          <p style={{ color: 'var(--cds-text-secondary)', fontSize: '0.75rem', margin: '0.25rem 0 0.75rem' }}>
            The winning configuration, trained once{finalCaption ? ` on the ${finalCaption}` : ''}.
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
            title={`Loss${smooth ? ' (train smoothed)' : ''}`}
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
    </div>
  )
}
