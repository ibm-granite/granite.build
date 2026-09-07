'use client'

import { useMemo } from 'react'
import { InlineLoading, InlineNotification } from '@carbon/react'
import { LineChart } from '@carbon/charts-react'
import { useChartsTheme } from '../hooks/useTheme'
import { useMetricStream } from '../hooks/useMetricStream'
import { getTrialMetrics } from '../api/autotunex'
import { metricChartOptions } from './metricChartOptions'
import { METRIC_PALETTE, emaChartRows, splitMetricRows, toChartRows } from './trialMetrics'
import type { TuningStatus } from '../types'

type Scope = 'own' | 'all'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

const TRAIN = 'Training loss'
const EVAL = 'Eval loss'

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
  jobId: string
  trialId: string
  status: TuningStatus
  /** This trial's colour in the job-level charts, so the two views agree. */
  color: string
  scope: Scope
}

/**
 * One trial's own curves, from GET /jobs/{id}/trials/{trialId}/metrics.
 *
 * Train and eval loss share a y-axis here, which is legitimate precisely because
 * they are the same measure of the same run — the gap between them is what the
 * reader is looking for. Learning rate and gradient norm get their own axes;
 * putting them on this one would be a second scale pretending to be comparable.
 *
 * X is the global step, not the epoch: within a single trial there is nothing to
 * normalise against, and steps are what the logs and checkpoints are numbered by.
 */
export function TrialMetricsPanel({ jobId, trialId, status, color, scope }: Props) {
  const theme = useChartsTheme()
  const isActive = ACTIVE_STATUSES.has(status)

  const {
    data: rows = [],
    isLoading,
    isError,
    error,
  } = useMetricStream(
    ['autotunex-trial-metrics', jobId, trialId, scope],
    (afterId) => getTrialMetrics(jobId, trialId, { afterId, scope }),
    { isActive }
  )

  const split = useMemo(() => splitMetricRows(rows), [rows])
  const evalColor = METRIC_PALETTE[theme][2] === color ? METRIC_PALETTE[theme][1] : METRIC_PALETTE[theme][2]

  const lossRows = useMemo(() => {
    const raw = toChartRows(split.trainSteps, 'global_step', (r) => r.loss)
    const train = emaChartRows(raw).map((r) => ({ ...r, group: TRAIN }))
    const evals = toChartRows(split.evals, 'global_step', (r) => r.extra?.eval_loss).map((r) => ({
      ...r,
      group: EVAL,
    }))
    return [...train, ...evals]
  }, [split.trainSteps, split.evals])

  const lrRows = useMemo(
    () =>
      toChartRows(split.trainSteps, 'global_step', (r) => r.learning_rate).map((r) => ({
        ...r,
        group: 'Learning rate',
      })),
    [split.trainSteps]
  )
  const gradRows = useMemo(
    () =>
      emaChartRows(toChartRows(split.trainSteps, 'global_step', (r) => r.grad_norm)).map((r) => ({
        ...r,
        group: 'Grad norm',
      })),
    [split.trainSteps]
  )

  if (isLoading) return <InlineLoading description="Loading metrics…" />

  if (isError) {
    return (
      <InlineNotification
        kind="error"
        title="Couldn't load this trial's metrics"
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
        title={isActive ? 'Waiting for the first logged step' : 'This trial logged no step metrics'}
        lowContrast
        hideCloseButton
      />
    )
  }

  const shared = { theme, xTitle: 'Global step', height: '220px' } as const

  return (
    <div>
      <Chart
        title="Loss — train (smoothed) and eval"
        rows={lossRows}
        options={metricChartOptions({
          ...shared,
          yTitle: 'Loss',
          colorScale: { [TRAIN]: color, [EVAL]: evalColor },
          points: true,
        })}
      />
      <Chart
        title="Learning rate"
        rows={lrRows}
        options={metricChartOptions({
          ...shared,
          yTitle: 'Learning rate',
          colorScale: { 'Learning rate': color },
          logY: true,
        })}
      />
      <Chart
        title="Gradient norm (smoothed)"
        rows={gradRows}
        options={metricChartOptions({
          ...shared,
          yTitle: 'Grad norm',
          colorScale: { 'Grad norm': color },
        })}
      />
    </div>
  )
}
