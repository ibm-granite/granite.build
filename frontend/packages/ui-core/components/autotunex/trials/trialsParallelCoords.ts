// The axes and polylines behind the trials parallel-coordinates plot: one vertical
// axis per searched hyperparameter, then the two outcome axes, and one polyline per
// trial crossing all of them.
//
// Kept free of React and of runtime imports so it can be unit-tested — the same
// split trialMetrics.ts, trialsRadar.ts, trialProgress.ts, trialCompareGrouping.ts
// and trialHyperparams.ts already make. The no-imports part is load-bearing, not
// stylistic: this module is require()d directly by its test, and `node --test`
// resolves extensionless relative specifiers only for imports that type stripping
// erases, so a value import from a sibling fails at load with ERR_MODULE_NOT_FOUND.
// That is why the caller passes the resolved metric in rather than this module
// importing `primaryMetric` from trialsRadar.ts.

import type { Trial } from '../../../types'

/** Seconds-valued metric every trial reports, plotted as its own outcome axis. */
const TOTAL_TIME = 'total_time'

// Axis order for the plot. Fixed rather than derived from the data, for the same
// reason COLUMN_PRIORITY in trialHyperparams.ts is fixed: an order derived from each
// axis's measured effect on the metric would reshuffle the axes as trials arrive
// mid-run, moving an axis the reader was just looking at.
//
// Listed left to right, and the list is ordered so the hyperparameters that most
// often decide the outcome come LAST — immediately left of the outcome axes. On this
// form a relationship between neighbouring axes reads as a short parallel run and
// the same relationship between distant axes reads as a long crossing, so seating
// learning_rate and r beside Loss is what keeps the plot readable rather than
// spaghetti. Keys not listed sort ahead of these, keeping hyperparamColumns' order.
const PLOT_AXIS_ORDER = [
  'bias',
  'gradient_accumulation_steps',
  'lr_scheduler_type',
  'warmup_ratio',
  'alpha_ratio',
  'lora_dropout',
  'per_device_train_batch_size',
  'r',
  'learning_rate',
]

export interface ParallelAxis {
  /** Config key, or `total_time` / the metric name for the two outcome axes. */
  key: string
  label: string
  /** Outcome axes are drawn right of a divider and carry the "↑ better" note. */
  isOutcome: boolean
  /** Tick text at the top and bottom of the axis. */
  topLabel: string
  bottomLabel: string
}

export interface ParallelLine {
  id: string
  /**
   * Where the trial sits on each axis, in `axes` order: 0 is the bottom of the
   * axis, 1 the top. `null` when the trial does not report that key — the
   * polyline breaks there rather than inventing a value.
   */
  positions: (number | null)[]
  /** Display text per axis, for the hover readout. */
  values: string[]
  /**
   * 0 = worst metric in the set, 1 = best, for the sequential colour ramp.
   * `null` when the trial reports no usable metric.
   */
  goodness: number | null
}

export interface ParallelMetric {
  name: string
  /** Whether a smaller number is the better result. From `isLowerBetter`. */
  lowerIsBetter: boolean
}

/** A key every plotted trial reports as a finite number is a numeric axis. */
function isNumericAxis(values: unknown[]): boolean {
  return values.length > 0 && values.every((v) => typeof v === 'number' && Number.isFinite(v))
}

function formatSeconds(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
}

// Matches formatHyperparamValue's thresholds so an axis tick and the same value's
// table cell read alike. Not imported from trialHyperparams.ts — see the note at the
// top of this file on why this module takes no value imports.
function formatNumber(value: number): string {
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  return magnitude < 1e-4 || magnitude >= 1e5 ? value.toExponential() : String(value)
}

/**
 * Position of `value` on an axis spanning `min`..`max`, as 0 (bottom) to 1 (top).
 *
 * A zero-width span puts everything at the top rather than dividing by zero. That
 * happens for a single plotted trial, and for `total_time` when every trial took
 * the same rounded number of seconds; either way one flat line across the axis is
 * the honest picture, and the top is where a lone trial belongs on an axis whose
 * scale it alone defines.
 */
function fraction(value: number, min: number, max: number, invert: boolean): number {
  if (max === min) return 1
  const t = (value - min) / (max - min)
  return invert ? 1 - t : t
}

/**
 * Build the axes and polylines for the parallel-coordinates plot.
 *
 * `hyperparamKeys` comes from `hyperparamColumns`, so the plot shows exactly the
 * hyperparameters the table shows as columns — the two cannot disagree about which
 * of them varied. They are then reordered by PLOT_AXIS_ORDER for legibility.
 *
 * Every axis carries its own real scale and its real tick values. Nothing is
 * remapped onto a shared 0..1 "goodness" scale the way the radar does it, which is
 * what let a 0.5% spread in loss read as the full distance from centre to rim.
 *
 * The two outcome axes are inverted when smaller is better, so on both of them the
 * top of the axis is the better result and a line that stays high did well. The
 * hyperparameter axes are never inverted: they carry no notion of better, so larger
 * is simply higher, and `topLabel` / `bottomLabel` say which is which.
 *
 * `boundsFrom` sets each axis's scale, defaulting to the plotted trials. Passing the
 * whole run holds the axes still as the reader ticks trials on and off, the same
 * reason `toRadarData` takes it.
 */
export function buildParallelCoords(
  trials: Trial[],
  hyperparamKeys: string[],
  metric: ParallelMetric,
  boundsFrom: Trial[] = trials
): { axes: ParallelAxis[]; lines: ParallelLine[] } {
  if (trials.length === 0) return { axes: [], lines: [] }

  const ordered = [...hyperparamKeys].sort((a, b) => {
    const rankA = PLOT_AXIS_ORDER.indexOf(a)
    const rankB = PLOT_AXIS_ORDER.indexOf(b)
    // Unlisted keys keep hyperparamColumns' order and sort ahead of listed ones.
    if (rankA === -1 && rankB === -1) return hyperparamKeys.indexOf(a) - hyperparamKeys.indexOf(b)
    if (rankA === -1) return -1
    if (rankB === -1) return 1
    return rankA - rankB
  })

  // Scales come from `boundsFrom` plus the plotted trials, so a plotted value can
  // never land outside its own axis however the caller picks `boundsFrom` — the
  // same guard toRadarData applies.
  const scaleTrials = [...new Map([...boundsFrom, ...trials].map((t) => [t.id, t])).values()]

  const configValue = (trial: Trial, key: string): unknown => {
    const config = trial?.config
    return config && typeof config === 'object' ? (config as Record<string, unknown>)[key] : undefined
  }
  const metricValue = (trial: Trial, key: string): number | undefined => {
    const raw = trial?.metrics?.[key]
    return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined
  }

  const axes: ParallelAxis[] = []
  // One positioner per axis, built alongside it so the two cannot fall out of order.
  const positionOf: ((trial: Trial) => number | null)[] = []
  const displayOf: ((trial: Trial) => string)[] = []

  for (const key of ordered) {
    const present = scaleTrials.map((t) => configValue(t, key)).filter((v) => v !== undefined && v !== null)

    if (isNumericAxis(present)) {
      const numbers = present as number[]
      const min = Math.min(...numbers)
      const max = Math.max(...numbers)
      axes.push({
        key,
        label: key,
        isOutcome: false,
        topLabel: formatNumber(max),
        bottomLabel: formatNumber(min),
      })
      positionOf.push((trial) => {
        const v = configValue(trial, key)
        return typeof v === 'number' && Number.isFinite(v) ? fraction(v, min, max, false) : null
      })
      displayOf.push((trial) => {
        const v = configValue(trial, key)
        return typeof v === 'number' && Number.isFinite(v) ? formatNumber(v) : '—'
      })
      continue
    }

    // Categorical. Sorted, not first-seen: a first-seen order would reshuffle the
    // axis as trials arrive, which is what the fixed axis order exists to avoid.
    const cats = [...new Set(present.map((v) => String(v)))].sort()
    axes.push({
      key,
      label: key,
      isOutcome: false,
      topLabel: cats[cats.length - 1] ?? '—',
      bottomLabel: cats[0] ?? '—',
    })
    positionOf.push((trial) => {
      const v = configValue(trial, key)
      if (v === undefined || v === null) return null
      const index = cats.indexOf(String(v))
      if (index === -1) return null
      return cats.length < 2 ? 1 : index / (cats.length - 1)
    })
    displayOf.push((trial) => {
      const v = configValue(trial, key)
      return v === undefined || v === null ? '—' : String(v)
    })
  }

  // ── Outcome axes ──────────────────────────────────────────────────────────
  // total_time first, then the metric, so the metric is the rightmost axis: it is
  // what the reader is here for, and the right edge is where a left-to-right read
  // ends. Both are inverted when smaller is better, so up is always better here.
  const outcomes: { key: string; lowerIsBetter: boolean; fmt: (v: number) => string }[] = [
    { key: TOTAL_TIME, lowerIsBetter: true, fmt: formatSeconds },
    { key: metric.name, lowerIsBetter: metric.lowerIsBetter, fmt: (v) => String(+v.toFixed(4)) },
  ]

  for (const outcome of outcomes) {
    const numbers = scaleTrials
      .map((t) => metricValue(t, outcome.key))
      .filter((v): v is number => v !== undefined)
    // An outcome no trial reports gets no axis at all, rather than a flat axis
    // labelled "—" that every line would cross at the same height.
    if (numbers.length === 0) continue

    const min = Math.min(...numbers)
    const max = Math.max(...numbers)
    const better = outcome.lowerIsBetter ? min : max
    const worse = outcome.lowerIsBetter ? max : min
    axes.push({
      key: outcome.key,
      label: outcome.key,
      isOutcome: true,
      topLabel: outcome.fmt(better),
      bottomLabel: outcome.fmt(worse),
    })
    positionOf.push((trial) => {
      const v = metricValue(trial, outcome.key)
      return v === undefined ? null : fraction(v, min, max, outcome.lowerIsBetter)
    })
    displayOf.push((trial) => {
      const v = metricValue(trial, outcome.key)
      return v === undefined ? '—' : outcome.fmt(v)
    })
  }

  // ── Goodness, for the colour ramp ─────────────────────────────────────────
  // Interpolated on the metric VALUE, not on rank: three trials that scored within
  // 0.0003 of each other really are the same outcome and should read as the same
  // colour. Rank would spread them across three visibly different steps and imply a
  // separation the numbers do not support.
  const metricNumbers = scaleTrials
    .map((t) => metricValue(t, metric.name))
    .filter((v): v is number => v !== undefined)
  const metricMin = metricNumbers.length ? Math.min(...metricNumbers) : 0
  const metricMax = metricNumbers.length ? Math.max(...metricNumbers) : 0

  const lines: ParallelLine[] = trials.map((trial) => {
    const score = metricValue(trial, metric.name)
    return {
      id: trial.id,
      positions: positionOf.map((f) => f(trial)),
      values: displayOf.map((f) => f(trial)),
      goodness:
        score === undefined ? null : fraction(score, metricMin, metricMax, metric.lowerIsBetter),
    }
  })

  return { axes, lines }
}
