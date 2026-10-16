import type { Trial } from '../../../types'

/** Carbon's radar row shape: one score per (trial, axis) pair. */
export interface RadarPoint {
  product: string
  feature: string
  score: number
}

/** Axis label for a metric key: `total_time` becomes `Total Time`. */
export function toFeatureLabel(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())
}

// `Trial.metrics` is an open `Record<string, number>` filled by the upstream, so
// this cannot be an exhaustive list. These are the measures where a SMALLER number
// is the better result; everything else is assumed higher-is-better, which is the
// right default for the accuracy / reward / f1 family a job may start reporting.
//
// Live jobs currently report exactly `loss`, `train_loss` and `total_time` -- all
// three lower-is-better, which is why every axis on the radar read backwards: the
// worst trial took the outer vertex and the winning trial collapsed toward the
// centre, on precisely the metric `bestTrialId` minimises.
const LOWER_IS_BETTER = /(^|_)(loss|time|runtime|latency|error|err|perplexity|ppl)(_|$)/

export function isLowerBetter(metricName: string): boolean {
  return LOWER_IS_BETTER.test(metricName.toLowerCase())
}

/**
 * The metric a trial is judged on: the key named by its own `metric`, falling back
 * to a literal `loss`.
 *
 * One accessor for the trials table, Compare and `bestTrialId`, which had drifted:
 * the table read only `metrics[metric]` while Compare's `lossOf` also fell back to
 * `metrics.loss` -- and claimed in its comment to match the table. For a trial
 * carrying `metrics.loss` but no `metric` the table printed an em dash for every
 * row and sorted them all to the end, so the documented "lowest loss first" order
 * silently degraded to API order while the header still showed a forced ascending
 * arrow -- and Compare, opened on those same rows, ranked them properly. Two
 * contradictory orderings of the same trials.
 *
 * Returns the name too, because the name is what decides the direction.
 *
 * The fallback is for a metric that is *absent*, not one that is present and
 * unusable: a trial scored on `reward` whose reward is NaN stays unranked rather
 * than being compared against the others on `loss`, which it was not judged on.
 */
export function primaryMetric(trial: Trial): { name: string; value: number } | null {
  const metrics = trial.metrics
  if (!metrics) return null
  const name = trial.metric && typeof metrics[trial.metric] === 'number' ? trial.metric : 'loss'
  const value = metrics[name]
  return typeof value === 'number' && Number.isFinite(value) ? { name, value } : null
}

/**
 * The metric a job's trials are ranked on: the first `metric` any trial names,
 * else `loss`.
 *
 * Decided for the job, not per trial: `primaryMetric` falls back to `loss` for a
 * trial whose own metric is absent, so judging direction from whichever trial
 * reported first made a reward job lower-is-better when that trial had only a
 * loss so far -- crowning the worst reward, compared against a loss.
 */
export function jobMetric(trials: Trial[]): string {
  return trials.find((t) => t.metric)?.metric ?? 'loss'
}

/** A trial's value on `metric`, or null when it has none (or only a fallback). */
export function scoreOn(trial: Trial, metric: string): number | null {
  const primary = primaryMetric(trial)
  return primary && primary.name === metric ? primary.value : null
}

/**
 * The trials, best first on the job's metric; trials with no value on it sink to
 * the end in their original order. The trials table, Compare and `bestTrialId`
 * all rank through this, so they cannot disagree on the order: both lists used to
 * sort ascending unconditionally, which put the tagged winner of a reward or
 * accuracy job last.
 *
 * Direction comes from `isLowerBetter`, the same predicate the radar scores its
 * axes with. There is nothing to plumb an authoritative objective direction from:
 * the tuning template has no `tune_config.mode` and `Trial` carries no direction,
 * so the metric name is the only signal available.
 */
export function rankBestFirst<T extends Trial>(trials: T[]): T[] {
  const metric = jobMetric(trials)
  const sign = isLowerBetter(metric) ? 1 : -1
  return trials
    .map((trial, index) => ({ trial, index, score: scoreOn(trial, metric) }))
    .sort((a, b) => {
      if (a.score === null || b.score === null) {
        if (a.score === b.score) return a.index - b.index
        return a.score === null ? 1 : -1
      }
      return sign * (a.score - b.score) || a.index - b.index
    })
    .map(({ trial }) => trial)
}

/** The best run by the job's metric -- first under `rankBestFirst`. */
export function bestTrialId(trials: Trial[]): string | undefined {
  const [best] = rankBestFirst(trials)
  return best && scoreOn(best, jobMetric(trials)) !== null ? best.id : undefined
}

// Carbon's RadarChart requires a complete grid: every group (trial) must carry a
// value for every axis (feature). If any (group, feature) pair is missing — e.g.
// one trial reports `loss` and another doesn't — the chart rejects with the name
// of the offending axis (that was the "Uncaught (in promise) Loss" error).
//
// So we take the *union* of metric names across all trials, then emit one entry
// per trial per axis.
//
// Each axis runs 0..1 over `boundsFrom` rather than over the plotted trials. One
// trial has no range of its own — min === max on every axis — so scaling it
// against itself would pin the whole blob to the centre point. Run-wide bounds
// also hold a blob's shape still as the selection grows, instead of reshaping
// every ticked trial each time another row is ticked.
//
// The axes plot *goodness*, not the raw value: see `isLowerBetter`.
export function toRadarData(trials: Trial[], boundsFrom: Trial[] = trials): RadarPoint[] {
  const withMetrics = trials.filter((t) => t.metrics && Object.keys(t.metrics).length > 0)
  if (withMetrics.length === 0) return []

  const metricNames = Array.from(new Set(withMetrics.flatMap((t) => Object.keys(t.metrics))))

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

  const data: RadarPoint[] = []
  for (const trial of withMetrics) {
    for (const name of metricNames) {
      const raw = trial.metrics[name]
      const { min, max } = bounds[name]
      let score: number

      if (typeof raw !== 'number' || !Number.isFinite(raw)) {
        // Not reported. Innermost, and deliberately NOT derived from the axis
        // minimum: on a lower-is-better axis the minimum is the *best* end, so
        // falling back to it would draw a trial that reported nothing as the
        // winner on that axis.
        score = 0
      } else if (min === max) {
        // Every trial in the bounds set reporting one value leaves nothing to
        // rank on that axis. Mid-radius reads as "no spread here"; 0 would read as
        // worst-in-run, which is a claim the data does not make.
        score = 0.5
      } else {
        const fraction = (raw - min) / (max - min)
        score = isLowerBetter(name) ? 1 - fraction : fraction
      }

      data.push({ product: trial.id, feature: toFeatureLabel(name), score })
    }
  }
  return data
}
