// Shaping for the per-step metrics behind the Hyperparameters tab charts —
// GET /jobs/{id}/metrics and its per-trial sibling. Kept free of React so it can
// be unit-tested (see apps/standalone/tests/trial-metrics.test.js), the same
// split `trialProgress.ts` makes.
//
// Two problems live here, and every chart depends on both being solved once:
//
//   1. Three different kinds of row share one stream (see `splitMetricRows`).
//   2. Two incomparable training phases share one job (see `derivePhases`).

import type { MetricPoint, Trial } from '../types'

export interface MetricSplit {
  /** Per-step rows: `loss`/`grad_norm`/`learning_rate` all carry values. */
  trainSteps: MetricPoint[]
  /** Epoch-boundary evals: the three scalars are null, numbers sit in `extra`. */
  evals: MetricPoint[]
  /** One end-of-training row per run: `extra.train_loss`, `total_flos`, timings. */
  summaries: MetricPoint[]
}

/**
 * Sorts the stream into its three row kinds.
 *
 * A null `loss` does not mean the value is missing — it means the row is not a
 * step row. Charting `rows.map(r => r.loss)` straight off the raw array puts a
 * hole in the line at every eval and every end-of-run summary, and a chart
 * looking for eval loss at the top level finds nothing at all, because eval loss
 * lives in `extra.eval_loss`. So nothing downstream reads raw rows; it reads one
 * of these three lists.
 */
export function splitMetricRows(rows: MetricPoint[]): MetricSplit {
  const trainSteps: MetricPoint[] = []
  const evals: MetricPoint[] = []
  const summaries: MetricPoint[] = []

  for (const row of rows) {
    if (row.split === 'eval') evals.push(row)
    else if (row.loss == null) summaries.push(row)
    else trainSteps.push(row)
  }
  return { trainSteps, evals, summaries }
}

export interface MetricPhases {
  /** Rows from the HPO search trials — the ones `/trials` knows about. */
  search: MetricPoint[]
  /** Rows from the final training run on the winning config. */
  final: MetricPoint[]
  /** Run ids attributed to the final phase, in first-seen order. */
  finalTrialIds: string[]
}

/**
 * Splits the stream into its two training phases.
 *
 * A job runs the HPO search first (each trial on a fraction of the data for
 * `hpo_num_epochs`), then trains the winning config once on the full data set.
 * Both phases write to the same metrics table, but they are not comparable: the
 * final run's descent over hundreds of steps sitting beside the search trials'
 * short flat stubs reads as "the last run won by a mile" when the real
 * difference is that it is the only one that saw all the data. So the two get
 * separate charts and never share a y-scale.
 *
 * DEFERRED — this is a local stand-in for a server-side field. The rows carry no
 * phase marker, so the only signal available is that the final run does not
 * appear in GET /jobs/{id}/trials, which lists search trials only. We asked
 * AutoTuneX for `phase: "search" | "final"` on the row and it was deliberately
 * deferred; when it lands, this function collapses to reading that field. Do not
 * "simplify" it by keying off the Ray experiment id prefix (`491c7_` vs
 * `11517_`) — that prefix is an implementation detail, not a contract.
 *
 * @param trialsLoaded whether GET /trials has actually resolved. Without this
 *   guard an in-flight trials query yields an empty id set, every row looks
 *   unrecognised, and the entire job misclassifies as one giant final run.
 */
export function derivePhases(
  rows: MetricPoint[],
  trialIds: string[],
  trialsLoaded: boolean
): MetricPhases {
  if (!trialsLoaded) return { search: rows, final: [], finalTrialIds: [] }

  const known = new Set(trialIds)
  const search: MetricPoint[] = []
  const final: MetricPoint[] = []
  const finalTrialIds: string[] = []

  for (const row of rows) {
    // A row with no trial id at all cannot be attributed; treat it as search so
    // it stays visible rather than being silently relabelled as the final run.
    if (!row.trial_id || known.has(row.trial_id)) {
      search.push(row)
      continue
    }
    if (!finalTrialIds.includes(row.trial_id)) finalTrialIds.push(row.trial_id)
    final.push(row)
  }
  return { search, final, finalTrialIds }
}

// Carbon's own categorical steps, reordered until they passed colour-vision
// validation — Carbon's default order fails twice: teal-70 (#005d5d) drops below
// the chroma floor and reads grey, and magenta-70 next to it separates by only
// ΔE 6.3 under deuteranopia. Dark is a selected set, not an inversion: purple-70
// and green-60 are too dark against Carbon's g100 layer, so they step up to
// purple-50 and green-50.
//
// Keys match `useChartsTheme()` in `@/hooks/useTheme`, which is what Carbon's
// `theme` option takes.
export const METRIC_PALETTE: Record<ChartsTheme, string[]> = {
  white: ['#1192e8', '#b28600', '#6929c4', '#198038', '#ee538b'],
  g100: ['#1192e8', '#b28600', '#a56eff', '#24a148', '#ee538b'],
}

/** Carbon grey-40 / grey-60 — the "context, not subject" stroke for emphasis. */
export const METRIC_DE_EMPHASIS: Record<ChartsTheme, string> = {
  white: '#a8a8a8',
  g100: '#6f6f6f',
}

/**
 * Past this many runs, categorical colour stops telling series apart and starts
 * burying the winner, so the charts switch to emphasis instead of reaching for
 * more hues. A generated 6th+ hue is indistinguishable from an existing one
 * under colour-vision deficiency.
 */
export const EMPHASIS_THRESHOLD = 5

export type ChartsTheme = 'white' | 'g100'

/**
 * Maps run id → colour for Carbon's `options.color.scale`.
 *
 * Colour follows the run, never its rank: the slot comes from the id's position
 * in `orderedIds`, so hiding one series never repaints the others. Callers must
 * therefore pass the full ordered run list, not the currently visible subset.
 *
 * Above `EMPHASIS_THRESHOLD` runs this returns the emphasis form — the best run
 * in the first palette slot, every other run in the de-emphasis grey.
 */
export function trialColorScale(
  orderedIds: string[],
  bestId: string | undefined,
  theme: ChartsTheme
): Record<string, string> {
  const palette = METRIC_PALETTE[theme]
  const scale: Record<string, string> = {}

  if (orderedIds.length > EMPHASIS_THRESHOLD) {
    for (const id of orderedIds) {
      scale[id] = id === bestId ? palette[0] : METRIC_DE_EMPHASIS[theme]
    }
    return scale
  }
  orderedIds.forEach((id, i) => {
    scale[id] = palette[i % palette.length]
  })
  return scale
}

/**
 * The best (lowest) run by its own reported metric. `metric` names which key of
 * `metrics` the run was scored on — AutoTuneX minimises it, matching
 * `tune_config.mode`.
 */
export function bestTrialId(trials: Trial[]): string | undefined {
  let best: { id: string; value: number } | undefined
  for (const trial of trials) {
    const value = trial.metric ? trial.metrics?.[trial.metric] : undefined
    if (typeof value !== 'number' || !Number.isFinite(value)) continue
    if (!best || value < best.value) best = { id: trial.id, value }
  }
  return best?.id
}

/** Carbon's tabular row shape for an axis chart with a numeric x scale. */
export interface ChartRow {
  group: string
  key: number
  value: number
}

export type MetricXKey = 'global_step' | 'epoch'

/**
 * Reshapes metric rows into Carbon's `{group, key, value}` rows, dropping any
 * point whose x or y is absent. Rows arrive ordered by `id`, which is ascending
 * by write time; sorting by x keeps a line monotonic even when two runs
 * interleave (`max_concurrent_trials > 1`).
 */
export function toChartRows(
  rows: MetricPoint[],
  xKey: MetricXKey,
  valueOf: (row: MetricPoint) => number | null | undefined
): ChartRow[] {
  const out: ChartRow[] = []
  for (const row of rows) {
    const key = row[xKey]
    const value = valueOf(row)
    if (typeof key !== 'number' || typeof value !== 'number') continue
    if (!Number.isFinite(key) || !Number.isFinite(value)) continue
    out.push({ group: row.trial_id ?? 'run', key, value })
  }
  return out.sort((a, b) => (a.group === b.group ? a.key - b.key : a.group < b.group ? -1 : 1))
}

/**
 * Exponential moving average within each group, in x order.
 *
 * Per-step loss is dominated by batch noise — on the reference job it swings
 * across 1.5 while the runs' reported losses sit inside 0.08 of each other — so
 * several raw lines on one plot are a single indistinguishable thicket. The
 * charts smooth by default and keep a Raw toggle; tooltips read the raw series,
 * so no reader is shown a smoothed number as if it were logged.
 */
export function emaChartRows(rows: ChartRow[], alpha = 0.34): ChartRow[] {
  const previous = new Map<string, number>()
  return rows.map((row) => {
    const last = previous.get(row.group)
    const value = last === undefined ? row.value : alpha * row.value + (1 - alpha) * last
    previous.set(row.group, value)
    return { ...row, value }
  })
}
