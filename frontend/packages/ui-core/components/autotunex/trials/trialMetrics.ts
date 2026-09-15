// Shaping for the per-step metrics behind the Hyperparameters tab charts —
// GET /jobs/{id}/metrics and its per-trial sibling. Kept free of React so it can
// be unit-tested (see apps/standalone/tests/trial-metrics.test.js), the same
// split `trialProgress.ts` makes.
//
// Two problems live here, and every chart depends on both being solved once:
//
//   1. Three different kinds of row share one stream (see `splitMetricRows`).
//   2. Two incomparable training phases share one job (see `derivePhases`).

import type { MetricPoint, Trial } from '../../../types'

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

/**
 * The rows belonging to the given runs — for drawing a selection rather than a
 * whole phase.
 *
 * A row with no `trial_id` is dropped. It belongs to the job's single unnamed run
 * (`groupOf` files it under `run`), which has no row in the trials table and so
 * can never be one of the ids passed here; keeping it would draw a curve the
 * reader did not ask for beside the ones they did.
 *
 * An empty `trialIds` returns nothing rather than everything: "show them all" is
 * the caller's decision, not a special case hidden in here.
 *
 * Safe to hand the result to `runOrigins`. This drops whole runs and leaves the
 * survivors' rows untouched, so each surviving run's earliest row — its origin on
 * the `elapsed` axis — is exactly what it was before the filter.
 */
export function rowsForTrials(rows: MetricPoint[], trialIds: string[]): MetricPoint[] {
  const wanted = new Set(trialIds)
  return rows.filter((row) => row.trial_id != null && wanted.has(row.trial_id))
}

// Carbon's own categorical steps, reordered until they passed colour-vision
// validation — Carbon's default order fails twice: teal-70 (#005d5d) drops below
// the chroma floor and reads grey, and magenta-70 next to it separates by only
// ΔE 6.3 under deuteranopia. Dark is a selected set, not an inversion: purple-70
// and green-60 are too dark against Carbon's g100 layer, so they step up to
// purple-50 and green-50.
//
// Slots 6-10 come from Carbon's 14-variant categorical pairing (option 1), the
// only set Carbon publishes above five — its pairings run 1..5 and then jump
// straight to 14. Candidates were filtered on 3:1 contrast against the layer and
// a chroma floor of 25, which independently rejected the same #005d5d the note
// above names, plus #012749 on light and #fff1f1 / #bae6ff on dark; the rest were
// chosen to maximise the worst-case pairwise ΔE across normal, deuteranopia and
// protanopia vision.
//
// APPEND ONLY — never reorder. The first five slots are load-bearing twice over:
// reordering repaints every trial a reader has already learned, and slot 2 of
// g100 is asserted literally in trial-metrics.test.js.
//
// Measured worst case (Machado severity 1.0 + ΔE76 — a different model from the
// 6.3 quoted above, so compare only within this note): light 23.9 at five slots,
// 13.3 at ten. Dark is 8.5 at both, because the pair that sets it is already in
// the first five — #1192e8 against #a56eff under deuteranopia, a consequence of
// the purple-70 → purple-50 step-up above. Widening to ten does not move that
// number, and fixing it would mean repainting an existing dark slot.
//
// Keys match `useChartsTheme()` in `@/hooks/useTheme`, which is what Carbon's
// `theme` option takes.
export const METRIC_PALETTE: Record<ChartsTheme, string[]> = {
  white: ['#1192e8', '#b28600', '#6929c4', '#198038', '#ee538b',
          '#9f1853', '#fa4d56', '#520408', '#009d9a', '#8a3800'],
  g100: ['#1192e8', '#b28600', '#a56eff', '#24a148', '#ee538b',
         '#8a3ffc', '#33b1ff', '#007d79', '#ff7eb6', '#fa4d56'],
}

/** Carbon grey-40 / grey-60 — the "context, not subject" stroke for emphasis. */
export const METRIC_DE_EMPHASIS: Record<ChartsTheme, string> = {
  white: '#a8a8a8',
  g100: '#6f6f6f',
}

/**
 * Past this many runs the charts switch to emphasis — best run in the first
 * palette slot, every other run in the de-emphasis grey — rather than reaching
 * for more hues. Ten is where `METRIC_PALETTE` runs out: past it a hue would have
 * to be invented rather than taken from Carbon, and an invented hue is
 * indistinguishable from an existing one under colour-vision deficiency.
 *
 * This counts every run in the job, not the selected subset. Colour follows the
 * run, so `trialColorScale` is always handed the full list — an 11-trial job is
 * in emphasis form even when the reader has ticked only two of them.
 */
export const EMPHASIS_THRESHOLD = 10

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

export type MetricXKey = 'global_step' | 'epoch' | 'elapsed'

/** Which run a row belongs to. A row with no trial id is the job's single run. */
function groupOf(row: MetricPoint): string {
  return row.trial_id ?? 'run'
}

const MS_PER_MINUTE = 60_000

/**
 * Run id -> the wall-clock ms of its earliest row, for the `elapsed` x axis.
 *
 * Call this on a whole phase's rows, never on one split series. The origin has
 * to be shared across a run's series: `splitMetricRows` hands out train steps
 * and evals separately, and an origin taken per series would re-anchor eval loss
 * at zero — drawing the first eval, which lands at the end of an epoch, as if it
 * were logged at the same moment as the first training step.
 *
 * Elapsed is measured per run rather than against one job-wide clock so that
 * every curve starts at zero and the runs stay comparable. Search trials execute
 * `max_concurrent_trials` at a time and the final run follows them, so on an
 * absolute clock the phases would sit side by side in disjoint time windows
 * instead of overlapping.
 */
export function runOrigins(rows: MetricPoint[]): Map<string, number> {
  const origins = new Map<string, number>()
  for (const row of rows) {
    if (!row.created_at) continue
    const at = Date.parse(row.created_at)
    if (!Number.isFinite(at)) continue
    const group = groupOf(row)
    const first = origins.get(group)
    if (first === undefined || at < first) origins.set(group, at)
  }
  return origins
}

/** A row's x value, in minutes for `elapsed`; undefined if it has no place on the axis. */
function xValueOf(
  row: MetricPoint,
  xKey: MetricXKey,
  origins?: Map<string, number>
): number | null | undefined {
  if (xKey !== 'elapsed') return row[xKey]
  const origin = origins?.get(groupOf(row))
  if (origin === undefined || !row.created_at) return undefined
  return (Date.parse(row.created_at) - origin) / MS_PER_MINUTE
}

/**
 * Reshapes metric rows into Carbon's `{group, key, value}` rows, dropping any
 * point whose x or y is absent. Rows arrive ordered by `id`, which is ascending
 * by write time; sorting by x keeps a line monotonic even when two runs
 * interleave (`max_concurrent_trials > 1`).
 *
 * @param origins needed only for `xKey: 'elapsed'` — see `runOrigins`. A run
 *   missing from the map has its rows dropped rather than plotted against some
 *   other run's zero.
 */
export function toChartRows(
  rows: MetricPoint[],
  xKey: MetricXKey,
  valueOf: (row: MetricPoint) => number | null | undefined,
  origins?: Map<string, number>
): ChartRow[] {
  const out: ChartRow[] = []
  for (const row of rows) {
    const key = xValueOf(row, xKey, origins)
    const value = valueOf(row)
    if (typeof key !== 'number' || typeof value !== 'number') continue
    if (!Number.isFinite(key) || !Number.isFinite(value)) continue
    out.push({ group: groupOf(row), key, value })
  }
  return out.sort((a, b) => (a.group === b.group ? a.key - b.key : a.group < b.group ? -1 : 1))
}

/**
 * Drops non-positive points, for the charts drawn on a log y axis.
 *
 * Carbon's LOG scale throws ("Data must have values greater than 0 if log scale
 * type is used.") when the axis domain's minimum is <= 0, and HF Trainer logs
 * `learning_rate: 0` on the final step of a linear-decay schedule. So an
 * ordinary completed run would otherwise take the chart -- and, with no error
 * boundary above it, the whole panel -- down.
 *
 * Applied at the log-axis call sites rather than inside `toChartRows`, because a
 * genuine `loss: 0` is a real point and belongs on a linear chart.
 */
export function positiveRows(rows: ChartRow[]): ChartRow[] {
  return rows.filter((r) => r.value > 0)
}
