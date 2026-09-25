// Shaping for the per-step metrics behind the Hyperparameters tab charts —
// GET /jobs/{id}/metrics and its per-trial sibling. Kept free of React so it can
// be unit-tested (see apps/standalone/tests/trial-metrics.test.js), the same
// split `trialProgress.ts` makes.
//
// Two problems live here, and every chart depends on both being solved once:
//
//   1. Three different kinds of row share one stream (see `splitMetricRows`).
//   2. Two incomparable training phases share one job (see `derivePhases`).

import type { MetricPoint } from '../../../types'

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
 * @param searchComplete whether every planned search trial has resolved (see
 *   `isSearchComplete`). `trialsLoaded` only covers a trials query that has never
 *   resolved, not a *stale* one: the metrics and trials queries are independent
 *   polls, so mid-search a newly started trial's metric rows can arrive before its
 *   trials row and look exactly like a final run. Only the search can mint a new
 *   trial id, and it cannot start one once all planned trials have resolved -- so
 *   until then an unrecognised id is a search trial, not a final run.
 */
export function derivePhases(
  rows: MetricPoint[],
  trialIds: string[],
  trialsLoaded: boolean,
  searchComplete: boolean
): MetricPhases {
  if (!trialsLoaded || !searchComplete) return { search: rows, final: [], finalTrialIds: [] }

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

/**
 * The rows the trials table can account for — the runs it lists, plus the job's
 * single unnamed run — for drawing a whole phase rather than a selection.
 *
 * `derivePhases` cannot always separate the two phases. While the search could
 * still start a trial it has to read every unrecognised id as a late search
 * trial rather than the final run (see `searchComplete` there), and a job that
 * finished without resolving all `num_trials` trial rows leaves it in that
 * branch for good. Its search phase then carries runs the table has no row for —
 * the final run among them — and they arrived on the search charts' shared
 * y-scale, where the final run's full-data descent beside the trials' one-epoch
 * stubs is the comparison those charts exist to prevent, under a legend entry
 * naming a trial the reader cannot find in the table.
 *
 * Unlike `rowsForTrials` this keeps a row with no `trial_id`. That row is the
 * job's single unnamed run, which owns no table row and so can never be named by
 * one; dropping it would blank the charts for a plain tuning job, whose every row
 * is untagged. The distinction is what makes these two functions separate: one
 * answers "which runs did the reader tick", where an untagged run cannot have
 * been ticked, and this one answers "which runs does the table know about", where
 * an untagged run is the table's whole subject.
 *
 * An empty `trialIds` therefore keeps the untagged rows and nothing else, rather
 * than returning everything: a table with no rows vouches for no named run.
 *
 * Safe to hand the result to `runOrigins`, for the same reason `rowsForTrials` is
 * — it drops whole runs and leaves the survivors' rows intact.
 */
export function rowsForKnownRuns(rows: MetricPoint[], trialIds: string[]): MetricPoint[] {
  const known = new Set(trialIds)
  return rows.filter((row) => row.trial_id == null || known.has(row.trial_id))
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
 * The number of distinct hues `METRIC_PALETTE` holds. Ten is where it runs out:
 * past it a hue would have to be invented rather than taken from Carbon, and an
 * invented hue is indistinguishable from an existing one under colour-vision
 * deficiency.
 *
 * So past this many runs the palette cycles and hues repeat — run 11's home slot is
 * run 1's. Two consequences: a ticked run whose home slot another ticked run holds
 * borrows a free one (see `selectionSlots`), and a view drawing every run with
 * nothing ticked switches to emphasis (see `emphasisColorScale`).
 *
 * This counts every run in the job, not the selected subset, because colour
 * follows the run and `trialColorScale` is always handed the full list.
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
 * Each run's home slot is its position modulo the palette, so past
 * `EMPHASIS_THRESHOLD` runs several runs share a home. The alternative — greying
 * every run but the best — made the colour a run showed depend on how many trials
 * the job happened to have. `slots`, from `selectionSlots`, overrides the home slot
 * for the ticked runs so that no two of them share a hue.
 */
export function trialColorScale(
  orderedIds: string[],
  theme: ChartsTheme,
  slots: Record<string, number> = {}
): Record<string, string> {
  const palette = METRIC_PALETTE[theme]
  const scale: Record<string, string> = {}
  orderedIds.forEach((id, i) => {
    scale[id] = palette[slots[id] ?? i % palette.length]
  })
  return scale
}

/**
 * The emphasis form of a `trialColorScale` result: the best run keeps its own
 * colour, every other run takes the de-emphasis grey.
 *
 * For a view that draws every run at once with nothing ticked, in a job past
 * `EMPHASIS_THRESHOLD` runs. There the runs are context rather than a comparison,
 * and a repeated hue would pair runs that have nothing to do with each other. The
 * best run keeps its permanent slot rather than taking slot 0, so it is the same
 * colour here as on its row checkbox and in its Metrics tab.
 */
export function emphasisColorScale(
  scale: Record<string, string>,
  bestId: string | undefined,
  theme: ChartsTheme
): Record<string, string> {
  const emphasis: Record<string, string> = {}
  for (const id of Object.keys(scale)) {
    emphasis[id] = id === bestId ? scale[id] : METRIC_DE_EMPHASIS[theme]
  }
  return emphasis
}

/**
 * Palette slot per ticked run, for `trialColorScale`'s `slots`.
 *
 * A run takes its home slot (its position in `orderedIds`, modulo the palette)
 * unless another ticked run already holds it, and then borrows the lowest free
 * slot. Refusing the tick instead disabled rows while the reader was still under
 * the selection cap, and the cap is the palette size, so a free slot always exists.
 *
 * Sticky: a run already in `previous` keeps its slot for as long as it stays in
 * `selectedIds`, even once its home frees up, so unticking one run never repaints
 * another's curve. Pass the previous result back in on every change; runs no longer
 * ticked are dropped, freeing their slots. Newly ticked runs are placed in
 * `selectedIds` order, which is tick order.
 *
 * At or below `EMPHASIS_THRESHOLD` runs no two homes collide, so every run gets its
 * home slot and colour follows the run outright.
 */
export function selectionSlots(
  orderedIds: string[],
  selectedIds: string[],
  previous: Record<string, number>
): Record<string, number> {
  const size = METRIC_PALETTE.white.length
  const slots: Record<string, number> = {}
  const taken = new Set<number>()
  for (const id of selectedIds) {
    if (previous[id] !== undefined) {
      slots[id] = previous[id]
      taken.add(previous[id])
    }
  }
  for (const id of selectedIds) {
    if (slots[id] !== undefined) continue
    const index = orderedIds.indexOf(id)
    if (index === -1) continue
    const home = index % size
    let slot = home
    if (taken.has(home)) {
      for (let s = 0; s < size; s++) {
        if (!taken.has(s)) {
          slot = s
          break
        }
      }
    }
    slots[id] = slot
    taken.add(slot)
  }
  return slots
}

// `primaryMetric` and `bestTrialId` live in `trialsRadar.ts`, beside the
// `isLowerBetter` predicate they have to agree with. Keeping the direction rule and
// its only consumers in one module is what stops them drifting apart again -- and a
// runtime import between these two modules is not available, because both are
// require()d directly by their unit tests, which resolve extensionless relative
// specifiers only for type-only imports.

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
  origins?: Map<string, number>,
  /**
   * Series name for a row. Defaults to the run it belongs to. The final-run charts
   * override it to label by measure ("Training loss" / "Eval loss") instead, and to
   * suffix the run id when the phase holds more than one run.
   */
  groupName: (row: MetricPoint) => string = groupOf
): ChartRow[] {
  const out: ChartRow[] = []
  for (const row of rows) {
    const key = xValueOf(row, xKey, origins)
    const value = valueOf(row)
    if (typeof key !== 'number' || typeof value !== 'number') continue
    if (!Number.isFinite(key) || !Number.isFinite(value)) continue
    out.push({ group: groupName(row), key, value })
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

/**
 * Fraction of the visible decades to leave clear at each end of a log y axis.
 *
 * Deliberately the same 0.1 Carbon pads a linear axis by, so the log charts get
 * the same visual breathing room as the loss charts beside them — the difference
 * is only that this one is measured in decades, which is the space the axis
 * actually draws in.
 */
const LOG_PADDING_RATIO = 0.1

/**
 * Minimum pad, in decades, for a series whose values never change. Carbon's own
 * pad is `(max - min) * ratio`, which is exactly 0 for a constant series, so its
 * domain collapses to `[v, v]` and the d3 log scale degenerates. A constant
 * learning rate is an ordinary schedule, not a broken run.
 */
const LOG_MIN_PAD_DECADES = 0.05

/**
 * An explicit y domain for a log axis, padded in decades.
 *
 * Carbon pads every axis domain by `(max - min) * paddingRatio` and applies that
 * linear pad whatever the scale type (`extendsDomain` -> the `nn` helper in
 * @carbon/charts). On a log axis that is almost no padding at all: a learning-rate
 * schedule spanning 1e-9..5e-6 gets a pad of ~10% of the max, which is log10(1.1)
 * = 0.04 of the 3.7 decades on screen — about 1% of the plot height, narrower than
 * the stroke, so every curve's peak was drawn flat-topped against the plot frame.
 * The floor was worse: the helper's LOG branch clamps the lower bound back to the
 * data minimum exactly, putting the lowest point *on* the bottom axis.
 *
 * `paddingRatio` lives in Carbon's `configuration-non-customizable`, so the only
 * way to reach this is to hand the axis a `domain` that already has the headroom.
 *
 * Carbon still re-pads what it is given — an explicit `domain` goes through
 * `extendsDomain` too — so the top ends up with slightly more room than the
 * bottom, whose clamp lands it back on exactly the value returned here. Both ends
 * clear the frame, which is the point; the asymmetry is a few percent and not
 * worth compensating for by second-guessing a constant we do not control.
 *
 * Returns undefined when there is nothing to measure, or when a non-positive
 * value is present. `positiveRows` runs ahead of this at every call site, so that
 * second case should not arise — and if it ever does, declining leaves Carbon's
 * own behaviour (it throws, loudly and on purpose) exactly as it was rather than
 * papering over it with a domain that silently hides the point.
 */
export function logDomain(rows: ChartRow[]): [number, number] | undefined {
  if (rows.length === 0) return undefined

  let min = Infinity
  let max = -Infinity
  for (const row of rows) {
    if (row.value < min) min = row.value
    if (row.value > max) max = row.value
  }
  if (!(min > 0) || !Number.isFinite(max)) return undefined

  const low = Math.log10(min)
  const high = Math.log10(max)
  const pad = Math.max((high - low) * LOG_PADDING_RATIO, LOG_MIN_PAD_DECADES)
  return [10 ** (low - pad), 10 ** (high + pad)]
}
