// Grouping logic behind the three labelled sections of the trial comparison
// view: Results, What differs, and Same for all. Kept separate from
// TrialCompare.tsx so the partition driving the section headings' counts is
// unit-testable (the frontend test harness has no jsdom).

// Keys dropped from the comparison entirely.
const IGNORE_KEYS = ['id']
// Result/metric keys — rendered in their own top section.
export const RESULT_KEYS = ['loss', 'train_loss', 'total_time']
// Flattened keys always hidden from the comparison, regardless of value —
// redundant with data shown elsewhere (model path) or only meaningful for
// online RL trials (reward function name/path).
const HIDDEN_KEYS = [
  'training_config.model_name_or_path',
  'training_rl_config.reward_function_name',
  'training_rl_config.reward_function_path',
]

// Display overrides for keys whose bare name reads wrong in the UI. The backend
// emits `loss` alongside `train_loss` and sets metric="loss", so the unqualified
// key really is the *evaluation* loss — labelling it "Loss" next to "Train loss"
// read as though one of the two were unqualified.
const KEY_LABELS: Record<string, string> = {
  loss: 'Eval loss',
}

/** Section-row label for a flattened compare key. */
export function labelForCompareKey(key: string): string {
  const override = KEY_LABELS[key]
  if (override) return override
  const text = key.replaceAll('_', ' ').trim()
  if (!text) return ''
  return text.charAt(0).toUpperCase() + text.slice(1)
}

export interface CompareKeyGroups {
  /** Outcome metrics that differ across the selected trials. */
  resultKeys: string[]
  /** Hyperparameters that differ across the selected trials. */
  differingKeys: string[]
  /** Keys identical across every selected trial. */
  sameKeys: string[]
}

export function isEmptyValue(value: unknown): boolean {
  if (value === null || value === undefined) return true
  if (typeof value === 'string') return value.trim() === ''
  if (Array.isArray(value)) return value.length === 0
  return false
}

// Every key any row carries, in first-seen order.
//
// The union, not rows[0]'s keys: each row is built independently from its own
// trial's config and metrics (see toCompareRow), so field sets genuinely differ
// between trials — TrialsTable's radar code notes the same about `loss`. Reading
// only the first row made anything it lacked invisible in all three sections and
// absent from the counts the headings claim are exhaustive. Which trial is first
// is decided by the loss sort, so the visible set even moved with the ranking.
function keyUniverse(rows: Record<string, any>[]): string[] {
  const keys: string[] = []
  const seen = new Set<string>()
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (seen.has(key)) continue
      seen.add(key)
      keys.push(key)
    }
  }
  return keys
}

// Keys whose value is not identical across every row.
export function findDifferingKeys(rows: Record<string, any>[]): Set<string> {
  const differing = new Set<string>()
  if (rows.length === 0) return differing
  for (const key of keyUniverse(rows)) {
    const values = rows.map((r) => r[key])
    // Compare by stringified value so arrays/objects don't count as always-differing.
    const first = JSON.stringify(values[0])
    if (!values.every((v) => JSON.stringify(v) === first)) differing.add(key)
  }
  return differing
}

// For each differing key, the minority ("odd one out") values — bolded in the UI.
// A key whose values split evenly (notably any differing key in a two-trial
// comparison) has no minority and is omitted, which is what gates the legend
// explaining the bold convention.
export function getOddOnesOut(
  rows: Record<string, any>[],
  keys: string[]
): Record<string, Set<string>> {
  const oddOnes: Record<string, Set<string>> = {}
  for (const key of keys) {
    const counts = new Map<string, number>()
    for (const row of rows) {
      if (!Object.prototype.hasOwnProperty.call(row, key)) continue
      const v = String(row[key])
      counts.set(v, (counts.get(v) ?? 0) + 1)
    }
    const entries = [...counts.entries()].sort((a, b) => a[1] - b[1])
    if (entries.length <= 1) continue
    const minCount = entries[0][1]
    const maxCount = entries[entries.length - 1][1]
    if (minCount === maxCount) continue
    const minority = new Set<string>()
    for (const [value, count] of entries) {
      if (count === minCount) minority.add(value)
    }
    oddOnes[key] = minority
  }
  return oddOnes
}

// Partition the visible keys into the view's three sections. Exhaustive and
// non-overlapping, so the counts shown in the section headings add up.
//
// A metric identical across every trial lands in `sameKeys` rather than
// `resultKeys` — matching the original behaviour of this view.
//
// `metricKeys` carries the trials' own primary-metric names (`trial.metric`),
// which are dynamic. RESULT_KEYS alone is a fixed list, so a job reporting e.g.
// eval_loss sorted by it correctly — lossOf() honours trial.metric — and then
// filed that number under "What differs" as if it were a hyperparameter, while
// Results showed no loss at all. It is the same number TrialsTable prints in its
// Loss column, which is what made the two views disagree.
export function groupCompareKeys(
  rows: Record<string, any>[],
  metricKeys: string[] = []
): CompareKeyGroups {
  if (rows.length === 0) return { resultKeys: [], differingKeys: [], sameKeys: [] }

  const isResultKey = (key: string) => RESULT_KEYS.includes(key) || metricKeys.includes(key)
  const differing = findDifferingKeys(rows)
  const visible = keyUniverse(rows).filter(
    (key) =>
      !HIDDEN_KEYS.includes(key) &&
      !IGNORE_KEYS.includes(key) &&
      !rows.every((row) => isEmptyValue(row[key]))
  )

  return {
    resultKeys: visible.filter((key) => isResultKey(key) && differing.has(key)),
    differingKeys: visible.filter((key) => !isResultKey(key) && differing.has(key)),
    sameKeys: visible.filter((key) => !differing.has(key)),
  }
}
