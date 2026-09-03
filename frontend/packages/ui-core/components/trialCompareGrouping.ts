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

// Keys whose value is not identical across every row.
export function findDifferingKeys(rows: Record<string, any>[]): Set<string> {
  const differing = new Set<string>()
  if (rows.length === 0) return differing
  for (const key in rows[0]) {
    if (!Object.prototype.hasOwnProperty.call(rows[0], key)) continue
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
export function groupCompareKeys(rows: Record<string, any>[]): CompareKeyGroups {
  if (rows.length === 0) return { resultKeys: [], differingKeys: [], sameKeys: [] }

  const differing = findDifferingKeys(rows)
  const visible = Object.keys(rows[0]).filter(
    (key) =>
      !HIDDEN_KEYS.includes(key) &&
      !IGNORE_KEYS.includes(key) &&
      !rows.every((row) => isEmptyValue(row[key]))
  )

  return {
    resultKeys: visible.filter((key) => RESULT_KEYS.includes(key) && differing.has(key)),
    differingKeys: visible.filter((key) => !RESULT_KEYS.includes(key) && differing.has(key)),
    sameKeys: visible.filter((key) => !differing.has(key)),
  }
}
