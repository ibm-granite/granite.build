export interface ValuesParseResult {
  /** Sorted, parsed candidate values — null when the input is not committable. */
  values: number[] | null
  error: boolean
}

/**
 * Parse a hyperparameter "Values" text field (a comma-separated candidate list)
 * for the Step 2 config editor.
 *
 * Pulled out as a pure function because this must run on *commit* (blur) only.
 * Running it per keystroke and refusing to update state on a parse failure pins
 * the controlled input to the last committed array, which makes the field
 * impossible to type in — transient states like "8,16," or "0.0000" are a normal
 * part of typing a list and must not be treated as terminal errors mid-edit.
 *
 * Empty entries are dropped rather than coerced (`Number('')` is `0`, which would
 * spuriously fail a positive `min_val` on every trailing comma).
 */
export function parseValuesInput(raw: string, minVal: number, maxVal: number): ValuesParseResult {
  const nums = raw
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .map(Number)

  const error = nums.length === 0 || nums.some((n) => Number.isNaN(n) || n < minVal || n > maxVal)
  if (error) return { values: null, error: true }

  return { values: [...nums].sort((a, b) => a - b), error: false }
}

/** Render a committed values array the way the source form does: comma-joined, no spaces. */
export function formatValues(values: unknown): string {
  return Array.isArray(values) ? values.join(',') : String(values ?? '')
}
