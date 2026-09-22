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

/**
 * Whether a `type: list` field's template default is a numeric list.
 *
 * The live default cannot decide its own type once it has been committed through a
 * string parser, so this is always asked of the *pristine* template value.
 */
export function isNumericList(template: unknown): boolean {
  return Array.isArray(template) && template.length > 0 && template.every((v) => typeof v === 'number')
}

/**
 * Parse a comma-separated `type: list` field whose template default is numeric.
 *
 * The generic list control committed through `parseCommaList`, which returns
 * `string[]`. That is correct for `tokenizer_config` (token strings) but corrupts a
 * numeric list: `tune_config.fidelity_schedule` is `type: list` with
 * `default: [0.1, 0.25, 0.5]`, so merely tabbing through that field rewrote it to
 * `["0.1","0.25","0.5"]` and the backend raised at BLDS init -- `blds.py` does
 * `any(p <= 0.0 or p > 1.0 for p in fidelity_schedule)`, which is a `TypeError`
 * between `str` and `float`. `normalizeTokenizerListFields` repairs only
 * `tokenizer_config`, so nothing caught it.
 *
 * Entries that are not finite numbers are dropped rather than coerced, mirroring
 * how `parseCommaList` already drops empty entries: `NaN` would serialise to `null`
 * and a mixed array fails exactly the way the all-string one did.
 */
export function parseNumericCommaList(value: unknown): number[] | null {
  const tokens = Array.isArray(value)
    ? value.map((v) => String(v))
    : typeof value === 'string'
      ? value.split(',')
      : []
  const nums = tokens
    .map((t) => t.trim())
    .filter((t) => t.length > 0)
    .map(Number)
    .filter((n) => Number.isFinite(n))
  return nums.length > 0 ? nums : null
}

/**
 * Cap for "Max concurrent trials", derived from the GPU budget and the GPUs each
 * trial takes.
 *
 * Guards the divisor. Carbon's `NumberInput` (without `allowEmpty`) reports
 * `Number('') === 0` when the user clears "Num GPUs per trial", so 0 is a normal
 * mid-edit value — and `maxGpus / 0` is `Infinity`, which renders as "Value must
 * be between 1 and Infinity" and, because `JSON.stringify(Infinity)` is
 * `"null"`, saves `max_concurrent_trials.default: null`. A non-positive or
 * non-finite trial size means no meaningful concurrency, so the cap is 1.
 */
export function maxConcurrentTrialsCap(maxGpus: number, gpusPerTrial: number): number {
  if (!Number.isFinite(maxGpus) || !Number.isFinite(gpusPerTrial) || gpusPerTrial <= 0) return 1
  return Math.max(1, Math.floor(maxGpus / gpusPerTrial))
}

/**
 * Next "Max concurrent trials" default after "Num GPUs per trial" changes.
 *
 * Clamps to the new ceiling rather than assigning it: raising GPUs per trial
 * lowers how many can run at once, but a deliberately smaller choice below that
 * ceiling has to stand -- assigning the cap silently raised a chosen 2 to 4 and
 * submitted that.
 *
 * The clamp is monotonically downward, which is why the mid-edit guard lives
 * here rather than at the call site. Carbon's `NumberInput` (without
 * `allowEmpty`) reports a cleared field as `Number('') === 0`, so the ordinary
 * "backspace, then type the new number" interaction fires the handler once with
 * 0. `maxConcurrentTrialsCap(max, 0)` floors to 1, and `Math.min(1, cap)` is 1
 * for every later keystroke -- so one transient 0 pinned concurrency to 1
 * permanently and saved it, with no validation error, because 1 sits inside
 * `[min_val, cap]`. The only symptom was a sweep running ~4x slower than asked.
 *
 * A non-positive or non-finite trial size is a mid-edit state, not a choice:
 * leave the previous value untouched and let the GPU field's own `invalidText`
 * surface the empty input.
 */
export function clampConcurrentTrials(prevDefault: number, maxGpus: number, gpusPerTrial: number): number {
  if (!Number.isFinite(gpusPerTrial) || gpusPerTrial <= 0) return prevDefault
  return Math.max(1, Math.min(prevDefault, maxConcurrentTrialsCap(maxGpus, gpusPerTrial)))
}

/**
 * Flattened paths of every `{default, min_val, max_val}` column whose default
 * falls outside its own bounds.
 *
 * The forms validated only the configuration name, while each numeric control
 * wrote its value through regardless of the range error it was already showing --
 * TimeInput calls onChange even when its own `isInvalid` is true. A 500-hour time
 * budget therefore posted 1800000 against the template's 1209600 ceiling. The
 * bounds come from the config itself, so this needs no constants of its own and
 * covers every numeric column rather than only the one that was reported.
 *
 * A null default is "unset" (an unset time budget means no limit) and NaN is what
 * Carbon reports for a cleared field -- neither is judged here; the field's own
 * invalid state covers those.
 */
export function findOutOfRangeFields(configData: unknown): string[] {
  const offenders: string[] = []

  const walk = (node: unknown, path: string) => {
    if (node === null || typeof node !== 'object' || Array.isArray(node)) return
    const record = node as Record<string, unknown>
    const { default: value, min_val: min, max_val: max } = record
    if (
      typeof value === 'number' &&
      Number.isFinite(value) &&
      typeof min === 'number' &&
      typeof max === 'number' &&
      (value < min || value > max)
    ) {
      offenders.push(path)
      return
    }
    for (const [key, child] of Object.entries(record)) {
      walk(child, path ? `${path}.${key}` : key)
    }
  }

  walk(configData, '')
  return offenders
}
