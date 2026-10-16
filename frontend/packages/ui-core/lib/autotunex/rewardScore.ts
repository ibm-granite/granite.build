/**
 * The score a reward function's test-run return value stands for, or null when
 * verl could not use it.
 *
 * The backend passes the return value through untouched (int, float, str, bool,
 * list or dict). verl takes a float, or a dict carrying a numeric "score" -- the
 * form the default template's docstring mentions -- and Python treats a bool as
 * 0/1. Anything else would fail at training time, so the test run must not count
 * it as a pass.
 */
export function rewardScore(value: unknown): number | null {
  if (typeof value === 'boolean') return Number(value)
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    const score = (value as Record<string, unknown>).score
    if (typeof score === 'number' && Number.isFinite(score)) return score
  }
  return null
}

/** Why `rewardScore` rejected a value, naming the Python type that came back. */
export function rewardScoreError(value: unknown): string {
  const pyType =
    value === null || value === undefined
      ? 'None'
      : Array.isArray(value)
        ? 'list'
        : typeof value === 'object'
          ? 'dict'
          : typeof value === 'string'
            ? 'str'
            : typeof value
  return `The reward must be a number, or a dict with a numeric "score" key (got ${pyType}).`
}
