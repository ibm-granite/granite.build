// Which hyperparameters the trials table shows as columns, what their headers read,
// and how their values format.
//
// Kept free of React and of runtime imports so it can be unit-tested — the same
// split trialMetrics.ts, trialsRadar.ts, trialProgress.ts and trialCompareGrouping.ts
// already make. The no-imports part is load-bearing, not stylistic: every
// unit-tested module here is require()d directly by its test, and `node --test`
// resolves extensionless relative specifiers only for imports that type stripping
// erases, so a value import from a sibling fails at load with ERR_MODULE_NOT_FOUND.
// `tsc` also rejects an explicit `./x.ts` specifier without
// allowImportingTsExtensions. trialsRadar.ts carries primaryMetric/bestTrialId for
// this same reason.

// Below this magnitude, or at/above the upper bound, a number reads better in
// exponential form: a learning-rate column of 0.000001 vs 0.000003 is far harder to
// scan than 1e-6 vs 3e-6.
const EXPONENTIAL_BELOW = 1e-4
const EXPONENTIAL_AT_OR_ABOVE = 1e5

/**
 * Display text for a hyperparameter value.
 *
 * Zero is special-cased ahead of the exponential rule: it is below the lower bound,
 * so it would otherwise render as "0e+0", and `lora_dropout: 0` is a real value.
 *
 * A non-finite number falls through to String(), matching what `formatCell` already
 * does for every other column — a NaN in a config is a data problem and should look
 * like one rather than like "not reported".
 */
export function formatHyperparamValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) return value.map(formatHyperparamValue).join(', ')
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value)
    if (value === 0) return '0'
    const magnitude = Math.abs(value)
    return magnitude < EXPONENTIAL_BELOW || magnitude >= EXPONENTIAL_AT_OR_ABOVE
      ? value.toExponential()
      : String(value)
  }
  return String(value)
}
