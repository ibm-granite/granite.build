/**
 * The train/validation record counts an auto-split upload will get, computed the
 * way the server does it (`_validation_indices` in services/datasets_io.py):
 * validation is `round(total * validationPct / 100)` -- Python's round, half to
 * even -- and train is the rest.
 *
 * Upload and Review used to compute this separately, flooring and rounding
 * respectively, and disagreed with each other for most totals.
 */
export function splitCounts(total: number, validationPct: number): { train: number; validation: number } {
  const exact = (total * validationPct) / 100
  let validation = Math.round(exact)
  if (Math.abs(exact % 1) === 0.5 && validation % 2 !== 0) validation -= 1
  return { train: total - validation, validation }
}
