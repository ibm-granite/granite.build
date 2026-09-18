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

import type { Trial } from '../../../types'

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

// Column headers, curated because a header wants shorter text than the raw key
// gives — "Per device train batch size" is a very wide column for "Batch size".
// Anything not listed falls back to the generic form, so a hyperparameter added
// upstream still gets a readable header without a code change.
//
// Not shared with `labelForCompareKey` in trialCompareGrouping.ts: that renders a
// whole dotted compare path, it cannot be imported here (see the note at the top of
// this file), and compare rows have room for longer labels than columns do.
const COLUMN_LABELS: Record<string, string> = {
  learning_rate: 'Learning rate',
  per_device_train_batch_size: 'Batch size',
  gradient_accumulation_steps: 'Grad accum steps',
  lr_scheduler_type: 'LR scheduler',
  lora_dropout: 'LoRA dropout',
  alpha_ratio: 'Alpha ratio',
  warmup_ratio: 'Warmup ratio',
  r: 'Rank (r)',
  bias: 'Bias',
}

/** Column header for a hyperparameter key. */
export function hyperparamColumnLabel(key: string): string {
  const curated = COLUMN_LABELS[key]
  if (curated) return curated
  const text = key.replaceAll('_', ' ').trim()
  if (!text) return ''
  return text.charAt(0).toUpperCase() + text.slice(1)
}

// Display order for the hyperparameter columns. A fixed list, not derived from
// variance: a variance-derived order would reshuffle the columns as new trials
// arrive mid-run. Keys not listed follow in first-seen order.
const COLUMN_PRIORITY = [
  'learning_rate',
  'per_device_train_batch_size',
  'r',
  'alpha_ratio',
  'warmup_ratio',
  'lr_scheduler_type',
]

// Row keys TrialsTable builds itself. The row object spreads hyperparameters after
// these, so a tuner naming a hyperparameter `loss` or `status` would silently
// replace a real column — exclude them rather than let that happen.
const RESERVED_ROW_KEYS = ['created_at', 'id', 'status', 'loss', 'total_time', 'isSelected']

/**
 * Hyperparameter keys the tuner searched, in display order.
 *
 * Read from each trial's `config.tuner_flags`, which is the tuner's own record of
 * which hyperparameters it varied — built from the template's `for_tuner` field
 * (`autotune.yaml`, see `autotune/config.py:152`) and used by every driver to
 * "separate tunable params (tuner_flags[k]=True) from fixed (False)".
 *
 * This is authoritative rather than inferred. The alternative — showing the keys
 * whose values differ across trials — is an empirical proxy that cannot work on a
 * single-trial job and that would add and remove columns as a run progresses.
 *
 * Unions across every trial, not just the first: trials in a job share a search
 * space, so in practice the sets agree, but a trial arriving with a partial config
 * must not drop a column another trial justifies.
 */
export function searchedHyperparams(trials: Trial[]): string[] {
  const found: string[] = []
  const seen = new Set<string>()

  for (const trial of trials) {
    const config = trial?.config
    if (!config || typeof config !== 'object') continue
    const flags = (config as Record<string, unknown>).tuner_flags
    if (!flags || typeof flags !== 'object' || Array.isArray(flags)) continue

    for (const [key, isSearched] of Object.entries(flags as Record<string, unknown>)) {
      if (isSearched !== true) continue
      if (seen.has(key) || RESERVED_ROW_KEYS.includes(key)) continue
      seen.add(key)
      found.push(key)
    }
  }

  // First-seen order captured before sorting. `sort` mutates in place, so a
  // comparator that called `found.indexOf` would be reading the array it is
  // reordering — it happens to survive V8's TimSort today, but it should not depend
  // on engine internals.
  const firstSeen = new Map(found.map((key, index) => [key, index]))

  return [...found].sort((a, b) => {
    const rankA = COLUMN_PRIORITY.indexOf(a)
    const rankB = COLUMN_PRIORITY.indexOf(b)
    if (rankA !== -1 && rankB !== -1) return rankA - rankB
    if (rankA !== -1) return -1
    if (rankB !== -1) return 1
    return (firstSeen.get(a) ?? 0) - (firstSeen.get(b) ?? 0)
  })
}
