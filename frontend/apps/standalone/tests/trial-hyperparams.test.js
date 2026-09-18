/**
 * Tests for the trials table's hyperparameter columns.
 *
 * Columns come from each trial's `config.tuner_flags` — the tuner's own record of
 * which hyperparameters it searched (`for_tuner` in autotune.yaml, see
 * autotunex/src/fm-tune/autotune/config.py:152) — rather than from guessing which
 * config values differ across trials.
 *
 * Kept in a pure module because the frontend test harness has no jsdom and cannot
 * render Carbon components, the same split trialMetrics.ts, trialsRadar.ts,
 * trialProgress.ts and trialCompareGrouping.ts already make.
 *
 * Usage: node --test tests/trial-hyperparams.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  formatHyperparamValue,
  hyperparamColumnLabel,
  searchedHyperparams,
} = require('../../../packages/ui-core/components/autotunex/trials/trialHyperparams.ts')

describe('formatHyperparamValue', () => {
  it('renders zero as "0", not in exponential form', () => {
    // `lora_dropout: 0` is a real value. Zero is below the 1e-4 threshold, so an
    // unguarded exponential rule renders it "0e+0".
    assert.equal(formatHyperparamValue(0), '0')
  })

  it('renders a small learning rate in exponential form', () => {
    // The point of the rule: 0.000001 vs 0.000003 is far harder to scan in a
    // column than 1e-6 vs 3e-6.
    assert.equal(formatHyperparamValue(0.000001), '1e-6')
    assert.equal(formatHyperparamValue(0.00009), '9e-5')
  })

  it('keeps ordinary magnitudes plain', () => {
    assert.equal(formatHyperparamValue(0.0001), '0.0001')
    assert.equal(formatHyperparamValue(0.05), '0.05')
    assert.equal(formatHyperparamValue(0.1), '0.1')
    assert.equal(formatHyperparamValue(1), '1')
    assert.equal(formatHyperparamValue(8), '8')
    assert.equal(formatHyperparamValue(99999), '99999')
  })

  it('switches to exponential at or above 1e5', () => {
    assert.equal(formatHyperparamValue(100000), '1e+5')
  })

  it('applies the threshold to the magnitude, so negatives behave the same', () => {
    assert.equal(formatHyperparamValue(-0.000002), '-2e-6')
    assert.equal(formatHyperparamValue(-0.05), '-0.05')
  })

  it('renders strings verbatim and booleans as words', () => {
    assert.equal(formatHyperparamValue('none'), 'none')
    assert.equal(formatHyperparamValue('linear'), 'linear')
    assert.equal(formatHyperparamValue(true), 'true')
    assert.equal(formatHyperparamValue(false), 'false')
  })

  it('formats array elements individually and joins them', () => {
    assert.equal(formatHyperparamValue([1, 2]), '1, 2')
    assert.equal(formatHyperparamValue([0.000001, 'x']), '1e-6, x')
  })

  it('renders a missing value as an em dash, matching formatCell', () => {
    assert.equal(formatHyperparamValue(null), '—')
    assert.equal(formatHyperparamValue(undefined), '—')
  })

  it('passes a non-finite number through rather than hiding it', () => {
    // formatCell's fallback is String(value), so NaN surfaces as "NaN" in every
    // other column. A data problem should look like one, not like "not reported".
    assert.equal(formatHyperparamValue(Number.NaN), 'NaN')
    assert.equal(formatHyperparamValue(Number.POSITIVE_INFINITY), 'Infinity')
  })
})

describe('hyperparamColumnLabel', () => {
  it('uses the curated header for each known hyperparameter', () => {
    // Curated because column headers want shorter text than the raw key gives:
    // "Per device train batch size" is a very wide column for "Batch size".
    assert.equal(hyperparamColumnLabel('learning_rate'), 'Learning rate')
    assert.equal(hyperparamColumnLabel('per_device_train_batch_size'), 'Batch size')
    assert.equal(hyperparamColumnLabel('gradient_accumulation_steps'), 'Grad accum steps')
    assert.equal(hyperparamColumnLabel('lr_scheduler_type'), 'LR scheduler')
    assert.equal(hyperparamColumnLabel('lora_dropout'), 'LoRA dropout')
    assert.equal(hyperparamColumnLabel('alpha_ratio'), 'Alpha ratio')
    assert.equal(hyperparamColumnLabel('warmup_ratio'), 'Warmup ratio')
    assert.equal(hyperparamColumnLabel('r'), 'Rank (r)')
    assert.equal(hyperparamColumnLabel('bias'), 'Bias')
  })

  it('falls back to a readable form for a key added upstream', () => {
    // So a new hyperparameter in autotune.yaml gets a sane header with no code change.
    assert.equal(hyperparamColumnLabel('some_new_knob'), 'Some new knob')
    assert.equal(hyperparamColumnLabel('beta'), 'Beta')
  })

  it('returns an empty string for an empty key rather than throwing', () => {
    assert.equal(hyperparamColumnLabel(''), '')
  })
})

// A trial shaped like the real payload: hyperparameters at the TOP LEVEL of config,
// with the fixed run config in nested sections. The driver contract
// (autotunex/src/fm-tune/CLAUDE.md:79) pops training_config, training_rl_config,
// tuner_flags and tune_config, and what remains at the top level is the
// hyperparameter set.
const trial = (id, hyperparams, flags) => ({
  id,
  status: 'completed',
  metrics: {},
  config: {
    ...hyperparams,
    training_config: { output_dir: '/tmp/x', model_name_or_path: '/models/m' },
    tune_config: { metric: 'loss', mode: 'min' },
    ...(flags === undefined ? {} : { tuner_flags: flags }),
  },
})

describe('searchedHyperparams', () => {
  it('returns only the keys tuner_flags marks true', () => {
    const t = trial(
      'a',
      { r: 8, learning_rate: 0.000001, bias: 'none' },
      { r: true, learning_rate: false, bias: true }
    )
    assert.deepEqual(searchedHyperparams([t]), ['r', 'bias'])
  })

  it('works from a single trial — it needs no cross-trial comparison', () => {
    // This is why tuner_flags beats "which values differ": a one-trial job still
    // knows what the tuner searched.
    const t = trial('a', { r: 8 }, { r: true })
    assert.deepEqual(searchedHyperparams([t]), ['r'])
  })

  it('unions across trials so a partial config cannot drop a column', () => {
    // Both keys are ranked, so the priority order decides: r (rank 3) before
    // warmup_ratio (rank 5), regardless of which trial contributed which.
    const a = trial('a', { r: 8 }, { r: true })
    const b = trial('b', { warmup_ratio: 0.1 }, { warmup_ratio: true })
    assert.deepEqual(searchedHyperparams([a, b]), ['r', 'warmup_ratio'])
  })

  it('keeps unranked keys in first-seen order behind the ranked ones', () => {
    const t = trial('a', {}, { zeta: true, alpha: true, r: true })
    assert.deepEqual(searchedHyperparams([t]), ['r', 'zeta', 'alpha'])
  })

  it('applies the priority order, with unranked keys after it', () => {
    const t = trial(
      'a',
      {},
      {
        some_new_knob: true,
        r: true,
        learning_rate: true,
        warmup_ratio: true,
        per_device_train_batch_size: true,
      }
    )
    assert.deepEqual(searchedHyperparams([t]), [
      'learning_rate',
      'per_device_train_batch_size',
      'r',
      'warmup_ratio',
      'some_new_knob',
    ])
  })

  it('is unaffected by the order of the trials it is given', () => {
    const a = trial('a', {}, { r: true })
    const b = trial('b', {}, { learning_rate: true })
    assert.deepEqual(searchedHyperparams([a, b]), searchedHyperparams([b, a]))
  })

  it('ignores a truthy-but-not-true flag', () => {
    // The contract is boolean; a strict check stops a malformed payload inventing
    // columns.
    const t = trial('a', { r: 8 }, { r: 1, bias: 'yes', warmup_ratio: true })
    assert.deepEqual(searchedHyperparams([t]), ['warmup_ratio'])
  })

  it('returns an empty list when nothing is flagged or tuner_flags is absent', () => {
    assert.deepEqual(searchedHyperparams([trial('a', { r: 8 }, { r: false })]), [])
    assert.deepEqual(searchedHyperparams([trial('a', { r: 8 }, undefined)]), [])
    assert.deepEqual(searchedHyperparams([]), [])
  })

  it('tolerates a malformed trial rather than throwing', () => {
    assert.deepEqual(searchedHyperparams([{ id: 'a', status: 'error', metrics: {} }]), [])
    assert.deepEqual(searchedHyperparams([{ id: 'a', status: 'error', metrics: {}, config: null }]), [])
    assert.deepEqual(searchedHyperparams([trial('a', {}, null)]), [])
    assert.deepEqual(searchedHyperparams([trial('a', {}, 'nope')]), [])
  })

  it('never returns a key that would overwrite a real table column', () => {
    // The row object spreads hyperparameters after the fixed keys, so a tuner
    // naming a hyperparameter `loss` or `status` would silently replace that column.
    const t = trial(
      'a',
      {},
      { loss: true, status: true, id: true, created_at: true, total_time: true, isSelected: true, r: true }
    )
    assert.deepEqual(searchedHyperparams([t]), ['r'])
  })
})
