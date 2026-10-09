/**
 * Tests for the trials table's hyperparameter columns.
 *
 * Columns are every top-level hyperparameter on each trial's `config` —
 * `config.tuner_flags` is ignored entirely, because production data showed its
 * flags do not track which values actually vary (see hyperparamColumns's doc
 * comment).
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
  hyperparamColumns,
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

describe('hyperparamColumns', () => {
  it('returns the full real-payload column set in priority + first-seen order', () => {
    // The regression this whole change fixes: learning_rate, per_device_train_batch_size,
    // alpha_ratio, lr_scheduler_type and warmup_ratio are all flagged false in real
    // tuner_flags payloads yet vary across trials, while bias is flagged true yet is
    // constant. The new rule shows every top-level hyperparameter regardless.
    const t = trial(
      'a',
      {
        alpha_ratio: 0.5,
        bias: 'none',
        gradient_accumulation_steps: 4,
        learning_rate: 0.00001,
        lora_dropout: 0.1,
        lr_scheduler_type: 'linear',
        per_device_train_batch_size: 8,
        r: 16,
        warmup_ratio: 0.03,
      },
      { learning_rate: false, per_device_train_batch_size: false, alpha_ratio: false, lr_scheduler_type: false, warmup_ratio: false, bias: true }
    )
    assert.deepEqual(hyperparamColumns([t]), [
      'learning_rate',
      'per_device_train_batch_size',
      'r',
      'alpha_ratio',
      'warmup_ratio',
      'lr_scheduler_type',
      'bias',
      'gradient_accumulation_steps',
      'lora_dropout',
    ])
  })

  it('excludes the nested config sections, including training_rl_config', () => {
    const t = trial('a', { r: 8 }, { r: true })
    t.config.training_rl_config = { kl_coef: 0.1 }
    assert.deepEqual(hyperparamColumns([t]), ['r'])
  })

  it('ignores tuner_flags values — a flag of false still yields the column', () => {
    // This is the bug being fixed: tuner_flags does not track which values vary,
    // so a hyperparameter flagged false must still show up as a column.
    const t = trial('a', { learning_rate: 0.00001 }, { learning_rate: false })
    assert.deepEqual(hyperparamColumns([t]), ['learning_rate'])
  })

  it('treats a top-level array value as a hyperparameter column', () => {
    const t = trial('a', { target_modules: ['q_proj', 'v_proj'] }, undefined)
    assert.deepEqual(hyperparamColumns([t]), ['target_modules'])
  })

  it('unions across trials so a key present on only one trial still yields a column', () => {
    const a = trial('a', { r: 8 }, undefined)
    const b = trial('b', { warmup_ratio: 0.1 }, undefined)
    assert.deepEqual(hyperparamColumns([a, b]), ['r', 'warmup_ratio'])
  })

  it('applies the priority order, with unranked keys after it in first-seen order', () => {
    const t = trial(
      'a',
      {
        some_new_knob: 1,
        r: 8,
        learning_rate: 0.1,
        warmup_ratio: 0.1,
        per_device_train_batch_size: 8,
      },
      undefined
    )
    assert.deepEqual(hyperparamColumns([t]), [
      'learning_rate',
      'per_device_train_batch_size',
      'r',
      'warmup_ratio',
      'some_new_knob',
    ])
  })

  it('is unaffected by the order of the trials it is given', () => {
    const a = trial('a', { r: 8 }, undefined)
    const b = trial('b', { learning_rate: 0.1 }, undefined)
    assert.deepEqual(hyperparamColumns([a, b]), hyperparamColumns([b, a]))
  })

  it('never returns a reserved row key that would overwrite a real table column', () => {
    // The row object spreads hyperparameters after the fixed keys, so a config
    // naming a hyperparameter `loss` or `status` would silently replace that column.
    const t = trial(
      'a',
      { loss: 1, status: 'x', id: 'y', created_at: 'z', total_time: 1, isSelected: true, r: 8 },
      undefined
    )
    assert.deepEqual(hyperparamColumns([t]), ['r'])
  })

  it('returns an empty list for degenerate inputs rather than throwing', () => {
    assert.deepEqual(hyperparamColumns([]), [])
    assert.deepEqual(hyperparamColumns([{ id: 'a', status: 'error', metrics: {} }]), [])
    assert.deepEqual(hyperparamColumns([{ id: 'a', status: 'error', metrics: {}, config: null }]), [])
    assert.deepEqual(hyperparamColumns([{ id: 'a', status: 'error', metrics: {}, config: 'nope' }]), [])
  })

  it('drops hyperparameters that are constant across all trials, on the real production payload', () => {
    // Verified against the live API: bias renders "none" on every trial and
    // gradient_accumulation_steps renders "1" on every trial — a column that
    // repeats one value on every row costs width and says nothing.
    const trials = [
      trial('491c7_00000', {
        alpha_ratio: 1,
        bias: 'none',
        gradient_accumulation_steps: 1,
        learning_rate: 0.000001,
        lora_dropout: 0,
        lr_scheduler_type: 'linear',
        per_device_train_batch_size: 8,
        r: 8,
        warmup_ratio: 0.1,
      }),
      trial('491c7_00001', {
        alpha_ratio: 1,
        bias: 'none',
        gradient_accumulation_steps: 1,
        learning_rate: 0.000003,
        lora_dropout: 0,
        lr_scheduler_type: 'linear',
        per_device_train_batch_size: 4,
        r: 8,
        warmup_ratio: 0.2,
      }),
      trial('491c7_00002', {
        alpha_ratio: 1,
        bias: 'none',
        gradient_accumulation_steps: 1,
        learning_rate: 0.000005,
        lora_dropout: 0.05,
        lr_scheduler_type: 'linear',
        per_device_train_batch_size: 8,
        r: 16,
        warmup_ratio: 0.2,
      }),
      trial('491c7_00003', {
        alpha_ratio: 2,
        bias: 'none',
        gradient_accumulation_steps: 1,
        learning_rate: 0.000003,
        lora_dropout: 0.05,
        lr_scheduler_type: 'cosine',
        per_device_train_batch_size: 8,
        r: 8,
        warmup_ratio: 0.1,
      }),
    ]
    assert.deepEqual(hyperparamColumns(trials), [
      'learning_rate',
      'per_device_train_batch_size',
      'r',
      'alpha_ratio',
      'warmup_ratio',
      'lr_scheduler_type',
      'lora_dropout',
    ])
  })

  it('drops a key whose value is identical across every trial', () => {
    const a = trial('a', { bias: 'none', r: 8 })
    const b = trial('b', { bias: 'none', r: 16 })
    assert.deepEqual(hyperparamColumns([a, b]), ['r'])
  })

  it('keeps a key that varies between only two of several trials', () => {
    const a = trial('a', { bias: 'none', r: 8 })
    const b = trial('b', { bias: 'none', r: 8 })
    const c = trial('c', { bias: 'none', r: 16 })
    assert.deepEqual(hyperparamColumns([a, b, c]), ['r'])
  })

  it('treats a key absent on one trial but present on another as varying, even if the shared key is constant', () => {
    const a = trial('a', { r: 8 })
    const b = trial('b', { r: 8, warmup_ratio: 0.1 })
    assert.deepEqual(hyperparamColumns([a, b]), ['warmup_ratio'])
  })

  it('returns every hyperparameter for a single trial, since a lone value cannot be compared for variance', () => {
    // Guard: with fewer than two trials every value is trivially identical to
    // itself, so the varies-filter would empty the list. A single trial must
    // return every hyperparameter instead of none.
    const t = trial('a', {
      alpha_ratio: 1,
      bias: 'none',
      learning_rate: 0.00001,
      r: 8,
    })
    assert.deepEqual(hyperparamColumns([t]), ['learning_rate', 'r', 'alpha_ratio', 'bias'])
  })

  it('compares array values by JSON.stringify: a differing array is kept, an identical one is dropped', () => {
    const a = trial('a', { target_modules: ['q_proj', 'v_proj'], seed_list: [1, 2] })
    const b = trial('b', { target_modules: ['q_proj', 'k_proj'], seed_list: [1, 2] })
    assert.deepEqual(hyperparamColumns([a, b]), ['target_modules'])
  })

  it('orders the surviving columns by priority, then by first-seen, once constant keys are dropped', () => {
    const a = trial('a', {
      some_new_knob: 1,
      bias: 'none',
      r: 8,
      learning_rate: 0.1,
      warmup_ratio: 0.1,
      per_device_train_batch_size: 8,
    })
    const b = trial('b', {
      some_new_knob: 2,
      bias: 'none',
      r: 16,
      learning_rate: 0.2,
      warmup_ratio: 0.1,
      per_device_train_batch_size: 8,
    })
    assert.deepEqual(hyperparamColumns([a, b]), ['learning_rate', 'r', 'some_new_knob'])
  })
})
