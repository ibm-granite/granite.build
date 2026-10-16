/**
 * Regression test for the hyperparameter "Values" field in Step 2 → Tuners.
 *
 * The field was effectively read-only: validation ran on every React onChange
 * (keystroke) and the handler bailed out without setting state whenever the text
 * didn't parse, so the controlled input snapped back to the last committed array.
 * Intermediate states that occur while typing a comma-separated list ("8,16,",
 * "0.0000") must therefore be treated as *not yet committable* rather than as
 * hard errors, and parsing must only run on commit (blur).
 *
 * Usage: node --test tests/hyperparam-values.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { parseValuesInput, formatValues, reconcileDefault, liveInvalidFields } = require('../../../packages/ui-core/lib/autotunex/hyperparamValues.ts')

describe('parseValuesInput', () => {
  it('parses and sorts a valid comma-separated list', () => {
    assert.deepEqual(parseValuesInput('32,8,16', 1, 64), { values: [8, 16, 32], error: false })
  })

  it('tolerates spaces around entries', () => {
    assert.deepEqual(parseValuesInput('0.5, 1, 2', 0, 4), { values: [0.5, 1, 2], error: false })
  })

  it('ignores a trailing comma instead of coercing it to 0', () => {
    // Number('') === 0, which would spuriously fail a positive min_val. Typing
    // "8,16," on the way to "8,16,32" must stay valid.
    assert.deepEqual(parseValuesInput('8,16,', 1, 64), { values: [8, 16], error: false })
  })

  it('accepts small floats at the bottom of a range (learning rates)', () => {
    assert.deepEqual(parseValuesInput('0.000001,0.000003', 0.000001, 0.00005), { values: [0.000001, 0.000003], error: false })
  })

  it('flags a value above max', () => {
    assert.deepEqual(parseValuesInput('8,128', 1, 64), { values: null, error: true })
  })

  it('flags a value below min', () => {
    // The AutoTuneX error-state screenshot: 0.0000005 is below the 0.000001 floor.
    assert.deepEqual(parseValuesInput('0.000001,0.000003,0.0000005', 0.000001, 0.00005), { values: null, error: true })
  })

  it('flags non-numeric text', () => {
    assert.deepEqual(parseValuesInput('8,abc', 1, 64), { values: null, error: true })
  })

  it('flags a fully empty field (a hyperparameter needs at least one candidate)', () => {
    assert.deepEqual(parseValuesInput('', 1, 64), { values: null, error: true })
    assert.deepEqual(parseValuesInput('  ,  ', 1, 64), { values: null, error: true })
  })

  it('never mutates the caller-visible order in place', () => {
    const raw = '32,8,16'
    const first = parseValuesInput(raw, 1, 64)
    const second = parseValuesInput(raw, 1, 64)
    assert.deepEqual(first.values, second.values)
  })
})

describe('formatValues', () => {
  it('comma-joins without spaces, matching the source form', () => {
    assert.equal(formatValues([8, 16, 32]), '8,16,32')
  })

  it('round-trips a committed array back through the parser', () => {
    const { values } = parseValuesInput(formatValues([8, 16, 32]), 1, 64)
    assert.deepEqual(values, [8, 16, 32])
  })

  it('handles null/undefined without throwing', () => {
    assert.equal(formatValues(null), '')
    assert.equal(formatValues(undefined), '')
  })
})

describe('reconcileDefault', () => {
  // The default search algorithm (lds/blds) looks the default up with
  // values.index(default), so a default outside a newly committed numeric list
  // raised ValueError when the job started. The string branch already did this.
  it('keeps a default that is still a candidate', () => {
    assert.equal(reconcileDefault([0.0001, 0.0002], 0.0002), 0.0002)
  })

  it('moves a default that is no longer a candidate to the first one', () => {
    assert.equal(reconcileDefault([0.001, 0.002], 0.0001), 0.001)
  })
})

describe('liveInvalidFields', () => {
  // An entry was never removed when its field stopped being rendered, so after
  // switching Tuner type the submit gate kept refusing a field that was not on
  // screen, and only discarding every edit got the user out.
  const errors = {
    'tuners_config-lora-learning_rate': { error: true, message: 'x' },
    'tuners_config-vera-rank': { error: true, message: 'x' },
    'tuners_rl_config-grpo-beta': { error: true, message: 'x' },
    'tuners_config-vera-alpha': { error: false, message: 'x' },
  }

  it('reports only the selected tuners\' fields', () => {
    assert.deepEqual(liveInvalidFields(errors, 'vera', 'grpo'), ['tuners_config-vera-rank', 'tuners_rl_config-grpo-beta'])
  })

  it('drops a field whose tuner is no longer selected', () => {
    assert.deepEqual(liveInvalidFields(errors, 'lora', 'none'), ['tuners_config-lora-learning_rate'])
  })

  it('does not match a tuner whose name only starts the same', () => {
    assert.deepEqual(liveInvalidFields({ 'tuners_config-lora_plus-r': { error: true, message: 'x' } }, 'lora', 'none'), [])
  })
})

