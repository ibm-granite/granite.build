/**
 * Regression test for the submit gate on the configuration forms.
 *
 * The forms validated only the configuration NAME. Every numeric control showed
 * its own range error and still wrote the value through: TimeInput calls
 * onChange even when its own `isInvalid` is true, so entering a time budget of
 * 500 hours turned the field red, left Create enabled, and posted
 * time_budget_s.default = 1800000 against a template max of 1209600 (2 weeks).
 *
 * The bounds are the template's own min_val/max_val, so this needs no constants
 * of its own and covers every numeric column rather than just the time budget.
 *
 * Usage: node --test tests/config-range-validation.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { findOutOfRangeFields } = require('../../../packages/ui-core/lib/autotunex/hyperparamValues.ts')

const TEMPLATE = {
  tune_config: {
    num_samples: { default: 10, min_val: 1, max_val: 100, type: 'int' },
    time_budget_s: { default: null, min_val: 60, max_val: 1209600, type: 'int' },
    search_alg: { default: 'random', options: ['random', 'bayesopt'], type: 'string' },
  },
  training_config: {
    num_train_epochs: { default: 3, min_val: 1, max_val: 100, type: 'int' },
    model_name_or_path: 'ibm-granite/granite-4.0-h-micro',
  },
}

const clone = () => JSON.parse(JSON.stringify(TEMPLATE))

describe('findOutOfRangeFields', () => {
  it('passes a config straight from the template', () => {
    assert.deepEqual(findOutOfRangeFields(TEMPLATE), [])
  })

  it('accepts a null default as "unset", not as out of range', () => {
    // An unset time budget means "no limit" and is a valid, expected state.
    assert.deepEqual(findOutOfRangeFields(TEMPLATE), [])
  })

  it('reports a value above its own max', () => {
    const config = clone()
    // 500 hours in seconds, against the template's two-week ceiling.
    config.tune_config.time_budget_s.default = 1800000
    assert.deepEqual(findOutOfRangeFields(config), ['tune_config.time_budget_s'])
  })

  it('reports a value below its own min', () => {
    const config = clone()
    config.tune_config.num_samples.default = 0
    assert.deepEqual(findOutOfRangeFields(config), ['tune_config.num_samples'])
  })

  it('reports every offending field, in order', () => {
    const config = clone()
    config.tune_config.num_samples.default = 500
    config.training_config.num_train_epochs.default = -1
    assert.deepEqual(findOutOfRangeFields(config), [
      'tune_config.num_samples',
      'training_config.num_train_epochs',
    ])
  })

  it('accepts the boundary values themselves', () => {
    const config = clone()
    config.tune_config.num_samples.default = 1
    config.training_config.num_train_epochs.default = 100
    assert.deepEqual(findOutOfRangeFields(config), [])
  })

  it('ignores a NaN default, which the range check cannot judge', () => {
    // Carbon reports Number('') as NaN for a cleared field; the field's own
    // invalid state covers that, and claiming "out of range" would be wrong.
    const config = clone()
    config.tune_config.num_samples.default = Number.NaN
    assert.deepEqual(findOutOfRangeFields(config), [])
  })

  it('ignores scalars, strings and missing bounds', () => {
    assert.deepEqual(findOutOfRangeFields({ s: { a: 'text', b: 7, c: null } }), [])
    assert.deepEqual(findOutOfRangeFields({}), [])
  })

  it('tolerates a null or undefined config', () => {
    assert.deepEqual(findOutOfRangeFields(null), [])
    assert.deepEqual(findOutOfRangeFields(undefined), [])
  })
})
