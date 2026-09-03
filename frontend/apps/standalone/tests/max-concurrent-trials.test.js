/**
 * Regression test for the "Num GPUs per trial" → "Max concurrent trials"
 * derivation in the Step 2 config editor (General tab and the generic
 * num_gpus_per_trial field).
 *
 * Carbon's NumberInput has no `allowEmpty` here, so clearing the field reports
 * Number('') === 0. The handlers divided by that value unguarded:
 * Math.floor(max_val / 0) === Infinity was written into
 * tune_config.max_concurrent_trials.default, the helper text read "Value must be
 * between 1 and Infinity", and because JSON.stringify(Infinity) === 'null' the
 * saved configuration posted max_concurrent_trials.default: null.
 *
 * Usage: node --test tests/max-concurrent-trials.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { maxConcurrentTrialsCap } = require('../../../packages/ui-core/lib/autotunex/hyperparamValues.ts')

describe('maxConcurrentTrialsCap', () => {
  it('divides the GPU budget by the per-trial size', () => {
    assert.equal(maxConcurrentTrialsCap(8, 2), 4)
    assert.equal(maxConcurrentTrialsCap(8, 1), 8)
  })

  it('floors a non-integral result', () => {
    assert.equal(maxConcurrentTrialsCap(8, 3), 2)
  })

  it('returns 1 — never Infinity — when the field is cleared to 0', () => {
    const cap = maxConcurrentTrialsCap(8, 0)
    assert.equal(cap, 1)
    assert.ok(Number.isFinite(cap), 'must stay finite')
  })

  it('survives a JSON round-trip as a number, not null', () => {
    // The actual saved-config corruption: JSON.stringify(Infinity) === 'null'.
    const saved = JSON.parse(JSON.stringify({ default: maxConcurrentTrialsCap(8, 0) }))
    assert.equal(saved.default, 1)
  })

  it('returns 1 for NaN or negative input', () => {
    assert.equal(maxConcurrentTrialsCap(8, NaN), 1)
    assert.equal(maxConcurrentTrialsCap(8, -2), 1)
    assert.equal(maxConcurrentTrialsCap(NaN, 2), 1)
  })

  it('never drops below 1 when a trial needs more GPUs than exist', () => {
    assert.equal(maxConcurrentTrialsCap(4, 8), 1)
  })
})
