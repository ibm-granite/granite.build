/**
 * Regression test for the Start Tuning wizard's launch gate.
 *
 * The last step re-checked only the experiment name. An earlier step re-entered
 * from Review and left invalid -- the model cleared, "Choose Existing" leaving no
 * configuration, the reward code cleared -- kept Review reachable, and Launch
 * created and uploaded the dataset before the server rejected the job.
 *
 * Usage: node --test tests/launch-readiness.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { firstIncompleteStep } = require('../app/dashboard/autotunex/start-tuning/launchReadiness.ts')

const gate = (valid) => (step) => valid[step]

describe('firstIncompleteStep', () => {
  it('is null when every step before the last passes its own gate', () => {
    assert.equal(firstIncompleteStep(gate([true, true, true, false]), 3), null)
  })

  it('names the earliest failing step', () => {
    assert.equal(firstIncompleteStep(gate([true, false, false, true]), 3), 1)
  })

  it('does not judge the last step itself', () => {
    // The last step's own gate is the Launch button's; it is checked there.
    assert.equal(firstIncompleteStep(gate([true, true, true, true, false]), 4), null)
  })

  it('catches a cleared model at step 0', () => {
    assert.equal(firstIncompleteStep(gate([false, true, true, true]), 3), 0)
  })
})
