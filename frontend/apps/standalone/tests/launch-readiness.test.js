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

const { firstIncompleteStep, invalidCompletedSteps } = require('../app/dashboard/autotunex/start-tuning/launchReadiness.ts')

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

describe('invalidCompletedSteps', () => {
  it('flags a completed step that no longer passes', () => {
    assert.deepEqual(invalidCompletedSteps([true, true, true, true], gate([false, true, true, true]), 3, 3), [true, false, false, false])
  })

  it('does not flag a step that was never completed', () => {
    assert.deepEqual(invalidCompletedSteps([true, false, false, false], gate([true, false, false, false]), 3, 3), [false, false, false, false])
  })

  it('leaves the step being edited and the last step alone', () => {
    assert.deepEqual(invalidCompletedSteps([true, true, true, true], gate([false, false, true, false]), 3, 1), [true, false, false, false])
  })

  it('agrees with firstIncompleteStep about which step blocks Launch', () => {
    const valid = gate([true, false, true, true])
    const flagged = invalidCompletedSteps([true, true, true, true], valid, 3, 3)
    assert.equal(flagged.indexOf(true), firstIncompleteStep(valid, 3))
  })
})

describe('the launch plan covers a HuggingFace import', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const dir = path.join(__dirname, '..', 'app', 'dashboard', 'autotunex', 'start-tuning')
  const wizard = fs.readFileSync(path.join(dir, 'StartTuningWizard.tsx'), 'utf8')
  const review = fs.readFileSync(path.join(dir, 'steps', 'Step3ReviewLaunch.tsx'), 'utf8')

  it('plans the import from the same condition handleLaunch imports on', () => {
    // An HF launch has no uploadedFile, so a plan keyed only on uploadDataset
    // drew no dataset row at all while the import ran.
    assert.match(wizard, /importHfDataset: !\(datasetId \|\| existingDatasetId\) && !!pendingHfImport,/)
    assert.match(wizard, /uploadDataset: !\(datasetId \|\| existingDatasetId\) && !pendingHfImport && !!uploadedFile,/)
    assert.match(wizard, /if \(!finalDatasetId && pendingHfImport\) \{/)
  })

  it('draws the import row from the plan, not from the upload flag', () => {
    const row = review.indexOf('<span>Import from HuggingFace</span>')
    assert.ok(row > -1, 'the import row should exist')
    const gate = review.lastIndexOf('{launchPlan?.', row)
    assert.equal(review.slice(gate, gate + '{launchPlan?.importHfDataset'.length), '{launchPlan?.importHfDataset')
  })
})

describe('Launch is disabled by an invalid earlier step, not only refused on click', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const wizard = fs.readFileSync(path.join(__dirname, '..', 'app', 'dashboard', 'autotunex', 'start-tuning', 'StartTuningWizard.tsx'), 'utf8')

  it('disables the Launch button on launchBlockedBy', () => {
    assert.match(wizard, /onClick=\{handleLaunch\} disabled=\{!canProceed \|\| isLaunching \|\| launchBlockedBy !== null\}/)
  })
})
