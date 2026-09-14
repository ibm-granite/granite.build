/**
 * Tests for reconciling a restored wizard draft with what still exists.
 *
 * The autosave wrote a draft on every field change but nothing ever read it, so a
 * mid-wizard reload lost everything. Wiring the restore up is only safe if it also
 * handles the two things a draft cannot carry:
 *
 *  - An uploaded File is not serializable, so a draft that reached Step 2 by
 *    uploading has no dataset behind it. Only a draft that picked an EXISTING
 *    dataset may treat Step 1 as done.
 *  - A draft lives up to 24 hours, so the dataset or configuration it names may
 *    have been deleted since.
 *
 * Restoring either blindly would mark a step complete with nothing behind it and
 * let the user walk to Review and launch against a dead id.
 *
 * Usage: node --test tests/wizard-draft.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { resolveDraft } = require('../app/dashboard/autotunex/start-tuning/wizardDraft.ts')

const draftAt = (overrides = {}) => ({
  savedAt: new Date().toISOString(),
  currentStep: 3,
  completedSteps: [true, true, true, false, false],
  selectedGoal: 'sft',
  selectedAlgorithm: 'lora',
  selectedModel: 'ibm-granite/granite-4.0-h-micro',
  modelSource: 'huggingface',
  datasetForm: { name: 'my-set', description: '' },
  existingDatasetId: 'ds-1',
  splitRatio: 80,
  selectedConfigId: 'cfg-1',
  experimentName: 'run-1',
  ...overrides,
})

describe('resolveDraft', () => {
  it('restores a fully resolvable draft untouched', () => {
    const { draft, notes } = resolveDraft(draftAt(), { dataset: true, config: true })
    assert.equal(draft.currentStep, 3)
    assert.deepEqual(draft.completedSteps, [true, true, true, false, false])
    assert.equal(draft.existingDatasetId, 'ds-1')
    assert.equal(draft.selectedConfigId, 'cfg-1')
    assert.deepEqual(notes, [])
  })

  it('rewinds to Step 2 when the configuration is gone', () => {
    const { draft, notes } = resolveDraft(draftAt(), { dataset: true, config: false })
    assert.equal(draft.selectedConfigId, null)
    assert.equal(draft.currentStep, 2)
    assert.deepEqual(draft.completedSteps, [true, true, false, false, false])
    assert.equal(draft.existingDatasetId, 'ds-1', 'the dataset is still fine')
    assert.match(notes.join(' '), /configuration .* no longer exists/)
  })

  it('rewinds to Step 1 when the dataset is gone', () => {
    const { draft, notes } = resolveDraft(draftAt(), { dataset: false, config: true })
    assert.equal(draft.existingDatasetId, null)
    assert.equal(draft.currentStep, 1)
    assert.deepEqual(draft.completedSteps, [true, false, false, false, false])
    assert.match(notes.join(' '), /dataset .* no longer exists/)
  })

  it('rewinds to Step 1 when Step 1 was completed by uploading a file', () => {
    // The File cannot be serialized, so there is nothing to launch against.
    const { draft, notes } = resolveDraft(draftAt({ existingDatasetId: null }), {
      dataset: false,
      config: true,
    })
    assert.equal(draft.currentStep, 1)
    assert.equal(draft.completedSteps[1], false)
    assert.match(notes.join(' '), /uploaded file cannot be saved/)
  })

  it('takes the lowest rewind when both references are gone', () => {
    const { draft, notes } = resolveDraft(draftAt(), { dataset: false, config: false })
    assert.equal(draft.currentStep, 1)
    assert.deepEqual(draft.completedSteps, [true, false, false, false, false])
    assert.equal(notes.length, 2)
  })

  it('notes an unsaved config, which saveDraft stores as null', () => {
    // selectedConfigId === '__pending__' is written as null: it has no id to return to.
    const { draft, notes } = resolveDraft(draftAt({ selectedConfigId: null }), {
      dataset: true,
      config: false,
    })
    assert.equal(draft.currentStep, 2)
    assert.match(notes.join(' '), /had not saved yet/)
  })

  it('stays quiet about steps the draft never claimed to complete', () => {
    // A draft abandoned on Step 0 has no dataset or config by definition; saying so
    // would be noise, not information.
    const early = draftAt({
      currentStep: 0,
      completedSteps: [false, false, false, false, false],
      existingDatasetId: null,
      selectedConfigId: null,
    })
    const { draft, notes } = resolveDraft(early, { dataset: false, config: false })
    assert.equal(draft.currentStep, 0)
    assert.deepEqual(notes, [])
  })

  it('does not mutate the draft it was given', () => {
    const original = draftAt()
    const snapshot = JSON.parse(JSON.stringify(original))
    resolveDraft(original, { dataset: false, config: false })
    assert.deepEqual(original, snapshot)
  })
})
