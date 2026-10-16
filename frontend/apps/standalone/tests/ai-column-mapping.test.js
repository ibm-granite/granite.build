/**
 * Tests for the AI column-mapping normalisation shared by the Upload tab and the
 * HuggingFace import.
 *
 * The AI returns keys from its own vocabulary ("input_col") against a dataset-types
 * dict keyed the same way, while the wizard's mapping state is keyed by target
 * column name ("input"). Every case below is one this translation gets silently
 * wrong if it regresses -- and a wrong mapping is not visible until a tuning run
 * has already consumed the projected dataset.
 *
 * Usage: node --test tests/ai-column-mapping.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  aiMappingToColumnMapping,
  adoptedAlgorithm,
} = require('../../../packages/ui-core/lib/autotunex/aiColumnMapping.ts')

// Mirrors the shape of /autotune_dataset_types' `columns` entry for an SFT type.
const SFT_DICT = {
  input_col: { name: 'input', required: true },
  output_col: { name: 'output', required: true },
}
const SFT_TARGETS = ['input', 'output']

describe('aiMappingToColumnMapping', () => {
  it('translates a dict key to its target column name', () => {
    const { mapping, suggestedFields } = aiMappingToColumnMapping(
      { input_col: 'instruction', output_col: 'response' },
      ['instruction', 'response'],
      { targetColumns: SFT_TARGETS, columnsDict: SFT_DICT }
    )
    assert.deepEqual(mapping, { input: 'instruction', output: 'response' })
    assert.deepEqual([...suggestedFields].sort(), ['input', 'output'])
  })

  it('falls back to stripping a trailing _col when the dict has no such key', () => {
    const { mapping } = aiMappingToColumnMapping(
      { input_col: 'instruction' },
      ['instruction'],
      { targetColumns: SFT_TARGETS, columnsDict: {} }
    )
    assert.deepEqual(mapping, { input: 'instruction' })
  })

  it('accepts a key that is already a target column name', () => {
    const { mapping } = aiMappingToColumnMapping(
      { output: 'response' },
      ['response'],
      { targetColumns: SFT_TARGETS, columnsDict: {} }
    )
    assert.deepEqual(mapping, { output: 'response' })
  })

  it('drops an entry whose source column is absent from the data', () => {
    const { mapping, suggestedFields } = aiMappingToColumnMapping(
      { input_col: 'instruction', output_col: 'no_such_column' },
      ['instruction'],
      { targetColumns: SFT_TARGETS, columnsDict: SFT_DICT }
    )
    assert.deepEqual(mapping, { input: 'instruction' })
    assert.deepEqual([...suggestedFields], ['input'])
  })

  it('drops an entry with an empty source column', () => {
    const { mapping } = aiMappingToColumnMapping(
      { input_col: '', output_col: 'response' },
      ['response'],
      { targetColumns: SFT_TARGETS, columnsDict: SFT_DICT }
    )
    assert.deepEqual(mapping, { output: 'response' })
  })

  it('drops a target outside the algorithm target list', () => {
    // The AI suggested DPO keys while the selected algorithm is still SFT.
    const { mapping } = aiMappingToColumnMapping(
      { chosen: 'good', rejected: 'bad' },
      ['good', 'bad'],
      { targetColumns: SFT_TARGETS, columnsDict: SFT_DICT }
    )
    assert.deepEqual(mapping, {})
  })

  it('returns an empty mapping when every entry filters out, so the caller can fall back', () => {
    const { mapping, suggestedFields } = aiMappingToColumnMapping(
      { mystery_key: 'nope' },
      ['instruction'],
      { targetColumns: SFT_TARGETS, columnsDict: SFT_DICT }
    )
    assert.deepEqual(mapping, {})
    assert.equal(suggestedFields.size, 0)
  })

  it('uses the target list it is given, so a changed algorithm maps its own targets', () => {
    const dpoDict = {
      prompt_col: { name: 'prompt' },
      chosen_col: { name: 'chosen' },
      rejected_col: { name: 'rejected' },
    }
    const { mapping } = aiMappingToColumnMapping(
      { prompt_col: 'question', chosen_col: 'good', rejected_col: 'bad' },
      ['question', 'good', 'bad'],
      { targetColumns: ['prompt', 'chosen', 'rejected'], columnsDict: dpoDict }
    )
    assert.deepEqual(mapping, { prompt: 'question', chosen: 'good', rejected: 'bad' })
  })

  it('ignores a dict entry with no name rather than mapping to undefined', () => {
    const { mapping } = aiMappingToColumnMapping(
      { input_col: 'instruction' },
      ['instruction'],
      { targetColumns: SFT_TARGETS, columnsDict: { input_col: {} } }
    )
    // No name in the dict, so the _col-stripping fallback resolves it.
    assert.deepEqual(mapping, { input: 'instruction' })
  })

  it('returns an empty mapping for an empty AI response', () => {
    const { mapping } = aiMappingToColumnMapping({}, ['a'], {
      targetColumns: SFT_TARGETS,
      columnsDict: SFT_DICT,
    })
    assert.deepEqual(mapping, {})
  })
})

describe('adoptedAlgorithm', () => {
  const ALGOS = [
    { id: 'lora', category: 'instruction' },
    { id: 'sft', category: 'instruction' },
    { id: 'dpo', category: 'preference' },
  ]

  it('does not adopt a dataset-type key, which is what the endpoint actually returns', () => {
    // The real regression: with selectedGoal null this used to write
    // "dataset_type_a" straight into the selected algorithm.
    assert.equal(
      adoptedAlgorithm({
        tuningType: 'dataset_type_a',
        current: 'lora',
        selectedGoal: null,
        algorithms: ALGOS,
      }),
      'lora'
    )
  })

  it('does not adopt a dataset-type key when a goal is set either', () => {
    assert.equal(
      adoptedAlgorithm({
        tuningType: 'dataset_type_a',
        current: 'lora',
        selectedGoal: 'instruction',
        algorithms: ALGOS,
      }),
      'lora'
    )
  })

  it('adopts a real algorithm id whose category matches the goal', () => {
    assert.equal(
      adoptedAlgorithm({
        tuningType: 'sft',
        current: 'lora',
        selectedGoal: 'instruction',
        algorithms: ALGOS,
      }),
      'sft'
    )
  })

  it('refuses a real algorithm id whose category does not match the goal', () => {
    assert.equal(
      adoptedAlgorithm({
        tuningType: 'dpo',
        current: 'lora',
        selectedGoal: 'instruction',
        algorithms: ALGOS,
      }),
      'lora'
    )
  })

  it('adopts a real algorithm id when no goal constrains it', () => {
    assert.equal(
      adoptedAlgorithm({
        tuningType: 'dpo',
        current: 'lora',
        selectedGoal: null,
        algorithms: ALGOS,
      }),
      'dpo'
    )
  })

  it('keeps the current algorithm when the suggestion names nothing', () => {
    assert.equal(
      adoptedAlgorithm({
        tuningType: undefined,
        current: 'lora',
        selectedGoal: 'instruction',
        algorithms: ALGOS,
      }),
      'lora'
    )
    assert.equal(
      adoptedAlgorithm({ tuningType: '', current: 'lora', selectedGoal: null, algorithms: ALGOS }),
      'lora'
    )
  })
})
