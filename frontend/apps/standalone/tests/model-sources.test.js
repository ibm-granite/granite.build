/**
 * The PVC (`dmf`) model source was retired. New tunings can only be launched
 * against Huggingface or Local, but jobs already in the database still carry
 * `dmf` and must keep reading "PVC" in the detail view.
 *
 * Usage: node --test tests/model-sources.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  MODEL_SOURCE_LABELS,
  MODEL_SOURCE_OPTIONS,
  modelSourceLabel,
} = require('../app/dashboard/autotunex/modelSources.ts')

describe('selectable model sources', () => {
  it('offers Huggingface and Local, in that order', () => {
    assert.deepEqual(
      MODEL_SOURCE_OPTIONS.map((o) => o.value),
      ['huggingface', 'custom_path'],
    )
  })

  it('does not offer PVC', () => {
    assert.ok(!MODEL_SOURCE_OPTIONS.some((o) => o.value === 'dmf'), 'dmf is still selectable')
    assert.ok(!('dmf' in MODEL_SOURCE_LABELS), 'dmf is still a selectable label')
  })

  it('gives every selectable source a unique radio id', () => {
    const ids = MODEL_SOURCE_OPTIONS.map((o) => o.id)
    assert.equal(new Set(ids).size, ids.length)
  })
})

describe('modelSourceLabel', () => {
  it('labels the sources that are still selectable', () => {
    assert.equal(modelSourceLabel('huggingface'), 'Huggingface')
    assert.equal(modelSourceLabel('custom_path'), 'Local')
  })

  it('still labels jobs launched against the retired PVC source', () => {
    assert.equal(modelSourceLabel('dmf'), 'PVC')
  })

  it('falls back to the raw value for anything else', () => {
    assert.equal(modelSourceLabel('some_future_backend'), 'some_future_backend')
  })
})
