// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * The PVC (`dmf`) model source was retired. New tunings can only be launched
 * against Hugging Face or Local, but jobs already in the database still carry
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
  toWireModelSource,
} = require('../app/dashboard/autotunex/modelSources.ts')

describe('selectable model sources', () => {
  it('offers Hugging Face, Tuned Models and Local, in that order', () => {
    assert.deepEqual(
      MODEL_SOURCE_OPTIONS.map((o) => o.value),
      ['huggingface', 'tuned_model', 'custom_path'],
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
    assert.equal(modelSourceLabel('huggingface'), 'Hugging Face')
    assert.equal(modelSourceLabel('custom_path'), 'Local')
    assert.equal(modelSourceLabel('tuned_model'), 'Tuned Models')
  })

  it('still labels jobs launched against the retired PVC source', () => {
    assert.equal(modelSourceLabel('dmf'), 'PVC')
  })

  it('falls back to the raw value for anything else', () => {
    assert.equal(modelSourceLabel('some_future_backend'), 'some_future_backend')
  })
})

describe('toWireModelSource', () => {
  it('sends a tuned model as huggingface', () => {
    assert.equal(toWireModelSource('tuned_model'), 'huggingface')
  })

  it('passes the wire sources through', () => {
    assert.equal(toWireModelSource('huggingface'), 'huggingface')
    assert.equal(toWireModelSource('custom_path'), 'custom_path')
  })
})
