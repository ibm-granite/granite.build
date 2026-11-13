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
 * Tests for the shared preview-table cell/header helpers.
 *
 * Extracted out of PreviewTable.tsx into a plain .ts module (no JSX, no
 * value imports) so node --test can load it directly -- node --test cannot
 * require a .tsx file.
 *
 * Usage: node --test tests/preview-cell.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { previewCellText, derivePreviewHeaders } = require('../../../packages/ui-core/lib/autotunex/previewCell.ts')

describe('previewCellText', () => {
  it('renders null and undefined as empty', () => {
    assert.equal(previewCellText(null), '')
    assert.equal(previewCellText(undefined), '')
  })

  it('renders falsy-but-present values as themselves, not empty', () => {
    // Guards against a future "simplification" to `!value ? '' : ...`, which
    // would also blank out 0 and false.
    assert.equal(previewCellText(0), '0')
    assert.equal(previewCellText(false), 'false')
  })

  it('renders a string as-is and an object as JSON', () => {
    assert.equal(previewCellText('a'), 'a')
    assert.equal(previewCellText({ a: 1 }), '{"a":1}')
  })

  it('truncates to maxCellChars when given, and not otherwise', () => {
    const long = 'x'.repeat(10)
    assert.equal(previewCellText(long, 3), 'xxx...')
    assert.equal(previewCellText(long), long)
  })
})

describe('derivePreviewHeaders', () => {
  it('derives the union of keys across ragged rows, not just row 0', () => {
    const headers = derivePreviewHeaders([{ a: 1 }, { b: 2 }])
    assert.deepEqual(headers.map((h) => h.key), ['a', 'b'])
  })
})
