// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the Builds search helpers (packages/ui-core/lib/buildSearch.ts):
 * turning a search term into gbserver's `name` wildcard, recognising a build id
 * for direct lookup, and the page clamp list views share.
 *
 * Usage: node --test tests/build-search.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  clampPage,
  isBuildId,
  nameContainsWildcard,
  pageSlice,
} = require('../../../packages/ui-core/lib/buildSearch.ts')

describe('nameContainsWildcard', () => {
  it('wraps the term in * so gbserver matches names containing it', () => {
    assert.equal(nameContainsWildcard('sft'), '*sft*')
  })

  it('trims surrounding spaces', () => {
    assert.equal(nameContainsWildcard('  sft '), '*sft*')
  })

  it('escapes * and \\ so they match literally', () => {
    assert.equal(nameContainsWildcard('a*b'), '*a\\*b*')
    assert.equal(nameContainsWildcard('back\\slash'), '*back\\\\slash*')
  })

  it('leaves % and _ alone; gbserver treats them as literal', () => {
    assert.equal(nameContainsWildcard('100%_x'), '*100%_x*')
  })
})

describe('isBuildId', () => {
  it('recognises a build UUID, in either case and with surrounding spaces', () => {
    assert.ok(isBuildId('3f2b9c1e-8a4d-4c2b-9f1e-0a1b2c3d4e5f'))
    assert.ok(isBuildId(' 3F2B9C1E-8A4D-4C2B-9F1E-0A1B2C3D4E5F '))
  })

  it('rejects names and partial ids, which go to the name search', () => {
    for (const term of ['granite-sft', '3f2b9c1e', '3f2b9c1e-8a4d-4c2b-9f1e', '']) {
      assert.equal(isBuildId(term), false, term)
    }
  })
})

describe('pageSlice and clampPage', () => {
  it('slices 1-based pages', () => {
    assert.deepEqual(pageSlice([1, 2, 3, 4, 5], 2, 2), [3, 4])
    assert.deepEqual(pageSlice([1, 2, 3, 4, 5], 3, 2), [5])
  })

  it('keeps a page that is still valid', () => {
    assert.equal(clampPage(3, 100, 20), 3)
  })

  it('moves to the last page when the list shrinks under the current one', () => {
    // DatasetList on page 5 when a search, `days` or `completedOnly` leaves 23 rows.
    assert.equal(clampPage(5, 23, 10), 3)
  })

  it('lands on page 1 for an empty list or a bad page', () => {
    assert.equal(clampPage(4, 0, 20), 1)
    assert.equal(clampPage(0, 50, 20), 1)
  })
})
