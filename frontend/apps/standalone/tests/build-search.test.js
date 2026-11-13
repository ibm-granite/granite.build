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
 * Tests for the Builds page's client-side search (packages/ui-core/lib/buildSearch.ts)
 * and the page clamp DatasetList uses.
 *
 * gbserver's `name` filter is exact-match, so while a term is active the page
 * fetches one window of recent builds and matches/paginates it locally. The
 * mode-switch cases cover React Query placeholder data: while the next request
 * loads, the previous result (fetched in the other mode) is still on screen.
 *
 * Usage: node --test tests/build-search.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  buildsView,
  clampPage,
  matchesBuildSearch,
  pageSlice,
} = require('../../../packages/ui-core/lib/buildSearch.ts')

const build = (name, extra = {}) => ({ name, uuid: `id-${name}`, username: 'alice', space_name: 'public', ...extra })
const names = (view) => view.items.map((b) => b.name)

// 30 builds newest-first, as the window request returns them: b29 ... b0.
const WINDOW = Array.from({ length: 30 }, (_, i) => build(`b${29 - i}`))

describe('matchesBuildSearch', () => {
  it('matches name, id, owner and space, case-insensitively', () => {
    const b = build('Granite-SFT', { uuid: 'abc-123', username: 'Bob', space_name: 'Research' })
    for (const term of ['sft', 'SFT', 'abc-1', 'bob', 'research']) assert.ok(matchesBuildSearch(b, term), term)
    assert.equal(matchesBuildSearch(b, 'lora'), false)
  })

  it('treats a blank term as matching everything and ignores surrounding spaces', () => {
    assert.ok(matchesBuildSearch(build('x'), '   '))
    assert.ok(matchesBuildSearch(build('granite-sft'), '  sft '))
  })

  it('tolerates a build without owner or space', () => {
    assert.ok(matchesBuildSearch({ name: 'n', uuid: 'u' }, 'n'))
    assert.equal(matchesBuildSearch({ name: 'n', uuid: 'u' }, 'alice'), false)
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

describe('buildsView', () => {
  it('shows nothing before any data', () => {
    assert.deepEqual(buildsView(undefined, undefined, { term: '', page: 1, pageSize: 20 }), {
      items: [],
      total: 0,
      truncated: false,
    })
  })

  it('passes a server page through untouched', () => {
    const page = { items: WINDOW.slice(0, 5), total: 30 }
    const view = buildsView(page, 'page', { term: '', page: 2, pageSize: 5 })
    assert.deepEqual(names(view), ['b29', 'b28', 'b27', 'b26', 'b25'])
    assert.equal(view.total, 30)
    assert.equal(view.truncated, false)
  })

  it('matches within the window and paginates the matches', () => {
    // Newest-first matches for "1": b21, b19..b10, b1 (12).
    const view = buildsView({ items: WINDOW, total: 30 }, 'window', { term: '1', page: 2, pageSize: 5 })
    assert.deepEqual(names(view), ['b15', 'b14', 'b13', 'b12', 'b11'])
    assert.equal(view.total, 12)
  })

  it('flags truncation only when the window is smaller than the filtered total', () => {
    const full = buildsView({ items: WINDOW, total: 30 }, 'window', { term: 'b', page: 1, pageSize: 5 })
    assert.equal(full.truncated, false)
    const short = buildsView({ items: WINDOW, total: 900 }, 'window', { term: 'b', page: 1, pageSize: 5 })
    assert.equal(short.truncated, true)
  })

  describe('placeholder data from the other mode', () => {
    it('entering a search shows the previous page as it was, with no truncation notice', () => {
      // The reported flash: a 20-row page with total 340 read as a truncated window.
      const previousPage = { items: WINDOW.slice(0, 20), total: 340 }
      const view = buildsView(previousPage, 'page', { term: 'b2', page: 1, pageSize: 20 })
      assert.equal(view.truncated, false)
      assert.equal(view.items.length, 20, 'not re-filtered as if it were the window')
      assert.equal(view.total, 340)
    })

    it('clearing a search shows page 1 of the window, not all of its rows', () => {
      // The reported flash: all 500 window rows rendered unpaginated.
      const previousWindow = { items: WINDOW, total: 340 }
      const view = buildsView(previousWindow, 'window', { term: '', page: 1, pageSize: 5 })
      assert.deepEqual(names(view), ['b29', 'b28', 'b27', 'b26', 'b25'])
      assert.equal(view.total, 340, 'the server total, as the page being fetched will report')
      assert.equal(view.truncated, false)
    })
  })
})
