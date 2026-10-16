/**
 * Regression test for the `selectedIds` shadow kept alongside Carbon's own
 * DataTable selection in the tunings, configurations and datasets tables.
 *
 * The three tables used to empty the shadow whenever the visible set changed, on
 * the premise that Carbon rebuilds its checkboxes from the rows it is handed. It
 * does not — `normalize()` carries `isSelected` forward for every id it already
 * knows. Rows stayed ticked while the shadow was empty, so Delete confirmed a
 * count of 0, deleted nothing and closed as a success.
 *
 * Usage: node --test tests/table-selection.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { pruneSelection } = require('../../../packages/ui-core/lib/autotunex/tableSelection.ts')

describe('pruneSelection', () => {
  it('keeps ids that are still on screen — the rows Carbon leaves ticked', () => {
    // The reported repro: tick a row, then change Items per page 10 -> 20. The
    // row is still rendered, so Carbon keeps it ticked and the shadow must too.
    assert.deepEqual(pruneSelection(['a'], ['a', 'b', 'c']), ['a'])
  })

  it('drops ids that left the visible set — the rows Carbon drops itself', () => {
    assert.deepEqual(pruneSelection(['a', 'b'], ['b', 'c']), ['b'])
  })

  it('empties the selection when no row survives', () => {
    assert.deepEqual(pruneSelection(['a', 'b'], ['c']), [])
    assert.deepEqual(pruneSelection(['a'], []), [])
  })

  it('returns the same array reference when nothing was dropped', () => {
    // Called from an effect on every render, so an unchanged result has to bail
    // out of the state update rather than loop.
    const prev = ['a', 'b']
    assert.equal(pruneSelection(prev, ['a', 'b', 'c']), prev)
  })

  it('leaves an already-empty selection alone', () => {
    const prev = []
    assert.equal(pruneSelection(prev, ['a']), prev)
  })

  it('accepts a Set without rebuilding it', () => {
    assert.deepEqual(pruneSelection(['a', 'b'], new Set(['a'])), ['a'])
  })

  it('preserves the caller order of the surviving ids', () => {
    assert.deepEqual(pruneSelection(['c', 'a', 'b'], ['a', 'b', 'c']), ['c', 'a', 'b'])
  })
})
