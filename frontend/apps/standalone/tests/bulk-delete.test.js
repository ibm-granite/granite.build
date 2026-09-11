/**
 * The AutoTuneX list views delete a multi-select one id at a time. Aborting on
 * the first rejection left the already-deleted rows on screen and re-DELETEd
 * them on the next confirm, so `deleteEach` must attempt every id and report
 * only the survivors.
 *
 * Usage: node --test tests/bulk-delete.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { deleteEach, isBulkDeleteError } = require(
  '../../../packages/ui-core/lib/autotunex/bulkDelete.ts'
)

describe('deleteEach', () => {
  it('resolves and attempts every id when all succeed', async () => {
    const seen = []
    await deleteEach(['a', 'b', 'c'], async (id) => { seen.push(id) })
    assert.deepEqual(seen, ['a', 'b', 'c'])
  })

  it('attempts the ids after a failure instead of aborting', async () => {
    const seen = []
    const err = await deleteEach(['a', 'b', 'c'], async (id) => {
      seen.push(id)
      if (id === 'b') throw new Error('409 in use')
    }).then(() => null, (e) => e)

    assert.deepEqual(seen, ['a', 'b', 'c'], 'c must still be attempted')
    assert.ok(isBulkDeleteError(err))
    assert.deepEqual(err.failedIds, ['b'], 'only the survivor is reported back')
  })

  it('reports every failure and keeps the first error for status messaging', async () => {
    const first = new Error('first')
    const err = await deleteEach(['a', 'b', 'c'], async (id) => {
      if (id === 'a') throw first
      if (id === 'c') throw new Error('second')
    }).then(() => null, (e) => e)

    assert.ok(isBulkDeleteError(err))
    assert.deepEqual(err.failedIds, ['a', 'c'])
    assert.equal(err.firstError, first)
  })

  it('resolves on an empty selection without calling the deleter', async () => {
    let calls = 0
    await deleteEach([], async () => { calls++ })
    assert.equal(calls, 0)
  })

  it('isBulkDeleteError rejects an unrelated error', () => {
    assert.equal(isBulkDeleteError(new Error('plain')), false)
    assert.equal(isBulkDeleteError(undefined), false)
  })
})
