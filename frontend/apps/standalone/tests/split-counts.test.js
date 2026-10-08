/**
 * Regression test for the train/validation counts the wizard shows for an
 * auto-split upload.
 *
 * The server takes round(total * pct / 100) records for validation (Python's
 * round, half to even) and the rest for training. Upload floored the train share
 * and Review rounded each side, so 101 records read 80/21 on Upload and 81/20 on
 * Review and the server.
 *
 * Usage: node --test tests/split-counts.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { splitCounts } = require('../../../packages/ui-core/lib/autotunex/splitCounts.ts')

describe('splitCounts', () => {
  it('matches the server for the case Upload got wrong', () => {
    assert.deepEqual(splitCounts(101, 20), { train: 81, validation: 20 })
  })

  it('always sums to the total', () => {
    for (let total = 1; total <= 200; total++) {
      const { train, validation } = splitCounts(total, 20)
      assert.equal(train + validation, total, `total ${total}`)
    }
  })

  it('rounds half to even, as Python does', () => {
    assert.deepEqual(splitCounts(10, 25), { train: 8, validation: 2 }) // 2.5 -> 2
    assert.deepEqual(splitCounts(14, 25), { train: 10, validation: 4 }) // 3.5 -> 4
  })

  it('reports an empty validation share instead of hiding it', () => {
    // The server rejects a split that leaves either side empty; Upload used to show
    // 1 validation record for a 2-record file.
    assert.deepEqual(splitCounts(2, 20), { train: 2, validation: 0 })
  })
})
