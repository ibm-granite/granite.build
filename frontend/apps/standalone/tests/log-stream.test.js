/**
 * Tests for the live log stream's gap handling.
 *
 * The bug this guards: the poll only ever fetches the newest page, and the merge
 * unions by id without checking contiguity. A job emitting more than one page
 * between ticks left a hole — hold 101-300, poll returns 601-800 — that rendered as
 * one continuous block, and `loadMore` could never repair it because it walks back
 * from the oldest held id, never into an interior gap.
 *
 * Usage: node --test tests/log-stream.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  mergeLogs,
  logGapCursor,
  gapReconciled,
} = require('../../../packages/ui-core/lib/autotunex/logStream.ts')

const entry = (id) => ({ id, timestamp: '2026-09-03T12:00:00Z', level: 'INFO', filename: 'x.py', message: `line ${id}` })
const range = (from, to) => {
  const out = []
  for (let id = to; id >= from; id--) out.push(entry(id))
  return out
}

describe('mergeLogs', () => {
  it('unions by id, newest first', () => {
    const merged = mergeLogs(range(1, 3), [entry(5), entry(4)])
    assert.deepEqual(merged.map((l) => l.id), [5, 4, 3, 2, 1])
  })

  it('deduplicates an overlapping page without disturbing history', () => {
    const merged = mergeLogs(range(1, 5), range(3, 7))
    assert.deepEqual(merged.map((l) => l.id), [7, 6, 5, 4, 3, 2, 1])
  })
})

describe('logGapCursor', () => {
  it('reports the cursor when the polled page starts after everything held', () => {
    // The reported repro: hold 101-300, a tick emits >200 lines, poll returns 601-800.
    assert.equal(logGapCursor(range(101, 300), range(601, 800)), 601)
  })

  it('is null when the polled page overlaps the held range', () => {
    assert.equal(logGapCursor(range(101, 300), range(111, 310)), null)
  })

  it('is null when the polled page is entirely inside the held range', () => {
    assert.equal(logGapCursor(range(101, 300), range(150, 200)), null)
  })

  it('is null on the first poll, when nothing is held yet', () => {
    assert.equal(logGapCursor([], range(1, 200)), null)
  })

  it('is null when the poll returns nothing', () => {
    assert.equal(logGapCursor(range(1, 200), []), null)
  })

  it('flags an exactly-adjacent page rather than assuming ids are gap-free', () => {
    // Costs one backfill request that immediately overlaps and stops. The opposite
    // error — assuming id+1 means adjacent — would silently keep a real gap, since
    // these ids are only guaranteed ordered, not contiguous per job.
    assert.equal(logGapCursor(range(101, 300), range(301, 500)), 301)
  })

  it('does not depend on the order of either page', () => {
    const heldAscending = range(101, 300).slice().reverse()
    const polledAscending = range(601, 800).slice().reverse()
    assert.equal(logGapCursor(heldAscending, polledAscending), 601)
  })
})

describe('gapReconciled', () => {
  it('is true once a backfill page reaches the previously-held range', () => {
    assert.equal(gapReconciled(range(250, 450), 300), true)
  })

  it('is false while the backfill is still above the held range', () => {
    assert.equal(gapReconciled(range(401, 600), 300), false)
  })

  it('is true for an empty page — there is nothing left to walk', () => {
    assert.equal(gapReconciled([], 300), true)
  })

  it('is true when the page lands exactly on the held boundary', () => {
    assert.equal(gapReconciled(range(300, 500), 300), true)
  })
})
