/**
 * Tests for the shared byte formatter.
 *
 * Extracted from two private copies that had drifted: one stopped at GB, the
 * other went to TB. The shared version takes the superset, so the boundary cases
 * are the point of these tests -- a unit array that loses its last entry silently
 * renders 1024.0 GB instead of 1.0 TB.
 *
 * Usage: node --test tests/format-bytes.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { formatBytes } = require('../../../packages/ui-core/lib/autotunex/formatBytes.ts')

describe('formatBytes', () => {
  it('reports a falsy size as 0 B', () => {
    assert.equal(formatBytes(0), '0 B')
    assert.equal(formatBytes(NaN), '0 B')
  })

  it('keeps sub-kilobyte sizes in bytes', () => {
    assert.equal(formatBytes(512), '512.0 B')
  })

  it('steps up through each unit at its boundary', () => {
    assert.equal(formatBytes(1024), '1.0 KB')
    assert.equal(formatBytes(1024 ** 2), '1.0 MB')
    assert.equal(formatBytes(1024 ** 3), '1.0 GB')
  })

  it('reaches TB, which one of the two original copies could not', () => {
    assert.equal(formatBytes(1024 ** 4), '1.0 TB')
  })

  it('clamps at the largest unit rather than inventing one', () => {
    assert.equal(formatBytes(1024 ** 5), '1024.0 TB')
  })

  it('formats the HF import size limit', () => {
    // 5 GiB, the hf_import.max_bytes the modal displays.
    assert.equal(formatBytes(5 * 1024 ** 3), '5.0 GB')
  })
})
