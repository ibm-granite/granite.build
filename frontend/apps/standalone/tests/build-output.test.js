/**
 * Verifies the Next.js static export (out/) has the expected routes.
 * Run after `yarn build` — tests will fail if the output directory is missing.
 *
 * Usage: node --test tests/build-output.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')

const OUT = path.join(__dirname, '../out')

function exists(rel) {
  return fs.existsSync(path.join(OUT, rel))
}

function notExists(rel) {
  return !fs.existsSync(path.join(OUT, rel))
}

describe('build output — required routes', () => {
  it('out/ directory was produced', () => {
    assert.ok(exists('.'), 'Run `yarn build` before running tests')
  })

  it('/dashboard/builds/ exists', () => {
    assert.ok(exists('dashboard/builds/index.html'), 'Builds list page missing from output')
  })

  it('/dashboard/builds/_ (detail shell) exists', () => {
    assert.ok(exists('dashboard/builds/_/index.html'), 'Build detail shell page missing from output')
  })

  it('/dashboard/artifacts/ exists', () => {
    assert.ok(exists('dashboard/artifacts/index.html'), 'Artifacts page missing from output')
  })

  it('/dashboard/analytics/ exists', () => {
    assert.ok(exists('dashboard/analytics/index.html'), 'Analytics page missing from output')
  })

  it('/dashboard/data-processing/ exists', () => {
    assert.ok(exists('dashboard/data-processing/index.html'), 'Data Processing page missing from output')
  })

})

describe('build output — deleted routes must be absent', () => {
  it('/dashboard/workloads/ was deleted (Phase 1)', () => {
    assert.ok(notExists('dashboard/workloads'), '/dashboard/workloads/ should not exist — delete app/dashboard/workloads/')
  })

  it('/dashboard/plans/ was deleted (Phase 1)', () => {
    assert.ok(notExists('dashboard/plans'), '/dashboard/plans/ should not exist — delete app/dashboard/plans/')
  })
})
