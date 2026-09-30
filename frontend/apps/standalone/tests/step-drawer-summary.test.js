/**
 * Unit tests for the build-drawer header logic and for picking a retried
 * target's current attempt. Both read the target run's own status and
 * timestamps (see buildrunner.py, _apply_run_timestamps) rather than
 * rebuilding them from step rows or list order.
 *
 * Usage: node --test tests/step-drawer-summary.test.js
 */

const { describe, it, before } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { registerHooks } = require('node:module')

// The modules under test are TypeScript with extensionless imports (bundler
// resolution). Node strips the types; this hook supplies the extension.
registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context)
    } catch (err) {
      if (err.code !== 'ERR_MODULE_NOT_FOUND' && err.code !== 'MODULE_NOT_FOUND') throw err
      return next(`${specifier}.ts`, context)
    }
  },
})

const ROOT = path.join(__dirname, '..')
let stepDrawerSummary
let attemptOrder
let isLaterAttempt

before(async () => {
  ;({ stepDrawerSummary } = await import(
    path.join(ROOT, 'app/dashboard/builds/[buildId]/stepDrawerSummary.ts')
  ))
  ;({ attemptOrder, isLaterAttempt } = await import(
    path.join(ROOT, '../../packages/ui-core/api/targetAttempts.ts')
  ))
})

const T0 = '2026-09-26T10:00:00Z'
const T2 = '2026-09-26T10:02:00Z'
const T5 = '2026-09-26T10:05:00Z'
const T60 = '2026-09-26T11:00:00Z'

function target(overrides = {}) {
  return { uuid: 't', target_name: 't', status: 'success', steps: [], ...overrides }
}
const step = (name) => ({ step_name: name })

/** Keep the current attempt among runs arriving in the given order. */
function pick(runs) {
  let current
  for (const r of runs) if (isLaterAttempt(r, current)) current = r
  return current
}

describe('stepDrawerSummary — timing from the target run', () => {
  it('no target (planned) has no status or summary', () => {
    assert.deepEqual(stepDrawerSummary(undefined), {
      status: undefined,
      subtitle: 'Target',
      summary: undefined,
    })
  })

  it('uses target.started_at / finished_at, not the build span', () => {
    const r = stepDrawerSummary(
      target({ started_at: T0, finished_at: T2, steps: [step('a')] }),
      { status: 'success', finished_at: T60 },
    )
    assert.equal(r.status, 'success')
    assert.equal(r.subtitle, 'Step · a')
    assert.match(r.summary, /^Completed in 2m · /)
  })

  it('a target with no step rows still gets a duration', () => {
    const r = stepDrawerSummary(target({ started_at: T0, finished_at: T5 }), {
      status: 'success',
      finished_at: T60,
    })
    assert.equal(r.subtitle, 'Target')
    assert.match(r.summary, /^Completed in 5m · /)
  })

  it('failed target reads "Ran for"', () => {
    const r = stepDrawerSummary(target({ status: 'failed', started_at: T0, finished_at: T2 }))
    assert.match(r.summary, /^Ran for 2m · /)
  })

  it('running target under a running build measures to now', () => {
    const started = new Date(Date.now() - 65_000).toISOString()
    const r = stepDrawerSummary(target({ status: 'running', started_at: started }), {
      status: 'running',
    })
    assert.match(r.summary, /^Running for 1m \d+s · started /)
  })

  it('non-terminal target under a stopped build is bounded by the build finish', () => {
    const r = stepDrawerSummary(target({ status: 'running', started_at: T0 }), {
      status: 'cancelled',
      finished_at: T5,
    })
    assert.match(r.summary, /^Ran for 5m · /)
  })

  it('a target that never started under a stopped build borrows no build timestamp', () => {
    const r = stepDrawerSummary(target({ status: 'pending' }), {
      status: 'cancelled',
      finished_at: T5,
    })
    assert.equal(r.summary, undefined)
  })

  it('a target that failed straight from PENDING shows only its finish time', () => {
    const r = stepDrawerSummary(target({ status: 'failed', finished_at: T2 }), {
      status: 'failed',
      finished_at: T5,
    })
    assert.ok(r.summary)
    assert.doesNotMatch(r.summary, /Ran for|Completed in/)
  })

  it('multi-step subtitle lists steps in order', () => {
    const r = stepDrawerSummary(target({ steps: [step('a'), step('b')] }))
    assert.equal(r.subtitle, '2 steps · a → b')
  })
})

describe('current attempt of a retried target', () => {
  const failed1 = { id: 1, started_at: T0, finished_at: T2 }

  it('a run with no timestamps sorts newest', () => {
    assert.equal(attemptOrder({}), Infinity)
  })

  it('a queued retry (no timestamps) replaces the failed attempt, either order', () => {
    const queued = { id: 2 }
    assert.equal(pick([failed1, queued]).id, 2)
    assert.equal(pick([queued, failed1]).id, 2)
  })

  it('a running or finished retry has a later started_at', () => {
    const retry = { id: 2, started_at: T5 }
    assert.equal(pick([retry, failed1]).id, 2)
    assert.equal(pick([failed1, retry]).id, 2)
  })

  it('an attempt that failed from PENDING sorts by finished_at between its neighbours', () => {
    const fromPending = { id: 2, finished_at: T5 }
    const third = { id: 3, started_at: T60 }
    assert.equal(pick([fromPending, failed1]).id, 2)
    assert.equal(pick([third, fromPending, failed1]).id, 3)
  })

  it('a restart of a succeeded target shows the new queued run', () => {
    const succeeded = { id: 1, started_at: T0, finished_at: T2 }
    assert.equal(pick([{ id: 2 }, succeeded]).id, 2)
  })
})
