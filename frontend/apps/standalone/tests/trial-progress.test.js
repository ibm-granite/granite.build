/**
 * Tests for the tuning run progress shown above the trials table.
 *
 * Users could not tell how far a run had got or how many trials were still
 * coming: `num_trials` (the planned total) was parsed in api/autotunex.ts but
 * rendered nowhere. Trials that do not exist yet are the whole point of the
 * request, so "queued" has to count planned-but-not-created trials, not just
 * the pending rows already returned.
 *
 * The estimate is deliberately conservative — it is withheld unless the run is
 * actually running, the planned total is known, and at least one trial has
 * finished to give a duration sample. A confidently wrong ETA is worse than none.
 *
 * Usage: node --test tests/trial-progress.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { computeTrialProgress } = require('../../../packages/ui-core/components/trialProgress.ts')

const T0 = Date.parse('2026-08-26T10:00:00Z')
const NOW = T0 + 10 * 60_000 // 10 minutes into the run

function trial(status, seconds, overrides = {}) {
  return {
    status,
    created_at: new Date(T0).toISOString(),
    updated_at: new Date(T0 + (seconds ?? 0) * 1000).toISOString(),
    metrics: seconds === null ? {} : { total_time: seconds },
    ...overrides,
  }
}

function run(trials, numTrials, jobStatus = 'running', now = NOW) {
  return computeTrialProgress({
    trials,
    numTrials,
    jobStatus,
    jobCreatedAt: new Date(T0).toISOString(),
    jobUpdatedAt: new Date(NOW).toISOString(),
    now,
  })
}

describe('computeTrialProgress counts', () => {
  it('counts trials by status', () => {
    const p = run([trial('completed', 60), trial('completed', 90), trial('running', null), trial('error', 30)], 12)
    assert.equal(p.completed, 2)
    assert.equal(p.running, 1)
    assert.equal(p.failed, 1)
  })

  it('counts planned-but-not-yet-created trials as queued', () => {
    // 12 planned, 3 exist (2 done + 1 running) -> 9 still to come
    const p = run([trial('completed', 60), trial('completed', 60), trial('running', null)], 12)
    assert.equal(p.queued, 9)
  })

  it('counts an existing pending trial as queued alongside the uncreated ones', () => {
    const p = run([trial('completed', 60), trial('pending', null)], 5)
    assert.equal(p.queued, 4) // 1 pending + 3 not yet created
  })

  it('treats a terminated trial as failed rather than completed', () => {
    const p = run([trial('terminated', 10)], 4)
    assert.equal(p.failed, 1)
    assert.equal(p.completed, 0)
  })
})

describe('computeTrialProgress planned total', () => {
  it('reports percent complete against the planned total', () => {
    const p = run([trial('completed', 60), trial('completed', 60), trial('completed', 60)], 12)
    assert.equal(p.planned, 12)
    assert.equal(p.percent, 25)
  })

  it('reports no planned total or percent when num_trials is absent', () => {
    const p = run([trial('completed', 60)], undefined)
    assert.equal(p.planned, null)
    assert.equal(p.percent, null)
  })

  it('never exceeds 100 percent when more trials ran than planned', () => {
    const p = run([trial('completed', 60), trial('completed', 60), trial('completed', 60)], 2)
    assert.equal(p.percent, 100)
  })
})

describe('computeTrialProgress elapsed', () => {
  it('measures elapsed against now while the run is active', () => {
    const p = run([trial('running', null)], 4, 'running')
    assert.equal(p.elapsedSeconds, 600)
  })

  it('freezes elapsed at the last update once the run has finished', () => {
    const p = run([trial('completed', 60)], 1, 'completed', NOW + 3_600_000)
    assert.equal(p.elapsedSeconds, 600)
  })
})

describe('computeTrialProgress estimate', () => {
  it('projects the median completed duration across the remaining trials', () => {
    // medians of 60/120/180 = 120s; 10 planned - 3 done = 7 remaining; 1 running
    const p = run([trial('completed', 60), trial('completed', 120), trial('completed', 180), trial('running', null)], 10)
    assert.equal(p.etaSeconds, 840)
  })

  it('divides the remaining work by how many trials run at once', () => {
    // median 100s, 8 remaining, 2 running concurrently -> 400s
    const p = run(
      [trial('completed', 100), trial('completed', 100), trial('running', null), trial('running', null)],
      10
    )
    assert.equal(p.etaSeconds, 400)
  })

  it('falls back to the trial timestamps when total_time is missing', () => {
    // no metrics.total_time; updated_at - created_at = 200s. 4 planned - 1 done = 3 remaining
    const p = run([trial('completed', null, { updated_at: new Date(T0 + 200_000).toISOString() })], 4)
    assert.equal(p.etaSeconds, 600)
  })

  it('withholds an estimate until at least one trial has finished', () => {
    const p = run([trial('running', null)], 10)
    assert.equal(p.etaSeconds, null)
  })

  it('withholds an estimate when the planned total is unknown', () => {
    const p = run([trial('completed', 60)], undefined)
    assert.equal(p.etaSeconds, null)
  })

  it('withholds an estimate for a run that is no longer running', () => {
    const p = run([trial('completed', 60)], 10, 'completed')
    assert.equal(p.etaSeconds, null)
  })

  it('withholds an estimate once every planned trial has completed', () => {
    const p = run([trial('completed', 60), trial('completed', 60)], 2)
    assert.equal(p.etaSeconds, null)
  })
})
