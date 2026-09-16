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

const {
  computeTrialProgress,
  isSearchComplete,
} = require('../../../packages/ui-core/components/autotunex/trials/trialProgress.ts')

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

function run(trials, numTrials, jobStatus = 'running', now = NOW, jobFinishedAt = undefined) {
  return computeTrialProgress({
    trials,
    numTrials,
    jobStatus,
    jobCreatedAt: new Date(T0).toISOString(),
    jobUpdatedAt: new Date(NOW).toISOString(),
    jobFinishedAt,
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

// The final full-dataset run on the winning config owns no trial row, so it is
// absent from GET /jobs/{id}/trials and invisible in the table. On the job that
// prompted this it was 40m of an 83m run — half the wall clock, sitting under a
// "Trial 4 of 4 complete" heading beside rows that summed to 41m. The split is
// inferred from timestamps (job start -> last trial end -> run stopped) so the
// parts add up to the total exactly, and is withheld whenever that inference
// cannot be trusted.
describe('computeTrialProgress phase split', () => {
  // Job starts at T0. One trial starts 1 minute in and runs 5 minutes, so the
  // search phase ends at T0+6m. The run stops at T0+30m, leaving 24m after it.
  const late = (mins) => new Date(T0 + mins * 60_000).toISOString()
  const searchTrial = trial('completed', 300, { created_at: late(1) })

  function finished(trials, finishedAt, jobStatus = 'completed') {
    return computeTrialProgress({
      trials,
      numTrials: trials.length,
      jobStatus,
      jobCreatedAt: new Date(T0).toISOString(),
      jobUpdatedAt: new Date(NOW).toISOString(),
      jobFinishedAt: finishedAt,
      now: NOW,
    })
  }

  it('splits the run into search and final-run phases', () => {
    const p = finished([searchTrial], late(30))
    assert.equal(p.searchSeconds, 360)
    assert.equal(p.finalRunSeconds, 1440)
  })

  it('makes the parts sum to the total exactly', () => {
    const p = finished([searchTrial], late(30))
    assert.equal(p.searchSeconds + p.finalRunSeconds, p.elapsedSeconds)
  })

  it('measures the search phase from job start, so setup is not lost', () => {
    // The trial itself ran 300s but started a minute after the job did; charging
    // that minute to neither phase is what stopped the parts adding up.
    const p = finished([searchTrial], late(30))
    assert.equal(p.searchSeconds, 360)
  })

  it('withholds the split while the run is still active', () => {
    const p = finished([searchTrial], undefined, 'running')
    assert.equal(p.searchSeconds, null)
    assert.equal(p.finalRunSeconds, null)
  })

  it('withholds the split for an error or terminated run', () => {
    // Trials there can lack durations, which would understate the last trial's
    // end and charge the difference to a "final run" that never happened.
    for (const status of ['error', 'terminated']) {
      const p = finished([searchTrial], late(30), status)
      assert.equal(p.finalRunSeconds, null, `${status} should not report a phase split`)
    }
  })

  it('withholds the split when nothing follows the last trial', () => {
    // A search-only run: stopping 10s after the last trial is teardown, not a
    // final run worth naming.
    const p = finished([searchTrial], new Date(T0 + 6 * 60_000 + 10_000).toISOString())
    assert.equal(p.searchSeconds, null)
    assert.equal(p.finalRunSeconds, null)
  })

  it('withholds the split with no trials at all', () => {
    const p = finished([], late(30))
    assert.equal(p.searchSeconds, null)
    assert.equal(p.finalRunSeconds, null)
  })

  // A trial that never reported `metrics.total_time` used to have its end
  // collapsed onto its own start, so the whole of its run was subtracted from the
  // search phase and charged to the "final full-dataset run" instead: a 21m
  // search / 9m final run was displayed as 1m / 29m. `trialDuration` already had
  // the timestamp fallback for exactly this case; the split now uses it.
  it('falls back to the trial timestamps when total_time is missing', () => {
    const noMetrics = trial('completed', null, {
      created_at: late(1),
      updated_at: late(21),
    })
    const p = finished([noMetrics], late(30))
    assert.equal(p.searchSeconds, 1260)
    assert.equal(p.finalRunSeconds, 540)
  })

  it('withholds the split when no trial has any duration evidence', () => {
    // Neither a reported duration nor a usable span: the last trial's end is
    // genuinely unknown, so understating it silently is worse than one aggregate.
    const noEvidence = trial('completed', null, { created_at: late(1), updated_at: late(1) })
    const p = finished([noEvidence], late(30))
    assert.equal(p.searchSeconds, null)
    assert.equal(p.finalRunSeconds, null)
  })

  it('takes the last end across trials, not the last one listed', () => {
    // Trials arrive sorted by loss, not by time, so the newest end can sit
    // anywhere in the array.
    const early = trial('completed', 60, { created_at: late(1) })
    const latest = trial('completed', 300, { created_at: late(5) })
    const p = finished([latest, early], late(30))
    assert.equal(p.searchSeconds, 600) // last end is T0+10m
    assert.equal(p.finalRunSeconds, 1200)
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

  // `updated_at` is any write to the job row, not the moment the run stopped, so
  // tagging or touching a finished job used to stretch its elapsed time forever.
  // TuningsTable and TuningDetailTabs already read `finished_at` first; these pin
  // the third display to the same source so one job cannot report two durations.
  it('measures elapsed to finished_at, not to a later updated_at', () => {
    // finished_at 5 minutes in; updated_at (NOW) is 10 minutes in.
    const finishedAt = new Date(T0 + 5 * 60_000).toISOString()
    const p = run([trial('completed', 60)], 1, 'completed', NOW, finishedAt)
    assert.equal(p.elapsedSeconds, 300)
  })

  it('ignores finished_at while the run is still active', () => {
    // A stale or early finished_at must not cut short a live run's clock; the
    // active branch measures to `now` regardless.
    const finishedAt = new Date(T0 + 5 * 60_000).toISOString()
    const p = run([trial('running', null)], 4, 'running', NOW, finishedAt)
    assert.equal(p.elapsedSeconds, 600)
  })

  it('falls back to updated_at when finished_at is absent', () => {
    const p = run([trial('completed', 60)], 1, 'completed', NOW + 3_600_000, undefined)
    assert.equal(p.elapsedSeconds, 600)
  })

  it('falls back to updated_at when finished_at is unparseable', () => {
    // The adapter types finished_at as `string | undefined`, so an empty string
    // can reach here; Date.parse gives NaN and must not become the answer.
    for (const bad of ['', 'not-a-date']) {
      const p = run([trial('completed', 60)], 1, 'completed', NOW + 3_600_000, bad)
      assert.equal(p.elapsedSeconds, 600, `finished_at ${JSON.stringify(bad)} should fall back`)
    }
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

  // `queued` already discounts failed trials (via notYetCreated), so projecting
  // them anyway made one summary contradict itself: "0 queued" beside an ETA
  // covering two trials that will never run.
  it('does not project trials that have already failed', () => {
    const p = run([trial('completed', 60), trial('completed', 60), trial('error', 30), trial('error', 30)], 4)
    assert.equal(p.queued, 0)
    assert.equal(p.etaSeconds, null)
  })

  it('projects only the trials that are actually still to come', () => {
    // 6 planned, 2 done, 1 failed, 1 running -> 3 still to run at median 60s.
    const p = run(
      [trial('completed', 60), trial('completed', 60), trial('error', 30), trial('running', null)],
      6
    )
    assert.equal(p.etaSeconds, 180)
  })

  it('withholds an estimate once every planned trial has completed', () => {
    const p = run([trial('completed', 60), trial('completed', 60)], 2)
    assert.equal(p.etaSeconds, null)
  })
})

describe('computeTrialProgress percent counts resolved work', () => {
  it('reaches 100 when the last outstanding trial fails', () => {
    // The reported defect: a sweep finishing 3-of-4 with one error sat at 75% on a
    // bar that never reached 'finished'. Failed trials are not coming back, so they
    // are not outstanding — the same reasoning `remaining`/`etaSeconds` already use.
    const p = run(
      [trial('completed', 60), trial('completed', 90), trial('completed', 75), trial('error', 30)],
      4,
      'completed'
    )
    assert.equal(p.completed, 3, 'the label still reports successes only')
    assert.equal(p.failed, 1)
    assert.equal(p.percent, 100)
  })

  it('counts a terminated trial as resolved too', () => {
    const p = run([trial('completed', 60), trial('terminated', 30)], 2, 'terminated')
    assert.equal(p.percent, 100)
  })

  it('still tracks partial progress mid-sweep', () => {
    const p = run([trial('completed', 60), trial('running', null)], 4)
    assert.equal(p.percent, 25)
  })

  it('never exceeds 100 if more trials resolve than were planned', () => {
    const p = run([trial('completed', 60), trial('completed', 60), trial('error', 10)], 2, 'completed')
    assert.equal(p.percent, 100)
  })

  it('stays null when the job never reported a planned total', () => {
    const p = run([trial('error', 10)], null, 'error')
    assert.equal(p.percent, null)
  })
})

describe('isSearchComplete', () => {
  it('is false while any planned trial is unresolved', () => {
    assert.equal(isSearchComplete([trial('completed', 60), trial('running', null)], 4), false)
    assert.equal(isSearchComplete([trial('completed', 60), trial('pending', null)], 2), false)
  })

  it('is true once every planned trial has resolved', () => {
    assert.equal(isSearchComplete([trial('completed', 60), trial('completed', 60)], 2), true)
  })

  it('counts failed and terminated trials as resolved — the search will not retry them', () => {
    assert.equal(isSearchComplete([trial('completed', 60), trial('error', 10)], 2), true)
    assert.equal(isSearchComplete([trial('terminated', 10), trial('error', 10)], 2), true)
  })

  it('is false when the planned total is unknown, rather than guessing', () => {
    assert.equal(isSearchComplete([trial('completed', 60)], null), false)
    assert.equal(isSearchComplete([trial('completed', 60)], undefined), false)
    assert.equal(isSearchComplete([trial('completed', 60)], 0), false)
  })

  it('is true when more trials resolved than were planned', () => {
    assert.equal(isSearchComplete([trial('completed', 60), trial('completed', 60)], 1), true)
  })
})
