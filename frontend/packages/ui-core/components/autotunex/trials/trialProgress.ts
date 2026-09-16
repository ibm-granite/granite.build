// Progress of a tuning run, derived from the job's planned trial count and the
// trials it has produced so far. Kept free of React so it can be unit-tested.

import type { TuningStatus } from '../../../types'

interface ProgressTrial {
  status: TuningStatus
  created_at: string
  updated_at: string
  metrics?: Record<string, number>
}

export interface TrialProgressInput {
  trials: ProgressTrial[]
  /** Planned total from the job (`num_trials`); absent on older jobs. */
  numTrials?: number
  jobStatus: TuningStatus
  jobCreatedAt: string
  jobUpdatedAt: string
  /** When the run actually stopped (`job.finished_at`). Preferred over
   *  `jobUpdatedAt` for elapsed time; absent until a task has finished. */
  jobFinishedAt?: string
  now: number
}

export interface TrialProgress {
  /** Planned trial total, or null when the job never reported one. */
  planned: number | null
  completed: number
  running: number
  /** Not started yet: pending trials plus planned trials that don't exist yet. */
  queued: number
  failed: number
  percent: number | null
  elapsedSeconds: number
  /** Rough projection, or null whenever it cannot be justified. */
  etaSeconds: number | null
  /**
   * Job start to the last trial's end — the search phase, setup included.
   * Null unless the phase split is trustworthy; see `phaseSplit` below.
   */
  searchSeconds: number | null
  /**
   * The last trial's end to the run stopping. For an autotune job this is the
   * final full-dataset run on the winning config, which owns no trial row and is
   * otherwise invisible — it was half the wall clock on the job that prompted
   * this. Inferred as the trailing remainder, so it also carries the model save
   * and cluster teardown (seconds, against a phase measured in minutes).
   * Null unless the phase split is trustworthy.
   */
  finalRunSeconds: number | null
}

const FAILED_STATUSES: TuningStatus[] = ['error', 'terminated']
const WAITING_STATUSES: TuningStatus[] = ['pending', 'paused']
const ACTIVE_JOB_STATUSES: TuningStatus[] = ['running', 'pending']

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

// How long a finished trial took: its reported metric when present, else the
// span between its own timestamps.
function trialDuration(trial: ProgressTrial): number | null {
  const reported = trial.metrics?.total_time
  if (typeof reported === 'number' && Number.isFinite(reported) && reported > 0) return reported
  const span = (Date.parse(trial.updated_at) - Date.parse(trial.created_at)) / 1000
  return Number.isFinite(span) && span > 0 ? span : null
}

export function computeTrialProgress(input: TrialProgressInput): TrialProgress {
  const { trials, numTrials, jobStatus, jobCreatedAt, jobUpdatedAt, jobFinishedAt, now } = input

  const planned = typeof numTrials === 'number' && numTrials > 0 ? numTrials : null
  const completedTrials = trials.filter((t) => t.status === 'completed')
  const completed = completedTrials.length
  const running = trials.filter((t) => t.status === 'running').length
  const failed = trials.filter((t) => FAILED_STATUSES.includes(t.status)).length
  const waiting = trials.filter((t) => WAITING_STATUSES.includes(t.status)).length

  // The trials a user is really asking about are the ones that don't exist yet:
  // the job reports how many it plans to run long before it creates their rows.
  const notYetCreated = planned !== null ? Math.max(0, planned - trials.length) : 0

  // `finished_at` is when the run actually stopped; `updated_at` only stands in
  // for it, because any later write to the job row bumps `updated_at` and would
  // inflate this figure permanently. TuningsTable and TuningDetailTabs already
  // prefer it the same way, so all three agree on one job's duration. An absent
  // or unparseable value falls back instead of poisoning the result with NaN.
  const finishedMs = jobFinishedAt ? Date.parse(jobFinishedAt) : NaN
  const stoppedMs = Number.isFinite(finishedMs) ? finishedMs : Date.parse(jobUpdatedAt)
  const elapsedEndMs = ACTIVE_JOB_STATUSES.includes(jobStatus) ? now : stoppedMs
  const elapsedSeconds = Math.max(0, Math.floor((elapsedEndMs - Date.parse(jobCreatedAt)) / 1000))

  // Phase split. Reported only for a cleanly completed job: while the run is live
  // the trailing phase has no end yet, and on an error/terminated job the trials
  // can lack durations, which would silently understate the last trial's end and
  // charge the difference to the final run.
  //
  // A trial's end is its start plus its own duration rather than its `updated_at`,
  // for the same reason the job uses `finished_at` — any later write to the row
  // moves `updated_at`. Treating a missing duration as zero collapsed that trial's
  // end onto its own start, so its whole run was taken off the search phase and
  // charged to the final run instead; `trialDuration` falls back to the trial's
  // own span, and a trial with no evidence either way contributes no end at all
  // rather than a wrong one.
  let searchSeconds: number | null = null
  let finalRunSeconds: number | null = null
  if (jobStatus === 'completed' && trials.length > 0) {
    const startMs = Date.parse(jobCreatedAt)
    const trialEnds = trials
      .map((t) => {
        const duration = trialDuration(t)
        return duration === null ? null : Date.parse(t.created_at) + duration * 1000
      })
      .filter((end): end is number => end !== null)
    const lastEndMs = trialEnds.length > 0 ? Math.max(...trialEnds) : NaN
    const searchMs = lastEndMs - startMs
    const finalMs = stoppedMs - lastEndMs
    // Every timestamp has to parse and the phases have to be ordered, or the
    // split is nonsense and one aggregate is the honest thing to show. The
    // one-minute floor keeps a job with no real final phase from reporting a
    // "final run" that is really just teardown.
    if (Number.isFinite(searchMs) && Number.isFinite(finalMs) && searchMs > 0 && finalMs >= 60_000) {
      searchSeconds = Math.floor(searchMs / 1000)
      finalRunSeconds = Math.floor(finalMs / 1000)
    }
  }

  // Only project when the run is live, the total is known, work remains, and at
  // least one finished trial gives a duration to extrapolate from.
  let etaSeconds: number | null = null
  // Failed trials are not coming back, so they are not work remaining. Counting
  // them projected time for trials that will never run, next to a `queued` figure
  // that already discounts them — one summary contradicting itself.
  const remaining = planned !== null ? Math.max(0, planned - completed - failed) : 0
  if (jobStatus === 'running' && planned !== null && remaining > 0) {
    const durations = completedTrials
      .map(trialDuration)
      .filter((d): d is number => d !== null)
    if (durations.length > 0) {
      const concurrency = Math.max(1, running)
      etaSeconds = Math.round((median(durations) * remaining) / concurrency)
    }
  }

  return {
    planned,
    completed,
    running,
    queued: waiting + notYetCreated,
    failed,
    percent: planned !== null ? Math.min(100, Math.round((completed / planned) * 100)) : null,
    elapsedSeconds,
    etaSeconds,
    searchSeconds,
    finalRunSeconds,
  }
}
