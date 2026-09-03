// Progress of a tuning run, derived from the job's planned trial count and the
// trials it has produced so far. Kept free of React so it can be unit-tested.

import type { TuningStatus } from '../types'

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
  const { trials, numTrials, jobStatus, jobCreatedAt, jobUpdatedAt, now } = input

  const planned = typeof numTrials === 'number' && numTrials > 0 ? numTrials : null
  const completedTrials = trials.filter((t) => t.status === 'completed')
  const completed = completedTrials.length
  const running = trials.filter((t) => t.status === 'running').length
  const failed = trials.filter((t) => FAILED_STATUSES.includes(t.status)).length
  const waiting = trials.filter((t) => WAITING_STATUSES.includes(t.status)).length

  // The trials a user is really asking about are the ones that don't exist yet:
  // the job reports how many it plans to run long before it creates their rows.
  const notYetCreated = planned !== null ? Math.max(0, planned - trials.length) : 0

  const elapsedEndMs = ACTIVE_JOB_STATUSES.includes(jobStatus) ? now : Date.parse(jobUpdatedAt)
  const elapsedSeconds = Math.max(0, Math.floor((elapsedEndMs - Date.parse(jobCreatedAt)) / 1000))

  // Only project when the run is live, the total is known, work remains, and at
  // least one finished trial gives a duration to extrapolate from.
  let etaSeconds: number | null = null
  const remaining = planned !== null ? planned - completed : 0
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
  }
}
