'use client'

import { ProgressBar } from '@carbon/react'
import { computeTrialProgress } from './trialProgress'
import type { JobDetail, Trial } from '../types'

function formatDuration(seconds: number): string {
  if (seconds <= 0) return '0s'
  const total = Math.floor(seconds)
  const hours = Math.floor(total / 3600)
  const mins = Math.floor((total % 3600) / 60)
  const secs = total % 60
  if (hours > 0) return `${hours}h ${mins}m`
  if (mins > 0) return `${mins}m ${secs}s`
  return `${secs}s`
}

interface Props {
  job: JobDetail
  /** The job's trials, fetched separately — GET /jobs/{id} no longer nests them. */
  trials: Trial[]
}

export function TrialProgressSummary({ job, trials }: Props) {
  // Recomputed on each render; the detail page polls every 15s while the run is
  // active, so elapsed and the estimate advance at that cadence.
  const progress = computeTrialProgress({
    trials,
    numTrials: job.num_trials,
    jobStatus: job.status,
    jobCreatedAt: job.created_at,
    jobUpdatedAt: job.updated_at,
    now: Date.now(),
  })

  // With no planned total and no trials yet there is nothing to report; the
  // caller's "no trial data" notice says it better.
  if (progress.planned === null && trials.length === 0) return null

  const parts: string[] = []
  if (progress.running > 0) parts.push(`${progress.running} running`)
  if (progress.queued > 0) parts.push(`${progress.queued} queued`)
  if (progress.failed > 0) parts.push(`${progress.failed} failed`)
  parts.push(`${formatDuration(progress.elapsedSeconds)} elapsed`)
  // Labelled as an estimate because trials differ in batch size and epoch count,
  // so a median-based projection can be well off.
  if (progress.etaSeconds !== null) {
    parts.push(`~${formatDuration(progress.etaSeconds)} remaining (rough estimate)`)
  }

  const label =
    progress.planned !== null
      ? `Trial ${progress.completed} of ${progress.planned} complete`
      : `${progress.completed} ${progress.completed === 1 ? 'trial' : 'trials'} complete`

  return (
    <div style={{ maxWidth: '32rem', marginBottom: '1.5rem' }}>
      <ProgressBar
        label={label}
        helperText={parts.join(' · ')}
        // A null value renders Carbon's indeterminate bar, which is the honest
        // display when the job never reported a planned total.
        value={progress.percent ?? undefined}
        max={100}
        size="small"
        status={progress.planned !== null && progress.percent === 100 ? 'finished' : 'active'}
      />
    </div>
  )
}
