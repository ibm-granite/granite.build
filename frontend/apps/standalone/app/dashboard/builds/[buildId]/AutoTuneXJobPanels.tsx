'use client'

import type { JobDetail } from '@granite-build/ui-core/types'
import { TrialsTable } from '@granite-build/ui-core/components/autotunex/trials/TrialsTable'
import { TuningLogViewer } from '@granite-build/ui-core/components/autotunex/tunings/TuningLogViewer'

/**
 * Trials & Logs for the AutoTuneX tuning job linked to a build, mirroring the
 * AutoTuneX tuning detail page.
 *
 * BuildDetails owns the linked-job lookup (see useLinkedTuningJob) and only mounts
 * these once it holds a job, so neither panel has a loading, error or empty state.
 *
 * The scope travels with the job: anything fetched *about* the job has to use the
 * same one, or an admin who resolved another user's job then 403s on its logs and
 * sees an empty pane.
 */

export function AutoTuneXTrialsPanel({ job }: { job: JobDetail }) {
  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <TrialsTable job={job} />
    </div>
  )
}

export function AutoTuneXLogsPanel({ job, scope }: { job: JobDetail; scope: 'own' | 'all' }) {
  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <TuningLogViewer jobId={job.id} status={job.status} scope={scope} />
    </div>
  )
}
