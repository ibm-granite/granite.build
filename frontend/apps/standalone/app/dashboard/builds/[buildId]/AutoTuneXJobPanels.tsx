'use client'

import type { JobDetail } from '@granite-build/ui-core/types'
import { TrialsTable } from '@granite-build/ui-core/components/autotunex/trials/TrialsTable'
import { TuningLogViewer } from '@granite-build/ui-core/components/autotunex/tunings/TuningLogViewer'
import { TuningResultsPanel } from '@granite-build/ui-core/components/autotunex/tunings/TuningResultsPanel'

/**
 * Trials, Logs & Results for the AutoTuneX tuning job linked to a build, mirroring
 * the AutoTuneX tuning detail page.
 *
 * BuildDetails owns the linked-job lookup (see useLinkedTuningJob) and only mounts
 * these once it holds a job, so none of them has a loading, error or empty state
 * for the lookup itself. (TuningResultsPanel's own states are about the job's
 * assets, not the job.)
 *
 * The scope travels with the job: anything fetched *about* the job has to use the
 * same one, or an admin who resolved another user's job then 403s on its logs and
 * sees an empty pane. AutoTuneXTrialsPanel takes no `scope` prop for this because
 * it doesn't need one passed down: TrialsTable derives the identical scope itself
 * from the same cached admin check (`useAutotunexIsAdmin`, see
 * packages/ui-core/components/autotunex/trials/TrialsTable.tsx), so a prop
 * here would be redundant, not missing.
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

// No `scope` prop, for the same reason as AutoTuneXTrialsPanel: TuningResultsPanel
// derives the identical scope from the same cached admin check.
export function AutoTuneXResultsPanel({ job }: { job: JobDetail }) {
  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <TuningResultsPanel jobId={job.id} jobStatus={job.status} />
    </div>
  )
}
