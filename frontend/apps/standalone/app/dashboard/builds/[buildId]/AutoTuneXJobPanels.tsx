'use client'

import { SkeletonText, InlineNotification } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getJobByBuildId } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import { TrialsTable } from '@granite-build/ui-core/components/TrialsTable'
import { TuningLogViewer } from '@granite-build/ui-core/components/TuningLogViewer'

/**
 * Trials & Logs for the AutoTuneX tuning job linked to a build, mirroring the
 * AutoTuneX tuning detail page. Both panels share the linked-job query with
 * AutoTuneXPanel (same query key), so they resolve from cache.
 */

// Shared fetch of the tuning job linked to this build.
function useLinkedTuningJob(buildId: string) {
  // Same "admin of at least one space" gate used by the tunings/settings
  // tables — admins get `scope=all` so these panels can resolve a job linked
  // to the build even when it doesn't belong to the viewer. Same queryKey
  // shape as AutoTuneXPanel's job query, so this shares its cache entry.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isAdmin = spaces.some((s) => s.is_admin)

  return useQuery({
    queryKey: ['autotunex-job-by-build', buildId, isAdmin],
    queryFn: () => getJobByBuildId(buildId, isAdmin ? 'all' : 'own'),
    // TrialsTable polls trials based on job.status, so a frozen status here
    // means that poll never stops. Refetch every 15s while the job is running
    // or pending so the UI stays current.
    refetchInterval: (query) => {
      const s = (query.state.data as { status?: string } | undefined)?.status
      return s && new Set(['running', 'pending']).has(s) ? 15_000 : false
    },
    enabled: Boolean(buildId),
  })
}

function Loading() {
  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <SkeletonText paragraph lineCount={6} />
    </div>
  )
}

function NoJob() {
  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <InlineNotification
        kind="info"
        title="No Model Customization job linked to this build"
        hideCloseButton
        lowContrast
      />
    </div>
  )
}

export function AutoTuneXTrialsPanel({ buildId }: { buildId: string }) {
  const { data: job, isLoading } = useLinkedTuningJob(buildId)

  if (isLoading) return <Loading />
  if (!job) return <NoJob />

  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <TrialsTable job={job} />
    </div>
  )
}

export function AutoTuneXLogsPanel({ buildId }: { buildId: string }) {
  const { data: job, isLoading } = useLinkedTuningJob(buildId)

  if (isLoading) return <Loading />
  if (!job) return <NoJob />

  return (
    <div style={{ padding: '1rem 1.5rem' }}>
      <TuningLogViewer jobId={job.id} status={job.status} />
    </div>
  )
}
