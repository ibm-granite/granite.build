'use client'

import { Suspense } from 'react'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getJob } from '@granite-build/ui-core/api/autotunex'
import { useAutotunexIsAdmin } from '@granite-build/ui-core/hooks/useAutotunexIsAdmin'
import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { TuningStatusBadge } from '@granite-build/ui-core/components/autotunex/tunings/TuningStatusBadge'
import { TuningDetailTabs } from './TuningDetailTabs'
import styles from './page.module.scss'
import { useRouteId } from '@granite-build/ui-core/hooks/useRouteId'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

// useSearchParams() bails the page out to client-side rendering up to the
// nearest Suspense boundary during static export — without one here, the
// statically-exported HTML (built with no query param) and the client's
// first render (which already sees the real ?id=) disagree, tripping a
// hydration mismatch (React error #418). The fallback below matches what
// the static export produces so hydration has nothing to reconcile against.
//
// Same hydration-safe pattern as BuildDetailPageClient — see
// frontend/app/dashboard/builds/[buildId]/BuildDetailPageClient.tsx.
export default function TuningDetailPage() {
  return (
    <Suspense fallback={<TuningDetailFallback />}>
      <TuningDetailContent />
    </Suspense>
  )
}

function TuningDetailFallback() {
  return (
    <div style={{ padding: '2rem 1.5rem 1.5rem' }}>
      <PageHeader
        crumbs={[
          { label: 'Model Customization', to: '/dashboard/autotunex' },
          { label: '…' },
        ]}
      />
      <div className={styles.headerRow}>
        <SkeletonText width="300px" />
      </div>
    </div>
  )
}

function TuningDetailContent() {
  const tuningId = useRouteId('/dashboard/autotunex')

  // The same cached AutoTuneX admin check the tunings list uses. Admins get
  // `scope=all` so they can drill into tunings they don't own.
  const { isAdmin } = useAutotunexIsAdmin()

  const { data: job, isLoading, error } = useQuery({
    queryKey: ['autotunex-job', tuningId, isAdmin],
    queryFn: () => getJob(tuningId!, isAdmin ? 'all' : 'own'),
    refetchInterval: (query) => {
      const s = (query.state.data as { status?: string } | undefined)?.status
      return s && ACTIVE_STATUSES.has(s) ? 15_000 : false
    },
    enabled: Boolean(tuningId),
  })

  if (error) {
    return (
      <div style={{ padding: '1rem 1.5rem' }}>
        <InlineNotification kind="error" title="Failed to load tuning" subtitle={String(error)} />
      </div>
    )
  }

  return (
    <div>
      <div style={{ padding: '2rem 1.5rem 1.5rem' }}>
        <PageHeader
          crumbs={[
            { label: 'Model Customization', to: '/dashboard/autotunex' },
            { label: job?.experiment_name ?? '…' },
          ]}
        />
        <div className={styles.headerRow}>
          {isLoading ? (
            <SkeletonText width="300px" />
          ) : (
            <>
              <h4>{job?.experiment_name}</h4>
              {job && <TuningStatusBadge status={job.status} />}
            </>
          )}
        </div>
      </div>
      {job && <TuningDetailTabs job={job} />}
    </div>
  )
}
