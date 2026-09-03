'use client'

import { Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getJob } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { TuningStatusBadge } from '@granite-build/ui-core/components/TuningStatusBadge'
import { TuningDetailTabs } from './TuningDetailTabs'
import styles from './page.module.scss'

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
  // The real id lives in the ?id= query param, not location.hash — a hash read
  // in a mount-only effect breaks when navigating from one tuning's page to
  // another without an intervening route change, since both are the same "_"
  // route to Next's router and a one-time hash read never sees the new id.
  // useSearchParams() is reactive, but it isn't enough by itself: Next's router
  // patches window.history, so our own cosmetic replaceState below (stripping
  // the query param once we've adopted the id) makes useSearchParams() briefly
  // report no id again. Latching the id into state — only ever overwritten by
  // a new *non-empty* param value — survives that revert.
  const searchParams = useSearchParams()
  const paramId = searchParams.get('id')
  const [tuningId, setTuningId] = useState(paramId ?? '')

  useEffect(() => {
    if (paramId && paramId !== tuningId) {
      setTuningId(paramId)
    }
  }, [paramId, tuningId])

  useEffect(() => {
    if (tuningId) {
      window.history.replaceState(null, '', `/dashboard/autotunex/${tuningId}/`)
    }
  }, [tuningId])

  // Reused verbatim from the tunings list (`["spaces"]` queryKey) so the admin
  // check shares the same React Query cache entry rather than issuing a
  // duplicate `listSpaces()` fetch. Admins get `scope=all` so they can drill
  // into tunings they don't own.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isAdmin = spaces.some((s) => s.is_admin)

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
