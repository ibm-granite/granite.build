'use client'

import { getJobLogs } from '../api/autotunex'
import { useScrollingLogs } from '../hooks/useScrollingLogs'
import type { TuningJob } from '../types'
import { LogLines } from './LogLines'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

interface Props {
  jobId: string
  status: TuningJob['status']
  /** Cap the inline scroll container height (px). Omit to keep the SCSS 70vh default (detail page). */
  maxHeight?: number
  /** List scope to fetch logs under. Omit to use getJobLogs' 'own' default (detail page). */
  scope?: 'own' | 'all'
}

export function TuningLogViewer({ jobId, status, maxHeight, scope }: Props) {
  const isActive = ACTIVE_STATUSES.has(status)
  const { logs, isLoading, isLoadingMore, handleScroll } = useScrollingLogs({
    // scope is part of the key so 'own' vs 'all' fetches don't collide in the RQ cache.
    queryKey: ['autotunex-job-logs', jobId, scope ?? 'own'],
    fetchLogs: (opts) => getJobLogs(jobId, { ...opts, scope }),
    isActive,
  })

  return (
    <LogLines
      logs={logs}
      isLoading={isLoading}
      isLoadingMore={isLoadingMore}
      onScroll={handleScroll}
      maxHeight={maxHeight}
    />
  )
}
