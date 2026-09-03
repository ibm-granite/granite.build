'use client'

import { getTrialLogs } from '../api/autotunex'
import { useScrollingLogs } from '../hooks/useScrollingLogs'
import type { TuningStatus } from '../types'
import { LogLines } from './LogLines'

const ACTIVE_STATUSES = new Set<TuningStatus>(['running', 'pending'])

interface Props {
  jobId: string
  trialId: string
  status: TuningStatus
}

export function TrialLogViewer({ jobId, trialId, status }: Props) {
  const isActive = ACTIVE_STATUSES.has(status)
  const { logs, isLoading, isLoadingMore, handleScroll } = useScrollingLogs({
    queryKey: ['autotunex-trial-logs', jobId, trialId],
    fetchLogs: (opts) => getTrialLogs(jobId, trialId, opts),
    isActive,
    pageSize: 50,
    pollIntervalMs: 60_000,
  })

  return <LogLines logs={logs} isLoading={isLoading} isLoadingMore={isLoadingMore} onScroll={handleScroll} />
}
