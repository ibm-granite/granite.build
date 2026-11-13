// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use client'

import { getTrialLogs } from '../../../api/autotunex'
import { useScrollingLogs } from '../../../hooks/useScrollingLogs'
import type { TuningStatus } from '../../../types'
import { LogLines } from '../shared/LogLines'

const ACTIVE_STATUSES = new Set<TuningStatus>(['running', 'pending'])

interface Props {
  jobId: string
  trialId: string
  status: TuningStatus
  /** List scope to fetch logs under. Must match the scope the enclosing job was
   *  resolved with: an admin viewing another user's job resolves it as 'all', and
   *  fetching its logs as 'own' 403s, which surfaces as an empty pane. */
  scope?: 'own' | 'all'
}

export function TrialLogViewer({ jobId, trialId, status, scope }: Props) {
  const isActive = ACTIVE_STATUSES.has(status)
  const { logs, isLoading, isLoadingMore, isError, handleScroll } = useScrollingLogs({
    // scope is part of the key so 'own' vs 'all' fetches don't collide in the RQ cache.
    queryKey: ['autotunex-trial-logs', jobId, trialId, scope ?? 'own'],
    fetchLogs: (opts) => getTrialLogs(jobId, trialId, { ...opts, scope }),
    isActive,
    pageSize: 50,
    pollIntervalMs: 60_000,
  })

  return (
    <LogLines
      logs={logs}
      isLoading={isLoading}
      isLoadingMore={isLoadingMore}
      isError={isError}
      onScroll={handleScroll}
    />
  )
}
