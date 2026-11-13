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

import { getJobLogs } from '../../../api/autotunex'
import { useScrollingLogs } from '../../../hooks/useScrollingLogs'
import type { TuningJob } from '../../../types'
import { LogLines } from '../shared/LogLines'

const ACTIVE_STATUSES = new Set(['running', 'pending'])

interface Props {
  jobId: string
  status: TuningJob['status']
  /** Cap the inline scroll container height (px). Omit to keep the SCSS 70vh default (detail page). */
  maxHeight?: number
  /** List scope to fetch logs under. Must match the scope the enclosing job was
   *  resolved with — fetching an admin's 'all'-resolved job as 'own' 403s. */
  scope?: 'own' | 'all'
}

export function TuningLogViewer({ jobId, status, maxHeight, scope }: Props) {
  const isActive = ACTIVE_STATUSES.has(status)
  const { logs, isLoading, isLoadingMore, isError, handleScroll } = useScrollingLogs({
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
      isError={isError}
      onScroll={handleScroll}
      maxHeight={maxHeight}
    />
  )
}
