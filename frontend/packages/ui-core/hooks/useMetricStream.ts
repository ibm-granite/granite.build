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

import { useEffect, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { collectKeysetPages } from '../api/autotunexAdapters'
import type { MetricPage, MetricPoint } from '../types'

/** Same cadence the trials table polls at, so the tab advances as one. */
const POLL_MS = 15_000

/**
 * Accumulating reader for the ascending keyset metric endpoints.
 *
 * The rows are append-only and the server pages forward from an id, so a live
 * job only needs the rows above the highest id already held — one small request
 * per tick rather than redrawing the whole run. The accumulator lives in a ref
 * keyed by the query key, so switching job or scope starts a fresh stream
 * instead of appending onto the previous one's rows.
 *
 * When a tick brings nothing new the previous array is returned by identity, so
 * React Query reports no change and nothing re-renders.
 */
export function useMetricStream(
  queryKey: unknown[],
  fetchPage: (afterId: number) => Promise<MetricPage>,
  opts: { isActive: boolean; enabled?: boolean }
) {
  const held = useRef<{ key: string; rows: MetricPoint[] }>({ key: '', rows: [] })
  const key = JSON.stringify(queryKey)

  const query = useQuery<MetricPoint[]>({
    queryKey,
    enabled: opts.enabled ?? true,
    refetchInterval: opts.isActive ? POLL_MS : false,
    queryFn: async () => {
      if (held.current.key !== key) held.current = { key, rows: [] }
      const rows = held.current.rows
      const fromId = rows.length ? rows[rows.length - 1].id : 0
      const fresh = await collectKeysetPages<MetricPoint>(
        async (afterId) => {
          const page = await fetchPage(afterId)
          return { items: page.metrics, hasMore: page.hasMore, nextAfterId: page.nextAfterId }
        },
        fromId
      )
      if (fresh.length) held.current = { key, rows: [...rows, ...fresh] }
      return held.current.rows
    },
  })

  // One last read when the job stops. `isActive` going false only cancels the
  // poll, so rows written between the last tick and the stop -- typically the
  // end-of-training summary the Final run tiles read -- never arrived until the
  // tab was refocused or reloaded.
  const wasActive = useRef(opts.isActive)
  const { refetch } = query
  const enabled = opts.enabled ?? true
  useEffect(() => {
    if (wasActive.current && !opts.isActive && enabled) void refetch()
    wasActive.current = opts.isActive
  }, [opts.isActive, enabled, refetch])

  return query
}
