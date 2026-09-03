import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { LogEntry } from '../types'

const DEFAULT_PAGE_SIZE = 200
const DEFAULT_POLL_MS = 10_000
const SCROLL_THRESHOLD_PX = 48

// Newest-first, deduped by id — merges a polled "latest" page with
// scroll-loaded older pages without disturbing already-loaded history.
function mergeLogs(existing: LogEntry[], incoming: LogEntry[]): LogEntry[] {
  const byId = new Map(existing.map((log) => [log.id, log]))
  for (const log of incoming) byId.set(log.id, log)
  return [...byId.values()].sort((a, b) => b.id - a.id)
}

interface UseScrollingLogsOptions {
  queryKey: unknown[]
  fetchLogs: (opts: { beforeId: number; limit: number }) => Promise<{ logs: LogEntry[]; hasMore: boolean }>
  isActive: boolean
  pageSize?: number
  pollIntervalMs?: number
}

export function useScrollingLogs({
  queryKey,
  fetchLogs,
  isActive,
  pageSize = DEFAULT_PAGE_SIZE,
  pollIntervalMs = DEFAULT_POLL_MS,
}: UseScrollingLogsOptions) {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [hasMore, setHasMore] = useState(true)
  const [isLoadingMore, setIsLoadingMore] = useState(false)

  // Polls the newest page and merges it in; does not affect older pages
  // already appended via scroll.
  const { data, isLoading } = useQuery({
    queryKey,
    queryFn: () => fetchLogs({ beforeId: 0, limit: pageSize }),
    refetchInterval: isActive ? pollIntervalMs : false,
  })

  useEffect(() => {
    if (!data) return
    setLogs((prev) => mergeLogs(prev, data.logs))
    setHasMore(data.hasMore)
  }, [data])

  async function loadMore() {
    if (isLoadingMore || !hasMore || logs.length === 0) return
    setIsLoadingMore(true)
    try {
      const oldestId = logs[logs.length - 1].id
      const next = await fetchLogs({ beforeId: oldestId, limit: pageSize })
      setLogs((prev) => mergeLogs(prev, next.logs))
      setHasMore(next.hasMore)
    } finally {
      setIsLoadingMore(false)
    }
  }

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget
    if (el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_THRESHOLD_PX) {
      void loadMore()
    }
  }

  return { logs, isLoading, isLoadingMore, handleScroll }
}
