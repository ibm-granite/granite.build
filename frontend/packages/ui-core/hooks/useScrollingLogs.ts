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
  // Two flags, not one. `pollHasMore` is what the *newest* page reports, which is
  // "entries exist before this page" and therefore stays true for any job with
  // history — so it cannot be the whole answer: once loadMore has walked back to
  // the real beginning, the next poll would otherwise flip pagination back on and
  // re-serve pages already held. `olderExhausted` is loadMore's own verdict and
  // wins over the poll.
  const [pollHasMore, setPollHasMore] = useState(true)
  const [olderExhausted, setOlderExhausted] = useState(false)
  const [isLoadingMore, setIsLoadingMore] = useState(false)
  const [loadMoreFailed, setLoadMoreFailed] = useState(false)

  // Reset when the caller switches subject. TuningDetailPageClient navigates from
  // one tuning to another *without* remounting (both are the same Next route), so
  // without this the previous job's lines stay merged into the next job's panel.
  // Done during render rather than in an effect so no frame ever paints job A's
  // logs under job B's heading.
  const subject = JSON.stringify(queryKey)
  const [prevSubject, setPrevSubject] = useState(subject)
  if (subject !== prevSubject) {
    setPrevSubject(subject)
    setLogs([])
    setPollHasMore(true)
    setOlderExhausted(false)
    setIsLoadingMore(false)
    setLoadMoreFailed(false)
  }

  // Polls the newest page and merges it in; does not affect older pages
  // already appended via scroll.
  const { data, isLoading, isError } = useQuery({
    queryKey,
    queryFn: () => fetchLogs({ beforeId: 0, limit: pageSize }),
    refetchInterval: isActive ? pollIntervalMs : false,
  })

  useEffect(() => {
    if (!data) return
    setLogs((prev) => mergeLogs(prev, data.logs))
    setPollHasMore(data.hasMore)
  }, [data])

  const hasMore = pollHasMore && !olderExhausted && !loadMoreFailed

  async function loadMore() {
    if (isLoadingMore || !hasMore || logs.length === 0) return
    setIsLoadingMore(true)
    try {
      const oldestId = logs[logs.length - 1].id
      const next = await fetchLogs({ beforeId: oldestId, limit: pageSize })
      setLogs((prev) => mergeLogs(prev, next.logs))
      if (!next.hasMore) setOlderExhausted(true)
    } catch {
      // Stop paginating instead of retrying on every scroll event: handleScroll
      // fires near the bottom, so an endpoint that 403s would otherwise be re-hit
      // for as long as the user keeps scrolling, one unhandled rejection each.
      setLoadMoreFailed(true)
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

  return { logs, isLoading, isLoadingMore, isError: isError || loadMoreFailed, handleScroll }
}
