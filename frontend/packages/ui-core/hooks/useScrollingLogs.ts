import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { LogEntry } from '../types'
import { gapReconciled, logGapCursor, mergeLogs } from '../lib/autotunex/logStream'

const DEFAULT_PAGE_SIZE = 200
const DEFAULT_POLL_MS = 10_000
const SCROLL_THRESHOLD_PX = 48
// Pages fetched per tick while closing a gap. Bounded so a very chatty job cannot
// turn one poll into an unbounded request storm; an unfinished walk resumes on the
// next tick, so the bound delays reconciliation rather than abandoning it.
const MAX_BACKFILL_PAGES_PER_TICK = 10

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

  // The history held as of the last commit, mirrored into a ref so the poll effect
  // can compare the incoming page against the *pre-merge* state without taking
  // `logs` as a dependency (which would re-run it on every merge).
  const heldRef = useRef<LogEntry[]>([])

  // An unfinished gap walk: where to keep fetching from, and the id that marks
  // reconnection with the history held when the gap was found. Survives across ticks
  // so the per-tick page bound only delays the repair.
  const pendingGapRef = useRef<{ beforeId: number; downTo: number } | null>(null)
  const backfillingRef = useRef(false)

  // Reset when the caller switches subject. TuningDetailPageClient navigates from
  // one tuning to another *without* remounting (both are the same Next route), so
  // without this the previous job's lines stay merged into the next job's panel.
  // Done during render rather than in an effect so no frame ever paints job A's
  // logs under job B's heading.
  const subject = JSON.stringify(queryKey)
  // The current subject, readable from a callback that has already been created:
  // loadMore's closure captures the subject of the render it came from, so it
  // needs this to notice that the reset below has since run.
  const subjectRef = useRef(subject)
  subjectRef.current = subject
  const [prevSubject, setPrevSubject] = useState(subject)
  if (subject !== prevSubject) {
    setPrevSubject(subject)
    setLogs([])
    setPollHasMore(true)
    setOlderExhausted(false)
    setIsLoadingMore(false)
    setLoadMoreFailed(false)
    // A gap belongs to the subject it was found in.
    pendingGapRef.current = null
    heldRef.current = []
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
    const held = heldRef.current
    setLogs((prev) => mergeLogs(prev, data.logs))
    setPollHasMore(data.hasMore)

    // The poll only asks for the newest page, so a job that emitted more than one
    // page between ticks leaves a hole that `loadMore` can never reach — it walks
    // back from the oldest held id, not into the middle. Close it here instead.
    if (!pendingGapRef.current) {
      const cursor = logGapCursor(held, data.logs)
      if (cursor !== null) {
        pendingGapRef.current = { beforeId: cursor, downTo: Math.max(...held.map((l) => l.id)) }
      }
    }
    if (pendingGapRef.current) void closeGap(subjectRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  useEffect(() => {
    heldRef.current = logs
  }, [logs])

  // Walks older pages from the gap's leading edge until it reconnects with the
  // history that was held when the gap was found. A failure leaves the gap in place
  // rather than retrying per scroll event; the next poll tick tries again.
  async function closeGap(issuedFor: string) {
    if (backfillingRef.current) return
    backfillingRef.current = true
    try {
      for (let page = 0; page < MAX_BACKFILL_PAGES_PER_TICK; page++) {
        const gap = pendingGapRef.current
        if (!gap) return
        const next = await fetchLogs({ beforeId: gap.beforeId, limit: pageSize })
        if (subjectRef.current !== issuedFor) return
        if (next.logs.length > 0) setLogs((prev) => mergeLogs(prev, next.logs))
        if (gapReconciled(next.logs, gap.downTo) || !next.hasMore) {
          pendingGapRef.current = null
          return
        }
        pendingGapRef.current = { ...gap, beforeId: Math.min(...next.logs.map((l) => l.id)) }
      }
    } catch {
      // Leave the gap pending; the next tick retries.
    } finally {
      backfillingRef.current = false
    }
  }

  const hasMore = pollHasMore && !olderExhausted && !loadMoreFailed

  async function loadMore() {
    if (isLoadingMore || !hasMore || logs.length === 0) return
    // The request can outlive the subject it was issued for -- the viewer is not
    // remounted when the page moves to another job -- and applying it anyway
    // merged the old job's lines into the new panel and, if the old job had no
    // more history, latched the new one's pagination off for good. That is exactly
    // what the render-phase reset above exists to prevent.
    const issuedFor = subject
    setIsLoadingMore(true)
    try {
      const oldestId = logs[logs.length - 1].id
      const next = await fetchLogs({ beforeId: oldestId, limit: pageSize })
      if (subjectRef.current !== issuedFor) return
      setLogs((prev) => mergeLogs(prev, next.logs))
      if (!next.hasMore) setOlderExhausted(true)
    } catch {
      // Stop paginating instead of retrying on every scroll event: handleScroll
      // fires near the bottom, so an endpoint that 403s would otherwise be re-hit
      // for as long as the user keeps scrolling, one unhandled rejection each.
      if (subjectRef.current === issuedFor) setLoadMoreFailed(true)
    } finally {
      // A stale run must not clear the flag a newer load is using.
      if (subjectRef.current === issuedFor) setIsLoadingMore(false)
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
