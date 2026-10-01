import type { LogEntry } from '../../types'

/**
 * Newest-first, deduped by id — merges a polled "latest" page with scroll-loaded
 * older pages without disturbing already-loaded history.
 */
export function mergeLogs(existing: LogEntry[], incoming: LogEntry[]): LogEntry[] {
  const byId = new Map(existing.map((log) => [log.id, log]))
  for (const log of incoming) byId.set(log.id, log)
  return [...byId.values()].sort((a, b) => b.id - a.id)
}

/**
 * The `beforeId` to backfill from when a polled page does not connect to the
 * history already held — or null when it does.
 *
 * The live poll only ever asks for the newest page, and `mergeLogs` unions by id
 * without checking that the page it is given is contiguous with what is held. A job
 * that emits more than `pageSize` lines between ticks therefore left a hole: hold
 * ids 101-300, poll returns 601-800, and the panel rendered one continuous block
 * with 301-600 missing and nothing indicating it. `loadMore` could not repair it
 * either, because it walks back from the *oldest* held id and so only ever fetches
 * older than the whole range, leaving an interior gap unreachable for the life of
 * the panel.
 *
 * Detection is by overlap, not by id arithmetic: these ids are only guaranteed to be
 * ordered, not gap-free per job, so "polledOldest === heldNewest + 1" is not a
 * reliable test for adjacency. Any page that starts strictly after everything held
 * is treated as needing reconciliation; when it turns out to have been adjacent
 * after all, the first backfill page overlaps the held range and the walk stops
 * immediately, which costs one request and cannot lose lines.
 */
export function logGapCursor(held: LogEntry[], polled: LogEntry[]): number | null {
  if (held.length === 0 || polled.length === 0) return null
  const heldNewest = Math.max(...held.map((l) => l.id))
  const polledOldest = Math.min(...polled.map((l) => l.id))
  return polledOldest > heldNewest ? polledOldest : null
}

/**
 * Whether a backfill page has reconnected with the history already held, i.e. it
 * reaches at or below the newest id that was held when the gap was found.
 */
export function gapReconciled(page: LogEntry[], heldNewestAtDetection: number): boolean {
  if (page.length === 0) return true
  return Math.min(...page.map((l) => l.id)) <= heldNewestAtDetection
}

/** A hole still to backfill: walk older pages from `beforeId` until one reaches `downTo`. */
export interface LogGap {
  beforeId: number
  downTo: number
}

/**
 * The gap to backfill after a poll, given the one still pending (if any).
 *
 * Detection must run on every poll, pending gap or not. It used to be skipped
 * while a walk was unfinished, so a second hole opened by the next tick went
 * unregistered -- and once the first walk finished it sat below the newest held
 * id, where `logGapCursor` (which only looks above it) and `loadMore` (which only
 * walks below the oldest) could never reach it. A second hole widens the pending
 * one instead: the walk restarts at the new leading edge and keeps the older lower
 * bound, so one walk covers both holes. The pages it re-fetches between them are
 * already held and `mergeLogs` dedups them.
 */
export function nextPendingGap(pending: LogGap | null, held: LogEntry[], polled: LogEntry[]): LogGap | null {
  const cursor = logGapCursor(held, polled)
  if (cursor === null) return pending
  if (pending) return { beforeId: cursor, downTo: pending.downTo }
  return { beforeId: cursor, downTo: Math.max(...held.map((l) => l.id)) }
}

