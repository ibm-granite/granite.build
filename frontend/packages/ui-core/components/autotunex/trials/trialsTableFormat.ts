// Display formatting for the trials table's cells. Kept out of TrialsTable.tsx
// for two reasons: the toolbar search filters on the same strings the cells
// render (Carbon's default filter matches raw values, so "5m 20" would miss a
// row showing "5m 20s" over a raw 320), and the frontend test harness has no
// jsdom and cannot require a .tsx file — see trialCompareGrouping.ts, which is
// split out for the same reason.

/**
 * A duration in seconds, to its two largest units. The one formatter for every
 * AutoTuneX "Total time": the trials table and Compare used to stop at minutes
 * while the tunings table and the Details tab carried hours, so the same 7200 s
 * read "120m 0s" in one column and "2h 0m" in the other.
 *
 * Deliberately not ui-core's lib/duration.ts, which formats gbserver build
 * durations: AutoTuneX runs need a days tier, and their timings arrive as
 * fractional seconds, which formatDurationSeconds would print unfloored.
 */
export function formatTime(seconds: number): string {
  if (seconds <= 0) return '0s'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${mins}m`
  if (mins > 0) return `${mins}m ${secs}s`
  return `${secs}s`
}

/**
 * Display text for one trials-table cell, keyed by its column.
 *
 * The branch order matters and matches what the cells rendered before this was
 * extracted: the missing-value fallback is last, so a `created_at` that somehow
 * arrives empty still goes through `new Date(...)` exactly as it used to rather
 * than silently becoming an em dash.
 */
export function formatCell(key: string, value: unknown): string {
  if (key === 'created_at') return new Date(value as string).toLocaleString()
  if (key === 'loss' && typeof value === 'number') return value.toFixed(4)
  if (key === 'total_time' && typeof value === 'number') return formatTime(value)
  return value === null || value === undefined ? '—' : String(value)
}
