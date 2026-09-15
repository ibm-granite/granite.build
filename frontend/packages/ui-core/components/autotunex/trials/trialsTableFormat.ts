// Display formatting for the trials table's cells. Kept out of TrialsTable.tsx
// for two reasons: the toolbar search filters on the same strings the cells
// render (Carbon's default filter matches raw values, so "5m 20" would miss a
// row showing "5m 20s" over a raw 320), and the frontend test harness has no
// jsdom and cannot require a .tsx file — see trialCompareGrouping.ts, which is
// split out for the same reason.

export function formatTime(seconds: number): string {
  if (seconds <= 0) return '0 s'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
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
