/**
 * Human-readable byte size.
 *
 * Was duplicated privately in SettingsDatasetView and TuningResultsPanel, and the
 * two copies had drifted: one stopped at GB. This is the superset.
 */
export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = bytes
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(1)} ${units[i]}`
}
