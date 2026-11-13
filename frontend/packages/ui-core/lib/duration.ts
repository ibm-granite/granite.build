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

/**
 * Elapsed-time formatting, shared by every place that shows a duration so the
 * dashboard tiles and the build drawer never drift apart.
 *
 * Format: `6s`, `2m 4s`, `2m` (trailing zero units dropped), `1h 3m`.
 */
export function formatDurationSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`
  const h = Math.floor(m / 60)
  const rm = m % 60
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`
}

/**
 * The same format, derived from a start/finish timestamp pair — gbserver records
 * no duration field on steps or targets.
 *
 * Returns undefined when either bound is missing or unparseable, or when the
 * pair is inverted: clock skew between the launcher and gbserver can put finish
 * before start, and showing a negative duration would be worse than showing
 * none.
 */
export function formatDurationBetween(
  from: string | undefined,
  to: string | undefined,
): string | undefined {
  if (!from || !to) return undefined
  const start = new Date(from).getTime()
  const end = new Date(to).getTime()
  if (Number.isNaN(start) || Number.isNaN(end)) return undefined
  const totalSeconds = Math.round((end - start) / 1000)
  if (totalSeconds < 0) return undefined
  return formatDurationSeconds(totalSeconds)
}
