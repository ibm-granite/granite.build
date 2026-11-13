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
 * Client-side Builds search over a bounded window of recent builds.
 *
 * gbserver's `name` filter on `/builds/` is an exact match, so a substring
 * search can't be pushed to the server. While a term is active the Builds page
 * fetches the `SEARCH_WINDOW` most recent builds matching its other filters
 * once, then matches and paginates them here.
 */

export const SEARCH_WINDOW = 500

/** Which request a builds result came from: one server page, or a search window. */
export type BuildsFetchMode = 'page' | 'window'

export interface SearchableBuild {
  name: string
  uuid: string
  username?: string
  space_name?: string
}

export interface BuildsResult<T> {
  items: T[]
  total: number
}

export interface BuildsView<T> {
  items: T[]
  total: number
  /** The window held fewer builds than match the filters, so older ones weren't searched. */
  truncated: boolean
}

export function matchesBuildSearch(build: SearchableBuild, term: string): boolean {
  const needle = term.trim().toLowerCase()
  if (!needle) return true
  return [build.name, build.uuid, build.username, build.space_name].some((f) =>
    (f ?? '').toLowerCase().includes(needle),
  )
}

export function pageSlice<T>(items: T[], page: number, pageSize: number): T[] {
  return items.slice((page - 1) * pageSize, page * pageSize)
}

/** The nearest valid 1-based page, for when the list shrank under the current one. */
export function clampPage(page: number, total: number, pageSize: number): number {
  const lastPage = Math.max(1, Math.ceil(total / pageSize))
  return Math.min(Math.max(1, page), lastPage)
}

/**
 * What the Builds table should show for `data`, given the current search state.
 *
 * `data` may be React Query placeholder data fetched in the other mode while the
 * new request is in flight, so `mode` is the mode `data` was fetched in, not the
 * one now wanted. A placeholder is shown as it was, never re-derived as if it
 * were the other kind:
 *
 * - entering a search over a server page: the page as it was, no notice;
 * - clearing a search over a window: page 1 of the window (it is newest-first,
 *   like the page being fetched) against the server total, not all its rows.
 */
export function buildsView<T extends SearchableBuild>(
  data: BuildsResult<T> | undefined,
  mode: BuildsFetchMode | undefined,
  { term, page, pageSize }: { term: string; page: number; pageSize: number },
): BuildsView<T> {
  if (!data) return { items: [], total: 0, truncated: false }
  if (mode !== 'window') return { items: data.items, total: data.total, truncated: false }
  if (!term.trim()) {
    return { items: pageSlice(data.items, page, pageSize), total: data.total, truncated: false }
  }
  const matches = data.items.filter((b) => matchesBuildSearch(b, term))
  return {
    items: pageSlice(matches, page, pageSize),
    total: matches.length,
    truncated: data.total > data.items.length,
  }
}
