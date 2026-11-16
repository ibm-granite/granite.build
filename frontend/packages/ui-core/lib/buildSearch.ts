// SPDX-License-Identifier: Apache-2.0

/**
 * Helpers for the Builds search box, and the page clamp list views share.
 *
 * Search runs on gbserver: `/builds/`'s `name` filter takes a `*` wildcard
 * (case-insensitive), so a term becomes `*term*`. A term shaped like a build id
 * is also looked up directly, as ids aren't part of the name (see searchBuilds
 * in api/gbserver.ts).
 */

/** Build ids are UUIDs (gbserver's get_uuid). */
const BUILD_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

export function isBuildId(term: string): boolean {
  return BUILD_ID_RE.test(term.trim())
}

/**
 * The gbserver `name` wildcard matching names that contain `term`. `*` and `\`
 * in the term are escaped, so they match literally rather than as wildcards.
 */
export function nameContainsWildcard(term: string): string {
  return `*${term.trim().replace(/[\\*]/g, (c) => `\\${c}`)}*`
}

export function pageSlice<T>(items: T[], page: number, pageSize: number): T[] {
  return items.slice((page - 1) * pageSize, page * pageSize)
}

/** The nearest valid 1-based page, for when the list shrank under the current one. */
export function clampPage(page: number, total: number, pageSize: number): number {
  const lastPage = Math.max(1, Math.ceil(total / pageSize))
  return Math.min(Math.max(1, page), lastPage)
}
