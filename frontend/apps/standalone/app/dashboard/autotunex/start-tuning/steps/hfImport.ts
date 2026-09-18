/**
 * Pure decision logic for the HuggingFace dataset-import flow.
 *
 * Split out of HfImportModal.tsx because `node --test` is the only automated
 * coverage this app has -- there is no jsdom harness -- and it can only load a
 * module Node itself can resolve. Every import here must therefore be
 * `import type`: type-stripping erases those before resolution, whereas a value
 * import of `@granite-build/ui-core/*` fails outright (no workspace-alias
 * resolution in Node). See wizardDraft.ts for the same constraint.
 */

// Matches the server's DatasetName: max 255, and no '/', '\' or '..' because the
// name becomes a filesystem path segment.
const NAME_MAX = 255
const ILLEGAL_RUN = /[^A-Za-z0-9._-]+/g
const DOT_RUN = /\.{2,}/g
const EDGE_PUNCTUATION = /^[.-]+|[.-]+$/g

/**
 * A dataset name from a repo id: the last path segment, sanitized.
 *
 * Dot runs are collapsed rather than merely stripped of illegal characters: '.'
 * is itself legal, so 'a..b' would otherwise pass this function and then 422 at
 * import on the server's '..' check.
 */
export function deriveDatasetName(repoId: string): string {
  const segment = repoId.split('/').filter(Boolean).pop() ?? ''
  const sanitized = segment
    .replace(ILLEGAL_RUN, '-')
    .replace(DOT_RUN, '.')
    .replace(EDGE_PUNCTUATION, '')
    .slice(0, NAME_MAX)
    .replace(EDGE_PUNCTUATION, '')
  return sanitized || 'dataset'
}

/** The server's own rule, applied client-side so a hand-edited name fails in the
 *  form rather than as a 422 after the user clicks Import. */
export function isDatasetNameValid(name: string): boolean {
  if (name.length === 0 || name.length > NAME_MAX) return false
  return !name.includes('/') && !name.includes('\\') && !name.includes('..')
}

/** Recovery for a 409: the same repo at a pinned revision is a legitimate second
 *  dataset, so the revision is what distinguishes it. Trims the head first so the
 *  result still fits, and re-trims edge punctuation so it stays valid. */
export function suffixWithRevision(name: string, revision: string): string {
  const short = revision.slice(0, 7)
  if (!short) return name
  const suffix = `-${short}`
  const head = name.slice(0, NAME_MAX - suffix.length).replace(EDGE_PUNCTUATION, '')
  return `${head}${suffix}`
}

/**
 * The throwaway mapping the first preview call carries.
 *
 * POST /datasets/hf/preview requires `column_mapping` to be non-empty, but that
 * same response is what supplies the column list the mapping form is built from.
 * A blank source is safe: the server's apply_mapping skips a target whose source
 * is blank or absent instead of raising, so `columns` and `raw_rows` come back
 * unaffected. Only `survived` is meaningless, and the caller discards it.
 */
export function probeMapping(requiredColumns: string[]): Record<string, string> {
  return { [requiredColumns[0] ?? 'input']: '' }
}

/**
 * Whether every required target has a source column.
 *
 * Gates both the mapped re-preview and the Import button. Returns false for an
 * empty `requiredColumns`: with no known targets there is no non-empty
 * column_mapping to send, so importing cannot succeed and must stay blocked
 * rather than pass vacuously.
 */
export function isMappingComplete(
  mapping: Record<string, string>,
  requiredColumns: string[]
): boolean {
  if (requiredColumns.length === 0) return false
  return requiredColumns.every((column) => Boolean(mapping[column]))
}
