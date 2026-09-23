/**
 * Pure decision logic for the HuggingFace dataset-import flow.
 *
 * Split out of the HuggingFace import UI because `node --test` is the only automated
 * coverage this app has -- there is no jsdom harness -- and it can only load a
 * module Node itself can resolve. Every import here must therefore be
 * `import type`: type-stripping erases those before resolution, whereas a value
 * import of `@granite-build/ui-core/*` fails outright (no workspace-alias
 * resolution in Node). See wizardDraft.ts for the same constraint.
 */

import type { HfProvenance } from '@granite-build/ui-core/types'

/**
 * The share of the train split held back for validation when no separate
 * validation split is chosen.
 *
 * A constant, not a form field: the Upload path shows no ratio control either --
 * StartTuningWizard's `splitRatio` is a fixed 80 -- so a percentage input here
 * would be the one place in the wizard where the split is negotiable. Keep this at
 * `100 - splitRatio`; the two paths splitting differently is invisible to the user
 * and only shows up in the row counts after an import.
 */
export const HF_VALIDATION_PERCENTAGE = 20

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

export type SurvivalKind = 'hidden' | 'blocked' | 'warning' | 'ok'

export interface SurvivalSummary {
  kind: SurvivalKind
  text: string
}

/**
 * What to tell the user about a mapping's survival count, and whether to block.
 *
 * `hidden` before the mapping is complete: the server counts survivors over the
 * mapping's own keys, so a half-filled mapping yields a high number describing
 * only the columns chosen so far -- reassuring and close to meaningless.
 *
 * Zero survivors blocks (the mapping is simply wrong); anything else warns and
 * allows, because real Hub datasets are legitimately ragged and an 85%-clean one
 * may be exactly what someone wants.
 */
export function survivalSummary(input: {
  sampled: number
  survived: number
  mappingComplete: boolean
}): SurvivalSummary {
  const { sampled, survived, mappingComplete } = input
  if (!mappingComplete) return { kind: 'hidden', text: '' }
  if (sampled === 0) return { kind: 'blocked', text: 'The selected split returned no rows.' }
  if (survived === 0) {
    return {
      kind: 'blocked',
      text: `No rows survived this mapping (0 of ${sampled} sampled rows). Check the column mapping.`,
    }
  }
  if (survived < sampled) {
    return {
      kind: 'warning',
      text: `${survived} of ${sampled} sampled rows have every mapped column filled.`,
    }
  }
  return { kind: 'ok', text: `All ${sampled} sampled rows have every mapped column filled.` }
}

/** Preselect a config, still explicitly shown: a silently wrong pick is only
 *  discovered after a multi-hour tuning run. */
export function defaultConfig(configNames: string[]): string {
  if (configNames.includes('default')) return 'default'
  return configNames[0] ?? ''
}

/** Same contract as defaultConfig, for the train split. */
export function defaultTrainSplit(splitNames: string[]): string {
  if (splitNames.includes('train')) return 'train'
  return splitNames[0] ?? ''
}

// Carbon's Select needs a real option value; null is not one. Lives here rather
// than in useHfImport.ts so the rule functions below can return it -- useHfImport
// imports from this module, so the reverse direction would be circular.
export const NO_VALIDATION = '__none__'

/**
 * The validation split to offer when the user turns the split toggle off.
 *
 * Matches `validation` exactly and never falls back to `test`. Selecting a model
 * against the held-out test split is a methodology error, and an auto-pick is only
 * discovered after a multi-hour tuning run. `''` means "the user asked for a
 * separate split but this dataset names none obviously" -- it renders as a
 * placeholder and `canImport` blocks on it.
 */
export function preselectValidationSplit(candidates: string[]): string {
  return candidates.includes('validation') ? 'validation' : ''
}

/**
 * Re-settle the validation selection after the train split changed.
 *
 * The candidate list is `splitNames` minus the *incoming* train split, so a
 * selection can go stale: picking `test` as validation and then making `test` the
 * train split would otherwise post the same split as both.
 */
export function reconcileValidationSplit(current: string, nextCandidates: string[]): string {
  // The toggle hides with no candidates, so its off-state must not survive.
  if (nextCandidates.length === 0) return NO_VALIDATION
  // Checked before the membership test below: the sentinel is never a candidate.
  if (current === NO_VALIDATION) return NO_VALIDATION
  if (nextCandidates.includes(current)) return current
  return preselectValidationSplit(nextCandidates)
}

// Pinned locale: the default is the host's, which would make both this copy and
// its tests machine-dependent.
function formatCount(value: number): string {
  return value.toLocaleString('en-US')
}

/**
 * Whether the import hit the row cap, and what to say about it.
 *
 * Above `max_rows` the server takes the first N rather than refusing. The
 * frontend cannot know the total beforehand -- the preview samples 100 rows -- so
 * this reads it back out of the provenance the import wrote. Both splits are
 * checked: a truncated validation split with an untouched train split is a real
 * outcome that a train-only check would report as clean.
 *
 * Comparing the two row counts is not sufficient on its own. `_original_rows` sums
 * only the shards the import actually opened, so a cap that lands exactly on a
 * shard boundary leaves the counts equal while whole shards went unread -- which
 * this reported as a clean import. `_truncated` is the server's own verdict over
 * both signals and is authoritative when present; the row comparison remains the
 * fallback for imports made before the server recorded it. Deliberately not the
 * other way round: a `false` flag must never suppress row counts that plainly
 * disagree with it.
 */
export function truncationNotice(
  provenance: HfProvenance | null | undefined,
  maxRows: number
): string | null {
  if (!provenance) return null
  const splits: [
    string,
    number | undefined,
    number | undefined,
    boolean | null | undefined,
    number | null | undefined,
  ][] = [
    [
      'train',
      provenance.train_original_rows,
      provenance.train_retained_rows,
      provenance.train_truncated,
      provenance.train_unread_shards,
    ],
    [
      'validation',
      provenance.validation_original_rows,
      provenance.validation_retained_rows,
      provenance.validation_truncated,
      provenance.validation_unread_shards,
    ],
  ]
  const parts: string[] = []
  for (const [label, original, retained, truncated, unreadShards] of splits) {
    if (typeof retained !== 'number') continue
    if (typeof original === 'number' && retained < original) {
      parts.push(
        `Imported the first ${formatCount(retained)} of ${formatCount(original)} ${label} rows (capped at ${formatCount(maxRows)}).`
      )
      continue
    }
    if (truncated !== true) continue
    // The rows agree, so there is no honest total to quote -- `_original_rows` counts
    // only what was opened. Name the shards left unread instead.
    const remainder =
      typeof unreadShards === 'number' && unreadShards > 0
        ? `${formatCount(unreadShards)} further ${unreadShards === 1 ? 'shard was' : 'shards were'} not read`
        : 'more rows remain upstream'
    parts.push(
      `Imported ${formatCount(retained)} ${label} rows (capped at ${formatCount(maxRows)}); ${remainder}.`
    )
  }
  return parts.length > 0 ? parts.join(' ') : null
}

/**
 * The user-facing message for a failed request.
 *
 * Every backend error is an RFC 9457 problem detail whose `detail` is authored,
 * user-safe copy, so rendering it verbatim beats deriving a message from the
 * status code -- and it is the only way to distinguish the two 503s, which share
 * `title: "Service Unavailable"` and differ only in `detail`.
 *
 * Read structurally rather than via `axios.isAxiosError`, because keeping this
 * module free of value imports is what lets `node --test` load it.
 */
export function problemDetail(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } } | null | undefined)
    ?.response?.data?.detail
  return typeof detail === 'string' && detail.trim() !== '' ? detail : fallback
}

/**
 * The HTTP status of a failed request, or undefined when there is none.
 *
 * Read structurally rather than via `axios.isAxiosError`, for the same reason as
 * `problemDetail`: keeping this module free of value imports is what lets
 * `node --test` load it. Callers branch on this — 503 is retryable, 422 is not,
 * 409 means a duplicate dataset name — so it is worth a test.
 */
export function hfErrorStatus(err: unknown): number | undefined {
  const status = (err as { response?: { status?: unknown } } | null | undefined)?.response?.status
  return typeof status === 'number' ? status : undefined
}

export type PollDecision = 'ready' | 'error' | 'wait' | 'timeout'

/**
 * What the import poll should do this tick.
 *
 * `ready` and `error` outrank an expired deadline: a run that finished on the
 * same tick it timed out has finished. A tick with no status (the GET failed)
 * waits while there is time left and times out afterwards — which is also why a
 * post-deadline fetch rejection surfaces the authored timeout copy rather than a
 * transport error the user cannot act on.
 */
export function pollStep(input: { status: string | undefined; expired: boolean }): PollDecision {
  if (input.status === 'ready') return 'ready'
  if (input.status === 'error') return 'error'
  if (input.expired) return 'timeout'
  return 'wait'
}

/** Identity of a mapped preview: which repo, config, split and mapping produced it. */
export function mappedPreviewKey(input: {
  repoId: string
  config: string
  trainSplit: string
  mappingKey: string
}): string {
  return [input.repoId, input.config, input.trainSplit, input.mappingKey].join('|')
}

/**
 * Whether the import may be submitted. Every clause is a way to ship a lossy or
 * impossible import, which is why this lives here rather than inline in the form.
 */
export function canImport(input: {
  hasRepo: boolean
  hasConfig: boolean
  hasTrainSplit: boolean
  mappingComplete: boolean
  nameValid: boolean
  survivalKind: SurvivalKind
  importing: boolean
  /** True when no separate validation split is chosen, so the percentage applies. */
  splitFromTrain: boolean
  validationPercentage: number
}): boolean {
  if (!input.hasRepo || !input.hasConfig || !input.hasTrainSplit) return false
  if (!input.mappingComplete || !input.nameValid || input.importing) return false
  if (input.survivalKind !== 'ok' && input.survivalKind !== 'warning') return false
  if (
    input.splitFromTrain &&
    !(
      Number.isInteger(input.validationPercentage) &&
      input.validationPercentage >= 1 &&
      input.validationPercentage <= 50
    )
  ) {
    return false
  }
  return true
}

/**
 * Drop mapping entries whose target is no longer required.
 *
 * The AI suggestion may change the selected algorithm, which changes the required
 * columns. The mapping then still holds the previous algorithm's targets, and
 * those stale keys would ride along in the import request body -- the server
 * would receive a projection mixing both algorithms' columns.
 *
 * Returns the same object when nothing needs dropping, so the effect that calls
 * this can run on every required-columns change without looping.
 */
export function pruneMapping(
  mapping: Record<string, string>,
  requiredColumns: string[]
): Record<string, string> {
  const keys = Object.keys(mapping)
  const kept = keys.filter((key) => requiredColumns.includes(key))
  if (kept.length === keys.length) return mapping
  const next: Record<string, string> = {}
  for (const key of kept) next[key] = mapping[key]
  return next
}
