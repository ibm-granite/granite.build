/**
 * Pure pagination + adapter helpers for the AutoTuneX v0.3.5 client
 * (`@/api/autotunex`, which re-exports every name here). Split out so this
 * module has no runtime imports beyond type-only ones — `tests/autotunex-
 * adapters.test.js` `require()`s it directly under plain `node --test`, which
 * (unlike the Next.js/tsc toolchain) cannot resolve the `@/…` path alias for a
 * value import. Keep it that way: no non-type imports here.
 */
import type {
  AiMappingSuggestion,
  Configuration,
  ConfigurationJobRef,
  ConfigData,
  ListParams,
  ListResult,
  Trial,
  TuningAsset,
  TuningJob,
  TuningStatus,
} from '../types'

// ── Pagination ──────────────────────────────────────────────────────────────────

/** Converts the frontend's {page,pageSize,q,scope} into the API's ?limit&offset&q&scope. */
export function pageQuery(p: ListParams): Record<string, string | number> {
  const limit = Math.min(p.pageSize, 100)
  // offset is derived from the CLAMPED limit, not the requested pageSize: with a
  // pageSize above the cap the two disagree and whole records are skipped —
  // page 2 of pageSize 150 asked for offset 150 while only fetching 100, so
  // records 100-149 were unreachable.
  const offset = Math.max(0, (p.page - 1) * limit)
  const params: Record<string, string | number> = { limit, offset, scope: p.scope ?? 'own' }
  if (p.q) params.q = p.q
  return params
}

/** Reshapes a `{items,total,limit,offset}` envelope into `{items,total}`, mapping each item through `adapt`. */
export function toListResult<T>(data: { items?: unknown[]; total?: number }, adapt: (raw: any) => T): ListResult<T> {
  return { items: (data.items ?? []).map(adapt), total: data.total ?? 0 }
}

/**
 * Drains an offset-paginated `{items,total}` endpoint into one array.
 *
 * `fetchPage` is injected rather than imported so this module stays free of
 * runtime imports (see the header comment — the test `require()`s this file
 * directly under `node --test`).
 *
 * Termination, in order of what each condition is for:
 *
 *  1. A short page (`items.length < limit`) is the correctness guarantee. Every
 *     terminating case reduces to it, an empty page included (`0 < limit`), so a
 *     server reporting a `total` larger than the rows it actually serves stops
 *     here rather than looping.
 *  2. `out.length >= total` is only an optimization: when `total` is accurate and
 *     the last page happens to be exactly full, it saves the one extra request
 *     that would otherwise be spent discovering an empty page. It is never
 *     load-bearing — `total` is server-reported, and trials are appended while a
 *     job runs, so it can be stale in either direction mid-drain.
 */
export async function collectPages<T>(
  fetchPage: (limit: number, offset: number) => Promise<{ items?: unknown[]; total?: number }>,
  adapt: (raw: any) => T,
  limit = 100
): Promise<T[]> {
  const out: T[] = []
  for (let offset = 0; ; offset += limit) {
    const page = await fetchPage(limit, offset)
    const items = page.items ?? []
    for (const raw of items) out.push(adapt(raw))
    if (items.length < limit) break
    if (out.length >= (page.total ?? Infinity)) break
  }
  return out
}

// ── Trials ───────────────────────────────────────────────────────────────────────

export function adaptTrial(raw: Record<string, unknown>): Trial {
  return {
    id: (raw.id as string) ?? '',
    job_id: (raw.job_id as string) ?? '',
    status: (raw.status as TuningStatus) ?? 'pending',
    config: (raw.config as Record<string, any>) ?? {},
    metric: (raw.metric as string) ?? undefined,
    metrics: (raw.metrics as Record<string, number>) ?? {},
    created_at: (raw.created_at as string) ?? '',
    updated_at: (raw.updated_at as string) ?? '',
  }
}

// ── Jobs ─────────────────────────────────────────────────────────────────────────

/** Maps the JobSummary fields shared by list rows and the detail record. */
export function adaptJob(raw: Record<string, unknown>): TuningJob {
  return {
    id: raw.id as string,
    user_id: raw.user_id as string,
    status: (raw.status as TuningStatus) ?? 'pending',
    seed: (raw.seed as number) ?? 0,
    config_id: raw.config_id as string,
    config_name: raw.config_name as string,
    dataset_id: raw.dataset_id as string,
    dataset: raw.dataset as string,
    model: raw.model as string,
    experiment_name: raw.experiment_name as string,
    user: (raw.user as string) ?? '',
    created_at: raw.created_at as string,
    updated_at: raw.updated_at as string,
  }
}

// ── Configurations ────────────────────────────────────────────────────────────

function adaptConfigurationJobRefs(raw: unknown): ConfigurationJobRef[] {
  if (!Array.isArray(raw)) return []
  return raw.map((r) => {
    const ref = (r ?? {}) as Record<string, unknown>
    return {
      id: ref.id as string,
      experiment_name: ref.experiment_name as string | undefined,
      status: (ref.status as TuningStatus) ?? 'pending',
    }
  })
}

export function adaptConfiguration(raw: Record<string, unknown>): Configuration {
  return {
    id: raw.id as string,
    user_id: raw.user_id as string,
    name: raw.name as string,
    tuner_type: raw.tuner_type as string,
    rl_tuner_type: (raw.rl_tuner_type as string | null | undefined) ?? null,
    config_data: (raw.config_data as ConfigData | null | undefined) ?? null,
    created_at: raw.created_at as string | undefined,
    updated_at: raw.updated_at as string | undefined,
    associated_jobs: adaptConfigurationJobRefs(raw.associated_jobs),
  }
}

// ── Results / output assets ────────────────────────────────────────────────────

/**
 * Maps one `AssetSummary` row from GET /jobs/{id}/result-report. Only
 * `filename`/`size` are guaranteed server-side; the rest default to null.
 * `published: false` is preserved (only null/undefined fall through to null).
 */
export function adaptAsset(raw: Record<string, unknown>): TuningAsset {
  return {
    filename: (raw.filename as string) ?? '',
    size: (raw.size as number) ?? 0,
    modified: (raw.modified as string | null) ?? null,
    path: (raw.path as string | null) ?? null,
    file_hash: (raw.file_hash as string | null) ?? null,
    published: (raw.published as boolean | null) ?? null,
  }
}

// ── AI-assisted column mapping ────────────────────────────────────────────────

export function adaptSuggestion(raw: Record<string, unknown>): AiMappingSuggestion {
  return {
    dataset_format: raw.dataset_format as string,
    tuning_type: raw.tuning_type as string,
    confidence: raw.confidence as number,
    column_mapping: (raw.column_mapping as Record<string, string>) ?? {},
    column_confidence: raw.column_confidence as Record<string, number> | undefined,
    reasoning: raw.reasoning as string | undefined,
  }
}
