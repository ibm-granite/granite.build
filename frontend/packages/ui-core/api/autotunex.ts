/**
 * API client for the AutoTuneX backend (FastAPI service, proxied via
 * /api/autotunex/* — see `autotunexApiBase` in `@/api/client`).
 *
 * Dev mode: calls go through the Next.js dev proxy. Standalone builds target
 * AUTOTUNEX_API_URL directly when it's baked in at build time. `getHFModels`/
 * `getHFModelCard` are the two exceptions — they call the public HuggingFace
 * API directly via bare `axios`, not through this backend.
 *
 * Targets AutoTuneX API v0.3.5 (`/api/v1/*`, offset-paginated list envelopes
 * `{items,total,limit,offset}`). `pageQuery`/`toListResult` and the `adaptX()`
 * mappers live in `@/api/autotunexAdapters` (re-exported below) and are the
 * shared helpers every list endpoint (`getJobs`/`getConfigurations`/
 * `getDatasets`) funnels through — hiding the page↔offset math and envelope
 * reshaping behind the frontend-friendly `ListParams`/`ListResult<T>`
 * contract from `@/types`.
 */
import type {
  AiMappingSuggestion,
  Configuration,
  ConfigData,
  Dataset,
  DatasetStatus,
  Estimation,
  GbTask,
  HuggingFaceModel,
  JobDetail,
  JobRead,
  ListParams,
  ListResult,
  LogEntry,
  ModelSource,
  PendingConfigData,
  PendingConfigUpdate,
  Resources,
  RewardFunctionValidationResult,
  Trial,
  TuningAsset,
  TuningForm,
  TuningJob,
} from '../types'
import axios from 'axios'
import { autotunexApiBase } from './client'
import { normalizeVerlRows } from '../lib/autotunex/verlNormalize'
import {
  adaptAsset,
  adaptConfiguration,
  adaptJob,
  adaptSuggestion,
  adaptTrial,
  collectPages,
  pageQuery,
  toListResult,
} from './autotunexAdapters'

// Re-exported so `import { adaptJob, pageQuery, … } from './autotunex'`
// keeps working for tests/consumers — the implementations live in
// `@/api/autotunexAdapters` purely so that leaf module stays free of
// non-type-only imports (see its header comment for why that matters).
export { adaptAsset, adaptConfiguration, adaptJob, adaptSuggestion, adaptTrial, collectPages, pageQuery, toListResult }

const client = axios.create({ baseURL: autotunexApiBase('') })

type Scope = 'own' | 'all'

// ── Feature gates ──────────────────────────────────────────────────────────────
// These three v0.2 endpoints have no v0.3.5 equivalent yet. Flip a flag to
// `true` once the server ships the endpoint to re-enable the feature — no
// other code changes needed.
// These endpoints shipped in the AutoTuneX backend after the initial migration;
// the flags remain as a kill-switch (flip to false to fall back to the graceful
// "temporarily unavailable" UI without touching call sites).
export const AUTOTUNEX_FEATURES = {
  estimation: true, // POST /jobs/estimate-usages
  rewardValidation: true, // POST /reward-functions/validate
  testSolutions: true, // POST /jobs/generate-test-solutions
} as const

// ── HuggingFace models ────────────────────────────────────────────────────────

export async function getHFModels(search = '', limit = 10): Promise<HuggingFaceModel[]> {
  const params = new URLSearchParams({ search, limit: String(limit), config: 'true' })
  const { data } = await axios.get<{ models: HuggingFaceModel[] } | HuggingFaceModel[]>(
    `https://huggingface.co/api/models?${params.toString()}`
  )
  return Array.isArray(data) ? data : []
}

export async function getHFModelCard(modelId: string): Promise<string> {
  const { data } = await axios.get<string>(`https://huggingface.co/${modelId}/raw/main/README.md`, {
    responseType: 'text',
  })
  return data
}

// ── Configurations ────────────────────────────────────────────────────────────

export async function getConfigurationTemplate(): Promise<ConfigData> {
  const { data } = await client.get<Record<string, unknown>>('/configurations/template')
  return data as unknown as ConfigData
}

export async function getConfigurations(p: ListParams): Promise<ListResult<Configuration>> {
  const { data } = await client.get('/configurations', { params: pageQuery(p) })
  return toListResult(data, adaptConfiguration)
}

export async function getConfiguration(id: string, scope: Scope = 'own'): Promise<Configuration> {
  const { data } = await client.get<Record<string, unknown>>(`/configurations/${id}`, { params: { scope } })
  return adaptConfiguration(data)
}

export async function createConfiguration(payload: PendingConfigData): Promise<Configuration> {
  const { data } = await client.post<Record<string, unknown>>('/configurations', {
    name: payload.name,
    tuner_type: payload.tuner_type,
    rl_tuner_type: payload.rl_tuner_type,
    config_data: payload.config_data,
  })
  return adaptConfiguration(data)
}

export async function updateConfiguration(
  configId: string,
  payload: PendingConfigUpdate,
  scope: Scope = 'own'
): Promise<Configuration> {
  const { data } = await client.put<Record<string, unknown>>(
    `/configurations/${configId}`,
    {
      name: payload.name,
      tuner_type: payload.tuner_type,
      rl_tuner_type: payload.rl_tuner_type,
      config_data: payload.config_data,
    },
    { params: { scope } }
  )
  return adaptConfiguration(data)
}

export async function deleteConfiguration(id: string, scope: Scope = 'own'): Promise<void> {
  await client.delete(`/configurations/${id}`, { params: { scope } })
}

// ── Datasets ───────────────────────────────────────────────────────────────────

export function adaptDataset(raw: Record<string, unknown>): Dataset {
  const rawPreview = raw.preview as Record<string, unknown> | null | undefined
  return {
    id: raw.id as string,
    user_id: raw.user_id as string,
    name: raw.name as string,
    description: raw.description as string,
    status: (raw.status as DatasetStatus) ?? 'empty',
    status_detail: raw.status_detail as string | undefined,
    train_file: (raw.train_file as string) ?? '',
    train_records: (raw.train_records as number) ?? 0,
    train_file_size: (raw.train_file_size as number) ?? 0,
    validation_file: (raw.validation_file as string) ?? '',
    validation_records: (raw.validation_records as number) ?? 0,
    validation_file_size: (raw.validation_file_size as number) ?? 0,
    artifact_id: (raw.artifact_id as string) ?? '',
    artifact_url: (raw.artifact_url as string) ?? '',
    created_at: raw.created_at as string,
    updated_at: raw.updated_at as string,
    data_format: raw.data_format as Dataset['data_format'],
    associated_jobs: (raw.associated_jobs as unknown[]) ?? [],
    // Coerce json.dumps'd verl fields (prompt/reward_model/extra_info) back to
    // native array/object so downstream consumers — notably the Reward Function
    // step's verl-strict test-case pre-fill — see the declared types. No-op for
    // already-native rows and non-RL datasets.
    preview: rawPreview
      ? {
          train: normalizeVerlRows(rawPreview.train as Record<string, any>[]),
          validation: normalizeVerlRows(rawPreview.validation as Record<string, any>[]),
        }
      : undefined,
  }
}

export async function getDatasets(p: ListParams): Promise<ListResult<Dataset>> {
  const { data } = await client.get('/datasets', { params: pageQuery(p) })
  return toListResult(data, adaptDataset)
}

export async function getDataset(
  id: string,
  opts?: { preview?: boolean; previewRows?: number; scope?: Scope }
): Promise<Dataset> {
  const params: Record<string, string | number | boolean> = { scope: opts?.scope ?? 'own' }
  if (opts?.preview) {
    params.preview = true
    params.preview_rows = opts?.previewRows ?? 50
  }
  const { data } = await client.get<Record<string, unknown>>(`/datasets/${id}`, { params })
  return adaptDataset(data)
}

export async function createDataset(payload: { name: string; description: string }): Promise<Dataset> {
  const { data } = await client.post<Record<string, unknown>>('/datasets', payload)
  return adaptDataset(data)
}

export async function deleteDataset(id: string, scope: Scope = 'own'): Promise<void> {
  await client.delete(`/datasets/${id}`, { params: { scope } })
}

// Multipart upload (replaces the old resumable-tus flow — the v1 API only
// offers `multipart/form-data`). Response is 202 with status:"uploading";
// callers poll `getDataset(id)` until status is `ready`/`error`.
export interface UploadDatasetOptions {
  trainFile: File
  validationFile?: File | null
  validationPercentage?: number | null
  columnMapping?: Record<string, string> | null
}

export async function uploadDataset(
  datasetId: string,
  opts: UploadDatasetOptions,
  onProgress?: (percent: number) => void
): Promise<Dataset> {
  const fd = new FormData()
  fd.append('train_file', opts.trainFile)
  if (opts.validationFile) fd.append('validation_file', opts.validationFile)
  if (opts.validationPercentage != null) fd.append('validation_percentage', String(opts.validationPercentage))
  if (opts.columnMapping) fd.append('column_mapping', JSON.stringify(opts.columnMapping))
  const { data } = await client.post<Record<string, unknown>>(`/datasets/${datasetId}/upload`, fd, {
    onUploadProgress: (e) => onProgress?.(e.total ? Math.round((e.loaded / e.total) * 100) : 0),
  })
  return adaptDataset(data)
}

// ── Dataset type metadata (backend-informed column requirements) ─────────────

function stripColSuffix(key: string): string {
  return key.endsWith('_col') ? key.slice(0, -4) : key
}

export async function getAutotuneDatasetTypes(): Promise<Record<string, any>> {
  const { data } = await client.get<Record<string, { desc?: string; columns?: Record<string, unknown> }>>(
    '/datasets/intelligence/formats'
  )
  const normalized: Record<string, any> = {}
  for (const [typeKey, typeVal] of Object.entries(data ?? {})) {
    const columns: Record<string, unknown> = {}
    for (const [colKey, colVal] of Object.entries(typeVal.columns ?? {})) {
      columns[stripColSuffix(colKey)] = colVal
    }
    normalized[typeKey] = { desc: typeVal.desc, columns }
  }
  return normalized
}

// ── AI-assisted column mapping ────────────────────────────────────────────────

export interface SuggestColumnMappingPayload {
  sample_data: Record<string, any>[]
  column_names: string[]
  column_samples: Record<string, string[]>
  // The backend field is `target_format` (a dataset-format key from
  // /datasets/intelligence/formats) and its model forbids extra fields, so the
  // old `target_dataset_type` name is rejected with 422. Omitted when undefined.
  target_format?: string
}

export async function suggestColumnMappingAI(payload: SuggestColumnMappingPayload): Promise<AiMappingSuggestion> {
  const { data } = await client.post<Record<string, unknown>>('/datasets/intelligence/suggest-mapping', payload)
  return adaptSuggestion(data)
}

// ── Job estimation & reward function (gated — see AUTOTUNEX_FEATURES) ────────

export async function estimateUsage(payload: Estimation): Promise<Resources | { unavailable: true }> {
  if (!AUTOTUNEX_FEATURES.estimation) return { unavailable: true }
  const { data } = await client.post<Resources>('/jobs/estimate-usages', payload)
  return data
}

export async function validateRewardFunction(
  code: string,
  functionName: string,
  testExecution: boolean = false,
  testInputs?: Record<string, any> | Record<string, any>[]
): Promise<RewardFunctionValidationResult | { unavailable: true }> {
  if (!AUTOTUNEX_FEATURES.rewardValidation) return { unavailable: true }
  const { data } = await client.post<RewardFunctionValidationResult>('/reward-functions/validate', {
    code,
    function_name: functionName,
    test_execution: testExecution,
    test_inputs: testInputs,
  })
  return data
}

export async function generateTestSolutions(
  prompts: Array<Array<{ role: string; content: string }>>
): Promise<{ solutions: string[] } | { unavailable: true }> {
  if (!AUTOTUNEX_FEATURES.testSolutions) return { unavailable: true }
  const { data } = await client.post<{ solutions: string[] }>('/jobs/generate-test-solutions', { prompts })
  return data
}

// ── Tuning jobs (Tunings list / detail view) ──────────────────────────────────

function adaptJobDetail(raw: Record<string, unknown>): JobDetail {
  return {
    ...adaptJob(raw),
    model_source: raw.model_source as ModelSource,
    precision: raw.precision as string | undefined,
    tuning_type: raw.tuning_type as string | undefined,
    rl_tuner_type: raw.rl_tuner_type as string | undefined,
    autotune: raw.autotune != null ? Boolean(raw.autotune) : undefined,
    num_trials: raw.num_trials as number | undefined,
    output_artifacts: (raw.output_artifacts as Record<string, unknown> | null) ?? null,
  }
}

// GET /jobs/{id} adds the two fields by-build-id withholds. No `trials` on either
// shape — they moved to their own paged endpoint (`getJobTrials`), so reading
// `raw.trials` here would silently produce `[]` for every job.
function adaptJobRead(raw: Record<string, unknown>): JobRead {
  return {
    ...adaptJobDetail(raw),
    config_snapshot: raw.config_snapshot as Record<string, unknown> | undefined,
    tasks: (raw.tasks as GbTask[]) ?? [],
  }
}

export async function getJobs(p: ListParams): Promise<ListResult<TuningJob>> {
  const { data } = await client.get('/jobs', { params: pageQuery(p) })
  return toListResult(data, adaptJob)
}

export async function startJob(tuning: TuningForm): Promise<{ id: string }> {
  const { data } = await client.post<Record<string, unknown>>('/jobs', tuning)
  return { id: data.id as string }
}

export async function getJob(id: string, scope: Scope = 'own'): Promise<JobRead> {
  const { data } = await client.get<Record<string, unknown>>(`/jobs/${id}?shape=lean`, { params: { scope } })
  return adaptJobRead(data)
}

/**
 * Fetches the tuning job linked to a gbserver build. Returns null when no job
 * is associated with the build (404), so callers can render nothing for builds
 * that merely carry the "autotunex" tag without a real linked job.
 *
 * This endpoint returns the leaner `JobDetail` — no `tasks`, no
 * `config_snapshot`. Use `getJob` if you need either.
 */
export async function getJobByBuildId(buildId: string, scope: Scope = 'own'): Promise<JobDetail | null> {
  try {
    const { data } = await client.get<Record<string, unknown>>(`/jobs/by-build-id/${buildId}`, { params: { scope } })
    return adaptJobDetail(data)
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) return null
    throw err
  }
}

// ── Trials ────────────────────────────────────────────────────────────────────
// Trials used to arrive nested on the job detail response. They now have their
// own paged endpoint, so the detail read no longer serializes every trial's
// `config` and `metrics` blob on every poll tick.

/**
 * Every trial of a job, oldest first (the endpoint orders by `created_at ASC`
 * with an `id` tiebreaker, so the offsets are stable while a run appends rows).
 *
 * Drains all pages: the endpoint caps `limit` at 100, but `TrialsTable` sorts by
 * loss and paginates client-side over the whole array. A visible job with no
 * trials yet is an empty page, not a 404.
 */
export async function getJobTrials(jobId: string, scope: Scope = 'own'): Promise<Trial[]> {
  return collectPages(
    async (limit, offset) => {
      const { data } = await client.get<{ items?: unknown[]; total?: number }>(`/jobs/${jobId}/trials`, {
        params: { limit, offset, scope },
      })
      return data
    },
    (raw) => adaptTrial(raw as Record<string, unknown>),
    100
  )
}

export async function deleteJob(id: string, scope: Scope = 'own'): Promise<void> {
  await client.delete(`/jobs/${id}`, { params: { scope } })
}

// ── Logs ───────────────────────────────────────────────────────────────────────

function adaptLogEntry(raw: Record<string, unknown>): LogEntry {
  return {
    id: raw.id as number,
    timestamp: raw.timestamp as string,
    level: raw.level as string,
    filename: raw.filename as string,
    message: raw.message as string,
  }
}

export interface LogPage {
  logs: LogEntry[]
  hasMore: boolean
  nextBeforeId: number | null
}

export async function getJobLogs(
  jobId: string,
  opts?: { beforeId?: number; limit?: number; scope?: Scope }
): Promise<LogPage> {
  const { data } = await client.get<{
    logs: Record<string, unknown>[]
    has_more: boolean
    next_before_id?: number | null
  }>(`/jobs/${jobId}/logs`, {
    params: { before_id: opts?.beforeId ?? 0, limit: opts?.limit ?? 50, scope: opts?.scope ?? 'own' },
  })
  return {
    logs: (data.logs ?? []).map(adaptLogEntry),
    hasMore: Boolean(data.has_more),
    nextBeforeId: data.next_before_id ?? null,
  }
}

// Trial logs now live under the job (`/jobs/{jobId}/trials/{trialId}/logs`) —
// unlike the legacy `/job/trial/{trialId}/logs`, the jobId is required.
export async function getTrialLogs(
  jobId: string,
  trialId: string,
  opts?: { beforeId?: number; limit?: number; scope?: Scope }
): Promise<LogPage> {
  const { data } = await client.get<{
    logs: Record<string, unknown>[]
    has_more: boolean
    next_before_id?: number | null
  }>(`/jobs/${jobId}/trials/${trialId}/logs`, {
    params: { before_id: opts?.beforeId ?? 0, limit: opts?.limit ?? 50, scope: opts?.scope ?? 'own' },
  })
  return {
    logs: (data.logs ?? []).map(adaptLogEntry),
    hasMore: Boolean(data.has_more),
    nextBeforeId: data.next_before_id ?? null,
  }
}

// New in v0.3.5 — raw GB build logs for the job's underlying build, not paginated
// the same way as the DB-backed job/trial logs above.
export async function getJobGbLogs(jobId: string, opts?: { all?: boolean; scope?: Scope }): Promise<string[]> {
  const { data } = await client.get<string[]>(`/jobs/${jobId}/gb-logs`, {
    params: { all: opts?.all ?? false, scope: opts?.scope ?? 'own' },
  })
  return data ?? []
}

// ── Results / output assets ──────────────────────────────────────────────────
// The Results tab lists a completed job's downloadable output files from
// GET /jobs/{id}/result-report (computed on read from the job's artifact
// source, so it returns 409 while the job is still producing them). Assets can
// come back as a bare `[]` when the source is readable but empty.

export async function getJobAssets(jobId: string, scope: Scope = 'own'): Promise<TuningAsset[]> {
  const { data } = await client.get<Record<string, unknown>[]>(`/jobs/${jobId}/result-report`, {
    params: { scope },
  })
  return (data ?? []).map(adaptAsset)
}

// Direct-download URL builders. The file/archive endpoints stream with
// `Content-Disposition: attachment`, so these are consumed as plain <a href>
// links (bytes go straight to disk) rather than fetched through axios — the
// same approach as the AutoTuneX UI. AutoTuneX runs auth-free in standalone, so
// no credentials are attached. `path` is the asset's relative `path` and is
// URL-encoded so nested slashes survive as a single query value.
export function resultFileUrl(jobId: string, path: string, scope: Scope = 'own'): string {
  return autotunexApiBase(
    `/jobs/${jobId}/result-report/file?path=${encodeURIComponent(path)}&scope=${scope}`
  )
}

export function resultArchiveUrl(jobId: string, scope: Scope = 'own'): string {
  return autotunexApiBase(`/jobs/${jobId}/result-report/archive?scope=${scope}`)
}
