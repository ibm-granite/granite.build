/**
 * API client for the gbserver REST API (/api/v1/*).
 *
 * Endpoint reference (from granite.build/src/gbserver/api/):
 *   GET  /builds/           → { builds: StoredBuild[] }
 *   GET  /builds/{id}       → { build: StoredBuild }
 *   GET  /builds/{id}/status → { status: { build, target_runs } }
 *   GET  /builds/{id}/events → { build_id, events: StoredEvent[] }
 *   GET  /builds/tags       → string[]
 *   GET  /artifacts/        → { artifacts: ArtifactRegistration[] }
 *   GET  /artifacts/{id}    → ArtifactRegistration
 *   GET  /spaces/           → { spaces: StoredSpace[] }
 */
import axios from 'axios'
import { apiBase } from './client'
import type {
  Build,
  BuildStatus,
  BuildEvent,
  BuildStatusDetail,
  BuildTargetRun,
  BuildStepRun,
  Artifact,
  Space,
} from '../types'

const client = axios.create({ baseURL: apiBase('/api/v1') })

// ── Response adapters ─────────────────────────────────────────────────────────
// gbserver returns StoredBuild which uses uppercase Status enums and slightly
// different field names. We normalise here so the UI always works with our types.

export function adaptStatus(s: string): BuildStatus {
  return (s || '').toLowerCase() as BuildStatus
}

function adaptBuild(raw: Record<string, unknown>): Build {
  return {
    uuid: raw.uuid as string,
    name: raw.name as string,
    space_name: raw.space_name as string,
    username: raw.username as string,
    status: adaptStatus(raw.status as string),
    tags: (raw.tags as string[]) ?? [],
    source_uri: raw.source_uri as string | undefined,
    description: raw.description as string | undefined,
    created_time: raw.created_time as string,
    updated_time: raw.updated_time as string,
    finished_at: raw.finished_at as string | undefined,
    failure_reason: raw.failure_reason as string | undefined,
    resources: (() => {
      if (raw.resources != null) return raw.resources as Build['resources']
      if (raw.total_cpu || raw.total_memory || raw.total_gpu != null) {
        return {
          cpu: raw.total_cpu as string | undefined,
          memory: raw.total_memory as string | undefined,
          gpu: raw.total_gpu as number | undefined,
          storage: raw.total_storage as string | undefined,
        }
      }
      // Aggregate from targets[].resources (gbserver returns these on the build object)
      const targets = (raw.targets as Array<Record<string, unknown>>) ?? []
      if (targets.length > 0) {
        let gpuTotal = 0
        let cpu: string | undefined
        let memory: string | undefined
        for (const t of targets) {
          const res = t.resources as Record<string, unknown> | undefined
          if (!res) continue
          if (!cpu && res.cpu) cpu = String(res.cpu)
          if (!memory && res.memory) memory = String(res.memory)
          const replicas = Number(res.replicas ?? t.replicas ?? 1) || 1
          if (res.gpu) gpuTotal += (Number(res.gpu) || 0) * replicas
        }
        if (cpu || memory || gpuTotal > 0)
          return { cpu, memory, gpu: gpuTotal > 0 ? gpuTotal : undefined }
      }
      return undefined
    })(),
    build_archive: raw.build_archive as string | undefined,
  }
}

function stepNameFromUri(uri: string): string {
  // Prefer the subdirectory fragment: git+ssh://...#subdirectory=steps/dpk-ray → "dpk-ray"
  const fragment = uri.split('#')[1] ?? ''
  const subdir = fragment.match(/subdirectory=(.+)/)?.[1]
  if (subdir) return subdir.split('/').filter(Boolean).pop() ?? uri
  // Fallback: last non-empty path segment before any query/fragment
  return uri.split(/[/?#@]/).filter(Boolean).pop() ?? uri
}

function adaptStepRun(raw: Record<string, unknown>): BuildStepRun {
  const json = (raw.json as Record<string, unknown>) ?? {}
  const definitionUri = (raw.definition_uri as string) || ''
  return {
    step_name: (definitionUri ? stepNameFromUri(definitionUri) : undefined) || (raw.uuid as string),
    status: adaptStatus((raw.status as string) || (json.status as string)),
    uri: definitionUri || undefined,
    started_at: (raw.started_at as string) || (json.started_at as string),
    updated_at: (raw.finished_at as string) || (json.finished_at as string),
    log_path: (raw.log_path as string) || (json.log_path as string) || undefined,
  }
}

function adaptTargetRun(raw: Record<string, unknown>): BuildTargetRun {
  const steps: BuildStepRun[] = ((raw.steps as unknown[]) ?? []).map(
    (s) => adaptStepRun(s as Record<string, unknown>)
  )
  const inputArtifacts = (raw.input_artifacts as Record<string, string>) ?? {}
  const outputArtifacts = (raw.output_artifacts as Record<string, unknown[]>) ?? {}
  return {
    target_name: (raw.name as string) || (raw.uuid as string),
    status: adaptStatus(raw.status as string),
    started_at: raw.started_at as string | undefined,
    updated_at: raw.finished_at as string | undefined,
    steps,
    inputs: Object.fromEntries(Object.entries(inputArtifacts).map(([k, v]) => [k, String(v)])),
    outputs: Object.fromEntries(
      Object.entries(outputArtifacts).map(([k, v]) => [k, Array.isArray(v) ? String(v[0]) : String(v)])
    ),
  }
}

function adaptArtifact(raw: Record<string, unknown>): Artifact {
  return {
    uuid: raw.uuid as string,
    name: (raw.name as string) || (raw.uri as string),
    artifact_type: ((raw.type as string) || (raw.artifact_type as string) || 'FILESET') as import('../types').ArtifactType,
    status: (((raw.status as string) || 'success').toLowerCase()) as import('../types').ArtifactStatus,
    space_name: raw.space_name as string,
    username: raw.username as string,
    uri: raw.uri as string,
    build_id: raw.created_by_build_id as string | undefined,
    created_time: ((raw.created_at ?? raw.created_time) as string),
    updated_time: ((raw.updated_at ?? raw.updated_time ?? raw.created_at) as string),
    tags: (raw.tags as string[]) ?? [],
    archived: (raw.is_archived as boolean) ?? false,
    checksum: raw.checksum as string | undefined,
  }
}

// ── Spaces ────────────────────────────────────────────────────────────────────

export async function listSpaces(): Promise<Space[]> {
  const { data } = await client.get<{ spaces: Record<string, unknown>[] }>('/spaces/')
  return (data.spaces ?? []).map((s) => ({
    uuid: s.uuid as string,
    name: s.name as string,
    git_repo_uri: s.git_repo_uri as string | undefined,
    is_admin: (s.is_admin as boolean) ?? false,
  }))
}

// ── Builds ────────────────────────────────────────────────────────────────────

export interface ListBuildsParams {
  space_name?: string
  username?: string
  tags?: string[]
  status?: string | string[]
  sort?: string
  page_index?: number
  page_size?: number
}

export interface BuildListResult {
  items: Build[]
  total: number
  page: number
  page_size: number
}

export async function listBuilds(params: ListBuildsParams): Promise<BuildListResult> {
  // gbserver uses GET with query params: tag (multi), status (multi), sort (multi)
  const qp = new URLSearchParams()
  if (params.space_name)  qp.set('space_name', params.space_name)
  if (params.username)    qp.set('username', params.username)
  for (const s of Array.isArray(params.status) ? params.status : params.status ? [params.status] : []) qp.append('status', s.toUpperCase())
  if (params.sort)        qp.append('sort', params.sort)
  if (params.page_index != null) qp.set('page_index', String(params.page_index))
  if (params.page_size)   qp.set('page_size', String(params.page_size))
  for (const tag of params.tags ?? []) qp.append('tag', tag)

  const [{ data }, total] = await Promise.all([
    client.get<{ builds: Record<string, unknown>[]; total?: number; count?: number }>(
      `/builds/?${qp.toString()}`
    ),
    getBuildCount({
      space_name: params.space_name,
      username: params.username,
      status: params.status,
      tags: params.tags,
    }),
  ])
  const items = (data.builds ?? []).map(adaptBuild)
  const resolvedTotal = data.total ?? data.count ?? total
  const pageSize = params.page_size ?? items.length
  return { items, total: resolvedTotal, page: (params.page_index ?? 0) + 1, page_size: pageSize }
}

export async function getBuildCount(params: Pick<ListBuildsParams, 'space_name' | 'username' | 'status' | 'tags'>): Promise<number> {
  const qp = new URLSearchParams()
  if (params.space_name) qp.set('space_name', params.space_name)
  if (params.username)   qp.set('username', params.username)
  for (const s of Array.isArray(params.status) ? params.status : params.status ? [params.status] : []) qp.append('status', s.toUpperCase())
  for (const tag of params.tags ?? []) qp.append('tag', tag)
  const { data } = await client.get<{ count: number }>(`/builds/count?${qp.toString()}`)
  return data.count ?? 0
}

export async function getBuild(buildId: string): Promise<Build> {
  const { data } = await client.get<{ build: Record<string, unknown> }>(`/builds/${buildId}`)
  return adaptBuild(data.build ?? data as Record<string, unknown>)
}

// gbserver doesn't have a separate /describe endpoint — getBuild returns everything
export async function describeBuild(buildId: string): Promise<Build> {
  return getBuild(buildId)
}

export async function getBuildStatus(buildId: string): Promise<BuildStatusDetail> {
  const { data } = await client.get<{
    status: {
      build: Record<string, unknown>
      target_runs: Array<{
        target: Record<string, unknown>
        steps: Record<string, unknown>[]
      }>
    }
  }>(`/builds/${buildId}/status`)

  const s = data.status
  const build = adaptBuild(s.build)
  const targets: Record<string, BuildTargetRun> = {}

  for (const tr of s.target_runs ?? []) {
    // tr.target has input_artifacts / output_artifacts as param→uuid dicts.
    // tr.input_artifacts / tr.output_artifacts are full artifact object arrays — not used here.
    const adapted = adaptTargetRun({
      ...tr.target,
      steps: tr.steps,
    })
    if (adapted.target_name) {
      targets[adapted.target_name] = adapted
    }
  }

  return {
    details: {
      build_id: build.uuid,
      name: build.name,
      started_at: build.created_time,
      updated_at: build.updated_time,
      status: build.status,
    },
    history: [],
    targets,
  }
}

const LEVEL_EMOJI: Record<string, string> = {
  ERROR:   '🔴',
  WARN:    '⚠️',
  WARNING: '⚠️',
  INFO:    'ℹ️',
}

export async function getBuildEvents(buildId: string): Promise<BuildEvent[]> {
  const { data } = await client.get<{
    events: Array<{
      type?: string
      build_event: { timestamp: string; payload: Record<string, unknown> }
    }>
  }>(`/builds/${buildId}/events`)

  return (data.events ?? []).map((e) => {
    const ev = e.build_event ?? {}
    const payload = (ev.payload as Record<string, unknown>) ?? {}
    const msg = (payload.msg as string) || ''
    const level = ((payload.level as string) || '').toUpperCase()
    const emoji = LEVEL_EMOJI[level]

    // Prepend the level header (e.g. "## ℹ️ INFO\n\n---\n\n") when the payload
    // carries a level field, matching the format the GitHub PR bot uses.
    const description = emoji && msg
      ? `## ${emoji} ${level}\n\n---\n\n${msg}`
      : msg || (payload.status as string) || ''

    return {
      time: ev.timestamp as string,
      description,
    }
  })
}

export async function getBuildArchiveFiles(buildId: string): Promise<Record<string, string>> {
  const build = await getBuild(buildId)
  const archive = build.build_archive
  if (!archive) return {}
  const JSZip = (await import('jszip')).default
  const zip = await JSZip.loadAsync(archive, { base64: true })
  const entries = await Promise.all(
    Object.entries(zip.files)
      .filter(([, f]) => !f.dir)
      .map(async ([name, f]) => [name, await f.async('string')] as const)
  )
  return Object.fromEntries(entries)
}

export async function getBuildStepLog(logPath: string): Promise<string> {
  const { data } = await client.get<string>(logPath, {
    responseType: 'text',
    baseURL: '',
  })
  return data
}

// gbserver's /logs/logquery mirrors the IBM Cloud Logs query shape so the
// same endpoint backs `gb build log` regardless of deployment. In standalone
// mode it's served by LocalLogQueryAPI (src/gbserver/utils/local_logquery.py),
// which reads MESSAGE_EVENT rows for the build straight out of gbserver's own
// event store — no cloud logs service or K8s involved.
export interface BuildLiveLogsResult {
  lines: string[]
  total: number
}

export async function getBuildLiveLogs(buildId: string, limit = 500): Promise<BuildLiveLogsResult> {
  const endDate = Date.now()
  const startDate = endDate - 24 * 3600_000

  const { data } = await client.post<{
    logs: Array<{ text: string | null; timestamp: number | null }>
    total: number
  }>('/logs/logquery', {
    queryDef: {
      startDate,
      endDate,
      pageSize: limit,
      pageIndex: 0,
      type: 'freeText',
      queryParams: {
        jsonObject: { 'kubernetes.labels.granite-dot-build/build-id': [buildId] },
      },
      sortModel: [{ field: 'timestamp', ordering: 'asc', missing: '_last' }],
    },
  })

  const lines = (data.logs ?? [])
    .slice()
    .sort((a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0))
    .map((l) => {
      try {
        return (JSON.parse(l.text ?? '{}') as { log?: string }).log ?? ''
      } catch {
        return l.text ?? ''
      }
    })

  return { lines: lines.slice(-limit), total: data.total || lines.length }
}


function extractTagStrings(data: unknown): string[] {
  const arr: unknown[] = Array.isArray(data)
    ? data
    : Array.isArray((data as Record<string, unknown>)?.tags)
      ? (data as Record<string, unknown>).tags as unknown[]
      : []
  return arr
    .map((t) => {
      if (typeof t === 'string') return t
      if (t && typeof t === 'object') {
        const o = t as Record<string, unknown>
        const v = o.name ?? o.tag ?? o.value ?? o.label ?? o.id
        if (typeof v === 'string') return v
      }
      return null
    })
    .filter((t): t is string => typeof t === 'string' && t.length > 0)
}

export async function getBuildTags(spaceName?: string): Promise<string[]> {
  const { data } = await client.get<unknown>('/builds/tags', {
    params: spaceName ? { space_name: spaceName } : {},
  })
  return extractTagStrings(data)
}

export async function cancelBuild(buildId: string): Promise<void> {
  await client.delete(`/builds/${buildId}`)
}

// ── Artifacts ─────────────────────────────────────────────────────────────────

export interface ListArtifactsParams {
  space_name?: string
  tags?: string[]
  username?: string
}

export interface ArtifactListResult {
  items: Artifact[]
  total: number
  page: number
  page_size: number
}

export async function listArtifacts(params: ListArtifactsParams): Promise<ArtifactListResult> {
  const qp = new URLSearchParams()
  if (params.space_name) qp.set('space_name', params.space_name)
  if (params.username)   qp.set('username', params.username)
  for (const tag of params.tags ?? []) qp.append('tag', tag)

  const { data } = await client.get<{ artifacts: Record<string, unknown>[] }>(
    `/artifacts/?${qp.toString()}`
  )
  const artifacts = (data.artifacts ?? []).map(adaptArtifact)
  return { items: artifacts, total: artifacts.length, page: 1, page_size: artifacts.length }
}

export async function getArtifactTags(spaceName?: string): Promise<string[]> {
  const { data } = await client.get<unknown>('/artifacts/tags', {
    params: spaceName ? { space_name: spaceName } : {},
  })
  return extractTagStrings(data)
}

export async function getArtifact(artifactId: string): Promise<Artifact> {
  const { data } = await client.get<Record<string, unknown>>(`/artifacts/${artifactId}`)
  return adaptArtifact((data.artifact ?? data) as Record<string, unknown>)
}

export interface ArtifactContents {
  columns: string[]
  rows: (string | number | null)[][]
  total: number
}

export async function getArtifactContents(artifactId: string): Promise<ArtifactContents> {
  const { data } = await client.get<ArtifactContents>(`/artifacts/${artifactId}/contents`)
  return data
}

export async function getArtifactModelCard(artifactId: string): Promise<string> {
  const { data } = await client.get<{ content: string }>(`/artifacts/${artifactId}/model_card`)
  return data.content ?? ''
}

// ── Artifact lineage ──────────────────────────────────────────────────────────

export interface ArtifactLineageNodeRef {
  node_type: string
  name: string
  uri?: string
  url?: string
}

export interface ArtifactRunEntry {
  job_name: string
  run_id: string
  status: string
  inputs: ArtifactLineageNodeRef[]
  outputs: ArtifactLineageNodeRef[]
}

export interface ArtifactLineageResult {
  root_id: string
  runs: ArtifactRunEntry[]
  truncated: boolean
}

export interface GetArtifactLineageParams {
  artifact_name?: string
  artifact_url?: string
  artifact_type?: string
  max_depth?: number
  direction?: string
}

export async function getArtifactLineage(params: GetArtifactLineageParams): Promise<ArtifactLineageResult> {
  const { data } = await client.post<ArtifactLineageResult>('/lineage/artifact', params, { timeout: 45_000 })
  return data
}
