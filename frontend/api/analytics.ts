/**
 * API client for the gb-ui analytics server (/api/analytics/*).
 * The server always runs on :8090 alongside the client — no URL config needed.
 * All calls return null gracefully when the server is not running or not configured.
 */
import axios, { AxiosError } from 'axios'
import type {
  BuildStatusChartPoint,
  FailureTrendResponse,
  TrendHistoryResponse,
  AIAnalysis,
  QueueCapacity,
  NodePool,
  K8sResource,
  LeaderboardEntry,
  UserResourceDay,
} from '@/types'

const client = axios.create({ baseURL: '/api/analytics' })

import { getActiveEnv } from '@/config/activeEnv'

client.interceptors.request.use((config) => {
  const auth = localStorage.getItem('gb-ui-auth')
  if (auth) {
    try {
      const { token } = JSON.parse(auth)
      if (token) config.headers['Authorization'] = `Bearer ${token}`
    } catch {
      // ignore
    }
  }
  // Forward the active environment so the sidecar uses the right gbserver schema.
  const envId = getActiveEnv()
  if (envId) config.params = { ...config.params, env_id: envId }
  return config
})

export function isSidecarConfigured(): boolean {
  // Always attempt analytics calls — safeGet handles connection failures gracefully.
  // Standalone mode with no server running returns null, showing empty states.
  return true
}

// Wraps calls so they return null instead of throwing when sidecar is absent
async function safeGet<T>(path: string, params?: Record<string, unknown>): Promise<T | null> {
  if (!isSidecarConfigured()) return null
  try {
    const { data } = await client.get<T>(path, { params })
    return data
  } catch (err) {
    const status = (err as AxiosError).response?.status
    if ((err as AxiosError).code === 'ECONNREFUSED' || status === 404 || status === 503) {
      return null
    }
    throw err
  }
}

// ── Build status chart ────────────────────────────────────────────────────────

export async function getBuildStatusChart(
  daysBack = 30,
  excludeTests = false,
): Promise<BuildStatusChartPoint[] | null> {
  return safeGet('/builds/status-chart', { days_back: daysBack, exclude_tests: excludeTests })
}

// ── Failure trends ────────────────────────────────────────────────────────────

export interface FailureTrendParams {
  days_back?: number
  date_from?: string
  date_to?: string
  categories?: string[]
  exclude_tests?: boolean
  source?: 'llm_phase1' | 'llm_custom'
}

export async function getFailureTrends(
  params: FailureTrendParams = {},
): Promise<FailureTrendResponse | null> {
  if (!isSidecarConfigured()) return null
  try {
    const { data } = await client.post<FailureTrendResponse>('/builds/failure-trends', params)
    return data
  } catch {
    return null
  }
}

export interface RunAnalysisParams {
  mode: 'auto' | 'custom'
  categories?: string[]
  days_back?: number
}

export async function runAnalysis(params: RunAnalysisParams): Promise<{ started: boolean; mode: string } | null> {
  if (!isSidecarConfigured()) return null
  try {
    const { data } = await client.post<{ started: boolean; mode: string }>('/ai/run', params)
    return data
  } catch {
    return null
  }
}

export async function getAIDaemonStatus(): Promise<{ running: boolean; analyzing: boolean }> {
  if (!isSidecarConfigured()) return { running: false, analyzing: false }
  try {
    const { data } = await client.get<{ running: boolean; analyzing: boolean }>('/ai/status')
    return data
  } catch {
    return { running: false, analyzing: false }
  }
}

// ── AI analysis ───────────────────────────────────────────────────────────────

export async function getAIAnalysis(buildId: string): Promise<AIAnalysis[] | null> {
  return safeGet(`/builds/${buildId}/ai-analysis`)
}

export async function analyzeLogsContent(
  buildId: string,
  logContent: string,
  buildName?: string,
  status = 'running',
): Promise<AIAnalysis | null> {
  if (!isSidecarConfigured()) return null
  try {
    const { data } = await client.post<AIAnalysis>(`/builds/${buildId}/analyze-logs`, {
      log_content: logContent,
      build_name: buildName ?? '',
      status,
    })
    return data
  } catch (err) {
    const s = (err as AxiosError).response?.status
    if ((err as AxiosError).code === 'ECONNREFUSED' || s === 404 || s === 503) return null
    throw err
  }
}

export async function submitAIFeedback(
  buildId: string,
  updateId: string,
  feedback: {
    rating?: number
    helpful?: boolean
    corrected_root_cause?: string
    comment?: string
  },
): Promise<void> {
  if (!isSidecarConfigured()) return
  await client.post(`/builds/${buildId}/ai-feedback`, { update_id: updateId, ...feedback })
}

// ── Infrastructure ────────────────────────────────────────────────────────────

export async function getQueueCapacity(): Promise<QueueCapacity[] | null> {
  return safeGet('/infra/queues')
}

export async function getNodePools(): Promise<NodePool[] | null> {
  return safeGet('/infra/nodes')
}

export async function getLeaderboard(
  view: 'running_jobs' | 'gpu' | 'cpu' | 'memory' | 'total_builds' = 'running_jobs',
): Promise<LeaderboardEntry[] | null> {
  return safeGet('/infra/leaderboard', { view })
}

export async function getUserResources(daysBack = 14): Promise<UserResourceDay[] | null> {
  return safeGet('/infra/resource-usage', { days_back: daysBack })
}

export async function getBuildK8sResources(buildId: string): Promise<K8sResource[] | null> {
  return safeGet(`/infra/builds/${buildId}/k8s-resources`)
}

export interface BuildResources {
  build_id: string
  cpu?: string | null
  memory?: string | null
  gpu?: number | null
}

export async function getBuildResources(buildIds: string[]): Promise<BuildResources[] | null> {
  if (!buildIds.length) return []
  try {
    const qp = new URLSearchParams()
    for (const id of buildIds) qp.append('build_id', id)
    const { data } = await client.get<BuildResources[]>(`/infra/builds/resources?${qp}`)
    return data
  } catch (err) {
    const status = (err as AxiosError).response?.status
    if ((err as AxiosError).code === 'ECONNREFUSED' || status === 404 || status === 503) return null
    throw err
  }
}

export interface BuildLogsResponse {
  lines: string[]
  total: number
}

export async function getBuildLogs(
  buildId: string,
  container: 'main' | 'sidecar' = 'main',
  limit = 500,
  offset?: number,
): Promise<BuildLogsResponse> {
  const { data } = await client.get<BuildLogsResponse>(`/builds/${buildId}/logs`, {
    params: { container, limit, ...(offset !== undefined ? { offset } : {}) },
  })
  return data
}

// ── Saved trend analyses ──────────────────────────────────────────────────────

export async function saveTrendAnalysis(
  data: FailureTrendResponse,
  title: string | undefined,
  isPublic: boolean,
  author: string,
): Promise<{ success: boolean; update_id?: string } | null> {
  if (!isSidecarConfigured()) return null
  try {
    const { data: res } = await client.post('/builds/failure-trends/save', {
      data,
      title: title || undefined,
      is_public: isPublic,
      author,
    })
    return res
  } catch {
    return null
  }
}

export async function getTrendHistory(
  tab: 'mine' | 'public',
  author: string,
): Promise<TrendHistoryResponse | null> {
  return safeGet('/builds/failure-trends/history', { tab, author })
}

export async function getSavedTrend(
  updateId: string,
): Promise<{ update_id: string; data: FailureTrendResponse; title?: string } | null> {
  return safeGet(`/builds/failure-trends/${updateId}`)
}

export async function deleteSavedTrend(updateId: string, author: string): Promise<void> {
  if (!isSidecarConfigured()) return
  await client.delete(`/builds/failure-trends/${updateId}`, { params: { author } })
}

export async function toggleTrendVisibility(
  updateId: string,
  isPublic: boolean,
  author: string,
): Promise<void> {
  if (!isSidecarConfigured()) return
  await client.patch(`/builds/failure-trends/${updateId}/visibility`, null, {
    params: { is_public: isPublic, author },
  })
}
