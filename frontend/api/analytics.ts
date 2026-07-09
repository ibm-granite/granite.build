/**
 * API client for the gb-ui analytics routes (/api/analytics/*), served by
 * gbserver itself at the same origin — no separate URL config needed.
 * All calls return null gracefully when analytics is not configured.
 */
import axios, { AxiosError } from 'axios'
import { apiBase } from '@/api/client'
import type {
  BuildStatusChartPoint,
  FailureTrendResponse,
  TrendHistoryResponse,
  AIAnalysis,
} from '@/types'

const client = axios.create({ baseURL: apiBase('/api/analytics') })

// Wraps calls so they return null instead of throwing when analytics is unavailable
async function safeGet<T>(path: string, params?: Record<string, unknown>): Promise<T | null> {
  try {
    const { data } = await client.get<T>(path, { params })
    return data
  } catch (err) {
    const status = (err as AxiosError).response?.status
    if ((err as AxiosError).code === 'ECONNREFUSED' || status === 404 || status === 500 || status === 503) {
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
  try {
    const { data } = await client.post<{ started: boolean; mode: string }>('/ai/run', params)
    return data
  } catch {
    return null
  }
}

export async function getAIDaemonStatus(): Promise<{ running: boolean; analyzing: boolean; llm_configured: boolean }> {
  try {
    const { data } = await client.get<{ running: boolean; analyzing: boolean; llm_configured: boolean }>('/ai/status')
    return data
  } catch {
    return { running: false, analyzing: false, llm_configured: false }
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
  await client.post(`/builds/${buildId}/ai-feedback`, { update_id: updateId, ...feedback })
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
  await client.delete(`/builds/failure-trends/${updateId}`, { params: { author } })
}

export async function toggleTrendVisibility(
  updateId: string,
  isPublic: boolean,
  author: string,
): Promise<void> {
  await client.patch(`/builds/failure-trends/${updateId}/visibility`, null, {
    params: { is_public: isPublic, author },
  })
}
