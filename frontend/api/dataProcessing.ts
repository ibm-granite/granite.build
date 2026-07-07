import axios from 'axios'
import { apiBase } from '@/api/client'

const client = axios.create({ baseURL: apiBase('/api/analytics/data-processing') })

export interface DPBuild {
  uuid: string
  name: string
  username: string
  status: string
  created_time: string
  type: 'tokenization' | 'megatron' | 'e2e'
}

export interface DPNode {
  id: string
  type: 'parquet' | 'arrow' | 'megatron' | 'merged_text' | 'merged_bin'
  path: string
  short_name: string
  column: 0 | 1 | 2 | 3
}

export interface DPEdge {
  id: string
  source: string
  target: string
  label: string
  status: string
  builds: DPBuild[]
}

export interface DPDataset {
  name: string
  short_name: string
  arrow_path: string
  megatron_path: string
  parquet_path: string
  merged_text_path: string
  merged_bin_path: string
  latest_build_id: string | null
  latest_build_status: string | null
  latest_build_time: string
  build_count: number
  builds: DPBuild[]
}

export interface DPLineageResult {
  nodes: DPNode[]
  edges: DPEdge[]
  datasets: DPDataset[]
  scanned: number
  matched: number
  days: number
  warning?: string
}

export interface DPReportParams {
  megatron_path: string
  arrow_path?: string
  parquet_path?: string
  include_p1?: boolean
  include_tokens?: boolean
}

export interface DPReportResult {
  error?: string
  summary?: Record<string, string>
  paths?: Record<string, string>
  metadata?: Record<string, unknown>
}

export async function getDPLineage(days: number): Promise<DPLineageResult> {
  const { data } = await client.get<DPLineageResult>('/lineage', { params: { days } })
  return data
}

export async function getDPRecentDatasets(days: number): Promise<DPDataset[]> {
  const { data } = await client.get<{ datasets: DPDataset[] }>('/recent-datasets', { params: { days } })
  return data.datasets ?? []
}

export async function getDPNodeCounts(
  paths: { id: string; path: string }[],
): Promise<Record<string, number>> {
  const { data } = await client.get<{ counts: Record<string, number> }>('/node-counts', {
    params: { paths: JSON.stringify(paths) },
  })
  return data.counts ?? {}
}

export async function getDPPipelineStatus(
  paths: { id: string; path: string; build_id?: string }[],
): Promise<Record<string, Record<string, string>>> {
  const { data } = await client.get<{ statuses: Record<string, Record<string, string>> }>(
    '/pipeline-status',
    { params: { paths: JSON.stringify(paths) } },
  )
  return data.statuses ?? {}
}

export async function loadDPReport(params: DPReportParams): Promise<DPReportResult> {
  const { data } = await client.get<DPReportResult>('/report', { params })
  return data
}

export async function addScanPrefix(prefix: string): Promise<string[]> {
  const { data } = await client.post<{ prefixes: string[] }>('/scan-prefixes', null, {
    params: { prefix },
  })
  return data.prefixes ?? []
}

export async function removeScanPrefix(prefix: string): Promise<string[]> {
  const { data } = await client.delete<{ prefixes: string[] }>('/scan-prefixes', {
    params: { prefix },
  })
  return data.prefixes ?? []
}
