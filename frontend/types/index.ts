// ── Runtime config / environment switcher ────────────────────────────────────

export interface EnvironmentEntry {
  id: string
  label: string
  url: string
  dbSchema?: string
}

// Core build status union
export type BuildStatus =
  | 'running'
  | 'success'
  | 'failed'
  | 'pending'
  | 'submitted'
  | 'suspended'
  | 'cancelled'
  | 'deleted'
  | 'planned'

// ── Build hierarchy ──────────────────────────────────────────────────────────

export interface BuildStepRun {
  step_name: string
  status: BuildStatus
  uri?: string
  started_at?: string
  updated_at?: string
  log_path?: string
}

export interface BuildTargetRun {
  target_name: string
  status: BuildStatus
  started_at?: string
  updated_at?: string
  steps: BuildStepRun[]
  inputs?: Record<string, string>
  outputs?: Record<string, string>
}

export interface Build {
  uuid: string
  name: string
  space_name: string
  username: string
  status: BuildStatus
  tags: string[]
  source_uri?: string
  description?: string
  created_time: string
  updated_time: string
  finished_at?: string
  targets?: BuildTargetRun[]
  resources?: {
    cpu?: string
    memory?: string
    gpu?: number
    storage?: string
  }
  failure_reason?: string
  failure_message?: string
  // Raw build.yaml contents (for Build Definition tab)
  build_archive?: unknown
}

export interface BuildEvent {
  time: string
  description: string
}

export interface BuildStatusDetail {
  details: {
    build_id: string
    name: string
    started_at: string
    updated_at: string
    status: BuildStatus
    source_pr?: string
  }
  history: BuildEvent[]
  targets: Record<string, BuildTargetRun>
}

// ── Artifacts ────────────────────────────────────────────────────────────────

export type ArtifactType = 'MODEL' | 'DATASET' | 'FILESET' | 'TABLE'

export interface Artifact {
  uuid: string
  name: string
  artifact_type: ArtifactType
  space_name: string
  username: string
  uri: string
  build_id?: string
  created_time: string
  updated_time: string
  tags: string[]
  description?: string
  archived: boolean
  checksum?: string
}

// ── Spaces ───────────────────────────────────────────────────────────────────

export interface Space {
  uuid: string
  name: string
  git_repo_uri?: string
  is_admin: boolean
}

// ── Analytics ────────────────────────────────────────────────────────────────

export interface BuildStatusChartPoint {
  date: string
  running: number
  success: number
  failed: number
  pending: number
  submitted: number
  suspended: number
  running_test: number
  success_test: number
  failed_test: number
  pending_test: number
  submitted_test: number
  suspended_test: number
}

export interface FailureTrendResponse {
  labels: string[]
  categories: string[]
  series: Record<string, number[]>
  builds_by_category: Record<string, CategorizedBuild[]>
  total_analyzed: number
  analysis_time_ms: number
}

export interface CategorizedBuild {
  build_id: string
  name: string
  username: string
  space_name: string
  created_at: string
  category: string
  confidence: number
  summary?: string
}

export interface TrendHistoryItem {
  update_id: string
  title?: string
  summary: string
  date_range_start: string
  date_range_end: string
  category_count: number
  total_builds: number
  is_public: boolean
  author: string
  created_at: string
}

export interface TrendHistoryResponse {
  items: TrendHistoryItem[]
  total_count: number
}

// ── AI Analysis ──────────────────────────────────────────────────────────────

export interface AIAnalysisIssue {
  type: string
  severity: 'critical' | 'high' | 'warning' | 'info' | string
  description: string
}

export interface AIAnalysis {
  update_id: string
  build_id: string
  source: 'llm_phase1' | 'llm_phase2' | 'human' | 'system'
  analysis_type?: string
  summary: string
  root_cause: string
  suggested_action: string
  issues: AIAnalysisIssue[]
  confidence: number
  model_name?: string
  error_category_1?: string
  error_category_2?: string
  kb_recommendation?: string
  parent_uid?: string
  created_at: string
  // User feedback
  feedback_rating?: number
  feedback_helpful?: boolean
  corrected_root_cause?: string
  feedback_comment?: string
  upvotes: number
  downvotes: number
}

// ── Infrastructure (analytics sidecar) ───────────────────────────────────────

export interface QueueCapacity {
  name: string
  cluster_name: string
  gpu_capacity: number
  gpu_used: number
  cpu_capacity_cores: number
  cpu_used_cores: number
  memory_capacity_gib: number
  memory_used_gib: number
  admitted_workloads: number
  pending_workloads: number
  reserving_workloads: number
}

export interface K8sResource {
  kind: string
  name: string
  namespace?: string
  cluster_name?: string
  status?: string
  build_status?: string
  failure_reason?: string
  failure_message?: string
  cpu?: string
  memory?: string
  gpu?: number
  storage?: string
  replicas?: number
  created_at?: string
  deleted_at?: string
}

export interface NodePool {
  pool_name: string
  cluster_name: string
  node_count: number
  ready_nodes: number
  cpu_allocatable_cores: number
  cpu_requested_cores: number
  memory_allocatable_gib: number
  memory_requested_gib: number
  gpu_allocatable: number
  gpu_requested: number
  running_pods: number
  pending_pods: number
  autoscale_enabled: boolean
  min_nodes?: number
  max_nodes?: number
}

export interface LeaderboardEntry {
  username: string
  running_jobs: number
  gpu_count: number
  cpu_cores: number
  memory_gib: number
  total_builds: number
}

export interface Metric {
  name: string
  value: string
  units?: string
  build_id?: string
  recorded_at: string
}

export interface UserResourceDay {
  username: string
  date: string
  build_count: number
  gpu_count: number
  cpu_cores: number
  memory_gib: number
}

// ── Flight Plans ─────────────────────────────────────────────────────────────

export interface Plan {
  plan_id: string
  name: string
  summary?: string
  markdown_body?: string
  status: string
  revision: number
  space_name?: string
  created_by?: string
  username?: string
  created_at?: string
  updated_at: string
  tags?: string[]
}

export interface LinkedBuild {
  plan_id: string
  build_id: string
  step_id: string
  revision: number
  execution_notes: string
  created_at: string
}

// ── API response wrappers ─────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}
