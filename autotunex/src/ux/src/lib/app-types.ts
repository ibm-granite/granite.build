// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import type { ToastNotificationProps } from 'carbon-components-svelte/src/Notification/ToastNotification.svelte';

export enum ModelSource {
	HuggingFace = 'huggingface',
	CustomPath = 'custom_path'
}

export type Job = {
	id: string;
	user_id: string;
	status: Status;
	seed: number;
	config_id: string;
	dataset_id: string;
	model: string;
	model_source: string;
	experiment_name: string;
	tuning_type: string;
	ray_address: null;
	precision: string;
	cleanup: number;
	autotune: number;
	created_at: Date;
	updated_at: Date;
	// The run's end (latest gb_tasks.updated_at) from the API, used for Total time.
	// A free-text string or null; not jobs.updated_at (that is a last-modified time).
	finished_at?: string | null;
};

export type Task = {
	id: string;
	job_id: string;
	build_id: string;
	status: string;
	type: string;
	pr_url: string;
	artifact_id: string;
	artifact_uri: string;
	build_status: BuildStatus;
	started_at: Date;
	updated_at: Date;
};

export type Tuning = {
	user: string;
	config_name: string;
	dataset: string;
	num_trials: number;
	task_id: string;
	build_id: string;
	task_status: Status;
	task_type: string;
	github_pr_url: string;
	artifact_id: string;
	artifact_uri: string;
	task_started_at: Date;
	task_updated_at: Date;
	trials?: Trial[];
	tasks?: Task[];
	config_snapshot?: Record<string, any> | null;
	// True when the live configuration has drifted from this job's snapshot
	// (config_data / tuner types differ). Detail response only; merged onto the
	// snapshot object by `getJobConfigSnapshot` for the drift banner.
	is_stale?: boolean;
	output_artifacts?: Record<string, any> | any[] | null;
	assets?: Assets[];
	build_status?: BuildStatus;
	logs?: Log[];
	gb_logs?: string[];
} & Job;

export type TuningForm = {
	config_id: string | undefined;
	dataset_id: string | undefined;
	model: string;
	model_source: ModelSource;
	experiment_name: string;
	autotune: boolean;
	reward_function_code?: string;
	reward_function_name?: string;
};

export type Pair = {
	input: any;
	output: any;
};
export type Dataset = {
	id: string;
	user_id: string;
	name: string;
	description: string;
	train_file: string;
	train_records: number;
	train_file_size: number;
	validation_file: string;
	validation_records: number;
	validation_file_size: number;
	artifact_id: string;
	artifact_url: string;
	created_at: Date;
	updated_at: Date;
	associated_jobs: Job[];
	validation_data?: Pair[];
	train_data?: Pair[];
	// From the API's preview.viewer_ready: false means the HuggingFace dataset viewer
	// is still precomputing, so empty preview rows mean "not ready yet", not "empty".
	// Undefined when no preview was requested or returned.
	viewer_ready?: boolean;
};

export interface Configuration {
	id: string;
	user_id: string;
	name: string;
	tuner_type: string;
	rl_tuner_type?: string | null;
	artifact_id: string;
	artifact_url: string;
	// ABSENT on list items. GET /configurations returns the lean shape and omits it
	// (the server does not even load the column); it is present on
	// GET /configurations/{id} and on the POST/PUT responses. Read the detail when
	// you need the search space — ConfigDisplay.fetchConfig and
	// Step2Configure.selectConfig both already fetch it on a miss and memoize it
	// back into the `configurations` store.
	//
	// Typed as required rather than optional despite that, deliberately. Marking it
	// optional adds 33 `possibly undefined` errors, every one of them in
	// ConfigDisplay, which only ever renders a *fetched detail* and so cannot
	// actually hit them. The real fix is to mirror the backend and split this into
	// list and detail types, which means retyping the `configurations` store and the
	// prop chain through Tunings -> CreateTuningForm -> ConfigDisplay — a refactor
	// worth doing on its own, not as a side effect of leaning one endpoint.
	config_data: ConfigData;
	associated_jobs: Job[];
	created_at: Date;
	updated_at: Date;
}

export interface DatasetUploadConfig {
	max_bytes: number;
	client_gzip_enabled: boolean;
	client_gzip_min_bytes: number;
	client_parquet_preview_max_bytes: number;
}

export interface AppConfig {
	dataset_upload: DatasetUploadConfig;
}

export type ConfigForm = {
	name?: string;
	tuner_type?: string;
	rl_tuner_type?: string;
} & ConfigData;

export interface ConfigData {
	tune_config: TuneConfig;
	tuners_config: TunersConfig;
	training_config: TrainingConfig;
	training_rl_config?: TrainingRlConfig;
	tuners_rl_config?: TunersRlConfig;
}

export interface TrainingConfig {
	[key: string]: HpoDatasetPercentage | InputColumn;
	seed: HpoDatasetPercentage;
	precision: InputColumn;
	max_length: HpoDatasetPercentage;
	input_column: InputColumn;
	warmup_ratio: HpoDatasetPercentage;
	output_column: InputColumn;
	hpo_num_epochs: HpoDatasetPercentage;
	num_train_epochs: HpoDatasetPercentage;
	use_chat_template: InputColumn;
	num_gpus_per_trial: HpoDatasetPercentage;
	num_cpus_per_worker: HpoDatasetPercentage;
	use_flash_attention: InputColumn;
	train_implementation: InputColumn;
	hpo_dataset_percentage: HpoDatasetPercentage;
}

export interface HpoDatasetPercentage {
	type: Type;
	values: null;
	default: number;
	max_val: number;
	min_val: number;
	description: string;
	search_alg?: string[];
}

export enum Type {
	Float = 'float',
	Int = 'int',
	String = 'string'
}

export interface InputColumn {
	type: string;
	values: string[] | null;
	default: string | number;
	max_val: number | null;
	min_val: number | null;
	description: string;
}

export interface NumberInputColumn {
	default: number | null;
	description: string;
	min_val: number;
	max_val: number;
	type: string;
}

export interface TuneConfig {
	[key: string]: InputColumn | HpoDatasetPercentage | NumberInputColumn | undefined;
	scheduler: InputColumn;
	search_alg: InputColumn;
	num_samples: HpoDatasetPercentage;
	max_discrepancy: HpoDatasetPercentage;
	max_concurrent_trials: HpoDatasetPercentage;
	time_budget_s?: NumberInputColumn;
}

export interface TunersConfig {
	[key: string]: Tuner;
	lora: Tuner;
	alora: Tuner;
}

export interface TunersRlConfig {
	[key: string]: Tuner;
}

export interface TrainingRlConfig {
	[key: string]: InputColumn | NumberInputColumn;
}

export interface Tuner {
	title: string;
	tuner_name: string;
	description: string;
	hyperparams: Hyperparams;
}

export interface Hyperparams {
	[key: string]: AlphaRatio | Bias | Field;
	r: AlphaRatio;
	bias: Bias;
	alpha_ratio: AlphaRatio;
	lora_dropout: AlphaRatio;
	learning_rate: AlphaRatio;
	lr_scheduler_type: Bias;
	gradient_accumulation_steps: AlphaRatio;
	per_device_train_batch_size: AlphaRatio;
	invocation_string: Field;
}

export interface AlphaRatio {
	type: Type;
	values: number[];
	default: number;
	max_val: number;
	min_val: number;
	options: Strategy[];
	strategy: Strategy;
	for_tuner: boolean;
	description: string;
}

export enum Strategy {
	Choice = 'choice',
	Loguniform = 'loguniform',
	Uniform = 'uniform'
}

export interface Bias {
	type: string;
	values: string[];
	default: string;
	max_val: null;
	min_val: null;
	options: Strategy[];
	strategy: Strategy;
	for_tuner: boolean;
	description: string;
}

export interface Field {
	type: Type;
	values: number[] | string[];
	default: number | string;
	max_val: number | null;
	min_val: number | null;
	options: Strategy[];
	strategy: Strategy;
	for_tuner: boolean;
	description: string;
}

export interface HuggingFaceModel {
	_id: string;
	id: string;
	likes: number;
	trendingScore: number;
	private: boolean;
	config: HuggingFaceModelConfig;
	downloads: number;
	tags: string[];
	pipeline_tag: string;
	library_name: LibraryName;
	createdAt: Date;
	modelId: string;
}

export interface HuggingFaceModelConfig {
	architectures: string[];
	model_type: string;
	chat_template_jinja?: string;
	processor_config?: ProcessorConfig;
}

export interface ProcessorConfig {
	chat_template: string;
}

export enum LibraryName {
	SentenceTransformers = 'sentence-transformers',
	Transformers = 'transformers'
}

export type Trial = {
	id: string;
	job_id: string;
	status: Status;
	config: Record<string, any>;
	created_at: Date;
	updated_at: Date;
	logs: any[];
	score: Score;
};

export type Metric = 'loss';

export type Status = 'COMPLETED' | 'ERROR' | 'RUNNING' | 'TERMINATED' | 'PENDING' | 'SUBMITTED';

export type Score = {
	id: string;
	job_id: string;
	trial_id: string;
	metric: Metric;
	metrics: Metrics;
	created_at: Date;
	updated_at: Date;
};

export type Metrics = {
	loss: number;
	total_time: number | string;
	train_loss: number;
};

export type User = {
	id: string;
	email: string;
	role: string;
	created_at: Date;
	// When the row last changed. Not a login time — an admin changing someone's
	// role moves this. The Users table read it as "Last login on" until
	// last_login_at existed, which is why a role edit looked like a login.
	updated_at: Date;
	// When the user last authenticated. Null only where the backend has no record
	// of a login at all — e.g. a user the tuning pipeline created and never
	// signed in as. Render it, never assume a date.
	last_login_at: Date | null;
};

export type DatasetForm = {
	name: string;
	description: string;
	train_file: File | null;
	validation_file: File | null;
	trainSetPercentage?: number;
};

export type DatasetType = {
	input: string;
	output: string;
};

// Wizard-related types for Start Tuning flow
export type DatasetFormatType =
	| 'preference_pairs'
	| 'kto_format'
	| 'standard_pairs'
	| 'prompt_only'
	| 'unknown';

export type ColumnMetadata = {
	name: string;
	detectedType: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'null';
	sampleValues: string[];
	nullCount: number;
	uniqueCount: number;
};

export type DatasetFormatInfo = {
	format: DatasetFormatType;
	columns: ColumnMetadata[];
	totalRecords: number;
	fileSize: number;
	fileName: string;
	compatibleMethods: string[];
};

export type ParsedDataRow = Record<string, any>;

// Column mapping: maps required column names to user's actual column names
export type ColumnMapping = Record<string, string>;

// AI-powered column mapping suggestion from LLM
export type AiMappingSuggestion = {
	dataset_type: string;
	dataset_type_desc: string;
	algorithm: string;
	confidence: number;
	column_mapping: Record<string, { source_column: string; confidence: number }>;
	reasoning: string;
};

// Algorithm definition for the wizard
export type AlgorithmOption = {
	id: string;
	name: string;
	category: 'sft' | 'offline_rl' | 'online_rl';
	requiredColumns: string[];
};

// Fine-tuning goal category for the Step 0 questionnaire
export type TuningGoal = 'sft' | 'offline_rl' | 'online_rl';

// Pending config data for deferred creation (not yet saved to API)
export type PendingConfigData = {
	name: string;
	tuner_type: string | null;
	rl_tuner_type: string | null;
	config_data: ConfigData;
};

// Pending config update for deferred save (edit of existing config)
export type PendingConfigUpdate = {
	configId: string;
	name: string;
	tuner_type: string | null;
	rl_tuner_type: string | null;
	config_data: ConfigData;
};

// Tracks multi-step launch progress
export type LaunchPhase =
	| 'creating_dataset'
	| 'uploading_files'
	| 'creating_config'
	| 'updating_config'
	| 'launching_job'
	| null;

export type Resources = {
	model_size_billion_params: number;
	gpu_memory_gb: number;
	cpu_memory_gb: number;
	num_gpus: number;
	weights_memory: number;
	optimizer_memory: number;
	gradients_memory: number;
	activations_memory: number;
};

export type Estimation = {
	model_name: string;
	gpu_memory: number;
	// Exactly one of config_id (a saved configuration) or config_data (an unsaved,
	// mid-wizard configuration) is set — tuner_type/rl_tuner_type accompany the latter.
	config_id?: string;
	config_data?: ConfigData;
	tuner_type?: string | null;
	rl_tuner_type?: string | null;
};

export type WizardDraft = {
	savedAt: string;
	currentStep: number;
	completedSteps: boolean[];
	selectedGoal: TuningGoal | null;
	selectedAlgorithm: string;
	selectedModel: string;
	modelSource: ModelSource;
	datasetForm: { name: string; description: string };
	existingDatasetId: string | null;
	splitRatio: number;
	selectedConfigId: string | null;
	experimentName: string;
	autotuneEnabled?: boolean;
};

export type Notification = Pick<
	ToastNotificationProps,
	'kind' | 'title' | 'subtitle' | 'caption' | 'timeout'
> & {
	show: boolean;
};

export type AppState = {
	isTuningsLoaded: boolean;
	isDatasetsLoaded: boolean;
	isConfigurationsLoaded: boolean;
};

export type FeatureFlags = {
	/** Show the legacy quick-create "Create new tuning" button on the Tunings page */
	quickCreateTuning: boolean;
	/** Show the "Custom Path" model source option (admin-only) */
	customPathModelSource: boolean;
};

export type Log = {
	id?: number;
	filename: string;
	level: 'INFO' | 'ERROR' | 'DEBUG' | 'WARN';
	message: string;
	timestamp: Date;
};

export type LogPage = {
	logs: Log[];
	has_more: boolean;
	next_before_id: number | null;
};

// Per-step training metrics (GET /jobs/{id}/trials/{trial_id}/metrics and
// /jobs/{id}/metrics). One row per HF Trainer log event; maps 1:1 to the backend
// MetricPointRead (no api-mappers entry needed). `split` is 'train' | 'eval' but
// kept as string — the backend column is not an enum.
export type MetricPoint = {
	id: number;
	trial_id: string | null;
	global_step: number;
	epoch: number | null;
	loss: number | null;
	grad_norm: number | null;
	learning_rate: number | null;
	split: string;
	extra: Record<string, unknown> | null;
	created_at: string | null;
};

export type MetricPage = {
	metrics: MetricPoint[];
	has_more: boolean;
	next_after_id: number | null;
};

export type Assets = {
	path: string;
	size: number;
	created: Date;
	filename: string;
	modified: Date;
	file_hash: string;
	file_size: number;
	published: boolean;
};

export interface BuildStatus {
	details: Details;
	targets: Target[];
	build_history: BuildHistory[];
}

export interface BuildHistory {
	time: Date;
	description: string;
}

export interface Details {
	name: string;
	status: string;
	build_id: string;
	source_pr: string;
	started_at: Date;
	updated_at: Date;
}

export interface Target {
	steps: Step[];
	status: string;
	build_id: string;
	target_id: string;
	target_name: string;
	input_artifacts: PutArtifact[];
	output_artifacts: PutArtifact[];
}

export interface PutArtifact {
	uri: string;
	artifact_id: string;
}

export interface Step {
	uri: string;
	status: string;
	step_id: string;
	started_at: Date;
}

export type ImportRowStatus =
	| 'ready'
	| 'name_required'
	| 'name_exists'
	| 'duplicate_in_batch'
	| 'invalid_missing_name'
	| 'invalid_missing_config_data'
	| 'parse_error'
	| 'import_failed';

export interface ImportPreviewRow {
	rowId: string; // client-side uuid for keyed-each
	sourceFile: string; // e.g. "my_config.yaml"
	originalName: string; // name as it appeared in the file
	editedName: string; // current value of the name input
	tunerType: string | null;
	rlTunerType: string | null;
	configData: Record<string, unknown>;
	status: ImportRowStatus;
	errorMessage: string | null; // human-readable reason for non-ready statuses
	skipped: boolean; // user clicked ✕ on this row
	edited: boolean; // editedName !== originalName (after any auto-suggest)
}

export interface ExportPreviewRow {
	rowId: string; // the Configuration id
	name: string;
	tunerType: string;
	rlTunerType: string | null;
	skipped: boolean;
}
