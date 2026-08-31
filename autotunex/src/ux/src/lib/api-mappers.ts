// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

// Anti-corruption layer: the backend returns Page[T] envelopes, lowercase
// RunStatus values, and nested trials/tasks. The UX components were written
// against a different, flatter shape with UPPERCASE statuses. Everything here
// translates one to the other so components stay unchanged.

import type {
	AiMappingSuggestion,
	Configuration,
	Dataset,
	Status,
	Task,
	Trial,
	Tuning
} from './app-types';

export type Page<T> = { items: T[]; total: number; limit: number; offset: number };

export const unwrapPage = <T>(res: Page<T>): T[] => res?.items ?? [];
export const pageTotal = (res: Page<unknown>): number => res?.total ?? 0;

// Backend RunStatus is lowercase; ShowStatus.svelte matches UPPERCASE (incl. PAUSED).
const STATUS_UP: Record<string, Status> = {
	pending: 'PENDING',
	running: 'RUNNING',
	paused: 'PAUSED' as Status,
	terminated: 'TERMINATED',
	error: 'ERROR',
	completed: 'COMPLETED'
};

export const mapStatus = (s: string): Status =>
	STATUS_UP[(s ?? '').toLowerCase()] ?? ((s ?? '').toUpperCase() as Status);

export const mapTrial = (t: any): Trial =>
	({
		id: t.id,
		job_id: t.job_id,
		status: mapStatus(t.status),
		config: t.config ?? {},
		created_at: t.created_at,
		updated_at: t.updated_at,
		logs: [], // no trial-log backend yet
		// Backend puts metric/metrics flat on the trial; UX expects a nested `score`.
		score: {
			id: t.id,
			job_id: t.job_id,
			trial_id: t.id,
			metric: t.metric,
			metrics: t.metrics ?? {},
			created_at: t.created_at,
			updated_at: t.updated_at
		}
	}) as unknown as Trial;

// The backend nests build tasks under `tasks[]` with the autotunex_jobs view's
// aliases (task_id / task_status / task_type / github_pr_url / task_started_at /
// task_updated_at), while Tasks.svelte and the `Task` type read the flatter
// id / status / type / pr_url / started_at / updated_at. Translate each task here so
// the admin Tasks tab renders — otherwise every cell is blank and Tasks.svelte's
// `row?.id.split('-')` throws on the undefined `id`. `task_status` is lowercase
// RunStatus; ShowStatus matches UPPERCASE, so route it through mapStatus. `job_id`
// is filled from the parent job — GbTaskRead does not carry it.
export const mapTask = (t: any, jobId: string): Task =>
	({
		id: t.task_id,
		job_id: jobId,
		build_id: t.build_id,
		status: t.task_status ? mapStatus(t.task_status) : t.task_status,
		type: t.task_type,
		pr_url: t.github_pr_url,
		artifact_id: t.artifact_id,
		artifact_uri: t.artifact_uri,
		build_status: t.build_status,
		started_at: t.task_started_at,
		updated_at: t.task_updated_at
	}) as unknown as Task;

export const mapJob = (job: any): Tuning => {
	const tasks: any[] = Array.isArray(job.tasks) ? job.tasks : [];
	const t0 = tasks[0] ?? {};
	return {
		...job,
		status: mapStatus(job.status),
		// Flatten the first task into the legacy top-level task fields, but keep tasks[].
		task_id: t0.task_id,
		build_id: t0.build_id,
		task_status: t0.task_status ? mapStatus(t0.task_status) : undefined,
		task_type: t0.task_type,
		github_pr_url: t0.github_pr_url,
		artifact_id: t0.artifact_id,
		artifact_uri: t0.artifact_uri,
		// The admin Status tab renders `tuning.build_status.build_history`; without this
		// lift it stays pinned on the "Loading details..." spinner even though the payload
		// carries tasks[0].build_status. Mirrors the build_id/github_pr_url lift above.
		build_status: t0.build_status,
		task_started_at: t0.task_started_at,
		task_updated_at: t0.task_updated_at,
		tasks: tasks.map((t) => mapTask(t, job.id)),
		trials: Array.isArray(job.trials) ? job.trials.map(mapTrial) : [],
		config_snapshot: job.config_snapshot,
		output_artifacts: job.output_artifacts
	} as unknown as Tuning;
};

// Backend ConfigurationRead has no `associated_jobs` (unlike DatasetRead, which defaults it
// to []), but the frontend Configuration type declares it non-optional and Configurations.svelte
// iterates it in the `Tunings` cell ({#each cell.value}) and reads `.length` on row-select —
// both throw on `undefined`. Default it here so every consumer sees the array the type promises.
// (The column stays empty until the backend exposes the reverse job→config relation.)
export const mapConfiguration = (c: any): Configuration =>
	({
		...c,
		associated_jobs: Array.isArray(c?.associated_jobs) ? c.associated_jobs : []
	}) as Configuration;

// The new API nests preview rows under `preview.{train,validation}` (populated only
// when `?preview=true` and status='ready'); the old frontend expects them flattened
// onto `train_data`/`validation_data`. Translate at the seam so DatasetDisplay's
// render gate works unchanged. Missing `preview` leaves the old fields untouched.
export const mapDataset = (d: any): Dataset =>
	({
		...d,
		train_data: d?.preview?.train ?? d?.train_data,
		validation_data: d?.preview?.validation ?? d?.validation_data
	}) as Dataset;

// suggest-mapping returns a FLAT `column_mapping: {target: source}` with a separate
// `column_confidence: {target: 0..1}` sidecar (plus `dataset_format`/`tuning_type`), while
// CreateDatasetForm reads the older nested `{target: {source_column, confidence}}` shape —
// so without translation it skips every entry (a string has no `.source_column`) and leaves
// each dropdown on its placeholder. Fold the confidence sidecar back into each entry here.
// `algorithm` is left empty on purpose: the backend echoes the autotune dataset-type key
// (e.g. `dataset_type_a`), not the UI's algorithm id (`lora`/`dpo`/…), so surfacing it would
// override the user's selection and break the ALGORITHM_TO_DATASET_TYPE lookups downstream.
export const mapMappingSuggestion = (s: any): AiMappingSuggestion => {
	const flat: Record<string, unknown> = s?.column_mapping ?? {};
	const conf: Record<string, number> = s?.column_confidence ?? {};
	const overall: number = typeof s?.confidence === 'number' ? s.confidence : 0;
	const nested: Record<string, { source_column: string; confidence: number }> = {};
	for (const [target, source] of Object.entries(flat)) {
		if (typeof source !== 'string') continue; // null/unmatched targets are already dropped server-side
		nested[target] = { source_column: source, confidence: conf[target] ?? overall };
	}
	return {
		dataset_type: s?.dataset_format ?? '',
		dataset_type_desc: '',
		algorithm: '',
		confidence: overall,
		column_mapping: nested,
		reasoning: s?.reasoning ?? ''
	};
};
