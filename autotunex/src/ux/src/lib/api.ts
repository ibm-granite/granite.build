// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { get } from 'svelte/store';
import { API_BASE, AUTH_BASE } from '$lib/config';
import { isAuthenticated, currentUser, authMode, appConfig } from './store';
import { Utils } from './utils';
import type {
	AppConfig,
	Assets,
	Configuration,
	Dataset,
	Estimation,
	LogPage,
	MetricPage,
	MetricPoint,
	Resources,
	Trial,
	Tuning,
	TuningForm,
	User
} from './app-types';
import type { UserMetaData } from './user';
import {
	unwrapPage,
	pageTotal,
	mapConfiguration,
	mapDataset,
	mapJob,
	mapMappingSuggestion,
	mapTrial
} from './api-mappers';
import type { Page } from './api-mappers';

// --- Paginated list getters (used by the three tables) ---------------------------

const pageQuery = (params: { limit: number; offset: number; q?: string }): string => {
	const qs = new URLSearchParams({
		limit: String(params.limit),
		offset: String(params.offset)
	});
	if (params.q && params.q.trim()) qs.set('q', params.q.trim());
	return qs.toString();
};

/**
 * Thrown by client methods whose backend endpoint does not exist yet. The UI gates
 * these features off via the `capabilities` map, so this is a loud backstop for any
 * stray call rather than a silent no-op. See the integration spec §9 register.
 */
export class EndpointNotAvailableError extends Error {
	constructor(feature: string) {
		super(`This feature ("${feature}") has no backend endpoint yet.`);
		this.name = 'EndpointNotAvailableError';
	}
}

export class DatasetUploadError extends Error {
	readonly title?: string;
	readonly detail?: string;
	readonly status?: number;

	constructor(message: string, options: { title?: string; detail?: string; status?: number } = {}) {
		super(message);
		this.name = 'DatasetUploadError';
		this.title = options.title;
		this.detail = options.detail;
		this.status = options.status;
	}
}

// A stalled upload has to fail; a merely slow one must not. `xhr.timeout` cannot
// tell them apart — it fires on total wall-clock time since send() whether or not
// bytes are still moving, so any fixed cap kills a healthy multi-GB transfer on a
// modest link (5 GiB inside 30 minutes needs a sustained ~3 MB/s). These two
// windows drive a watchdog instead, and `xhr.timeout` is deliberately not set.
//
// The first is re-armed by every upload progress event, so only a genuine stall
// trips it. The second bounds the wait *after* the last byte is sent: that phase
// emits no progress events at all, and the server is still writing the received
// file out, so it gets its own, more generous window rather than the stall one.
const UPLOAD_STALL_TIMEOUT_MS = 3 * 60 * 1000;
const UPLOAD_RESPONSE_TIMEOUT_MS = 10 * 60 * 1000;

export class API {
	constructor() {}

	// --- External (Hugging Face) — not the AutoTuneX backend ---------------------

	getHFModels = async (search = '', limit = 10) =>
		fetch(
			`https://huggingface.co/api/models?search=${encodeURIComponent(
				search
			)}&limit=${limit}&config=true`
		).then((response) => response.json());

	getHFModelCard = async (modelId: string) =>
		fetch(`https://huggingface.co/${modelId}/raw/main/README.md`).then((response) =>
			response.text()
		);

	// --- Auth (BFF) --------------------------------------------------------------

	/**
	 * Resolve the caller and reshape the backend `Principal` into the legacy
	 * `{ authenticated, user: { email, role, user_id } }` shape the UX expects. A 200
	 * means authenticated (standalone always resolves to an admin); a 401 means a real
	 * provider wants a login. Records the inferred auth mode for 401 recovery.
	 *
	 * `user_id` is carried through (not just `email`/`role`) so the UI can key
	 * system-account checks off the stable seed UUID rather than an email literal —
	 * see the guard in `Configurations.svelte`.
	 */
	me = async (): Promise<{
		authenticated: boolean;
		user?: { email: string; role: string; user_id: string | null; impersonator?: string | null };
	}> => {
		const res = await fetch(`${AUTH_BASE}/me`, { credentials: 'include' });
		if (!res.ok) {
			authMode.set('session'); // a 401 here means a real provider wants a login
			return { authenticated: false };
		}
		const p = await res.json(); // Principal { email, provider, user_id, is_admin, impersonator }
		authMode.set(p.email == null ? 'standalone' : 'session');
		return {
			authenticated: true,
			user: {
				email: p.email ?? 'standalone',
				role: p.is_admin ? 'admin' : 'user',
				user_id: p.user_id ?? null,
				impersonator: p.impersonator ?? null
			}
		};
	};

	login = async () => {
		// Backend 302s to the OIDC provider; the browser must follow it.
		window.location.href = `${AUTH_BASE}/login`;
	};

	logout = async () => {
		const res = await fetch(`${AUTH_BASE}/logout`, { method: 'POST', credentials: 'include' });
		return res.ok ? await res.json().catch(() => ({})) : {}; // may carry { end_session_endpoint }
	};

	assumeUser = async (userId: string) => {
		const res = await fetch(`${AUTH_BASE}/assume/${userId}`, {
			method: 'POST',
			credentials: 'include'
		});
		if (!res.ok) throw new Error(`Failed to assume user (${res.status})`);
		return res.json();
	};

	unassumeUser = async () => {
		const res = await fetch(`${AUTH_BASE}/unassume`, { method: 'POST', credentials: 'include' });
		if (!res.ok) throw new Error(`Failed to exit impersonation (${res.status})`);
		return res.json();
	};

	// --- Configurations (full CRUD) ----------------------------------------------

	// GET /configurations is paginated (limit max 100); loop offsets so pickers see
	// the whole list, mirroring getUsers below.
	getConfigurations = async (): Promise<Configuration[]> => {
		const limit = 100;
		const out: Configuration[] = [];
		for (let offset = 0; ; offset += limit) {
			const page = await fetch(`${API_BASE}/configurations?limit=${limit}&offset=${offset}`, {
				credentials: 'include'
			}).then(this.handleResponse);
			const batch = unwrapPage(page).map(mapConfiguration);
			out.push(...batch);
			if (batch.length < limit || out.length >= pageTotal(page)) break;
		}
		return out;
	};

	getConfigurationsPage = async (params: {
		limit: number;
		offset: number;
		q?: string;
	}): Promise<Page<Configuration>> => {
		const page = await fetch(`${API_BASE}/configurations?${pageQuery(params)}`, {
			credentials: 'include'
		}).then(this.handleResponse);
		return {
			items: unwrapPage(page).map(mapConfiguration),
			total: pageTotal(page),
			limit: params.limit,
			offset: params.offset
		};
	};

	getConfiguration = async (id: string) =>
		fetch(`${API_BASE}/configurations/${id}`, { credentials: 'include' })
			.then(this.handleResponse)
			.then(mapConfiguration);

	createConfiguration = async (config: any) =>
		fetch(`${API_BASE}/configurations`, {
			method: 'POST',
			body: JSON.stringify(config),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	updateConfiguration = async (config_id: string, config: any) =>
		fetch(`${API_BASE}/configurations/${config_id}`, {
			method: 'PUT',
			body: JSON.stringify(config),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	deleteConfiguration = async (config_id: string) =>
		fetch(`${API_BASE}/configurations/${config_id}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);

	// --- Jobs (read-only) --------------------------------------------------------

	getJobs = async (): Promise<Tuning[]> => {
		const page = await fetch(`${API_BASE}/jobs`, { credentials: 'include' }).then(
			this.handleResponse
		);
		return unwrapPage(page).map(mapJob);
	};

	getJobsPage = async (params: {
		limit: number;
		offset: number;
		q?: string;
	}): Promise<Page<Tuning>> => {
		const page = await fetch(`${API_BASE}/jobs?${pageQuery(params)}`, {
			credentials: 'include'
		}).then(this.handleResponse);
		return {
			items: unwrapPage(page).map(mapJob),
			total: pageTotal(page),
			limit: params.limit,
			offset: params.offset
		};
	};

	// The legacy `options` (include_logs/log_limit/all_logs) are accepted for
	// caller-compat but ignored — there is no log backend yet.
	getJob = async (id: string, _options?: unknown): Promise<Tuning> => {
		const job = await fetch(`${API_BASE}/jobs/${id}`, { credentials: 'include' }).then(
			this.handleResponse
		);
		return mapJob(job);
	};

	// --- Trials (their own endpoint) ---------------------------------------------

	// GET /jobs/{id}/trials is paginated (limit max 100); loop offsets so the Trials
	// table — which paginates client-side over the whole array — still sees every
	// trial. Mirrors getConfigurations/getDatasets/getUsers above.
	//
	// This used to read `(await this.getJob(jobId)).trials`, which cost a second
	// fetch of the whole job detail (trials and their results included) on top of
	// the one TuningDisplay had already made, on every load and every poll tick.
	getTrialsByJobId = async (jobId: string): Promise<Trial[]> => {
		const limit = 100;
		const out: Trial[] = [];
		for (let offset = 0; ; offset += limit) {
			const page = await fetch(`${API_BASE}/jobs/${jobId}/trials?limit=${limit}&offset=${offset}`, {
				credentials: 'include'
			}).then(this.handleResponse);
			const batch = unwrapPage(page).map(mapTrial);
			out.push(...batch);
			if (batch.length < limit || out.length >= pageTotal(page)) break;
		}
		return out;
	};

	// The UX nests metric/metrics under `score`; mapTrial already builds it.
	getResultsByJobId = async (jobId: string) =>
		(await this.getTrialsByJobId(jobId)).map((t: any) => t.score).filter(Boolean);

	// --- Derived from the nested job payload (no dedicated endpoint) --------------

	getAllTaskByJob = async (jobId: string) => (await this.getJob(jobId)).tasks ?? [];

	// `is_stale` is a top-level, read-time field on the job detail (not baked into the
	// frozen snapshot); merge it onto the snapshot object the drift banner reads.
	// `Object.assign` (not object spread) keeps the snapshot's index signature so callers
	// still read `snapshot.name`, `.config_data`, etc. without a type error.
	getJobConfigSnapshot = async (jobId: string) => {
		const job = await this.getJob(jobId);
		return job.config_snapshot
			? Object.assign({}, job.config_snapshot, { is_stale: job.is_stale ?? false })
			: null;
	};

	// --- Datasets (CRUD + upload) ------------------------------------------------

	// GET /datasets is paginated (limit max 100); loop offsets so pickers see the
	// whole list, mirroring getUsers below.
	getDatasets = async (): Promise<Dataset[]> => {
		const limit = 100;
		const out: Dataset[] = [];
		for (let offset = 0; ; offset += limit) {
			const page = await fetch(`${API_BASE}/datasets?limit=${limit}&offset=${offset}`, {
				credentials: 'include'
			}).then(this.handleResponse);
			const batch = unwrapPage(page).map(mapDataset);
			out.push(...batch);
			if (batch.length < limit || out.length >= pageTotal(page)) break;
		}
		return out;
	};

	getDatasetsPage = async (params: {
		limit: number;
		offset: number;
		q?: string;
	}): Promise<Page<Dataset>> => {
		const page = await fetch(`${API_BASE}/datasets?${pageQuery(params)}`, {
			credentials: 'include'
		}).then(this.handleResponse);
		return {
			items: unwrapPage(page).map(mapDataset),
			total: pageTotal(page),
			limit: params.limit,
			offset: params.offset
		};
	};

	/**
	 * Read-only, non-sensitive config the backend defines and the frontend needs
	 * (upload cap, client gzip/preview knobs). Unauthenticated by design.
	 *
	 * Deliberately NOT routed through `handleResponse`: this runs at boot, before
	 * `/auth/me` has settled `authMode`, and `handleResponse`'s 401 branch calls
	 * `window.location.reload()` for anything that is not yet known to be session
	 * mode. An edge proxy that 401s this first unauthenticated call (an OpenShift
	 * oauth-proxy, say) would then reload the page, re-issue it, and loop — and
	 * because that branch returns instead of throwing, the caller's own `.catch()`
	 * could not break the cycle. A plain fetch that rejects keeps the failure local
	 * to this one fire-and-forget request.
	 */
	getAppConfig = async (): Promise<AppConfig> => {
		const res = await fetch(`${API_BASE}/app-config`, { credentials: 'include' });
		if (!res.ok) throw new Error(`Failed to load app config (HTTP ${res.status}).`);
		return res.json();
	};

	getDataset = async (id: string, preview = false) =>
		fetch(`${API_BASE}/datasets/${id}${preview ? '?preview=true' : ''}`, {
			credentials: 'include'
		})
			.then(this.handleResponse)
			.then(mapDataset);

	createDataset = async (dataset: any) =>
		fetch(`${API_BASE}/datasets`, {
			method: 'POST',
			// data_format defaults to jsonl; description is optional — a blank one is sent
			// as null so the nullable column stores NULL rather than an empty string.
			body: JSON.stringify({
				data_format: 'jsonl',
				...dataset,
				description: dataset?.description || null
			}),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	deleteDataset = async (datasetId: string) =>
		fetch(`${API_BASE}/datasets/${datasetId}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);

	/**
	 * Upload a dataset's file(s) via the backend's single multipart endpoint, then
	 * poll until the async validation reaches a terminal status. Keeps the original
	 * signature (train + optional validation, or an auto-split percentage) so callers
	 * are unchanged; the tus resumable path is retired (registered as a future endpoint).
	 *
	 * On any HTTP or network failure, rejects with a {@link DatasetUploadError} (never
	 * the raw XMLHttpRequest) so a 413 or other problem+json error reaches the caller
	 * as a real, human-readable message instead of a generic "Network error."
	 */
	uploadDatasetChunked = async (
		datasetId: string,
		opts: {
			trainFile: File;
			validationFile?: File | null;
			columnMapping?: Record<string, string> | null;
			trainSetPercentage?: number | null; // TRAIN share, e.g. 80 for an 80/20 split
			chunkSize?: number; // accepted for signature-compat; unused (no chunking now)
			onProgress?: (percent: number) => void;
		}
	): Promise<any> => {
		// The boot-time fetch already put this in a store (see (release)/+layout.svelte),
		// so read that rather than spending a round trip per upload. `null` means the
		// fetch has not resolved or failed — fall back to the same defaults it would.
		const gzipConfig = get(appConfig)?.dataset_upload;
		const isCompressibleFormat = (file: File) => /\.(jsonl|json|csv)$/i.test(file.name);
		// Name the real cap in a 413 when we know it; stay silent when we don't,
		// rather than guessing a number at the user.
		const sizeLimitSuffix = gzipConfig?.max_bytes
			? ` of ${(gzipConfig.max_bytes / 1024 ** 3).toFixed(1)} GiB`
			: '';
		// Server contract: Content-Encoding: gzip applies to the WHOLE request,
		// decoded uniformly for every file part (no per-part signal exists) — so
		// this is one all-or-nothing decision for the whole upload, based on the
		// train file (the dominant cost driver). See Global Constraints.
		const wantsGzip =
			(gzipConfig?.client_gzip_enabled ?? true) &&
			isCompressibleFormat(opts.trainFile) &&
			opts.trainFile.size >= (gzipConfig?.client_gzip_min_bytes ?? 1024 * 1024);

		let trainBody: File | Blob = opts.trainFile;
		let validationBody: File | Blob | null = opts.validationFile ?? null;
		let gzipApplied = false;
		if (wantsGzip) {
			try {
				trainBody = await Utils.gzipFileAsync(opts.trainFile);
				validationBody = opts.validationFile
					? await Utils.gzipFileAsync(opts.validationFile)
					: null;
				gzipApplied = true;
			} catch {
				// Compression unsupported or failed — fall back to an uncompressed
				// upload rather than blocking it. Never send a partially-gzipped
				// request (the server treats Content-Encoding as whole-request).
				trainBody = opts.trainFile;
				validationBody = opts.validationFile ?? null;
				gzipApplied = false;
			}
		}

		const form = new FormData();
		form.append('train_file', trainBody, opts.trainFile.name);
		if (validationBody) {
			form.append('validation_file', validationBody, opts.validationFile!.name);
		} else if (opts.trainSetPercentage != null) {
			// Backend rejects a validation file AND a percentage together (422).
			// `trainSetPercentage` is the TRAIN share, but the API field is the
			// VALIDATION share — invert here or the split lands backwards.
			form.append('validation_percentage', String(100 - opts.trainSetPercentage));
		}
		if (opts.columnMapping) form.append('column_mapping', JSON.stringify(opts.columnMapping));

		// XHR (not fetch) so we keep byte-level upload progress.
		await new Promise<void>((resolve, reject) => {
			const xhr = new XMLHttpRequest();
			const STALLED = 'Upload timed out — the connection stalled.';
			let watchdog: ReturnType<typeof setTimeout> | undefined;
			let timeoutMessage = STALLED;
			let aborting = false;
			const disarm = () => {
				if (watchdog !== undefined) clearTimeout(watchdog);
				watchdog = undefined;
			};
			// Aborting routes through onabort below, which rejects with the message of
			// whichever window ran out. Re-arming is refused once we have decided to
			// abort: `xhr.abort()` fires the upload's own `loadend` first, and letting
			// that swap the message in would report a stalled transfer as a silent
			// server.
			const arm = (ms: number, message: string) => {
				disarm();
				if (aborting) return;
				timeoutMessage = message;
				watchdog = setTimeout(() => {
					aborting = true;
					xhr.abort();
				}, ms);
			};

			xhr.open('POST', `${API_BASE}/datasets/${datasetId}/upload`);
			xhr.withCredentials = true;
			if (gzipApplied) xhr.setRequestHeader('Content-Encoding', 'gzip');
			xhr.upload.onprogress = (e) => {
				arm(UPLOAD_STALL_TIMEOUT_MS, STALLED);
				if (e.lengthComputable && opts.onProgress) {
					opts.onProgress(Math.min(99, Math.round((e.loaded / e.total) * 100)));
				}
			};
			// Last byte handed off (or the transfer died): stop expecting progress
			// events and switch to bounding the server's response instead. On the
			// abort path this re-arms momentarily, then onabort disarms for good.
			xhr.upload.onloadend = () =>
				arm(UPLOAD_RESPONSE_TIMEOUT_MS, 'Upload timed out — the server did not respond.');
			xhr.onload = () => {
				disarm();
				if (xhr.status >= 200 && xhr.status < 300) {
					resolve();
					return;
				}
				let problem: { title?: string; detail?: string } = {};
				try {
					problem = JSON.parse(xhr.responseText);
				} catch {
					// Non-JSON error body (e.g. a proxy's own error page) — fall through.
				}
				const message =
					xhr.status === 413
						? `File exceeds the upload size limit${sizeLimitSuffix}.`
						: problem.detail || problem.title || `Upload failed (HTTP ${xhr.status}).`;
				reject(
					new DatasetUploadError(message, {
						title: problem.title,
						detail: problem.detail,
						status: xhr.status
					})
				);
			};
			xhr.onerror = () => {
				disarm();
				reject(new DatasetUploadError('Network error occurred during upload.', { status: 0 }));
			};
			xhr.onabort = () => {
				disarm();
				reject(new DatasetUploadError(timeoutMessage, { status: 0 }));
			};
			// Arm before send: a connection that never produces a first progress
			// event must not hang here forever.
			arm(UPLOAD_STALL_TIMEOUT_MS, STALLED);
			xhr.send(form);
		});

		// Backend flips to 'uploading' then validates off-request; poll until terminal.
		const terminal = await this.pollDatasetStatus(datasetId);
		opts.onProgress?.(100);
		if ((terminal as any)?.status === 'error') {
			const detail = (terminal as any).status_detail ?? '';
			throw new DatasetUploadError(detail || 'Dataset processing failed', {
				title: 'Dataset processing failed',
				detail
			});
		}
		return terminal;
	};

	// Poll GET /datasets/{id} until status leaves 'uploading'. Bounded by attempts.
	// Throws (rather than silently returning an in-progress dataset) if the budget
	// is exhausted while still 'uploading' — the caller must see that it did not
	// finish, not treat a partial result as success.
	pollDatasetStatus = async (datasetId: string, attempts = 60, delayMs = 2000): Promise<any> => {
		for (let i = 0; i < attempts; i++) {
			const ds = await this.getDataset(datasetId);
			if ((ds as any).status !== 'uploading') return ds;
			await new Promise((r) => setTimeout(r, delayMs));
		}
		const ds = await this.getDataset(datasetId);
		if ((ds as any).status === 'uploading') {
			throw new DatasetUploadError(
				'Dataset is still processing. This can take a while for large files — check back in a moment.',
				{ title: 'Still processing', status: 202 }
			);
		}
		return ds;
	};

	// --- Dataset intelligence (stateless LLM helpers) ----------------------------

	generateParsingStrategy = async (sample: string | any[], format: string, customPrompt?: string) =>
		fetch(`${API_BASE}/datasets/intelligence/parse-strategy`, {
			method: 'POST',
			body: JSON.stringify({
				sample,
				data_format: format,
				...(customPrompt ? { custom_prompt: customPrompt } : {})
			}),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	validateParsingStrategy = async (strategy: any, sample: string | any[]) =>
		fetch(`${API_BASE}/datasets/intelligence/validate-strategy`, {
			method: 'POST',
			body: JSON.stringify({ strategy, sample }),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	suggestColumnMapping = async (
		sampleData: Record<string, any>[],
		columnNames: string[],
		columnSamples: Record<string, string[]>,
		targetDatasetType?: string
	) => {
		const body: Record<string, any> = {
			sample_data: sampleData,
			column_names: columnNames,
			column_samples: columnSamples
		};
		if (targetDatasetType) body.target_format = targetDatasetType;
		return fetch(`${API_BASE}/datasets/intelligence/suggest-mapping`, {
			method: 'POST',
			body: JSON.stringify(body),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		})
			.then(this.handleResponse)
			.then(mapMappingSuggestion);
	};

	getAutotuneDatasetTypes = async () =>
		fetch(`${API_BASE}/datasets/intelligence/formats`, { credentials: 'include' }).then(
			this.handleResponse
		);

	// --- User metadata (caller's own counts) -------------------------------------

	getUserMetadata = async (): Promise<UserMetaData> =>
		fetch(`${API_BASE}/users/me/metadata`, { credentials: 'include' }).then(this.handleResponse);

	// --- No backend yet — gated in the UI, loud if called anyway -----------------

	// Starter template for a new configuration — the schema-less config_data shape,
	// sourced from the autotune core. Returns the raw object (not a Configuration);
	// a 503 (autotune not installed) surfaces via handleResponse's thrown problem-detail.
	getConfigurationTemplate = async (): Promise<any> =>
		fetch(`${API_BASE}/configurations/template`, { credentials: 'include' }).then(
			this.handleResponse
		);
	startJob = async (form: TuningForm): Promise<Tuning> => {
		const job = await fetch(`${API_BASE}/jobs`, {
			method: 'POST',
			body: JSON.stringify(form),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);
		return mapJob(job);
	};
	deleteJob = async (jobId: string) =>
		fetch(`${API_BASE}/jobs/${jobId}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);
	estimateUsage = async (payload: Estimation): Promise<Resources> =>
		fetch(`${API_BASE}/jobs/estimate-usages`, {
			method: 'POST',
			body: JSON.stringify(payload),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);
	// --- Logs (DB-backed job/trial logs are keyset-paginated, newest first) --------

	getLogs = async (jobId: string, beforeId = 0, limit = 50): Promise<LogPage> => {
		const qs = new URLSearchParams();
		if (beforeId > 0) qs.set('before_id', String(beforeId));
		if (limit !== 50) qs.set('limit', String(limit));
		const q = qs.toString();
		return fetch(`${API_BASE}/jobs/${jobId}/logs${q ? `?${q}` : ''}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	// The trial-log path fixes BOTH job and trial ids, so jobId is required here.
	getTrialLogs = async (
		jobId: string,
		trialId: string,
		beforeId = 0,
		limit = 50
	): Promise<LogPage> => {
		const qs = new URLSearchParams();
		if (beforeId > 0) qs.set('before_id', String(beforeId));
		if (limit !== 50) qs.set('limit', String(limit));
		const q = qs.toString();
		return fetch(`${API_BASE}/jobs/${jobId}/trials/${trialId}/logs${q ? `?${q}` : ''}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	// --- Training metrics (per-step; ascending keyset by id) ----------------------

	// Page a metrics endpoint to completion. The endpoint returns ascending keyset
	// pages (default limit 1000) and a trial is only ~10 points/epoch, so this is
	// usually a single request — but loop on has_more to be safe. Mirrors the
	// getLogs/getTrialLogs URL+auth pattern. No api-mappers entry: MetricPage maps
	// 1:1 to the frontend type. `handleResponse` returns undefined on a 401 (it
	// redirects), so the `?.` guards yield [] rather than throwing.
	pageMetrics = async (url: string): Promise<MetricPoint[]> => {
		const out: MetricPoint[] = [];
		let afterId = 0;
		for (;;) {
			const sep = url.includes('?') ? '&' : '?';
			const page: MetricPage = await fetch(`${url}${sep}after_id=${afterId}`, {
				credentials: 'include'
			}).then(this.handleResponse);
			out.push(...(page?.metrics ?? []));
			if (!page?.has_more || page.next_after_id == null || page.next_after_id <= afterId) break;
			afterId = page.next_after_id;
		}
		return out;
	};

	getTrialMetrics = async (jobId: string, trialId: string): Promise<MetricPoint[]> =>
		this.pageMetrics(`${API_BASE}/jobs/${jobId}/trials/${trialId}/metrics`);

	// Phase 2 (cross-trial overlay). Added now to unblock it; no Phase 1 UI consumes it.
	getJobMetrics = async (jobId: string): Promise<MetricPoint[]> =>
		this.pageMetrics(`${API_BASE}/jobs/${jobId}/metrics`);

	// Live build (gb) logs: oldest-first string lines. 503 when the reader is disabled.
	getGBLogs = async (jobId: string, all = false): Promise<string[]> => {
		const q = all ? '?all=true' : '';
		return fetch(`${API_BASE}/jobs/${jobId}/gb-logs${q}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};
	getAssetsByJobId = async (jobId: string): Promise<Assets[]> =>
		fetch(`${API_BASE}/jobs/${jobId}/result-report`, { credentials: 'include' }).then(
			this.handleResponse
		);

	// Downloads are plain authenticated GET URLs the browser navigates to directly, so
	// large files (model checkpoints) stream straight to disk instead of through JS. The
	// BFF session cookie rides along on the top-level GET; the server sets
	// Content-Disposition: attachment. Per-file downloads key on the relative `path`
	// (not the basename, which repeats across directories); download-all yields a ZIP.
	resultFileUrl = (jobId: string, path: string): string =>
		`${API_BASE}/jobs/${jobId}/result-report/file?path=${encodeURIComponent(path)}`;

	resultArchiveUrl = (jobId: string): string => `${API_BASE}/jobs/${jobId}/result-report/archive`;
	// --- User management (admin) -------------------------------------------------

	// GET /users is paginated (limit max 100); loop offsets so the whole list shows.
	getUsers = async (): Promise<User[]> => {
		const limit = 100;
		const users: User[] = [];
		for (let offset = 0; ; offset += limit) {
			const page = await fetch(`${API_BASE}/users?limit=${limit}&offset=${offset}`, {
				credentials: 'include'
			}).then(this.handleResponse);
			const batch = unwrapPage<User>(page);
			users.push(...batch);
			if (batch.length < limit || users.length >= pageTotal(page)) break;
		}
		return users;
	};

	setUserRole = async (userId: string, role: 'admin' | 'user'): Promise<User> =>
		fetch(`${API_BASE}/users/${userId}`, {
			method: 'PATCH',
			body: JSON.stringify({ role }),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);
	generateTestSolutions = async (prompts: Array<Array<Record<string, string>>>): Promise<any> =>
		fetch(`${API_BASE}/jobs/generate-test-solutions`, {
			method: 'POST',
			body: JSON.stringify({ prompts }),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);
	validateRewardFunction = async (
		code: string,
		functionName: string,
		testExecution: boolean,
		testInputs?: unknown
	): Promise<any> =>
		fetch(`${API_BASE}/reward-functions/validate`, {
			method: 'POST',
			body: JSON.stringify({
				code,
				function_name: functionName,
				test_execution: testExecution,
				test_inputs: testInputs ?? null
			}),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);
	startChat = async (
		messages: Array<{ role: string; content: string }>,
		context: Record<string, unknown> = {},
		thread_id?: string
	) => {
		try {
			const response = await fetch(`${API_BASE}/chat`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				credentials: 'include',
				body: JSON.stringify({ messages, context, thread_id })
			});

			if (response.ok) {
				return await response.json();
			} else {
				const errorData = await response.text();
				console.error('Chat error:', errorData);
				throw new Error('Chat request failed');
			}
		} catch (error) {
			console.error('Request failed:', error);
			throw error;
		}
	};

	startChatStream = (
		messages: Array<{ role: string; content: string }>,
		context: Record<string, unknown> = {},
		handlers: {
			onToolStart?: (name: string, label: string) => void;
			onToolEnd?: (name: string) => void;
			onToken?: (text: string) => void;
			onContext?: (ctx: Record<string, unknown>) => void;
			onRefresh?: (target: string) => void;
			onDone?: () => void;
			onError?: (message: string) => void;
		} = {},
		thread_id?: string
	): { done: Promise<void>; abort: () => void } => {
		const controller = new AbortController();

		const done = (async () => {
			try {
				const response = await fetch(`${API_BASE}/chat/stream`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Accept: 'text/event-stream'
					},
					credentials: 'include',
					body: JSON.stringify({ messages, context, thread_id }),
					signal: controller.signal
				});

				if (!response.ok || !response.body) {
					const errorText = response.ok ? 'No response body' : await response.text();
					handlers.onError?.(errorText || `Chat stream failed (${response.status})`);
					return;
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let buffer = '';

				const dispatch = (raw: string) => {
					const trimmed = raw.trim();
					if (!trimmed) return;
					// SSE frames may contain multiple "data:" lines; concatenate them
					const dataLines: string[] = [];
					for (const line of trimmed.split('\n')) {
						if (line.startsWith('data:')) {
							dataLines.push(line.slice(5).trimStart());
						}
					}
					if (dataLines.length === 0) return;
					const payload = dataLines.join('\n');
					let evt: Record<string, unknown>;
					try {
						evt = JSON.parse(payload);
					} catch (e) {
						console.warn('Bad SSE payload:', payload, e);
						return;
					}
					switch (evt.type) {
						case 'tool_start':
							handlers.onToolStart?.(String(evt.name ?? ''), String(evt.label ?? ''));
							break;
						case 'tool_end':
							handlers.onToolEnd?.(String(evt.name ?? ''));
							break;
						case 'token':
							if (typeof evt.text === 'string' && evt.text.length > 0) {
								handlers.onToken?.(evt.text);
							}
							break;
						case 'context':
							if (evt.context && typeof evt.context === 'object') {
								handlers.onContext?.(evt.context as Record<string, unknown>);
							}
							break;
						case 'refresh':
							handlers.onRefresh?.(String(evt.target ?? ''));
							break;
						case 'done':
							handlers.onDone?.();
							break;
						case 'error':
							handlers.onError?.(String(evt.message ?? 'Chat failed.'));
							break;
						default:
							break;
					}
				};

				while (true) {
					const { value, done: streamDone } = await reader.read();
					if (value) buffer += decoder.decode(value, { stream: true });
					let idx: number;
					while ((idx = buffer.indexOf('\n\n')) !== -1) {
						const frame = buffer.slice(0, idx);
						buffer = buffer.slice(idx + 2);
						dispatch(frame);
					}
					if (streamDone) {
						buffer += decoder.decode();
						if (buffer.trim()) dispatch(buffer);
						buffer = '';
						break;
					}
				}
			} catch (error) {
				if ((error as DOMException)?.name === 'AbortError') {
					// Caller aborted — silent.
					return;
				}
				console.error('Chat stream failed:', error);
				handlers.onError?.((error as Error)?.message ?? 'Chat stream failed.');
			}
		})();

		return { done, abort: () => controller.abort() };
	};

	// --- Shared response handler -------------------------------------------------

	handleResponse = async (response: Response) => {
		if (response.ok) {
			return response.status === 204 ? undefined : await response.json();
		}
		if (response.status === 401) {
			isAuthenticated.set(false);
			currentUser.set(null);
			// Session mode → send the browser to the backend login; standalone → reload.
			if (get(authMode) === 'session') window.location.href = `${AUTH_BASE}/login`;
			else window.location.reload();
			return;
		}
		throw await response.json().catch(() => ({ title: 'Request failed', status: response.status }));
	};
}
