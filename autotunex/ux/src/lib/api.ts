// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import type { User } from '$lib/user';
import { PUBLIC_AUTOTUNEX_API_URL } from '$env/static/public';
import { isAuthenticated, currentUser } from './store';
import type { Configuration, Dataset, Estimation, Log, Resources, Tuning } from './app-types';

export class API {
	constructor() {}

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

	getConfigurations = async (): Promise<Configuration[]> =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/configs`, { credentials: 'include' }).then(
			this.handleResponse
		);

	login = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/login`, { credentials: 'include' }).then((response) =>
			response.json()
		);

	me = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/me`, { credentials: 'include' }).then((response) =>
			response.json()
		);

	validate = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/validate`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			credentials: 'include'
		}).then((response) => response.json());

	logout = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/logout`, {
			method: 'POST',
			credentials: 'include'
		}).then((response) => response.json());

	getConfigurationTemplate = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/config`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getConfiguration = async (id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/config/${id}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getJobConfigSnapshot = async (jobId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/config`, { credentials: 'include' }).then(
			this.handleResponse
		);

	createConfiguration = async (config: any) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/config`, {
			method: 'POST',
			body: JSON.stringify(config),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		})
			.then(async (response) => {
				return this.handleResponse(response);
			})
			.catch(async (error) => {
				console.log(await error);
				console.error('Failed to create configuration:');
				throw error;
			});

	updateConfiguration = async (config_id: string, config: any) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/config/${config_id}`, {
			method: 'PUT',
			body: JSON.stringify(config),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	deleteConfiguration = async (config_id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/config/${config_id}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);

	getJobs = (): Promise<Tuning[]> => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/jobs`, { credentials: 'include' }).then(
			this.handleResponse
		);
	};

	getGBLogs = (job_id: string, all = false) => {
		const suffix = all ? '?all=true' : '';
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/gb/logs/${job_id}${suffix}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	getTrialsByJobId = (jobId: string) => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/trials`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	getTrialLogs = async (
		trialId: string,
		beforeId: number = 0,
		limit: number = 50
	): Promise<{ logs: Log[]; has_more: boolean }> => {
		const params = new URLSearchParams();
		if (beforeId > 0) params.set('before_id', String(beforeId));
		if (limit !== 50) params.set('limit', String(limit));
		const qs = params.toString();
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/trial/${trialId}/logs${qs ? `?${qs}` : ''}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	getResultsByJobId = (jobId: string) => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/results`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	getAssetsByJobId = (jobId: string) => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/result_report`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	getPublishedModels = () => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/models`, { credentials: 'include' }).then(
			this.handleResponse
		);
	};

	publishModel = (jobId: string, data: Record<string, any>) => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/model/${jobId}`, {
			method: 'POST',
			body: JSON.stringify(data),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);
	};

	deleteModel = (jobId: string) => {
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/model/${jobId}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);
	};

	startJob = async (tuning: any) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job`, {
			method: 'POST',
			body: JSON.stringify(tuning),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	pushToRits = async (job_id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/push_to_rits/${job_id}`, {
			method: 'POST',
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	getPushToRits = async (id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/push_to_rits/${id}`, { credentials: 'include' })
			.then(this.handleResponse)
			.catch((e) => {
				console.log('error', e);
			});

	getJob = async (
		id: string,
		options?: { include_logs?: boolean; log_limit?: number; all_logs?: boolean }
	): Promise<Tuning> => {
		const params = new URLSearchParams();
		if (options?.include_logs === false) params.set('include_logs', 'false');
		if (options?.log_limit !== undefined) params.set('log_limit', String(options.log_limit));
		if (options?.all_logs) params.set('all_logs', 'true');
		const qs = params.toString();
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${id}${qs ? `?${qs}` : ''}`, {
			credentials: 'include'
		})
			.then(this.handleResponse)
			.catch((e) => {
				console.log('error', e);
			});
	};

	getLogs = async (
		jobId: string,
		beforeId: number = 0,
		limit: number = 50
	): Promise<{ logs: Log[]; has_more: boolean }> => {
		const params = new URLSearchParams();
		if (beforeId > 0) params.set('before_id', String(beforeId));
		if (limit !== 50) params.set('limit', String(limit));
		const qs = params.toString();
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/logs${qs ? `?${qs}` : ''}`, {
			credentials: 'include'
		}).then(this.handleResponse);
	};

	deleteJob = async (job_id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${job_id}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);

	getDatasets = async (): Promise<Dataset[]> =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/datasets`, { credentials: 'include' }).then(
			this.handleResponse
		);

	searchDMFModels = async (query: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/search`, {
			method: 'POST',
			body: JSON.stringify({ query }),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	getDmfModelCard = async (params: {
		namespace: string;
		table?: string;
		model_label: string;
		revision: string;
	}) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/model_card`, {
			method: 'POST',
			body: JSON.stringify({
				namespace: params.namespace,
				table: params.table || 'model_shared',
				model_label: params.model_label,
				revision: params.revision
			}),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	getDataset = async (id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dataset/${id}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	deleteDataset = async (datasetId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dataset/${datasetId}`, {
			method: 'DELETE',
			credentials: 'include'
		}).then(this.handleResponse);

	createDataset = async (dataset: any) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dataset`, {
			method: 'POST',
			body: JSON.stringify(dataset),
			credentials: 'include',
			headers: {
				'Content-type': 'application/json; charset=UTF-8'
			}
		}).then(this.handleResponse);

	/**
	 * Upload a dataset's RAW file(s) to the backend via the tus resumable-upload
	 * protocol. Each file in the group (train + optional validation) becomes its
	 * own `tus.Upload`; the helper resolves when all uploads succeed.
	 *
	 * Modes (mirrors the single-shot endpoint):
	 *  - auto-split:       pass `trainSetPercentage`, omit `validationFile`
	 *  - custom validation: pass `validationFile`, omit `trainSetPercentage`
	 */
	uploadDatasetChunked = async (
		datasetId: string,
		opts: {
			trainFile: File;
			validationFile?: File | null;
			columnMapping?: Record<string, string> | null;
			trainSetPercentage?: number | null;
			chunkSize?: number;
			onProgress?: (percent: number) => void;
		}
	): Promise<any> => {
		const { Upload } = await import('tus-js-client');
		const endpoint = `${PUBLIC_AUTOTUNEX_API_URL}/datasets/tus`;
		// 16MB chunks: bounded request bodies for ingress limits and forward-compat
		// with a future S3 backend's 5MB minimum part size.
		const chunkSize = opts.chunkSize ?? 16 * 1024 * 1024;

		// Decide the expected file set for this dataset upload group.
		const hasValidation = !!opts.validationFile;
		const files: Array<{ file: File; role: 'source' | 'train' | 'validation' }> = hasValidation
			? [
					{ file: opts.trainFile, role: 'train' },
					{ file: opts.validationFile as File, role: 'validation' }
			  ]
			: [{ file: opts.trainFile, role: 'source' }];
		const expects = files.map((f) => f.role).join(',');

		// Aggregate progress across all files in the group (size-weighted).
		const totalBytes = files.reduce((sum, f) => sum + f.file.size, 0);
		const uploaded: Record<string, number> = {};
		const reportProgress = () => {
			if (!opts.onProgress) return;
			const done = Object.values(uploaded).reduce((a, b) => a + b, 0);
			opts.onProgress(Math.min(100, Math.round((done / Math.max(1, totalBytes)) * 100)));
		};

		const uploadOne = (file: File, role: string): Promise<void> =>
			new Promise<void>((resolve, reject) => {
				const metadata: Record<string, string> = {
					dataset_id: datasetId,
					filename: file.name,
					filetype: file.type || 'application/octet-stream',
					role,
					expects
				};
				if (opts.columnMapping) metadata.column_mapping = JSON.stringify(opts.columnMapping);
				if (!hasValidation && opts.trainSetPercentage != null) {
					metadata.train_set_percentage = String(opts.trainSetPercentage);
				}

				const upload = new Upload(file, {
					endpoint,
					chunkSize,
					retryDelays: [0, 3000, 5000, 10000, 20000],
					removeFingerprintOnSuccess: true,
					metadata,
					// tus-js-client does NOT send cookies by default. The server resolves
					// identity from the signed session cookie (get_current_user), so set
					// withCredentials on every tus XHR — otherwise finalize 401s.
					onBeforeRequest: (req) => {
						const xhr = req.getUnderlyingObject() as XMLHttpRequest;
						xhr.withCredentials = true;
					},
					onError: (error) => reject(error),
					onProgress: (bytesUploaded) => {
						uploaded[role] = bytesUploaded;
						reportProgress();
					},
					onSuccess: () => {
						uploaded[role] = file.size;
						reportProgress();
						resolve();
					}
				});

				// Reload-safe resume: continue any previously-interrupted upload of this file.
				upload
					.findPreviousUploads()
					.then((previous) => {
						if (previous.length) upload.resumeFromPreviousUpload(previous[0]);
						upload.start();
					})
					.catch(() => upload.start());
			});

		await Promise.all(files.map((f) => uploadOne(f.file, f.role)));
		opts.onProgress?.(100);
		// Finalization happens server-side on the last file's completion; the
		// post-upload refresh is driven by the caller after this resolves.
		return { status: 'uploaded' };
	};

	getUserMetadata = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/user/metadata`, { credentials: 'include' })
			.then(this.handleResponse)
			.catch((e) => {
				console.log('error', e);
			});

	startChat = async (
		messages: Array<{ role: string; content: string }>,
		context: Record<string, unknown> = {},
		thread_id?: string
	) => {
		try {
			const response = await fetch(`${PUBLIC_AUTOTUNEX_API_URL}/chat`, {
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
				const response = await fetch(`${PUBLIC_AUTOTUNEX_API_URL}/chat/stream`, {
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

	estimateUsage = async (payload: Estimation): Promise<Resources> =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/estimate_usages`, {
			method: 'POST',
			credentials: 'include',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(payload)
		}).then(this.handleResponse);

	getUsers = async (): Promise<User[]> =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/users`, { credentials: 'include' }).then(
			this.handleResponse
		);

	assumeUser = async (userId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/assume/${userId}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	unassumeUser = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/auth/unassume`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getUserData = async (userId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/user/${userId}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getModelsByUserId = async (userId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/user/${userId}/dmf/models`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getAllModels = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/dmf/all_models`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getAllTaskByJob = async (job_id: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/tasks/${job_id}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	getTask = async (taskId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/task/${taskId}`, { credentials: 'include' }).then(
			this.handleResponse
		);

	prepareDownload = async (jobId: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/job/${jobId}/prepare_download`, {
			method: 'POST',
			credentials: 'include',
			headers: { 'Content-type': 'application/json; charset=UTF-8' }
		}).then(this.handleResponse);

	/**
	 * Generate parsing strategy using LLM for raw data
	 */
	generateParsingStrategy = async (sample: string | any[], format: string, customPrompt?: string) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/datasets/parse-strategy`, {
			method: 'POST',
			body: JSON.stringify({
				sample,
				format,
				...(customPrompt ? { custom_prompt: customPrompt } : {})
			}),
			credentials: 'include',
			headers: {
				'Content-Type': 'application/json'
			}
		}).then(this.handleResponse);

	/**
	 * Validate a parsing strategy against sample data
	 */
	validateParsingStrategy = async (strategy: any, sample: string | any[]) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/datasets/validate-strategy`, {
			method: 'POST',
			body: JSON.stringify({ strategy, sample }),
			credentials: 'include',
			headers: {
				'Content-Type': 'application/json'
			}
		}).then(this.handleResponse);

	/**
	 * Generate solution strings from prompts using LLM
	 */
	generateTestSolutions = async (prompts: Array<Array<{ role: string; content: string }>>) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/generate-test-solutions`, {
			method: 'POST',
			body: JSON.stringify({ prompts }),
			credentials: 'include',
			headers: { 'Content-Type': 'application/json' }
		}).then(this.handleResponse);

	/**
	 * Validate and optionally test a reward function
	 */
	validateRewardFunction = async (
		code: string,
		functionName: string,
		testExecution: boolean = false,
		testInputs?: Record<string, any> | Record<string, any>[]
	) =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/reward-function/validate`, {
			method: 'POST',
			body: JSON.stringify({
				code,
				function_name: functionName,
				test_execution: testExecution,
				test_inputs: testInputs
			}),
			credentials: 'include',
			headers: {
				'Content-Type': 'application/json'
			}
		}).then(this.handleResponse);

	/**
	 * Get supported AutoTune dataset types
	 */
	getAutotuneDatasetTypes = async () =>
		fetch(`${PUBLIC_AUTOTUNEX_API_URL}/autotune_dataset_types`, {
			credentials: 'include'
		}).then(this.handleResponse);

	/**
	 * AI-powered column mapping suggestion using LLM
	 */
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
		if (targetDatasetType) body.target_dataset_type = targetDatasetType;
		return fetch(`${PUBLIC_AUTOTUNEX_API_URL}/datasets/suggest-mapping`, {
			method: 'POST',
			body: JSON.stringify(body),
			credentials: 'include',
			headers: {
				'Content-Type': 'application/json'
			}
		}).then(this.handleResponse);
	};

	handleResponse = async (response: Response) => {
		if (response.ok) {
			return await response.json();
		} else if (response.status === 401) {
			console.error('Authentication error');
			isAuthenticated.set(false);
			currentUser.set(null);
			window.location.reload();
		} else {
			throw response.json();
		}
	};
}
