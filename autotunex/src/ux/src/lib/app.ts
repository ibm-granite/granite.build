// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { get, writable } from 'svelte/store';
import type { AppState, Configuration, Dataset, Log, Notification, Tuning } from './app-types';
import { API } from './api';
import { capabilities } from './store';
import { LOG_POLL_INTERVAL } from './constants';

export const tunings = writable<Tuning[]>();
export const datasets = writable<Dataset[]>();
export const configurations = writable<Configuration[]>();
export const logsStore = writable<
	Record<string, { logs: Log[]; oldestId: number; hasMore: boolean; isComplete: boolean }>
>({});
export const trialLogsStore = writable<
	Record<string, { logs: Log[]; oldestId: number; hasMore: boolean; isComplete: boolean }>
>({});
export const datasetTypes = writable<Record<string, any>>({});
export const appState = writable<AppState>({
	isTuningsLoaded: false,
	isDatasetsLoaded: false,
	isConfigurationsLoaded: false
});

export const notifications = writable<Notification>({
	show: false,
	kind: 'info',
	title: '',
	subtitle: '',
	caption: new Date().toLocaleString(),
	timeout: 5000
});

const api = new API();
let jobsIntervalId: ReturnType<typeof setInterval> | null = null;

export const getCachedLogs = (jobId: string) => {
	return get(logsStore)[jobId] ?? null;
};

const ACTIVE_STATUSES = ['SUBMITTED', 'PENDING', 'RUNNING'];

/**
 * Fetch the first page of logs (newest first, descending).
 * For completed jobs, uses cache. For active jobs, bypasses cache.
 */
export const fetchAndCacheLogs = async (
	jobId: string,
	options?: { status?: string }
): Promise<void> => {
	if (!capabilities.logs) return; // no log backend yet
	const isActive = options?.status ? ACTIVE_STATUSES.includes(options.status) : false;

	if (!isActive) {
		const cached = getCachedLogs(jobId);
		if (cached) return;
	}

	const result = await api.getLogs(jobId, 0, 50);
	const logs = result.logs ?? [];
	const oldestId = logs.length > 0 ? logs[logs.length - 1].id ?? 0 : 0;
	logsStore.update((prev) => ({
		...prev,
		[jobId]: { logs, oldestId, hasMore: result.has_more, isComplete: false }
	}));
};

/**
 * Load the next page of older logs (scroll pagination).
 */
const loadingOlderLogs = new Set<string>();

export const loadOlderLogs = async (jobId: string): Promise<void> => {
	if (!capabilities.logs) return; // no log backend yet
	const cached = getCachedLogs(jobId);
	if (!cached || !cached.hasMore || loadingOlderLogs.has(jobId)) return;

	loadingOlderLogs.add(jobId);

	const result = await api.getLogs(jobId, cached.oldestId, 50);
	const newLogs = result.logs ?? [];
	const oldestId = newLogs.length > 0 ? newLogs[newLogs.length - 1].id ?? 0 : cached.oldestId;

	logsStore.update((prev) => {
		const entry = prev[jobId];
		if (!entry) return prev;
		return {
			...prev,
			[jobId]: {
				...entry,
				logs: [...entry.logs, ...newLogs],
				oldestId,
				hasMore: result.has_more
			}
		};
	});

	loadingOlderLogs.delete(jobId);
};

// Poll-based log refresh for active jobs
const activeLogPolls = new Map<string, ReturnType<typeof setInterval>>();

/**
 * Start polling newest logs for an active job.
 * Fetches the first page every LOG_POLL_INTERVAL ms and replaces the top of the store,
 * preserving any older pages already loaded via scroll.
 */
export const startLogPoll = async (jobId: string) => {
	if (!capabilities.logs) return; // no log backend yet
	if (activeLogPolls.has(jobId)) return;

	// Arm the interval synchronously — before the awaited initial fetch below — so an
	// overlapping startLogPoll (e.g. the tuning dialog's job-refresh poll re-invoking this
	// every INTERVAL_DURATION) is caught by the guard above and cannot leak a second interval.
	const intervalId = setInterval(async () => {
		const result = await api.getLogs(jobId, 0, 50);
		const newLogs = result.logs ?? [];
		const oldestId = newLogs.length > 0 ? newLogs[newLogs.length - 1].id ?? 0 : 0;

		logsStore.update((prev) => {
			const entry = prev[jobId];
			if (!entry) {
				return {
					...prev,
					[jobId]: { logs: newLogs, oldestId, hasMore: result.has_more, isComplete: false }
				};
			}

			// Find where the new page ends and older loaded pages begin
			const existingOlder = entry.logs.filter((log) => (log.id ?? 0) < oldestId);
			return {
				...prev,
				[jobId]: {
					...entry,
					logs: [...newLogs, ...existingOlder],
					oldestId:
						existingOlder.length > 0 ? existingOlder[existingOlder.length - 1].id ?? 0 : oldestId,
					hasMore: existingOlder.length > 0 ? entry.hasMore : result.has_more
				}
			};
		});
	}, LOG_POLL_INTERVAL);
	activeLogPolls.set(jobId, intervalId);

	// Immediate first fetch so a just-opened viewer shows logs without waiting a full interval.
	await fetchAndCacheLogs(jobId, { status: 'RUNNING' });
};

export const stopLogPoll = (jobId: string) => {
	const intervalId = activeLogPolls.get(jobId);
	if (intervalId) {
		clearInterval(intervalId);
		activeLogPolls.delete(jobId);
	}
};

// ---- Trial Logs ----

export const getCachedTrialLogs = (trialId: string) => {
	return get(trialLogsStore)[trialId] ?? null;
};

export const fetchAndCacheTrialLogs = async (
	jobId: string,
	trialId: string,
	options?: { status?: string }
): Promise<void> => {
	if (!capabilities.logs) return; // no trial-log backend yet
	const isActive = options?.status ? ACTIVE_STATUSES.includes(options.status) : false;

	if (!isActive) {
		const cached = getCachedTrialLogs(trialId);
		if (cached) return;
	}

	const result = await api.getTrialLogs(jobId, trialId, 0, 50);
	const logs = result.logs ?? [];
	const oldestId = logs.length > 0 ? logs[logs.length - 1].id ?? 0 : 0;
	trialLogsStore.update((prev) => ({
		...prev,
		[trialId]: { logs, oldestId, hasMore: result.has_more, isComplete: false }
	}));
};

const loadingOlderTrialLogs = new Set<string>();

export const loadOlderTrialLogs = async (jobId: string, trialId: string): Promise<void> => {
	if (!capabilities.logs) return; // no trial-log backend yet
	const cached = getCachedTrialLogs(trialId);
	if (!cached || !cached.hasMore || loadingOlderTrialLogs.has(trialId)) return;

	loadingOlderTrialLogs.add(trialId);

	const result = await api.getTrialLogs(jobId, trialId, cached.oldestId, 50);
	const newLogs = result.logs ?? [];
	const oldestId = newLogs.length > 0 ? newLogs[newLogs.length - 1].id ?? 0 : cached.oldestId;

	trialLogsStore.update((prev) => {
		const entry = prev[trialId];
		if (!entry) return prev;
		return {
			...prev,
			[trialId]: {
				...entry,
				logs: [...entry.logs, ...newLogs],
				oldestId,
				hasMore: result.has_more
			}
		};
	});

	loadingOlderTrialLogs.delete(trialId);
};

const activeTrialLogPolls = new Map<string, ReturnType<typeof setInterval>>();

export const startTrialLogPoll = async (jobId: string, trialId: string) => {
	if (!capabilities.logs) return; // no trial-log backend yet
	if (activeTrialLogPolls.has(trialId)) return;

	await fetchAndCacheTrialLogs(jobId, trialId, { status: 'RUNNING' });

	const intervalId = setInterval(async () => {
		const result = await api.getTrialLogs(jobId, trialId, 0, 50);
		const newLogs = result.logs ?? [];
		const oldestId = newLogs.length > 0 ? newLogs[newLogs.length - 1].id ?? 0 : 0;

		trialLogsStore.update((prev) => {
			const entry = prev[trialId];
			if (!entry) {
				return {
					...prev,
					[trialId]: { logs: newLogs, oldestId, hasMore: result.has_more, isComplete: false }
				};
			}

			const existingOlder = entry.logs.filter((log) => (log.id ?? 0) < oldestId);
			return {
				...prev,
				[trialId]: {
					...entry,
					logs: [...newLogs, ...existingOlder],
					oldestId:
						existingOlder.length > 0 ? existingOlder[existingOlder.length - 1].id ?? 0 : oldestId,
					hasMore: existingOlder.length > 0 ? entry.hasMore : result.has_more
				}
			};
		});
	}, 60000);

	activeTrialLogPolls.set(trialId, intervalId);
};

export const stopTrialLogPoll = (trialId: string) => {
	const intervalId = activeTrialLogPolls.get(trialId);
	if (intervalId) {
		clearInterval(intervalId);
		activeTrialLogPolls.delete(trialId);
	}
};

export const updateJob = (job: Tuning) => {
	tunings.update((prev) => {
		// check if job already exists
		const index = prev.findIndex((row) => row.id === job.id);

		// if not found, add new job
		if (index === -1) return [...prev, job];

		// if found → replace job cleanly
		const updated = [...prev];
		updated[index] = { ...updated[index], ...job };
		return updated;
	});
};

export const updateConfig = (config: Configuration) => {
	configurations.update((prev) => {
		// check if job already exists
		const index = prev.findIndex((row) => row.id === config.id);

		// if not found, add new job
		if (index === -1) return [...prev, config];

		// if found → replace job cleanly
		const updated = [...prev];
		updated[index] = { ...updated[index], ...config };
		return updated;
	});
};

export const updateDataset = (dataset: Dataset) => {
	datasets.update((prev) => {
		// check if job already exists
		const index = prev.findIndex((row) => row.id === dataset.id);

		// if not found, add new job
		if (index === -1) return [...prev, dataset];

		// if found → replace job cleanly
		const updated = [...prev];
		updated[index] = { ...updated[index], ...dataset };
		return updated;
	});
};

const fetchActiveJobs = async (jobIds: string[]) => {
	const results = await Promise.allSettled(
		jobIds.map((id) => api.getJob(id, { include_logs: false }))
	);
	results.forEach((result) => {
		if (result.status === 'fulfilled' && result.value) {
			updateJob(result.value);
		}
	});
};

tunings.subscribe((jobs) => {
	// if there's any active job, ensure the interval exists, else clear it
	const isActive = (jobs || []).some((job) =>
		['SUBMITTED', 'PENDING', 'RUNNING'].includes(job.status)
	);

	if (isActive) {
		// create interval only if not already running
		if (!jobsIntervalId) {
			jobsIntervalId = setInterval(() => {
				const currentJobs = get(tunings) || [];
				const activeIds = currentJobs
					.filter((j) => ['SUBMITTED', 'PENDING', 'RUNNING'].includes(j.status))
					.map((j) => j.id);

				if (activeIds.length === 0) {
					// stop when nothing active
					if (jobsIntervalId) {
						clearInterval(jobsIntervalId);
						jobsIntervalId = null;
					}
					return;
				}

				// Fetch the currently active job ids (always fresh)
				fetchActiveJobs(activeIds);
			}, 60000);
		}
	} else {
		// clear interval only if running
		if (jobsIntervalId) {
			clearInterval(jobsIntervalId);
			jobsIntervalId = null;
		}
	}
});

// tunings.subscribe((jobs) => {
// 	console.log('🚀 ~ jobs:', jobs);
// 	const isActive = jobs?.some((job) => ['SUBMITTED', 'PENDING', 'RUNNING'].includes(job.status));
// 	if (isActive) {
// 		const jobIds = jobs
// 			.filter((job) => ['SUBMITTED', 'PENDING', 'RUNNING'].includes(job.status))
// 			.map((job) => job.id);
// 		// create interval only if not already running
// 		if (!jobsIntervalId) {
// 			jobsIntervalId = setInterval(() => {
// 				fetchActiveJobs(jobIds);
// 			}, 20000);
// 		}
// 	} else {
// 		// clear interval only if running
// 		if (jobsIntervalId) {
// 			clearInterval(jobsIntervalId);
// 			jobsIntervalId = null;
// 		}
// 	}
// });
