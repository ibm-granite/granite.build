<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Button,
		CodeSnippet,
		Column,
		Grid,
		InlineLoading,
		InlineNotification,
		ProgressBar,
		Row,
		StructuredList,
		StructuredListBody,
		StructuredListCell,
		StructuredListHead,
		StructuredListRow,
		Tab,
		TabContent,
		Tabs
	} from 'carbon-components-svelte';
	import DisplayDict from '../tabs/DisplayDict.svelte';
	import Trials from '../tables/Trials.svelte';
	import { RadarChart } from '@carbon/charts-svelte';
	import { onDestroy, onMount } from 'svelte';
	import { API } from '$lib/api';
	import { INTERVAL_DURATION } from '$lib/constants';
	import { currentUser, capabilities } from '$lib/store';
	import { UserAdmin, Download } from 'carbon-icons-svelte';
	import Tasks from '../tables/Tasks.svelte';
	import type { Trial, Tuning } from '$lib/app-types';
	import {
		fetchAndCacheLogs,
		loadOlderLogs,
		logsStore,
		startLogPoll,
		stopLogPoll,
		tunings,
		updateJob
	} from '$lib/app';

	const api = new API();

	export let tuning_id: string;
	export let selectedTabId = 0;

	let intervalId: number;
	let showTasks: boolean = false;
	let resultLoading: boolean = false;
	let trialComparisionMode: boolean = false;
	let selectedTrialRows: Trial[] = [];
	let tuning: Tuning;

	let gbLogsLoadingAll = false;
	let gbLogsAllLoaded = false;
	let gbLogsDownloading = false;
	let gbLogsUnavailable = false;

	const sanitizeFilenamePart = (value: string | undefined | null) =>
		(value ?? '')
			.toString()
			.trim()
			.replace(/[^a-zA-Z0-9._-]+/g, '_');

	const downloadGbLogs = async () => {
		gbLogsDownloading = true;
		try {
			let logs: string[] = Array.isArray(tuning.gb_logs) ? tuning.gb_logs : [];
			if (!gbLogsAllLoaded) {
				const fetched = await api.getGBLogs(tuning_id, true);
				logs = Array.isArray(fetched) ? fetched : [];
				tuning.gb_logs = logs;
				gbLogsAllLoaded = true;
				updateJob(tuning);
			}
			const buildId = tuning.build_id || tuning_id;
			const expName = sanitizeFilenamePart(tuning.experiment_name) || 'job';
			const safeBuildId = sanitizeFilenamePart(buildId) || 'logs';
			const filename = `${expName}-${safeBuildId}.log`;

			const blob = new Blob([(logs ?? []).join('\n')], {
				type: 'text/plain;charset=utf-8'
			});
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);
		} catch (e) {
			gbLogsUnavailable = true;
			console.error('Error downloading GB logs:', e);
		} finally {
			gbLogsDownloading = false;
		}
	};

	// The backend streams the ZIP on the fly and sets Content-Disposition:
	// attachment, so a plain same-site GET navigation downloads it (the BFF
	// session cookie rides along on top-level GETs) without leaving the page or
	// buffering the archive in JS. A per-file link works the same way.
	const triggerBrowserDownload = (url: string) => {
		const link = document.createElement('a');
		link.href = url;
		link.rel = 'noopener';
		document.body.appendChild(link);
		link.click();
		link.remove();
	};

	const downloadAllAssets = () => triggerBrowserDownload(api.resultArchiveUrl(tuning_id));

	// At component level, outside reactive statements
	let trialColorRegistry = new Map();
	let nextColorIndex = 0;

	const COLOR_PALETTE = [
		'#0f62fe',
		'#24a148',
		'#da1e28',
		'#8a3ffc',
		'#ff832b',
		'#198038',
		'#002d9c',
		'#ee538b',
		'#009d9a',
		'#012749',
		'#8a3800',
		'#a56eff',
		'#005d5d',
		'#570408',
		'#fa4d56'
	];

	const getOrAssignColor = (trialId: string) => {
		if (!trialColorRegistry.has(trialId)) {
			trialColorRegistry.set(trialId, COLOR_PALETTE[nextColorIndex % COLOR_PALETTE.length]);
			nextColorIndex++;
		}
		return trialColorRegistry.get(trialId);
	};

	const toRadarData = (trials: Trial[]) => {
		if (!trials[0]?.score) return [];
		const metricNames = Object.keys(trials[0].score.metrics);
		const metrics = trials.map((t) => t.score.metrics as Record<string, number>);
		// Initialize min and max values for each metric
		const metricBounds: Record<string, number> = {};
		metricNames.forEach((metricName) => {
			metricBounds[`${metricName}_min`] = Number.MAX_VALUE;
			metricBounds[`${metricName}_max`] = Number.MIN_VALUE;
		});
		// First pass: Calculate min and max values for each metric
		metrics.forEach((m) => {
			metricNames.forEach((metricName) => {
				const value = m[metricName];
				if (value !== undefined) {
					metricBounds[`${metricName}_min`] = Math.min(metricBounds[`${metricName}_min`], value);
					metricBounds[`${metricName}_max`] = Math.max(metricBounds[`${metricName}_max`], value);
				}
			});
		});

		// Second pass: Normalize and create radar data
		const radarData: { product: string; feature: string; score: number }[] = [];
		trials.forEach((t) => {
			const m = t.score.metrics as Record<string, number>;
			metricNames.forEach((metricName) => {
				const value = m[metricName];
				if (value !== undefined) {
					const min = metricBounds[`${metricName}_min`];
					const max = metricBounds[`${metricName}_max`];

					// Avoid division by zero
					const normalizedScore = min === max ? 0 : (value - min) / (max - min);

					radarData.push({
						product: String(t.id),
						feature: metricName.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
						score: normalizedScore
					});
				}
			});
		});

		return radarData;
	};

	const radarOptions = (trials: Trial[]) => {
		const title =
			trials[0]?.score?.metric?.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()) ?? '';

		// Use registry to maintain consistent colors
		const colorScale = trials.reduce(
			(acc, trial) => {
				acc[String(trial.id)] = getOrAssignColor(trial.id);
				return acc;
			},
			{} as Record<string, string>
		);

		return {
			title: title,
			radar: {
				axes: {
					angle: 'feature',
					value: 'score'
				}
			},
			data: {
				groupMapsTo: 'product'
			},
			color: {
				scale: colorScale
			},
			toolbar: {
				numberOfIcons: 2
			}
		};
	};

	const getTabsData = async (id: string) => {
		if (!$tunings) {
			tunings.set([]);
		}
		let tuning = $tunings.find((job) => job.id === id);

		// GET /jobs returns the lean JobSummary shape (no tasks, autotune, build_id, or
		// config_snapshot), so a tuning taken from the $tunings list store is missing the
		// fields the detail tabs gate on — Trials on `autotune`, and Status/GB Logs/Tasks on
		// `build_id`/`github_pr_url` (derived from tasks). Always fetch the full JobRead from
		// GET /jobs/{id} and merge it over any list-sourced object so those tabs render.
		// Neither shape carries the trials themselves — they come from their own endpoint,
		// below, which is why this is now one detail fetch rather than two.
		const detail = await api.getJob(id, { include_logs: false });
		tuning = tuning ? { ...tuning, ...detail } : detail;

		// Logs: poll for active jobs, cached fetch for completed
		if (['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status)) {
			await startLogPoll(id);
		} else {
			fetchAndCacheLogs(id, { status: tuning?.status });
		}

		if (
			tuning?.autotune &&
			(!tuning?.trials || ['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status))
		) {
			try {
				tuning.trials = await api.getTrialsByJobId(id);
			} catch (error) {
				console.error('Error fetching trials data:', error);
				tuning.trials = [];
			}
		}
		return tuning;
	};

	const loadResultsData = async () => {
		if (resultLoading) return; // Prevent concurrent calls
		if (!capabilities.artifacts) {
			// No artifact backend yet — render the empty-results state instead of failing.
			tuning.assets = tuning.assets ?? [];
			updateJob(tuning);
			return;
		}
		resultLoading = true;
		try {
			if (tuning.status === 'COMPLETED' && !tuning.assets) {
				tuning.assets = await api.getAssetsByJobId(tuning_id);
			}
		} catch (error) {
			console.error('Results not loaded', error);
			tuning.assets = [];
		} finally {
			updateJob(tuning);
			resultLoading = false;
		}
	};

	onMount(async () => {
		tuning = await getTabsData(tuning_id);
		updateJob(tuning);
		if (!tuning.autotune && selectedTabId === 2) {
			selectedTabId = 1;
		}
		intervalId = setInterval(async () => {
			if (!['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status)) {
				clearInterval(intervalId);
				return;
			}
			tuning = await getTabsData(tuning_id);
		}, INTERVAL_DURATION);
	});

	onDestroy(() => {
		clearInterval(intervalId);
		stopLogPoll(tuning_id);
	});

	$: if (tuning?.status && !['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status)) {
		clearInterval(intervalId);
	}

	// A job is "settled" once it leaves the active set this component polls on. The reconcile
	// loop only assembles `build_status` on a terminal state, so a still-null `build_status`
	// means "not yet built" while active but "no build details at all" once settled — and
	// since polling has stopped (see the clearInterval above), a spinner would never resolve.
	// The Status tab uses this to show the empty state instead of an endless "Loading
	// details..." bar. (The GB Logs tab drives its own spinner off the in-flight fetch, so it
	// does not need this.)
	$: jobSettled = !!tuning?.status && !['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning.status);

	$: resultsTabIndex = tuning?.autotune ? 2 : 1;

	// Reactive statement to load results when Results tab is selected (handles both programmatic and user clicks)
	$: if (
		selectedTabId === resultsTabIndex &&
		tuning &&
		tuning?.status === 'COMPLETED' &&
		!tuning.assets &&
		!resultLoading
	) {
		loadResultsData();
	}
</script>

{#if tuning}
	<Tabs bind:selected={selectedTabId}>
		<Tab label="Details" />
		{#if tuning?.autotune}
			<Tab label="Trials" />
		{/if}
		<Tab label="Results" />
		{#if $currentUser?.role === 'admin' && (tuning?.github_pr_url || tuning?.build_id)}
			<Tab label="Status">
				<svelte:fragment slot="default">
					<div style="display: flex; align-items: center;">
						<UserAdmin style="margin-right: 0.5rem;" /> Status
					</div>
				</svelte:fragment>
			</Tab>
			<Tab
				label="GB Logs"
				on:click={async () => {
					if (tuning.status === 'COMPLETED' && tuning.gb_logs) {
						return;
					}
					tuning.gb_logs = undefined;
					gbLogsUnavailable = false;
					gbLogsAllLoaded = false;
					gbLogsLoadingAll = false;
					try {
						const logs = await api.getGBLogs(tuning_id);
						tuning.gb_logs = Array.isArray(logs) ? logs : [];
						updateJob(tuning);
					} catch (err) {
						// 503 = gb reader disabled in this deployment; 502 = upstream failure.
						// Either way, show a friendly note instead of hanging on the spinner.
						gbLogsUnavailable = true;
						tuning.gb_logs = [];
						console.warn('GB logs unavailable', err);
					}
				}}
			>
				<div style="display: flex; align-items: center;">
					<UserAdmin style="margin-right: 0.5rem;" /> GB Logs
				</div>
			</Tab>
			<Tab
				label="Tasks"
				on:click={() => {
					showTasks = true;
				}}
			>
				<svelte:fragment slot="default">
					<div style="display: flex; align-items: center;">
						<UserAdmin style="margin-right: 0.5rem;" /> Tasks
					</div>
				</svelte:fragment>
			</Tab>
		{/if}
		<svelte:fragment slot="content">
			<div style="height: 600px; overflow-y: scroll;">
				<TabContent>
					<DisplayDict dict={tuning} />
					<div style="margin-top: 1.5rem;">
						{#if $logsStore[tuning_id]}
							<div
								class="log-viewer"
								on:scroll={(e) => {
									const el = e.currentTarget;
									if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
										loadOlderLogs(tuning_id);
									}
								}}
							>
								{#each $logsStore[tuning_id].logs as log}
									<div class="log-line">
										{new Date(log.timestamp).toLocaleString()}
										{log.level} -- {log.filename} -- {log.message}
									</div>
								{/each}
								{#if $logsStore[tuning_id].hasMore}
									<div style="padding: 0.5rem 1rem; color: #f4f4f4;">
										<InlineLoading description="Loading older logs..." />
									</div>
								{/if}
							</div>
						{:else}
							<ProgressBar size="sm" helperText="Loading logs..." />
						{/if}
					</div>
				</TabContent>
				{#if tuning?.autotune}
					<TabContent>
						<Grid noGutter fullWidth style="padding:1rem">
							<Row>
								<Column>
									{#if tuning?.trials?.length !== 0}
										<Trials
											jobId={tuning_id}
											bind:showCompare={trialComparisionMode}
											bind:selectedRows={selectedTrialRows}
											bind:trials={tuning.trials}
										/>
									{:else}
										<InlineNotification
											kind="info"
											hideCloseButton
											title="No trial data available"
										/>
									{/if}
								</Column>
								{#if selectedTrialRows?.length > 0 && selectedTrialRows.every((trial) => trial.status === 'COMPLETED') && !trialComparisionMode}
									<Column md={3}>
										<div style="height:420px;">
											<RadarChart
												data={toRadarData(selectedTrialRows)}
												options={radarOptions(selectedTrialRows)}
											/>
										</div>
									</Column>
								{/if}
							</Row>
						</Grid>
					</TabContent>
				{/if}
				<TabContent>
					{#if tuning.assets && tuning.assets?.length > 0}
						<div style="display: flex; align-items: center; justify-content: space-between;">
							<h4>Output assets</h4>
							<Button size="small" kind="tertiary" icon={Download} on:click={downloadAllAssets}>
								Download all assets
							</Button>
						</div>
						<Grid noGutter fullWidth style="padding:1rem">
							<Row>
								<Column>
									<StructuredList condensed style="margin-bottom: 2rem">
										<StructuredListHead>
											<StructuredListRow head>
												<StructuredListCell head>File name</StructuredListCell>
												<StructuredListCell head>File size</StructuredListCell>
												<StructuredListCell head>Created on</StructuredListCell>
											</StructuredListRow>
										</StructuredListHead>
										<StructuredListBody>
											{#each tuning.assets as asset}
												<StructuredListRow>
													<StructuredListCell>
														<!-- Key on the relative path, not the basename: many
														     files share a name (e.g. adapters.safetensors) across
														     directories. The server sets Content-Disposition:
														     attachment, so this downloads rather than navigates. -->
														<a href={api.resultFileUrl(tuning_id, asset.path ?? asset.filename)}>
															{asset.filename}
														</a>
													</StructuredListCell>
													<StructuredListCell>
														{#if asset.size < 1048576}
															{(asset.size / 1024).toFixed(2)} KB
														{:else}
															{(asset.size / (1024 * 1024)).toFixed(2)}
															MB
														{/if}
													</StructuredListCell>
													<StructuredListCell>
														{new Date(asset.modified).toLocaleString()}
													</StructuredListCell>
												</StructuredListRow>
											{/each}
										</StructuredListBody>
									</StructuredList>
								</Column>
							</Row>
						</Grid>
					{:else if resultLoading}
						<ProgressBar size="sm" helperText="Loading result details..." />
					{:else}
						<div style="padding: 16px;">
							<InlineNotification kind="info" title="No results data available" hideCloseButton />
						</div>
					{/if}
				</TabContent>

				{#if $currentUser?.role === 'admin' && (tuning?.github_pr_url || tuning?.build_id)}
					<TabContent>
						{#if !tuning?.build_status && !jobSettled}
							<ProgressBar size="sm" helperText="Loading details..." />
						{:else if !tuning?.build_status?.build_history || tuning.build_status.build_history.length === 0}
							<div style="padding: 16px;">
								<InlineNotification
									kind="info"
									title="No status details yet"
									subtitle="Build history is not available for this job."
									hideCloseButton
								/>
							</div>
						{:else}
							<CodeSnippet
								type="multi"
								code={tuning?.build_status?.build_history?.map((item) => item.description).join('')}
								wrapText
								style="max-width: 100%; word-break: break-word;"
								expanded
							/>
						{/if}
					</TabContent>
					<TabContent>
						{#if gbLogsUnavailable}
							<div style="padding: 16px;">
								<InlineNotification
									kind="info"
									title="Live build logs unavailable"
									subtitle="Live build logs are not available in this deployment."
									hideCloseButton
								/>
							</div>
						{:else if !tuning.gb_logs}
							<ProgressBar size="sm" helperText="Loading details..." />
						{:else if !Array.isArray(tuning.gb_logs) || tuning.gb_logs.length === 0}
							<div style="padding: 16px;">
								<InlineNotification
									kind="info"
									title="No logs found"
									subtitle="No GB logs are available for this job yet."
									hideCloseButton
								/>
							</div>
						{:else}
							<div
								style="display: flex; justify-content: flex-end; gap: 0.5rem; margin-bottom: 0.5rem;"
							>
								{#if !gbLogsAllLoaded}
									<Button
										size="small"
										kind="tertiary"
										disabled={gbLogsLoadingAll || gbLogsDownloading}
										on:click={async () => {
											gbLogsLoadingAll = true;
											try {
												const logs = await api.getGBLogs(tuning_id, true);
												tuning.gb_logs = Array.isArray(logs) ? logs : [];
												gbLogsAllLoaded = true;
												updateJob(tuning);
											} catch (err) {
												gbLogsUnavailable = true;
												console.warn('GB logs unavailable', err);
											} finally {
												gbLogsLoadingAll = false;
											}
										}}
									>
										{#if gbLogsLoadingAll}
											<InlineLoading description="Loading all logs..." />
										{:else}
											Load all logs
										{/if}
									</Button>
								{/if}
								<Button
									size="small"
									kind="tertiary"
									icon={Download}
									disabled={gbLogsLoadingAll || gbLogsDownloading}
									on:click={downloadGbLogs}
								>
									{#if gbLogsDownloading}
										<InlineLoading description="Preparing download..." />
									{:else}
										Download log
									{/if}
								</Button>
							</div>
							<CodeSnippet
								type="multi"
								code={tuning.gb_logs.join('\n')}
								wrapText
								style="max-width: 100%; word-break: break-word;"
								expanded
							/>
						{/if}
					</TabContent>
					<TabContent>
						{#if showTasks}
							<Tasks job_id={tuning_id} />
						{/if}
					</TabContent>
				{/if}
			</div>
		</svelte:fragment>
	</Tabs>
{:else}
	<ProgressBar size="sm" helperText="Loading tuning details..." />
{/if}

<style>
	.log-viewer {
		max-height: 296px;
		overflow-y: auto;
		background-color: #161616;
		color: #f4f4f4;
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.75rem;
		line-height: 1.4;
		padding: 0.5rem 1rem;
		word-break: break-word;
		white-space: pre-wrap;
	}
	.log-line {
		padding: 1px 0;
	}
</style>
