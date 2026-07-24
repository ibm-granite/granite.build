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
		NotificationActionButton,
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
	import DmfMetadataForm from '../forms/DmfMetadataForm.svelte';
	import { Utils } from '$lib/utils';
	import { INTERVAL_DURATION, DMF_UI_URL, buildDmfUrl } from '$lib/constants';
	import { currentUser } from '$lib/store';
	import { UserAdmin, Download } from 'carbon-icons-svelte';
	import { PUBLIC_AUTOTUNEX_API_URL } from '$env/static/public';
	import Tasks from '../tables/Tasks.svelte';
	import type { Task, Trial, Tuning } from '$lib/app-types';
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

	export let tuning_id;
	export let selectedTabId = 0;

	let intervalId: number;
	let ritIntervalId: number;
	let isLoading: boolean = false;
	let showTasks: boolean = false;
	let publishing: boolean = false;
	let resultLoading: boolean = false;
	let trialComparisionMode: boolean = false;
	let selectedTrialRows: Trial[] = [];
	let tuning: Tuning;
	let rits: Task;

	let downloadPreparing = false;
	let downloadStatus: string | null = null;
	let downloadTaskId: string | null = null;
	let downloadPollingId: ReturnType<typeof setInterval> | null = null;

	let gbLogsLoadingAll = false;
	let gbLogsAllLoaded = false;
	let gbLogsDownloading = false;

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
			console.error('Error downloading GB logs:', e);
		} finally {
			gbLogsDownloading = false;
		}
	};

	let dmfMetadata = {
		label: '',
		variant: '',
		type: '',
		size: ''
	};

	const fetchModels = async () => {
		isLoading = true;
		tuning.dmf = await api.getPublishedModels();
		isLoading = false;
	};

	const handleDownloadAll = async () => {
		downloadPreparing = true;
		downloadStatus = 'PENDING';
		try {
			const result = await api.prepareDownload(tuning_id);
			downloadTaskId = result.task_id;
			downloadStatus = result.status;

			if (result.status === 'COMPLETED') {
				triggerDownload(result.task_id);
				resetDownloadState();
				return;
			}

			downloadPollingId = setInterval(async () => {
				try {
					const task = await api.getTask(downloadTaskId!);
					downloadStatus = task?.status;
					if (task?.status === 'COMPLETED') {
						clearInterval(downloadPollingId!);
						triggerDownload(downloadTaskId!);
						resetDownloadState();
					} else if (['ERROR', 'TERMINATED'].includes(task?.status)) {
						clearInterval(downloadPollingId!);
						downloadPreparing = false;
						downloadStatus = 'ERROR';
					}
				} catch (e) {
					console.error('Error polling download task:', e);
				}
			}, 5000);
		} catch (error) {
			console.error('Error preparing download:', error);
			downloadPreparing = false;
			downloadStatus = 'ERROR';
		}
	};

	const triggerDownload = (taskId: string) => {
		window.open(
			`${PUBLIC_AUTOTUNEX_API_URL}/job/${tuning_id}/download_all_assets?task_id=${taskId}`,
			'_blank'
		);
	};

	const resetDownloadState = () => {
		downloadPreparing = false;
		downloadStatus = null;
		downloadTaskId = null;
		downloadPollingId = null;
	};

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

	const setDmfMetadata = (data: Tuning | null) => {
		if (!data) return;
		let tempString = data['model'].replace('-', '/');
		let result_array = tempString.split('/');
		dmfMetadata.label = data.experiment_name;
		dmfMetadata.variant = 'instruct';
		dmfMetadata.type = result_array[1];
		dmfMetadata.size = Utils.extractParameterLength(data?.model)!;
	};

	const getTabsData = async (id: string) => {
		if (!$tunings) {
			tunings.set([]);
		}
		let tuning = $tunings.find((job) => job.id === id);

		if (!tuning) {
			tuning = await api.getJob(id, { include_logs: false });
		} else if ((tuning.github_pr_url || tuning.build_id) && !tuning.build_status) {
			// List endpoint strips build_status; fetch full detail for GB jobs so the Status tab renders.
			const detail = await api.getJob(id, { include_logs: false });
			if (detail) tuning = { ...tuning, ...detail };
		}

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
		if (tuning) {
			setDmfMetadata(tuning ? structuredClone(tuning) : null);
		}
		return tuning;
	};

	const pollRitsTask = async (tuning_id: string) => {
		try {
			rits = await api.getPushToRits(tuning_id);
		} catch (error) {
			console.error('Error fetching job logs:', error);
		} finally {
			if (!ritIntervalId && (rits?.status === 'RUNNING' || rits?.status === 'PENDING')) {
				ritIntervalId = setInterval(() => pollRitsTask(tuning_id), 60000);
			}
			if (
				ritIntervalId &&
				rits &&
				(rits?.status === 'COMPLETED' || rits?.status === 'TERMINATED')
			) {
				clearInterval(ritIntervalId);
			}
		}
	};

	const loadResultsData = async () => {
		if (resultLoading) return; // Prevent concurrent calls
		resultLoading = true;
		try {
			if (tuning.status === 'COMPLETED' && !tuning.assets) {
				tuning.assets = await api.getAssetsByJobId(tuning_id);
			}
			if (tuning?.assets && tuning.assets?.length > 0 && !tuning.dmf) {
				isLoading = true;
				tuning.dmf = await api.getPublishedModels();
				isLoading = false;
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
		if (tuning_id && tuning?.status === 'COMPLETED') {
			pollRitsTask(tuning_id);
		}
		intervalId = setInterval(async () => {
			if (!['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status)) {
				clearInterval(intervalId);
				return;
			}
			tuning = await getTabsData(tuning_id);
			if (tuning.assets && tuning?.assets?.length > 0) {
				fetchModels();
			}
		}, INTERVAL_DURATION);
	});

	onDestroy(() => {
		clearInterval(intervalId);
		clearInterval(ritIntervalId);
		if (downloadPollingId) clearInterval(downloadPollingId);
		stopLogPoll(tuning_id);
	});

	$: if (tuning?.status && !['SUBMITTED', 'PENDING', 'RUNNING'].includes(tuning?.status)) {
		clearInterval(intervalId);
	}

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
					gbLogsAllLoaded = false;
					gbLogsLoadingAll = false;
					let logs = await api.getGBLogs(tuning_id);
					tuning.gb_logs = logs;
					updateJob(tuning);
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
					<DisplayDict dict={tuning} {rits} on:deploy={(e) => pollRitsTask(e.detail)} />
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
					{#if tuning?.dmf?.detail}
						<InlineNotification
							kind="error"
							title="DMF issue:"
							hideCloseButton
							subtitle="Error occured while fetching models"
						>
							<svelte:fragment slot="actions">
								<NotificationActionButton on:click={fetchModels} bind:disabled={isLoading}>
									{#if !isLoading}
										Retry
									{:else}
										<InlineLoading />
									{/if}
								</NotificationActionButton>
							</svelte:fragment>
						</InlineNotification>
					{/if}
					{#if tuning.assets && tuning.assets?.length > 0}
						<div style="display: flex; align-items: center; justify-content: space-between;">
							<h4>Output assets</h4>
							<Button
								size="small"
								kind="tertiary"
								icon={Download}
								disabled={downloadPreparing}
								on:click={handleDownloadAll}
							>
								{#if downloadPreparing}
									<InlineLoading description="Preparing download..." />
								{:else if downloadStatus === 'ERROR'}
									Retry download
								{:else}
									Download all assets
								{/if}
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
														<a
															href={`${PUBLIC_AUTOTUNEX_API_URL}/job/${tuning_id}/result_report/${asset.filename}`}
															target="_blank"
															>{asset.filename}
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
							<Row>
								{#if !tuning?.is_published && !tuning?.dmf && tuning?.dmf?.length >= 0}
									<Column>
										{#if tuning?.dmf && !tuning.dmf
												.map((model) => model.revision)
												.includes(tuning_id)}
											{#if !publishing}
												<DmfMetadataForm bind:data={dmfMetadata} />
												<Button
													on:click={(e) => {
														publishing = true;
														api.publishModel(tuning_id, dmfMetadata).then(async () => {
															tuning.dmf = await api.getPublishedModels();
															publishing = false;
														});
													}}
													expressive
													size="xl"
												>
													Push to Data Model Factory
												</Button>
											{:else}
												<InlineLoading status="active" description="Publishing..." />
											{/if}
										{:else if !publishing}
											<InlineLoading
												status="finished"
												description={`Published on ${new Date(
													tuning.dmf?.find(
														(model) => model.revision === tuning_id
													)?.files[0]?.created
												).toLocaleString()}`}
											/>
											{#if DMF_UI_URL}
												<Button
													target="_blank"
													href={buildDmfUrl(
														`autotunex.${tuning?.experiment_name}`,
														tuning_id
													)}
													kind="secondary"
												>
													Open in DMF
												</Button>
											{/if}
											<Button
												on:click={(e) => {
													publishing = true;
													api.deleteModel(tuning_id).finally(async () => {
														tuning.dmf = await api.getPublishedModels();
														publishing = false;
													});
												}}
												kind="ghost"
											>
												Unpublish
											</Button>
										{:else}
											<InlineLoading status="active" description="Deleting..." />
										{/if}
									</Column>
								{/if}
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
						{#if !tuning?.build_status}
							<ProgressBar size="sm" helperText="Loading details..." />
						{:else if !tuning?.build_status?.build_history || tuning.build_status.build_history.length === 0}
							<div style="padding: 16px;">
								<InlineNotification
									kind="info"
									title="No status details yet"
									subtitle="Build history is not available for this job yet."
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
						{#if !tuning.gb_logs}
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
												tuning.gb_logs = logs;
												gbLogsAllLoaded = true;
												updateJob(tuning);
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
