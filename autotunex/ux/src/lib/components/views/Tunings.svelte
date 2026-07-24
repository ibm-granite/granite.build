<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import '@carbon/charts-svelte/styles.css';
	import { showLoader, userMetadata, featureFlags } from '$lib/store';
	import {
		Button,
		Column,
		DataTable,
		DataTableSkeleton,
		Grid,
		InlineLoading,
		InlineNotification,
		Link,
		NotificationActionButton,
		Pagination,
		ProgressBar,
		Row,
		Tag,
		Toolbar,
		ToolbarBatchActions,
		ToolbarContent,
		ToolbarSearch
	} from 'carbon-components-svelte';
	import DatasetNotifier from '../DatasetNotifier.svelte';
	import {
		appState,
		configurations,
		fetchAndCacheLogs,
		loadOlderLogs,
		logsStore,
		notifications,
		publishedModels,
		startLogPoll,
		stopLogPoll,
		tunings
	} from '$lib/app';
	import { Utils } from '$lib/utils';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';
	import ShowStatus from '../ShowStatus.svelte';
	import { Compare, TrashCan, View } from 'carbon-icons-svelte';
	import CompareDialog from '../CompareDialog.svelte';
	import CreateDialog from '../CreateDialog.svelte';
	import CreateTuningForm from '../forms/CreateTuningForm.svelte';
	import { ModelSource, type Configuration, type TuningForm } from '$lib/app-types';
	import ViewDialog from '../ViewDialog.svelte';

	import DeleteDialog from '../DeleteDialog.svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import ConfigDisplay from '../displays/ConfigDisplay.svelte';
	import DatasetDisplay from '../displays/DatasetDisplay.svelte';
	import TuningDisplay from '../displays/TuningDisplay.svelte';

	const api = new API();

	let pageCount: number = 1;
	let pageSize: number = 10;
	let selectedTabId: number = 0;
	let filteredRowIds: string[] = [];
	let selectedRowIds: string[] = [];
	let tuning: TuningForm | undefined;
	let selectedConfigId: string | null = null;
	let snapshotConfig: Configuration | null = null;
	let isSnapshotStale: boolean = false;
	let selectedDatasetname: string | null = null;
	let isModelsLoading: boolean = false;
	let entityName: string = '';

	let config: Configuration | null = null;
	let configClone: Configuration | null = null;

	// Flags to change state of modal
	let openNew = false;
	let openView = false;
	let openDelete = false;
	let openCompare = false;
	let loaded = false;

	let headers = [
		{ key: 'experiment_name', value: 'Experiment' },
		{
			key: 'created_at',
			value: 'Created on',
			display: (date: Date) => new Date(date).toLocaleString()
		},
		{ key: 'status', value: 'Status' },
		{ key: 'model', value: 'Model' },
		// { key: 'tuning_type', value: 'Type' },
		// { key: 'model_source', value: 'Source', width: '120px' },
		{ key: 'config_name', value: 'Configuration' },
		{ key: 'dataset', value: 'Data set' },
		{ key: 'total_time', value: 'Total time' }
	];

	$: if (!openView) {
		selectedConfigId = null;
		snapshotConfig = null;
		isSnapshotStale = false;
		selectedDatasetname = null;
		entityName = 'tuning';
		goto($page.url.pathname);
	}

	const fetchJobs = async () => {
		try {
			if ($appState.isTuningsLoaded) {
				return;
			}
			let jobs = await api.getJobs();

			tunings.update((prev) => {
				// guard prev in case it's undefined
				const prevArr = Array.isArray(prev) ? prev : [];

				// Start with existing jobs in a map by id (keeps insertion order of prev)
				const map = new Map(prevArr.map((job) => [job.id, job]));

				for (const job of jobs) {
					const oldJob = map.get(job.id);
					// merge: keep UI-only fields from oldJob, let API fields override
					map.set(job.id, oldJob ? { ...oldJob, ...job } : job);
				}
				return Array.from(map.values());
			});
			appState.update((prev) => {
				return { ...prev, isTuningsLoaded: true };
			});
			return jobs;
		} catch (error) {
			let msg = await error;
			console.log('🚀 ~ fetchJobs ~ msg:', msg);
			notifications.set({
				show: true,
				caption: new Date().toLocaleString(),
				kind: 'error',
				title: 'Error',
				subtitle: msg?.detail || 'Error Occured while fetching jobs',
				timeout: 5000
			});
		}
	};

	const fetchModels = async () => {
		try {
			if ($appState.isPublishedModelsLoaded) {
				return;
			}
			isModelsLoading = true;
			let models = await api.getPublishedModels();
			publishedModels.set(models);
			appState.update((prev) => {
				return { ...prev, isPublishedModelsLoaded: true };
			});
			return models;
		} catch (error) {
			console.error('🚀 ~ fetchModels ~ error:', await error);
		} finally {
			isModelsLoading = false;
		}
	};

	onMount(async () => {
		let jobsData = await fetchJobs();
		if (!jobsData || jobsData?.length === 0) {
			loaded = true;
			publishedModels.set([]);
			return;
		}
		await fetchModels();
		loaded = true;
	});

	const createTuning = async () => {
		try {
			showLoader.set(true);

			// Start the job -- config is already saved by "Apply Changes" in CreateTuningForm
			if (tuning?.experiment_name) {
				tuning.experiment_name = tuning.experiment_name.trim().replace(/\s+/g, '_');
			}
			await api.startJob(tuning);
			appState.update((prev) => {
				return { ...prev, isTuningsLoaded: false };
			});
			await fetchJobs();
			openNew = false;
		} catch (error) {
			console.error(error);
		} finally {
			showLoader.set(false);
			tuning = undefined;
		}
	};

	const deleteTuning = async () => {
		try {
			showLoader.set(true);
			for (let id of selectedRowIds) {
				await api.deleteJob(id);
				let updatedTuning = $tunings.filter((row) => !selectedRowIds.includes(row.id));
				tunings.set(updatedTuning);
			}
			selectedRowIds = [];
			openDelete = !openDelete;
		} catch (error) {
			console.error(await error);
		} finally {
			showLoader.set(false);
		}
	};

	$: selectedRows = $tunings?.filter(
		(row) => selectedRowIds.filter((r_id) => r_id === row.id).length > 0
	);
</script>

{#if $userMetadata && $userMetadata.number_of_datasets === 0}
	<DatasetNotifier
		on:create={() => {
			userMetadata.update((prev) => {
				return { ...prev, number_of_datasets: prev.number_of_datasets + 1 };
			});
		}}
	/>
{/if}
<Grid noGutter fullWidth>
	{#if loaded && !Array.isArray($publishedModels)}
		<InlineNotification
			kind="error"
			title="DMF issue:"
			hideCloseButton
			subtitle="Error occured while fetching models"
		>
			<svelte:fragment slot="actions">
				<NotificationActionButton on:click={fetchModels} bind:disabled={isModelsLoading}>
					{#if !isModelsLoading}
						Retry
					{:else}
						<InlineLoading />
					{/if}
				</NotificationActionButton>
			</svelte:fragment>
		</InlineNotification>
	{/if}
	<Row>
		<Column>
			{#if $appState.isTuningsLoaded && Array.isArray($publishedModels)}
				<DataTable
					zebra
					sortable
					batchSelection
					selectable
					expandable
					page={pageCount}
					{pageSize}
					{headers}
					sortDirection="descending"
					title="Tunings"
					sortKey="created_at"
					bind:selectedRowIds
					description="Shows your past tunings along with their status and performance metrics."
					rows={$tunings?.map((job) => {
						const total_time = Utils.getTimeElapsed(
							job.created_at,
							job.updated_at,
							job.status === 'RUNNING'
						);
						const is_published = $publishedModels?.map((item) => item.revision)?.includes(job.id);
						return { ...job, total_time, is_published };
					})}
					on:click:row--expand={async (e) => {
						const { id, status } = e.detail.row;
						if (!e.detail.expanded) {
							stopLogPoll(id);
							return;
						}

						if (['SUBMITTED', 'PENDING', 'RUNNING'].includes(status)) {
							await startLogPoll(id);
						} else {
							await fetchAndCacheLogs(id, { status });
						}
					}}
				>
					<Toolbar>
						<ToolbarBatchActions>
							{#if selectedRowIds.length > 1}
								<Button
									icon={Compare}
									on:click={() => {
										openCompare = !openCompare;
									}}
								>
									Compare
								</Button>
							{:else}
								<Button
									icon={View}
									on:click={() => {
										openView = true;
									}}
								>
									View
								</Button>
							{/if}
							<Button
								icon={TrashCan}
								on:click={(e) => {
									openDelete = !openDelete;
								}}
							>
								Delete
							</Button>
						</ToolbarBatchActions>
						<ToolbarContent>
							<ToolbarSearch persistent shouldFilterRows bind:filteredRowIds />
							{#if $featureFlags.quickCreateTuning}
								<Button kind="tertiary" on:click={() => (openNew = !openNew)}
									>Create New Tuning</Button
								>
							{/if}
							<Button href="/autotune/start-tuning">Start Tuning Wizard</Button>
						</ToolbarContent>
					</Toolbar>
					<svelte:fragment slot="cell" let:row let:cell>
						{#if cell.key === 'experiment_name'}
							<Link
								href="#"
								on:click={(e) => {
									selectedRowIds = [row.id];
									entityName = row.experiment_name;
									openView = !openView;
								}}>{cell.value}</Link
							>
						{:else if cell.key === 'dataset'}
							<Link
								href="#"
								on:click={() => {
									selectedDatasetname = row.dataset_id;
									selectedRowIds = [row.id];
									entityName = row.dataset;
									openView = !openView;
								}}>{cell.value}</Link
							>
						{:else if cell.key === 'model'}
							{#if cell.value?.startsWith('/')}
								<span title={cell.value}>{cell.value.split('/').slice(-2).join('/')}</span>
							{:else}
								<Link href={`https://huggingface.co/${cell.value}`} target="_blank"
									>{cell.value}</Link
								>
							{/if}
						{:else if cell.key === 'config_name'}
							<Link
								href="#"
								on:click={async () => {
									const snapshot = await api.getJobConfigSnapshot(row.id);
									if (snapshot) {
										snapshotConfig = {
											id: row.config_id,
											name: snapshot.name,
											tuner_type: snapshot.tuner_type,
											rl_tuner_type: snapshot.rl_tuner_type,
											config_data: snapshot.config_data,
											user_id: '',
											artifact_id: '',
											artifact_url: '',
											associated_jobs: [],
											created_at: new Date(),
											updated_at: new Date()
										};
										isSnapshotStale = snapshot.is_stale ?? false;
									}
									selectedConfigId = row.config_id;
									selectedRowIds = [row.id];
									entityName = row.config_name;
									openView = !openView;
								}}>{cell.value}</Link
							>
						{:else if cell.key === 'tuning_type'}
							{#if row['tuning_type'] && row['rl_tuner_type']}
								<!-- <Tag>{`Offline RL - ${row['rl_tuner_type']}`}</Tag> -->
								{`Offline RL - ${row['rl_tuner_type']}`}
							{:else if !row['tuning_type'] && row['rl_tuner_type']}
								<Tag>{`Online RL - ${row['rl_tuner_type']}`}</Tag>
							{:else}
								<!-- <Tag>{row['tuning_type']}</Tag> -->
								{row['tuning_type']}
							{/if}
						{:else if cell.key === 'status'}
							<ShowStatus status={cell.value} />
						{:else if cell.key === 'model_source'}
							<Tag
								style="margin:0"
								type={cell.value === ModelSource.DMF
									? 'blue'
									: cell.value === ModelSource.CustomPath
									  ? 'purple'
									  : 'cyan'}>{cell.value}</Tag
							>
						{:else}
							{cell.display ? cell.display(cell.value, row) : Utils.toUpperCase(cell.value)}
						{/if}
					</svelte:fragment>
					<svelte:fragment slot="expanded-row" let:row>
						{#if !$logsStore[row.id]}
							<ProgressBar size="sm" helperText="Loading logs..." />
						{:else}
							<div
								class="log-viewer"
								on:scroll={(e) => {
									const el = e.currentTarget;
									if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
										loadOlderLogs(row.id);
									}
								}}
							>
								{#each $logsStore[row.id].logs as log}
									<div class="log-line">
										{new Date(log.timestamp).toLocaleString()}
										{log.level} -- {log.filename} -- {log.message}
									</div>
								{/each}
								{#if $logsStore[row.id].hasMore}
									<div style="padding: 0.5rem 1rem;">
										<InlineLoading description="Loading older logs..." />
									</div>
								{/if}
							</div>
						{/if}
					</svelte:fragment>
				</DataTable>
				<Pagination
					bind:pageSize
					bind:page={pageCount}
					totalItems={filteredRowIds?.length}
					pageSizeInputDisabled
				/>
			{:else}
				<DataTableSkeleton />
			{/if}
		</Column>
	</Row>
</Grid>

<CreateDialog
	submitBtnDisable={!tuning?.experiment_name}
	primaryButtonText="OK"
	secondaryButtonText="Cancel"
	bind:open={openNew}
	entity="tuning"
	on:submit={() => createTuning()}
>
	<CreateTuningForm
		bind:tuning
		bind:config
		bind:configClone
		on:configSaved={(e) => {
			notifications.set({
				show: true,
				caption: new Date().toLocaleString(),
				kind: 'success',
				title: 'Configuration Saved',
				subtitle: e.detail.isNew
					? `New configuration "${e.detail.config.name}" created`
					: `Configuration "${e.detail.config.name}" updated`,
				timeout: 3000
			});
		}}
	/>
</CreateDialog>
<ViewDialog bind:open={openView} entity={entityName}>
	{#if selectedConfigId}
		<ConfigDisplay
			config_id={selectedConfigId}
			configuration={snapshotConfig}
			isStale={isSnapshotStale}
		/>
	{:else if selectedDatasetname}
		<DatasetDisplay datasetId={selectedDatasetname} />
	{:else}
		<TuningDisplay tuning_id={selectedRows[0].id} {selectedTabId} />
	{/if}
</ViewDialog>
<DeleteDialog
	entity={entityName}
	bind:open={openDelete}
	primaryButtonDisabled={selectedRows?.some((row) => row?.is_published && !row?.github_pr_url)}
	on:submit={deleteTuning}
>
	<slot name="delete" {selectedRows}>
		<p>This is a permanent action and cannot be undone.</p>
	</slot>
</DeleteDialog>

<CompareDialog entities="tunings" bind:open={openCompare} bind:rows={selectedRows} />

<style>
	.log-viewer {
		max-height: 300px;
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
