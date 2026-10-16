<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import '@carbon/charts-svelte/styles.css';
	import { showLoader, userMetadata, featureFlags, capabilities } from '$lib/store';
	import {
		Button,
		Column,
		DataTable,
		DataTableSkeleton,
		Grid,
		InlineLoading,
		Link,
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
		fetchAndCacheLogs,
		loadOlderLogs,
		logsStore,
		notifications,
		startLogPoll,
		stopLogPoll
	} from '$lib/app';
	import { Utils } from '$lib/utils';
	import { API } from '$lib/api';
	import ShowStatus from '../ShowStatus.svelte';
	import { Compare, TrashCan, View } from 'carbon-icons-svelte';
	import CompareDialog from '../CompareDialog.svelte';
	import CreateDialog from '../CreateDialog.svelte';
	import CreateTuningForm from '../forms/CreateTuningForm.svelte';
	import { ModelSource, type Configuration, type Tuning, type TuningForm } from '$lib/app-types';
	import ViewDialog from '../ViewDialog.svelte';

	import DeleteDialog from '../DeleteDialog.svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import ConfigDisplay from '../displays/ConfigDisplay.svelte';
	import DatasetDisplay from '../displays/DatasetDisplay.svelte';
	import TuningDisplay from '../displays/TuningDisplay.svelte';

	const api = new API();

	let rows: Tuning[] = [];
	let total = 0;
	// `page` is already imported from `$app/stores` (used as `$page` below), so the
	// pagination cursor keeps its pre-existing name to avoid a redeclaration.
	let pageCount: number = 1; // 1-based, drives Pagination
	let pageSize: number = 10;
	let q = '';
	let loaded = false;
	// True while a page/page-size change is refetching — drives the table skeleton
	// (see fetchPage). Distinct from `loaded`, which only gates the very first load.
	let fetching = false;
	let selectedTabId: number = 0;
	let selectedRowIds: string[] = [];
	let tuning: TuningForm | undefined;
	let selectedConfigId: string | null = null;
	let snapshotConfig: Configuration | null = null;
	let isSnapshotStale: boolean = false;
	let selectedDatasetname: string | null = null;
	let entityName: string = '';

	let config: Configuration | null = null;
	let configClone: Configuration | null = null;

	// Flags to change state of modal
	let openNew = false;
	let openView = false;
	let openDelete = false;
	let openCompare = false;

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

	let searchDebounce: ReturnType<typeof setTimeout> | undefined;
	function onSearchInput(value: string) {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => {
			q = value;
			pageCount = 1;
			// Do not call fetchPage() here — the (pageCount, pageSize, q) key change
			// below drives the single refetch.
		}, 300);
	}

	// Named handler (not an inline `as` cast in the markup) so svelte-check doesn't
	// trip on the type assertion — mirrors Trials.svelte's `handleTrialScroll`.
	function handleSearchInput(e: Event) {
		onSearchInput((e.target as HTMLInputElement)?.value ?? '');
	}

	// Carbon's clear button (✕) and Escape dispatch `clear`, not `input`, so
	// onSearchInput never sees the reset — handle it explicitly, or the table stays
	// filtered instead of returning to its original state. Cancel any pending
	// debounced input so a trailing keystroke can't re-apply the cleared term; the
	// (pageCount, pageSize, q) key change below drives the single refetch.
	function onSearchClear() {
		clearTimeout(searchDebounce);
		q = '';
		pageCount = 1;
	}

	// `showSkeleton` swaps the table for a loading skeleton while this fetch runs.
	// Callers pass it for page navigation only; create/delete refetches leave it
	// false (they already show the global loader), and search leaves it false so the
	// toolbar search input stays mounted and keeps focus.
	const fetchPage = async (showSkeleton = false) => {
		if (showSkeleton) fetching = true;
		try {
			const res = await api.getJobsPage({
				limit: pageSize,
				offset: (pageCount - 1) * pageSize,
				q
			});
			rows = res.items;
			total = res.total;
		} catch (error) {
			let msg = await error;
			notifications.set({
				show: true,
				caption: new Date().toLocaleString(),
				kind: 'error',
				title: 'Error',
				subtitle: msg?.detail || 'Error Occured while fetching jobs',
				timeout: 5000
			});
		} finally {
			loaded = true;
			fetching = false;
		}
	};

	// Fetch exactly once per distinct (pageCount, pageSize, q) — `loaded` must not
	// be part of the key: fetchPage itself flips it false -> true, and referencing
	// it here would re-trigger this block and double-fetch.
	let prevKey: string | null = null;
	$: {
		const key = `${pageCount} ${pageSize} ${q}`;
		if (key !== prevKey) {
			// Show the skeleton only for a genuine page turn: the page/page-size
			// portion changed while the search term did not (see fetchPage). A search
			// always changes `q` — and also resets the page to 1 — so skeletoning then
			// would unmount the toolbar search input mid-type and drop focus. Both
			// parts are read back off `prevKey` (format "<pageCount> <pageSize> <q>",
			// and `q` may itself contain spaces) so this stays correct through the
			// manual prevKey pre-syncs in createTuning/deleteTuning.
			const [prevPage, prevSize, ...prevQParts] = (prevKey ?? '').split(' ');
			const pageChanged =
				prevKey !== null &&
				prevQParts.join(' ') === q &&
				`${prevPage} ${prevSize}` !== `${pageCount} ${pageSize}`;
			// `rows` is about to be replaced by a different page, so a selection
			// made against the old page is stale — clear it here rather than in
			// onSearchInput/Pagination separately, so every path that changes the
			// key (search, page, pageSize) is covered by one guard. Skip on the
			// very first run (prevKey still null): that's the initial load, not a
			// real change, and there is nothing selected yet to clear.
			if (prevKey !== null) {
				selectedRowIds = [];
			}
			prevKey = key;
			fetchPage(pageChanged);
		}
	}

	const createTuning = async () => {
		try {
			showLoader.set(true);

			// Start the job -- config is already saved by "Apply Changes" in CreateTuningForm.
			// tuning is only undefined before the form is populated, which the submit
			// button already guards against (see submitBtnDisable below).
			if (!tuning) {
				return;
			}
			if (tuning.experiment_name) {
				tuning.experiment_name = tuning.experiment_name.trim().replace(/\s+/g, '_');
			}
			await api.startJob(tuning);
			// Follows the same cache-invalidation convention as isConfigurationsLoaded/
			// isDatasetsLoaded (see Configurations.svelte/Datasets.svelte), though unlike
			// those two, nothing currently reads isTuningsLoaded back — ChatBox.svelte
			// sets it true after a chat-triggered job start. Kept for that symmetry and
			// for whatever reader is added next; not a dead write to remove.
			appState.update((prev) => {
				return { ...prev, isTuningsLoaded: false };
			});
			// Reset to page 1 and pre-sync prevKey (synchronously, before the explicit
			// fetchPage below) so the reactive pageCount/pageSize/q watcher doesn't also
			// fire a second fetch once its flush runs.
			pageCount = 1;
			prevKey = `${pageCount} ${pageSize} ${q}`;
			await fetchPage();
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
			}
			selectedRowIds = [];
			openDelete = !openDelete;
			// Follows the same cache-invalidation convention as isConfigurationsLoaded/
			// isDatasetsLoaded — see the comment on the same call in createTuning above.
			// Not a dead write.
			appState.update((prev) => ({ ...prev, isTuningsLoaded: false }));
			await fetchPage();
			// Deleting the last row(s) on a page beyond the first can strand the user on
			// an empty page — step back one page and refetch rather than leave them there.
			if (rows.length === 0 && pageCount > 1) {
				pageCount -= 1;
				prevKey = `${pageCount} ${pageSize} ${q}`;
				await fetchPage();
			}
		} catch (error) {
			console.error(await error);
		} finally {
			showLoader.set(false);
		}
	};

	$: selectedRows = rows?.filter(
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
	<Row>
		<Column>
			{#if !loaded || fetching}
				<DataTableSkeleton {headers} rows={pageSize} zebra />
			{:else}
				<DataTable
					zebra
					sortable
					batchSelection
					selectable
					expandable
					{headers}
					sortDirection="descending"
					title="Tunings"
					sortKey="created_at"
					bind:selectedRowIds
					description="Shows your past tunings along with their status and performance metrics."
					rows={rows.map((job) => {
						// Total time = job start -> end. A running job ticks live to now;
						// a finished job uses finished_at (latest gb_tasks.updated_at). Jobs
						// with no build task (e.g. local-backend) have no end -> show a dash.
						// NOT job.updated_at: it is ON UPDATE CURRENT_TIMESTAMP, so any write
						// bumps it and inflates the duration.
						const total_time =
							job.status === 'RUNNING'
								? Utils.getTimeElapsed(job.created_at, new Date(), true)
								: job.finished_at
								  ? Utils.getTimeElapsed(job.created_at, job.finished_at)
								  : '—';
						return { ...job, total_time };
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
								disabled={!capabilities.jobDelete}
								on:click={(e) => {
									openDelete = !openDelete;
								}}
							>
								Delete
							</Button>
						</ToolbarBatchActions>
						<ToolbarContent>
							<ToolbarSearch
								persistent
								value={q}
								on:input={handleSearchInput}
								on:clear={onSearchClear}
							/>
							{#if $featureFlags.quickCreateTuning}
								<Button
									kind="tertiary"
									disabled={!capabilities.jobSubmit}
									on:click={() => (openNew = !openNew)}>Create New Tuning</Button
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
							<Tag style="margin:0" type={cell.value === ModelSource.CustomPath ? 'purple' : 'cyan'}
								>{cell.value}</Tag
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
			{/if}
			{#if loaded}
				<Pagination bind:pageSize bind:page={pageCount} totalItems={total} pageSizeInputDisabled />
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
<DeleteDialog entity={entityName} bind:open={openDelete} on:submit={deleteTuning}>
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
