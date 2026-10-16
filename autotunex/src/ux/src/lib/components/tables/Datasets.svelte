<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { DataTableSkeleton, Link } from 'carbon-components-svelte';
	import Table from '../Table.svelte';
	import CreateDatasetForm from '../forms/CreateDatasetForm.svelte';
	import { API } from '$lib/api';
	import DatasetDisplay from '../displays/DatasetDisplay.svelte';
	import { showLoader, userMetadata, currentUser } from '$lib/store';
	import type { Dataset, DatasetForm, ColumnMapping } from '$lib/app-types';
	import { appState, notifications } from '$lib/app';
	import { Utils } from '$lib/utils';

	const api = new API();

	// The seed/system account that owns the datasets shipped out of the box — the
	// same id the backend guards on (`core/constants.py`, `services/scoping.py`),
	// and the same literal the three config-side components already pin.
	const SYSTEM_USER_ID = '00000000-0000-0000-0000-000000000001';

	let dataset: DatasetForm;
	let columnMapping: ColumnMapping = {};
	let selectedId: string[];
	let selectedTabId: number;
	let uploadProgress: number = 0;
	let openView: boolean = false;
	let isUploading: boolean = false;
	let openCreateDataset: boolean = false;
	let createdDatasetId: string | null = null;

	// Clear the remembered (retry) dataset id whenever the create dialog is
	// closed, so a cancelled-then-reopened flow for a DIFFERENT dataset can't
	// reuse a stale id. The success path also nulls it before closing.
	$: if (!openCreateDataset) createdDatasetId = null;

	// Per-id try/catch for the same reasons as the configurations table: one rejection
	// (a 403 on shared starter content, a 409 from a dataset a tuning still references)
	// must not abort the rest of the batch, the counter must only move for a delete
	// that happened, and the failure has to be surfaced — `Table` removes the rows
	// optimistically on submit, so a swallowed error reads as success until the
	// refetch below puts them back.
	async function handleDelete(ids: string[]) {
		const failures: string[] = [];
		for (const id of ids) {
			try {
				await api.deleteDataset(id);
				userMetadata.update((prev) => ({
					...prev,
					number_of_datasets: prev.number_of_datasets - 1
				}));
			} catch (err) {
				failures.push(Utils.problemDetail(err, 'Could not delete dataset'));
			}
		}
		if (failures.length > 0) {
			notifications.set({
				show: true,
				kind: 'error',
				title:
					failures.length === ids.length
						? 'Delete failed'
						: `Deleted ${ids.length - failures.length} of ${ids.length}`,
				subtitle: failures[0],
				timeout: 6000
			});
		}
		// Deliberately invalidates the tuning-wizard/form cache — see the comment on
		// the same call in createDataset above. Not a dead write.
		appState.update((prev) => ({ ...prev, isDatasetsLoaded: false }));
		await fetchPage();
		// Deleting the last row(s) on a page beyond the first can strand the user on
		// an empty page — step back one page and refetch rather than leave them there.
		if (rows.length === 0 && page > 1) {
			page -= 1;
			prevKey = `${page} ${pageSize} ${q}`;
			await fetchPage();
		}
	}

	let datasetHeaders = [
		{ key: 'name', value: 'Name' },
		{ key: 'train_records', value: 'Training samples' },
		{ key: 'validation_records', value: 'Validation samples' },
		{
			key: 'created_at',
			value: 'Created on',
			display: (date: Date) => new Date(date).toLocaleString()
		}
	];

	let rows: Dataset[] = [];
	let total = 0;
	let page = 1;
	let pageSize = 10;
	let q = '';
	let loaded = false;
	// True while a page/page-size change is refetching — drives the table skeleton
	// (see fetchPage). Distinct from `loaded`, which only gates the very first load.
	let fetching = false;

	// `showSkeleton` swaps the table for a loading skeleton while this fetch runs.
	// Callers pass it for page navigation only; create/delete refetches leave it
	// false (they already show the global loader), and search leaves it false so the
	// toolbar search input stays mounted and keeps focus.
	const fetchPage = async (showSkeleton = false) => {
		if (showSkeleton) fetching = true;
		try {
			const res = await api.getDatasetsPage({ limit: pageSize, offset: (page - 1) * pageSize, q });
			// Preserve the existing zero-fill for record counts.
			rows = res.items.map((data) => ({
				...data,
				train_records: data.train_records ?? 0,
				validation_records: data.validation_records ?? 0
			}));
			total = res.total;
		} catch (error) {
			notifications.set({
				show: true,
				kind: 'error',
				title: 'Error',
				subtitle: 'Could not load datasets',
				timeout: 5000
			});
		} finally {
			loaded = true;
			fetching = false;
		}
	};

	function onSearch(value: string) {
		q = value;
		page = 1;
	}

	// Fetch exactly once per distinct (page, pageSize, q) — `loaded` must not be
	// part of the key: fetchPage itself flips it false -> true, and referencing it
	// here would re-trigger this block and double-fetch.
	let prevKey: string | null = null;
	$: {
		const key = `${page} ${pageSize} ${q}`;
		if (key !== prevKey) {
			// Show the skeleton only for a genuine page turn: the page/page-size
			// portion changed while the search term did not (see fetchPage). A search
			// always changes `q` — and also resets the page to 1 — so skeletoning then
			// would unmount the toolbar search input mid-type and drop focus. Both
			// parts are read back off `prevKey` (format "<page> <pageSize> <q>", and
			// `q` may itself contain spaces) so this stays correct through the manual
			// prevKey pre-syncs in the create/delete handlers.
			const [prevPage, prevSize, ...prevQParts] = (prevKey ?? '').split(' ');
			const pageChanged =
				prevKey !== null &&
				prevQParts.join(' ') === q &&
				`${prevPage} ${prevSize}` !== `${page} ${pageSize}`;
			prevKey = key;
			fetchPage(pageChanged);
		}
	}

	const createDataset = async () => {
		// Only create the DB row on the first attempt; reuse the id on retry so
		// a re-submit after a failed upload does not hit 409 Conflict.
		if (!createdDatasetId) {
			const resp = await api.createDataset({
				name: dataset.name,
				description: dataset.description
			});
			if (!resp?.id) {
				console.error(resp);
				return;
			}
			createdDatasetId = resp.id;
		}

		const datasetId = createdDatasetId!;

		if (!dataset.train_file || !dataset.validation_file) {
			console.error('Both train and validation files are required');
			return;
		}
		isUploading = true;
		uploadProgress = 0;

		// Auto-split mode is signalled by the same file used for train and validation.
		const isAutoSplit = !!(
			dataset.trainSetPercentage && dataset.train_file === dataset.validation_file
		);

		try {
			// Upload the raw file(s) in one multipart request; mapping + split happen
			// server-side, so the browser never loads the whole dataset into memory.
			await api.uploadDatasetChunked(datasetId, {
				trainFile: dataset.train_file,
				validationFile: isAutoSplit ? undefined : dataset.validation_file,
				columnMapping,
				trainSetPercentage: isAutoSplit ? dataset.trainSetPercentage : undefined,
				onProgress: (percent) => {
					uploadProgress = percent;
				}
			});

			isUploading = false;
			uploadProgress = 100;
			// Deliberately invalidates the tuning-wizard/form cache — CreateTuningForm,
			// StartTuningWizard and Step1DatasetUpload check this flag on mount and only
			// refetch $datasets when it is false. Not a dead write.
			appState.update((prev) => ({ ...prev, isDatasetsLoaded: false }));
			// Reset to page 1 and pre-sync prevKey (synchronously, before the explicit
			// fetchPage below) so the reactive page/pageSize/q watcher doesn't also fire
			// a second fetch once its flush runs.
			page = 1;
			prevKey = `${page} ${pageSize} ${q}`;
			await fetchPage();
			dataset = {
				name: '',
				description: '',
				train_file: null,
				validation_file: null
			};
			columnMapping = {};
			createdDatasetId = null;
			userMetadata.update((prev) => {
				return { ...prev, number_of_datasets: prev.number_of_datasets + 1 };
			});
			openCreateDataset = false;
			showLoader.set(false);
		} catch (err: any) {
			isUploading = false;
			showLoader.set(false);
			notifications.set({
				show: true,
				kind: 'error',
				title: 'Dataset upload failed',
				subtitle: err?.message || 'Network error occurred during upload.',
				timeout: 5000
			});
		}
	};
	// A system-owned dataset is shared starter content: deleting one takes it from
	// every account at once, so the backend refuses it with a 403 for every caller
	// (`services/scoping.is_delete_protected`). Mirrored here — reason and exemption
	// both — exactly as in `Configurations.svelte`; see the comment there.
	$: isCuratingSystemContent =
		!!$currentUser?.impersonator && $currentUser?.user_id === SYSTEM_USER_ID;
	$: selectedSystemDatasets = (rows ?? []).filter(
		(row) => selectedId?.includes(row.id) && row.user_id === SYSTEM_USER_ID
	);
	$: deleteBlockedMessage =
		selectedSystemDatasets.length > 0 && !isCuratingSystemContent
			? `${selectedSystemDatasets.map((row) => `"${row.name}"`).join(', ')} ${
					selectedSystemDatasets.length === 1 ? 'is' : 'are'
			  } provided with AutoTuneX and cannot be deleted.`
			: null;
</script>

{#if loaded}
	<Table
		title="Data sets"
		entity="data set"
		entities="data sets"
		description="Shows your datasets."
		actionButtonText="Create New Dataset"
		headers={datasetHeaders}
		expandable={false}
		primaryButtonText="Save"
		submitBtnDisable={!dataset?.name || !dataset?.train_file || !dataset?.validation_file}
		bind:selectedRowIds={selectedId}
		bind:openView
		bind:openNew={openCreateDataset}
		{rows}
		serverSide
		loading={fetching}
		{total}
		bind:page
		bind:pageSize
		on:search={(e) => onSearch(e.detail)}
		{deleteBlockedMessage}
		on:delete={(e) => handleDelete(e.detail)}
		on:new={async () => {
			if (dataset?.train_file && dataset?.validation_file && dataset?.name) {
				showLoader.set(true);
				try {
					await createDataset();
				} catch (e) {
					showLoader.set(false);
					const body = await Promise.resolve(e).catch(() => null);
					const subtitle =
						(body && typeof body === 'object' && 'detail' in body && String(body.detail)) ||
						'Could not create dataset';
					notifications.set({
						show: true,
						kind: 'error',
						title: 'Create dataset failed',
						subtitle,
						timeout: 5000
					});
				}
			}
		}}
	>
		<svelte:fragment slot="cell" let:cell let:row>
			{#if cell.key === 'name'}
				<Link
					on:click={(e) => {
						selectedId = [row.id];
						openView = true;
					}}
					href="#">{cell.value}</Link
				>
			{:else if typeof cell.value === 'number'}
				<div style="text-align: center;">
					{cell.value.toLocaleString('en-US', {
						notation: 'compact',
						maximumFractionDigits: 2
					})}
				</div>
			{:else}
				{cell.display ? cell.display(cell.value, row) : cell.value}
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="create">
			<CreateDatasetForm
				bind:dataset
				bind:selectedTabId
				bind:isUploading
				bind:uploadProgress
				bind:columnMapping
			/>
		</svelte:fragment>
		<svelte:fragment slot="view" let:selectedRows>
			<DatasetDisplay datasetId={selectedRows[0].id} />
		</svelte:fragment>
		<svelte:fragment slot="delete" let:selectedRows>
			{#if selectedRows.some((row) => row.associated_jobs && row.associated_jobs.length > 0)}
				<p>The selected dataset has associated jobs. Please delete the jobs before proceeding.</p>
				<div style="padding-top: 1rem;">
					{#each selectedRows.filter((row) => row?.associated_jobs?.length > 0) as row}
						{#each row.associated_jobs || [] as associatedJob}
							<p>{associatedJob?.experiment_name}</p>
						{/each}
					{/each}
				</div>
			{:else}
				<p>This is a permanent action and cannot be undone.</p>
			{/if}
		</svelte:fragment>
	</Table>
{:else}
	<DataTableSkeleton headers={datasetHeaders} rows={pageSize} zebra />
{/if}
