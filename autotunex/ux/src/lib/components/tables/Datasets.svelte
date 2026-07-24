<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { DataTableSkeleton, Link } from 'carbon-components-svelte';
	import Table from '../Table.svelte';
	import CreateDatasetForm from '../forms/CreateDatasetForm.svelte';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';
	import DatasetDisplay from '../displays/DatasetDisplay.svelte';
	import { showLoader, userMetadata } from '$lib/store';
	import type { DatasetForm, ColumnMapping } from '$lib/app-types';
	import { appState, datasets, notifications } from '$lib/app';

	const api = new API();

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

	const fetchDatasets = async () => {
		try {
			if ($appState.isDatasetsLoaded) {
				return;
			}
			let datasetsData = await api.getDatasets();
			datasets.update((prev) => {
				// guard prev in case it's undefined
				const prevArr = Array.isArray(prev) ? prev : [];

				// Start with existing configs in a map by id (keeps insertion order of prev)
				const map = new Map(prevArr.map((job) => [job.id, job]));

				for (const dataset of datasetsData) {
					const oldDataset = map.get(dataset.id);
					// merge: keep UI-only fields from oldConfig, let API fields override
					map.set(dataset.id, oldDataset ? { ...oldDataset, ...dataset } : dataset);
				}
				return Array.from(map.values());
			});
			appState.update((prev) => {
				return { ...prev, isDatasetsLoaded: true };
			});
			return datasetsData;
		} catch (error) {
			console.log('fetchDatasets', error);
		}
	};

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
			// Upload the raw file(s) via tus (resumable); mapping + split happen
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
			appState.update((prev) => ({ ...prev, isDatasetsLoaded: false }));
			await fetchDatasets();
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

	onMount(async () => {
		await fetchDatasets();
	});
</script>

{#if $appState.isDatasetsLoaded}
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
		rows={$datasets.map((data) => {
			if (!data.train_records) {
				data.train_records = 0;
			}
			if (!data.validation_records) {
				data.validation_records = 0;
			}
			return data;
		})}
		on:delete={async (e) => {
			for (let id of e.detail) {
				await api.deleteDataset(id);
				let updateDatasets = $datasets.filter((item) => item.id !== id);
				datasets.set(updateDatasets);
				userMetadata.update((prev) => {
					return { ...prev, number_of_datasets: prev.number_of_datasets - 1 };
				});
			}
		}}
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
	<DataTableSkeleton />
{/if}
