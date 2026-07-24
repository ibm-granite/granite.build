<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { InlineNotification, Link } from 'carbon-components-svelte';
	import CreateDatasetForm from './forms/CreateDatasetForm.svelte';
	import CreateDialog from './CreateDialog.svelte';
	import { API } from '$lib/api';
	import { showLoader } from '$lib/store';
	import { createEventDispatcher } from 'svelte';
	import type { ColumnMapping } from '$lib/app-types';

	let dataset: any = {
		name: '',
		description: '',
		train_file: null,
		validation_file: null
	};
	let columnMapping: ColumnMapping = {};
	let selectedTabId: number;
	let api = new API();
	let openCreateDataset = false;
	let error: string;
	let createdDatasetId: string | null = null;

	// Clear the remembered (retry) dataset id whenever the create dialog is
	// closed, so a cancelled-then-reopened flow for a DIFFERENT dataset can't
	// reuse a stale id. The success path also nulls it before closing.
	$: if (!openCreateDataset) createdDatasetId = null;
	const dispatch = createEventDispatcher();

	const createDataset = async () => {
		// Only create the DB row on the first attempt; reuse the id on retry so
		// a re-submit after a failed upload does not hit 409 Conflict.
		if (!createdDatasetId) {
			const resp = await api.createDataset({
				name: dataset.name,
				description: dataset.description
			});
			if (!resp?.id) return;
			createdDatasetId = resp.id;
		}

		const datasetId = createdDatasetId!;

		if (!dataset.train_file || !dataset.validation_file) return;

		const isAutoSplit = !!(
			dataset.trainSetPercentage && dataset.train_file === dataset.validation_file
		);

		// Stream the raw file(s) to the backend in chunks; column mapping and the
		// train/validation split are applied server-side. The browser never reads
		// or re-serializes the whole file, so large datasets no longer crash it.
		await api.uploadDatasetChunked(datasetId, {
			trainFile: dataset.train_file,
			validationFile: isAutoSplit ? undefined : dataset.validation_file,
			columnMapping,
			trainSetPercentage: isAutoSplit ? dataset.trainSetPercentage : undefined
		});

		columnMapping = {};
		createdDatasetId = null;
		openCreateDataset = false;
		dispatch('create', dataset);
	};
</script>

<InlineNotification kind="info" subtitle="To run your first fine-tuning experiment please first">
	<svelte:fragment slot="subtitle">
		To run your first fine-tuning experiment please first <Link
			style="cursor: pointer"
			on:click={() => (openCreateDataset = true)}>create a dataset</Link
		>
	</svelte:fragment>
</InlineNotification>

<CreateDialog
	bind:open={openCreateDataset}
	entity="dataset"
	on:submit={async () => {
		if (dataset?.train_file && dataset?.validation_file && dataset?.name) {
			showLoader.set(true);
			try {
				await createDataset();
			} catch (e) {
				error = (await e)?.detail;
				console.error('error occured while uploading dataset', await e);
			} finally {
				showLoader.set(false);
			}
		}
	}}
	primaryButtonText="Save"
	submitBtnDisable={!dataset?.name || !dataset?.train_file || !dataset?.validation_file}
>
	{#if error}
		<InlineNotification kind="error" subtitle={error} />
	{/if}
	<CreateDatasetForm bind:dataset bind:selectedTabId bind:columnMapping />
</CreateDialog>
