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

	// In the script block rather than inline in the markup so the caught value can
	// be narrowed: template expressions are parsed as plain JS, so a type
	// annotation there is a syntax error.
	const submitDataset = async () => {
		if (!dataset?.train_file || !dataset?.validation_file || !dataset?.name) return;

		showLoader.set(true);
		try {
			await createDataset();
		} catch (e) {
			// A DatasetUploadError from the network/timeout/still-processing paths has
			// no `detail` at all — reading only that rendered nothing and left the
			// dialog silent. `.message` always carries a real sentence.
			const err = e as { detail?: string; message?: string } | null;
			error = err?.detail || err?.message || 'Failed to upload dataset. Please try again.';
			console.error('error occured while uploading dataset', err);
		} finally {
			showLoader.set(false);
		}
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
	on:submit={submitDataset}
	primaryButtonText="Save"
	submitBtnDisable={!dataset?.name || !dataset?.train_file || !dataset?.validation_file}
>
	{#if error}
		<InlineNotification kind="error" subtitle={error} />
	{/if}
	<CreateDatasetForm bind:dataset bind:selectedTabId bind:columnMapping />
</CreateDialog>
