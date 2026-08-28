<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { API } from '$lib/api';
	import { Utils } from '$lib/utils';
	import {
		ProgressBar,
		Tabs,
		Tab,
		TabContent,
		InlineNotification,
		Tag
	} from 'carbon-components-svelte';
	import Table from '../Table.svelte';
	import type { Dataset, DatasetFormatType } from '$lib/app-types';
	import { datasets, updateDataset } from '$lib/app';

	const api = new API();

	export let datasetId: string;

	let isLoading = false;
	let dataset: Dataset;

	let detectedFormat: DatasetFormatType = 'unknown';
	let trainPreviewHeaders: { key: string; value: string }[] = [];
	let trainPreviewRows: { id: string; [key: string]: any }[] = [];
	let valPreviewHeaders: { key: string; value: string }[] = [];
	let valPreviewRows: { id: string; [key: string]: any }[] = [];

	const formatLabels: Record<DatasetFormatType, string> = {
		preference_pairs: 'Preference Pairs (DPO)',
		kto_format: 'KTO Format',
		standard_pairs: 'Standard Pairs (SFT)',
		prompt_only: 'Prompt-Only (Online RL)',
		unknown: 'Unknown Format'
	};

	const formatTagKinds = {
		preference_pairs: 'blue',
		kto_format: 'purple',
		standard_pairs: 'teal',
		prompt_only: 'cyan',
		unknown: 'warm-gray'
	} as const;

	function buildPreview(data: any[]): {
		headers: { key: string; value: string }[];
		rows: { id: string; [key: string]: any }[];
	} {
		if (!data || data.length === 0) return { headers: [], rows: [] };

		const metadata = Utils.extractColumnMetadata(data);
		console.log('🚀 ~ buildPreview ~ metadata:', metadata);
		const columns = metadata.map((col) => col.name);
		const headers = columns.map((col) => ({
			key: col,
			value: Utils.toUpperCase(col) || col
		}));
		const rows = data.map((row, i) => {
			const processed: { id: string; [key: string]: any } = { id: String(i) };
			for (const col of columns) {
				const val = row[col];
				if (val === null || val === undefined) {
					processed[col] = '';
				} else if (typeof val === 'string') {
					processed[col] = val.length > 120 ? val.substring(0, 120) + '...' : val;
				} else {
					const str = JSON.stringify(val);
					processed[col] = str.length > 120 ? str.substring(0, 120) + '...' : str;
				}
			}
			return processed;
		});
		return { headers, rows };
	}

	const fetchDataset = async () => {
		try {
			isLoading = true;
			if (!$datasets) {
				datasets.set([]);
			}
			let data = $datasets.find((dataset) => dataset.id === datasetId)!;
			if ((data && !data.train_data && !data.validation_data) || !data) {
				// Request the preview explicitly: the backend only returns rows when
				// `?preview=true` (and status='ready'). mapDataset flattens them onto
				// train_data/validation_data for the render gate below.
				data = await api.getDataset(datasetId, true);
				if (data) {
					updateDataset(data);
				}
			}
			dataset = data;
		} catch (e) {
			console.log('error: ', await e);
		} finally {
			isLoading = false;
		}
	};

	$: if (datasetId) {
		fetchDataset();
	}

	$: if (dataset?.train_data && dataset.train_data.length > 0) {
		const result = buildPreview(dataset.train_data);
		trainPreviewHeaders = result.headers;
		trainPreviewRows = result.rows;
		const columns = Object.keys(dataset.train_data[0] || {});
		detectedFormat = Utils.detectDatasetFormat(columns);
	}

	$: if (dataset?.validation_data && dataset.validation_data.length > 0) {
		const result = buildPreview(dataset.validation_data);
		valPreviewHeaders = result.headers;
		valPreviewRows = result.rows;
	}
</script>

{#if !isLoading && dataset?.train_data && dataset?.validation_data}
	<Tabs>
		<Tab label="Train" />
		<Tab label="Validation" />
		<svelte:fragment slot="content">
			<TabContent style="min-height:42rem; max-height: 42rem; overflow-x: scroll;">
				{#if detectedFormat !== 'unknown'}
					<div style="margin-bottom: 0.5rem;">
						<Tag type={formatTagKinds[detectedFormat]}>{formatLabels[detectedFormat]}</Tag>
					</div>
				{/if}
				<Table
					rows={trainPreviewRows}
					headers={trainPreviewHeaders}
					size="short"
					batchSelection={false}
					selectable={false}
					expandable={false}
					showActionButton={false}
				/>
			</TabContent>
			<TabContent style="min-height:42rem; max-height: 42rem; overflow-x: scroll;">
				{#if detectedFormat !== 'unknown'}
					<div style="margin-bottom: 0.5rem;">
						<Tag type={formatTagKinds[detectedFormat]}>{formatLabels[detectedFormat]}</Tag>
					</div>
				{/if}
				<Table
					rows={valPreviewRows}
					headers={valPreviewHeaders}
					size="short"
					batchSelection={false}
					selectable={false}
					expandable={false}
					showActionButton={false}
				/>
			</TabContent>
		</svelte:fragment>
	</Tabs>
{:else if isLoading}
	<ProgressBar size="sm" helperText="Loading dataset details..." />
{:else}
	<InlineNotification
		title="Error"
		kind="error"
		subtitle="Unable to load dataset"
		hideCloseButton
		style="width: 100%"
	/>
{/if}
