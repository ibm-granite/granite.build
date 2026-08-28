<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Utils } from '$lib/utils';
	import {
		Form,
		Grid,
		Row,
		Column,
		TextInput,
		FileUploaderDropContainer,
		FileUploaderButton,
		FileUploaderItem,
		ProgressBar,
		Toggle,
		Tag,
		Dropdown,
		Select,
		SelectItem,
		InlineLoading,
		InlineNotification,
		Tile,
		Tabs,
		Tab,
		TabContent,
		Tooltip
	} from 'carbon-components-svelte';
	import Table from '../Table.svelte';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';
	import { datasetTypes } from '$lib/app';
	import { showLoader } from '$lib/store';
	import type {
		DatasetForm,
		DatasetType,
		ColumnMetadata,
		DatasetFormatType,
		ParsedDataRow,
		ColumnMapping,
		AiMappingSuggestion
	} from '$lib/app-types';

	export let isUploading = false;
	export let uploadProgress = 0;
	export let columnMapping: ColumnMapping = {};

	type ProcessedResult = {
		train: DatasetType[];
		test: DatasetType[];
		validation: DatasetType[];
	};

	// Kept for binding compatibility with existing parents; unused internally.
	export let selectedTabId = 0;
	export let dataset: DatasetForm & { trainSetPercentage?: number } = {
		name: '',
		description: '',
		train_file: null,
		validation_file: null,
		trainSetPercentage: 80
	};

	const api = new API();

	let uploadedFile: File | null = null;
	let uploadedValidationFile: File | null = null;
	let allData: DatasetType[] = [];
	let totalLines = 0;
	const trainSetPercentage = 80;
	let trainCount = 0;
	let validationCount = 0;
	let isSplitEnabled = true;
	let isProcessingFile = false;
	let processingError: string | null = null;
	const PREVIEW_LIMIT = 1000;

	// A Parquet file over the client preview cap is parsed for its column names
	// only, so `allData` is a single blank placeholder row (see
	// Utils.isPreviewDeferred): the column mapping is real, the preview cells and
	// the line-count estimate derived from them are not.
	$: isPreviewDeferred = Utils.isPreviewDeferred(uploadedFile);

	let columnMetadata: ColumnMetadata[] = [];
	let detectedFormat: DatasetFormatType = 'unknown';
	let previewHeaders: { key: string; value: string }[] = [];
	let previewRows: { id: string; [key: string]: any }[] = [];
	let valPreviewHeaders: { key: string; value: string }[] = [];
	let valPreviewRows: { id: string; [key: string]: any }[] = [];
	let activePreviewTab = 0;

	// Column mapping state
	let detectedAlgorithm: string = 'lora';
	let userColumns: string[] = [];
	let isAiSuggesting = false;
	let aiSuggestion: AiMappingSuggestion | null = null;
	let aiColumnConfidences: Record<string, number> = {};
	let aiSuggestedFields: Set<string> = new Set();
	let showAiReasoning = false;
	let showColumnMapping = false;

	// Reactive: column info from backend dataset types
	$: allColumns =
		Object.keys($datasetTypes).length > 0
			? Utils.getColumnsFromTypes(detectedAlgorithm, $datasetTypes)
			: [];
	$: requiredColumns =
		Object.keys($datasetTypes).length > 0
			? Utils.getRequiredColumnsFromTypes(detectedAlgorithm, $datasetTypes)
			: Utils.getRequiredColumns(detectedAlgorithm);
	$: allColumnNames = allColumns.map((c: { name: string }) => c.name);

	const datasetTypeItems = [
		{ id: 'lora', text: 'SFT (Standard Pairs)' },
		{ id: 'dpo', text: 'DPO (Preference Pairs)' },
		{ id: 'kto', text: 'KTO (Label Format)' },
		{ id: 'grpo', text: 'Online RL' }
	];

	const processedResult: ProcessedResult = {
		train: [],
		test: [],
		validation: []
	};

	onMount(async () => {
		dataset = {
			name: '',
			description: '',
			train_file: null,
			validation_file: null,
			trainSetPercentage: 80
		};
		selectedTabId = 0;
		resetMappingState();

		if (Object.keys($datasetTypes).length === 0) {
			try {
				const types = await api.getAutotuneDatasetTypes();
				datasetTypes.set(types);
			} catch (err) {
				console.warn('Failed to fetch dataset types:', err);
			}
		}
	});

	function shuffleArray<T>(array: T[]): T[] {
		const shuffled = [...array];
		for (let i = shuffled.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
		}
		return shuffled;
	}

	function resetMappingState() {
		columnMapping = {};
		userColumns = [];
		detectedAlgorithm = 'lora';
		showColumnMapping = false;
		isAiSuggesting = false;
		aiSuggestion = null;
		aiColumnConfidences = {};
		aiSuggestedFields = new Set();
		showAiReasoning = false;
	}

	function updateColumnMapping(requiredCol: string, userCol: string) {
		columnMapping = { ...columnMapping, [requiredCol]: userCol };
		aiSuggestedFields.delete(requiredCol);
		aiSuggestedFields = aiSuggestedFields;
	}

	function initMappingFromData(data: DatasetType[]) {
		if (data.length === 0) return;

		const columns = Object.keys(data[0] || {});
		userColumns = columns;
		columnMetadata = Utils.extractColumnMetadata(data as ParsedDataRow[]);
		detectedFormat = Utils.detectDatasetFormat(columns);
		detectedAlgorithm = Utils.suggestAlgorithm(columns);
		showColumnMapping = columns.length > 0;

		const columnsToMap =
			allColumnNames.length > 0 ? allColumnNames : Utils.getRequiredColumns(detectedAlgorithm);
		columnMapping = Utils.suggestColumnMapping(userColumns, columnsToMap);

		suggestMappingWithAI(data as ParsedDataRow[], columnMetadata);
	}

	async function suggestMappingWithAI(data: ParsedDataRow[], metadata: ColumnMetadata[]) {
		if (data.length === 0 || metadata.length === 0) return;

		isAiSuggesting = true;
		aiSuggestion = null;
		aiColumnConfidences = {};
		aiSuggestedFields = new Set();
		showAiReasoning = false;

		try {
			const colNames = metadata.map((c) => c.name);
			const colSamples: Record<string, string[]> = {};
			for (const col of metadata) {
				colSamples[col.name] = col.sampleValues.slice(0, 3);
			}

			const targetType = Utils.ALGORITHM_TO_DATASET_TYPE[detectedAlgorithm] || undefined;
			const result: AiMappingSuggestion = await api.suggestColumnMapping(
				data.slice(0, 8),
				colNames,
				colSamples,
				targetType
			);

			aiSuggestion = result;

			if (result.algorithm) {
				detectedAlgorithm = result.algorithm;
			}

			if (result.column_mapping) {
				const newMapping: ColumnMapping = {};
				const newConfidences: Record<string, number> = {};
				const newSuggested = new Set<string>();

				let types = $datasetTypes;
				if (Object.keys(types).length === 0) {
					try {
						types = await api.getAutotuneDatasetTypes();
						datasetTypes.set(types);
					} catch (e) {
						console.warn('Failed to fetch dataset types inline:', e);
					}
				}

				const algo = detectedAlgorithm;
				const aiAllCols =
					Object.keys(types).length > 0
						? Utils.getColumnsFromTypes(algo, types).map((c: { name: string }) => c.name)
						: Utils.getRequiredColumns(algo);

				const typeKey = Utils.ALGORITHM_TO_DATASET_TYPE[algo];
				const columnsDict = types[typeKey]?.columns || {};
				const dictKeyToName: Record<string, string> = {};
				for (const [key, col] of Object.entries(columnsDict)) {
					dictKeyToName[key] = (col as any).name;
				}

				for (const [aiKey, mapping] of Object.entries(result.column_mapping)) {
					if (!(mapping as any).source_column || !colNames.includes((mapping as any).source_column))
						continue;

					let matchedCol = dictKeyToName[aiKey];
					if (!matchedCol) {
						const normalized = aiKey.replace(/_col$/, '');
						matchedCol =
							aiAllCols.find(
								(rc: string) =>
									rc === aiKey || rc === normalized || rc === (mapping as any).source_column
							) || '';
					}

					if (matchedCol && aiAllCols.includes(matchedCol)) {
						newMapping[matchedCol] = (mapping as any).source_column;
						newConfidences[matchedCol] = (mapping as any).confidence;
						newSuggested.add(matchedCol);
					}
				}

				columnMapping = newMapping;
				aiColumnConfidences = newConfidences;
				aiSuggestedFields = newSuggested;
			}
		} catch (err) {
			console.warn('AI suggestion failed, using heuristic fallback:', err);
		} finally {
			isAiSuggesting = false;
		}
	}

	async function splitData(file: File, ratio: number) {
		isProcessingFile = true;
		processingError = null;

		try {
			const previewData = (await Utils.processUploadedFile(file, PREVIEW_LIMIT)) ?? [];

			if (previewData.length === 0) {
				throw new Error(
					'Could not parse any valid data from file. Supported formats: JSONL, JSON, CSV, Parquet.'
				);
			}

			const fileSize = file.size;
			const previewSize = JSON.stringify(previewData).length;
			const estimatedLines = Math.floor((fileSize / previewSize) * previewData.length);
			totalLines = estimatedLines > previewData.length ? estimatedLines : previewData.length;

			const trainSize = Math.floor((totalLines * ratio) / 100);
			trainCount = trainSize;
			validationCount = totalLines - trainSize;

			dataset.train_file = file;
			dataset.validation_file = file;
			dataset.trainSetPercentage = trainSetPercentage;

			const shuffledPreview = shuffleArray(previewData);
			const previewTrainSize = Math.floor((shuffledPreview.length * ratio) / 100);

			processedResult.train = shuffledPreview.slice(0, previewTrainSize).slice(0, 100);
			processedResult.validation = shuffledPreview.slice(previewTrainSize).slice(0, 100);

			const columns = Object.keys(previewData[0] || {});
			detectedFormat = Utils.detectDatasetFormat(columns);

			isProcessingFile = false;
		} catch (error: any) {
			console.error('Error processing file:', error);
			processingError = error.message || 'Failed to process file';
			isProcessingFile = false;
		}
	}

	async function handleCustomValidationUpload(file: File) {
		isProcessingFile = true;
		processingError = null;
		uploadedValidationFile = file;

		try {
			const validationData = (await Utils.processUploadedFile(file, PREVIEW_LIMIT)) ?? [];

			if (validationData.length === 0) {
				throw new Error('Validation file contains no valid data. Please check the file format.');
			}

			const fileSize = file.size;
			const previewSize = JSON.stringify(validationData).length;
			const estimatedLines = Math.floor((fileSize / previewSize) * validationData.length);
			validationCount =
				estimatedLines > validationData.length ? estimatedLines : validationData.length;

			processedResult.validation = validationData;
			dataset.validation_file = file;

			if (detectedFormat === 'unknown') {
				const columns = Object.keys(validationData[0] || {});
				detectedFormat = Utils.detectDatasetFormat(columns);
			}

			isProcessingFile = false;
		} catch (error: any) {
			console.error('Error processing validation file:', error);
			processingError = error.message || 'Failed to process validation file';
			isProcessingFile = false;
		}
	}

	async function handleFileDrop(file: File) {
		resetMappingState();
		uploadedFile = file;
		if (!dataset.name) {
			dataset.name = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9-_]/g, '-');
		}
		allData = (await Utils.processUploadedFile(file, PREVIEW_LIMIT)) ?? [];
		initMappingFromData(allData);
		await splitData(file, trainSetPercentage);
	}

	function clearTrainFile() {
		uploadedFile = null;
		uploadedValidationFile = null;
		allData = [];
		totalLines = 0;
		trainCount = 0;
		validationCount = 0;
		processedResult.train = [];
		processedResult.validation = [];
		previewHeaders = [];
		previewRows = [];
		valPreviewHeaders = [];
		valPreviewRows = [];
		dataset.train_file = null;
		dataset.validation_file = null;
		detectedFormat = 'unknown';
		processingError = null;
		resetMappingState();
	}

	function handleDatasetTypeSelect(id: string) {
		detectedAlgorithm = id;
		const columnsToMap =
			allColumnNames.length > 0 ? allColumnNames : Utils.getRequiredColumns(detectedAlgorithm);
		columnMapping = Utils.suggestColumnMapping(userColumns, columnsToMap);
		aiSuggestion = null;
		aiSuggestedFields = new Set();
	}

	// Handle toggle between auto-split and custom validation
	$: if (!isSplitEnabled) {
		if (allData.length > 0) {
			processedResult.train = allData.slice(0, 100);
			trainCount = totalLines;
			dataset.train_file = uploadedFile;
		}
		if (!uploadedValidationFile) {
			processedResult.validation = [];
			validationCount = 0;
			dataset.validation_file = null;
		}
	} else {
		uploadedValidationFile = null;
		if (uploadedFile) {
			(async () => {
				await splitData(uploadedFile, trainSetPercentage);
			})();
		}
	}

	// While uploading, replace the global spinner with our in-dialog progress card
	$: if (isUploading) {
		showLoader.set(false);
	}

	// Build preview headers/rows for Train and Validation independently
	function buildPreview(sourceData: DatasetType[]): {
		headers: { key: string; value: string }[];
		rows: { id: string; [key: string]: any }[];
	} {
		if (sourceData.length === 0) return { headers: [], rows: [] };
		const metadata = Utils.extractColumnMetadata(sourceData);
		const columns = metadata.map((col) => col.name);
		const headers = columns.map((col) => ({
			key: col,
			value: Utils.toUpperCase(col) || col
		}));
		const rows = sourceData.slice(0, 100).map((row, i) => {
			const processedRow: { id: string; [key: string]: any } = { id: String(i) };
			for (const col of columns) {
				const val = (row as Record<string, any>)[col];
				if (val === null || val === undefined) {
					processedRow[col] = '';
				} else if (typeof val === 'string') {
					processedRow[col] = val.length > 120 ? val.substring(0, 120) + '...' : val;
				} else {
					const str = JSON.stringify(val);
					processedRow[col] = str.length > 120 ? str.substring(0, 120) + '...' : str;
				}
			}
			return processedRow;
		});
		return { headers, rows };
	}

	$: {
		const trainView = buildPreview(processedResult.train);
		previewHeaders = trainView.headers;
		previewRows = trainView.rows;
		// Keep columnMetadata reflecting the currently-viewed train preview (back-compat).
		if (processedResult.train.length > 0) {
			columnMetadata = Utils.extractColumnMetadata(processedResult.train);
		}
	}

	$: {
		const valView = buildPreview(processedResult.validation);
		valPreviewHeaders = valView.headers;
		valPreviewRows = valView.rows;
	}

	// Empty-state helpers
	$: typeKey = Utils.ALGORITHM_TO_DATASET_TYPE[detectedAlgorithm];
	$: typeDesc = $datasetTypes[typeKey]?.desc;
	$: typeColumns = $datasetTypes[typeKey]?.columns;
	$: emptyStateExamples =
		Object.keys($datasetTypes).length > 0
			? Utils.getDatasetExamplesFromTypes(detectedAlgorithm, $datasetTypes)
			: Utils.getDatasetExamples(detectedAlgorithm);
	$: emptyStateFormats = typeColumns
		? Utils.generateFormatExamples(Object.values(typeColumns))
		: null;
</script>

<Form>
	<Grid noGutter fullWidth>
		<Row>
			<!-- LEFT: Dataset Settings tile -->
			<Column sm={4} md={4} lg={6}>
				<Tile style="padding: 1.25rem;">
					<h6 class="tile-heading">Dataset Settings</h6>

					<!-- Dataset Name -->
					<div class="name-field">
						<TextInput
							labelText="Dataset Name"
							bind:value={dataset.name}
							placeholder="my-dataset"
						/>
					</div>

					<!-- Dataset Type -->
					<div class="type-field">
						<Dropdown
							labelText="Dataset Type"
							size="sm"
							selectedId={detectedAlgorithm}
							items={datasetTypeItems}
							on:select={(e) => handleDatasetTypeSelect(e.detail.selectedItem.id)}
						/>
					</div>

					<!-- Split toggle -->
					<div class="split-toggle-row">
						<div class="toggle-label-with-tooltip">
							<span class="toggle-label-text">Split dataset</span>
							<Tooltip align="start" direction="bottom"
								>Automatically splits your uploaded file into training and validation sets using the
								specified ratio. Disable this to upload a separate validation file.</Tooltip
							>
						</div>
						<Toggle
							labelText=""
							hideLabel
							bind:toggled={isSplitEnabled}
							disabled={!uploadedFile}
							size="sm"
						/>
					</div>

					<!-- Upload area -->
					{#if !uploadedFile}
						<div class="drop-zone">
							<FileUploaderDropContainer
								labelText="Drag and drop a file here or click to upload"
								accept={['.jsonl', '.json', '.csv', '.parquet']}
								on:change={async (e) => {
									if (e.detail && e.detail.length > 0) {
										await handleFileDrop(e.detail[0]);
									}
								}}
							/>
							<p class="drop-zone-hint">Accepted formats: .jsonl, .json, .csv, .parquet</p>
						</div>
					{:else}
						<div class="file-row">
							<span class="file-label"
								>Train file <Tooltip align="start" direction="bottom"
									>The main dataset used to train the model. Learns patterns from your data.</Tooltip
								></span
							>
							<FileUploaderItem name={uploadedFile.name} status="edit" on:delete={clearTrainFile} />
						</div>
					{/if}

					<!-- Custom validation upload (only when split is off) -->
					{#if uploadedFile && !isSplitEnabled}
						<div class="reveal-block">
							{#if !uploadedValidationFile}
								<FileUploaderButton
									labelText="Upload validation file"
									kind="tertiary"
									size="small"
									accept={['.jsonl', '.json', '.csv', '.parquet']}
									on:change={async (e) => {
										if (e.detail && e.detail.length > 0) {
											await handleCustomValidationUpload(e.detail[0]);
										}
									}}
								/>
							{:else}
								<div class="file-row">
									<span class="file-label"
										>Validation file <Tooltip align="start" direction="bottom"
											>A separate dataset used to evaluate model performance during training.</Tooltip
										></span
									>
									<FileUploaderItem
										name={uploadedValidationFile.name}
										status="edit"
										on:delete={() => {
											uploadedValidationFile = null;
											processedResult.validation = [];
											validationCount = 0;
											dataset.validation_file = null;
										}}
									/>
								</div>
							{/if}
						</div>
					{/if}

					{#if isProcessingFile}
						<InlineLoading
							description="Processing..."
							style="margin-top: var(--cds-spacing-03, 0.5rem);"
						/>
					{/if}

					{#if processingError}
						<InlineNotification
							kind="error"
							title="Error"
							subtitle={processingError}
							hideCloseButton
							lowContrast
							style="margin-top: var(--cds-spacing-03, 0.5rem);"
						/>
					{/if}

					{#if isPreviewDeferred}
						<InlineNotification
							kind="info"
							title="Preview skipped"
							subtitle="This Parquet file is too large to preview in the browser, so only its column names were read. The preview rows and record counts below are placeholders — the file itself uploads and is validated in full on the server."
							hideCloseButton
							lowContrast
							style="margin-top: var(--cds-spacing-03, 0.5rem);"
						/>
					{/if}

					<!-- Column mapping -->
					{#if showColumnMapping && userColumns.length > 0}
						<hr class="section-divider" />
						<div class="mapping-title-row">
							<p class="section-title">Column Mapping</p>
							{#if isAiSuggesting}
								<InlineLoading description="AI analyzing..." />
							{:else if aiSuggestion}
								<Tag
									type="green"
									size="sm"
									interactive
									on:click={() => {
										showAiReasoning = !showAiReasoning;
									}}
								>
									AI Suggested ({Math.round(aiSuggestion.confidence * 100)}%)
									{showAiReasoning ? '▴' : '▾'}
								</Tag>
							{/if}
						</div>

						{#if aiSuggestion?.reasoning && showAiReasoning}
							<InlineNotification
								kind="info"
								title="AI Insight"
								subtitle={aiSuggestion.reasoning}
								hideCloseButton
								lowContrast
								style="margin-bottom: 0.75rem;"
							/>
						{/if}

						{#if !isAiSuggesting}
							<div class="mapping-header">
								<span>Field</span>
								<span>Source Column</span>
							</div>
							{#if allColumns.length > 0}
								{#each [...allColumns].sort((a, b) => Number(b.required) - Number(a.required)) as colInfo}
									<div class="mapping-row">
										<div class="mapping-label">
											{Utils.toUpperCase(colInfo.name)}
											{#if colInfo.desc}
												<Tooltip align="start" direction="top">{colInfo.desc}</Tooltip>
											{/if}
										</div>
										<Select
											labelText=""
											size="sm"
											selected={columnMapping[colInfo.name] || ''}
											on:change={(e) => {
												const target = e.target;
												if (target instanceof HTMLSelectElement)
													updateColumnMapping(colInfo.name, target.value);
											}}
										>
											<SelectItem value="" text={colInfo.required ? 'Select column...' : 'None'} />
											{#each userColumns as col}
												<SelectItem value={col} text={col} />
											{/each}
										</Select>
									</div>
								{/each}
							{:else}
								{#each requiredColumns as reqCol}
									<div class="mapping-row">
										<div class="mapping-label">
											{Utils.toUpperCase(reqCol)}
										</div>
										<Select
											labelText=""
											size="sm"
											selected={columnMapping[reqCol] || ''}
											on:change={(e) => {
												const target = e.target;
												if (target instanceof HTMLSelectElement)
													updateColumnMapping(reqCol, target.value);
											}}
										>
											<SelectItem value="" text="Select column..." />
											{#each userColumns as col}
												<SelectItem value={col} text={col} />
											{/each}
										</Select>
									</div>
								{/each}
							{/if}
						{/if}
					{/if}
				</Tile>
			</Column>

			<!-- RIGHT: Preview / empty state -->
			<Column sm={4} md={4} lg={10}>
				{#if uploadedFile && previewRows.length > 0 && previewHeaders.length > 0}
					<Tile class="preview-tile">
						<Tabs bind:selected={activePreviewTab}>
							<Tab
								label="Train ({trainCount.toLocaleString('en-US', {
									notation: 'compact',
									maximumFractionDigits: 2
								})})"
							/>
							<Tab
								label="Validation ({validationCount.toLocaleString('en-US', {
									notation: 'compact',
									maximumFractionDigits: 2
								})})"
							/>
							<svelte:fragment slot="content">
								<TabContent style="padding: 0.5rem 0;">
									<div style="overflow-x: auto;">
										<Table
											rows={previewRows}
											headers={previewHeaders}
											batchSelection={false}
											selectable={false}
											expandable={false}
											showActionButton={false}
										/>
									</div>
									{#if processedResult.train.length >= 100 && trainCount > 100}
										<div class="preview-footer">
											Showing first 100 of {trainCount.toLocaleString()} lines
										</div>
									{/if}
								</TabContent>
								<TabContent style="padding: 0.5rem 0;">
									{#if valPreviewRows.length > 0 && valPreviewHeaders.length > 0}
										<div style="overflow-x: auto;">
											<Table
												rows={valPreviewRows}
												headers={valPreviewHeaders}
												batchSelection={false}
												selectable={false}
												expandable={false}
												showActionButton={false}
											/>
										</div>
										{#if processedResult.validation.length >= 100 && validationCount > 100}
											<div class="preview-footer">
												Showing first 100 of {validationCount.toLocaleString()} lines
											</div>
										{/if}
									{:else}
										<p class="empty-state-hint">
											{isSplitEnabled
												? 'No validation rows yet — upload a file or increase the validation percentage.'
												: 'Upload a validation file to preview it here.'}
										</p>
									{/if}
								</TabContent>
							</svelte:fragment>
						</Tabs>
					</Tile>
				{:else}
					<Tile style="padding: 1.25rem;">
						<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
							<h6 class="tile-heading" style="margin: 0;">Expected Dataset Format</h6>
						</div>
						{#if typeDesc}
							<p class="helper-text-inline" style="margin-bottom: 0.75rem;">{typeDesc}</p>
						{/if}
						<Tabs>
							<Tab label="JSON" />
							<Tab label="JSONL" />
							<Tab label="CSV" />
							<Tab label="Parquet" />
							<svelte:fragment slot="content">
								<TabContent style="padding: 0.5rem 0;">
									<pre class="example-code">{emptyStateFormats?.json ||
											JSON.stringify(emptyStateExamples, null, 2)}</pre>
								</TabContent>
								<TabContent style="padding: 0.5rem 0;">
									<pre class="example-code">{emptyStateFormats?.jsonl ||
											emptyStateExamples.map((ex) => JSON.stringify(ex)).join('\n')}</pre>
								</TabContent>
								<TabContent style="padding: 0.5rem 0;">
									<pre class="example-code">{emptyStateFormats?.csv ||
											emptyStateExamples.map((ex) => Object.values(ex).join(', ')).join('\n')}</pre>
								</TabContent>
								<TabContent style="padding: 0.5rem 0;">
									<pre
										class="example-code">Apache Parquet is a columnar storage format.{'\n'}Use the same column structure as CSV/JSON.{'\n'}Generate with: df.to_parquet("data.parquet")</pre>
								</TabContent>
							</svelte:fragment>
						</Tabs>
						<p class="empty-state-hint">Upload a dataset to see a preview.</p>
					</Tile>
				{/if}
			</Column>
		</Row>
	</Grid>
</Form>

{#if isUploading}
	<div class="upload-progress-overlay">
		<div class="upload-progress-card">
			<InlineLoading description="Uploading dataset… {Math.round(uploadProgress)}%" />
			<ProgressBar labelText="" hideLabel value={uploadProgress} max={100} size="sm" />
		</div>
	</div>
{/if}

<style>
	.tile-heading {
		font-weight: 600;
		margin: 0 0 1.25rem 0;
	}

	.section-title {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		margin: 0;
	}

	.name-field {
		margin-bottom: 0.75rem;
	}

	.type-field {
		margin-bottom: 0.75rem;
	}

	.split-toggle-row {
		margin-bottom: 1rem;
	}

	.toggle-label-with-tooltip {
		display: flex;
		align-items: center;
		gap: 0;
		margin-bottom: 0.25rem;
	}

	.toggle-label-text {
		font-size: 0.75rem;
		font-weight: 400;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
	}

	.toggle-label-with-tooltip :global(.bx--tooltip__trigger) {
		margin-left: 0.25rem;
		display: inline-flex;
		align-items: center;
	}

	.section-divider {
		border: none;
		border-top: 1px solid var(--cds-border-subtle, #e0e0e0);
		margin: 1.25rem 0;
	}

	.helper-text-inline {
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
	}

	.reveal-block {
		margin-top: 0.75rem;
	}

	.mapping-title-row {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}

	.mapping-header {
		display: grid;
		grid-template-columns: 160px 1fr;
		gap: 0.75rem;
		font-size: 0.75rem;
		font-weight: 400;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		padding-bottom: 0.375rem;
		margin-bottom: var(--cds-spacing-03, 0.5rem);
	}

	.mapping-row {
		display: grid;
		grid-template-columns: 160px 1fr;
		gap: 0.75rem;
		align-items: end;
		margin-bottom: 1rem;
	}

	.mapping-label {
		font-size: 0.875rem;
		font-weight: 500;
		color: var(--cds-text-01, #161616);
		display: flex;
		align-items: center;
		gap: 0;
		height: 2rem;
	}

	.mapping-label :global(.bx--tooltip__trigger) {
		margin-left: 0.25rem;
		display: inline-flex;
		align-items: center;
		vertical-align: middle;
	}

	.mapping-label :global(.bx--tooltip__trigger svg) {
		vertical-align: middle;
	}

	.toggle-label-with-tooltip :global(.bx--tooltip__label),
	.file-label :global(.bx--tooltip__label),
	.mapping-label :global(.bx--tooltip__label) {
		display: inline-flex;
		align-items: center;
	}

	.file-label :global(.bx--tooltip),
	.mapping-label :global(.bx--tooltip) {
		white-space: normal;
	}

	.example-code {
		background: var(--cds-background-inverse, #262626);
		color: var(--cds-layer, #f4f4f4);
		padding: 0.75rem 1rem;
		border-radius: 4px;
		font-size: 0.8125rem;
		font-family: 'IBM Plex Mono', monospace;
		line-height: 1.5;
		overflow-x: auto;
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.drop-zone {
		border: 2px dashed var(--cds-icon-disabled, #a8a8a8);
		border-radius: 4px;
		padding: var(--cds-spacing-06, 1.5rem);
		text-align: center;
		transition:
			border-color 0.15s,
			background 0.15s;
	}

	.drop-zone:hover {
		border-color: var(--cds-interactive, #0f62fe);
		background: var(--cds-highlight, #edf5ff);
	}

	.drop-zone-hint {
		font-size: 0.75rem;
		color: var(--cds-text-02, #525252);
		margin: var(--cds-spacing-03, 0.5rem) 0 0 0;
	}

	.empty-state-hint {
		color: var(--cds-text-02, #525252);
		font-size: 0.8125rem;
		margin-top: 1rem;
		text-align: center;
	}

	:global(.preview-tile) {
		padding: 1.25rem;
		max-height: calc(100vh - 16rem);
		overflow: auto;
	}

	.preview-footer {
		padding: 0.5rem;
		text-align: center;
		color: var(--cds-text-02, #525252);
		font-size: 0.8125rem;
		background: #f4f4f4;
		margin-top: 0.5rem;
	}

	.file-row {
		display: grid;
		grid-template-columns: 120px 1fr;
		align-items: center;
		gap: 0.5rem;
	}

	.file-label {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		white-space: nowrap;
		display: inline-flex;
		align-items: center;
		gap: 0;
		flex-shrink: 0;
	}

	.file-label :global(.bx--tooltip__trigger) {
		margin-left: 0.25rem;
		display: inline-flex;
		align-items: center;
	}

	.file-row :global(.bx--file__selected-file) {
		flex: 1;
		min-width: 0;
		background-color: #f4f4f4;
	}

	.upload-progress-overlay {
		position: fixed;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 10001;
		background: rgba(22, 22, 22, 0.5);
	}

	.upload-progress-card {
		background: var(--cds-layer, #ffffff);
		border-radius: 4px;
		padding: 1.5rem 1.75rem;
		min-width: 360px;
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.upload-progress-card :global(.bx--inline-loading) {
		min-height: auto;
	}

	.upload-progress-card :global(.bx--inline-loading__text) {
		font-size: 0.875rem;
		font-weight: 500;
		color: var(--cds-text-01, #161616);
	}

	.name-field :global(.bx--text-input) {
		background-color: #f4f4f4 !important;
	}
	:global(.bx--select-input) {
		background-color: #f4f4f4 !important;
	}
	:global(.bx--list-box) {
		background-color: #f4f4f4 !important;
	}
</style>
