<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		FileUploaderButton,
		FileUploaderDropContainer,
		FileUploaderItem,
		TextInput,
		Tile,
		DataTable,
		Select,
		SelectItem,
		Dropdown,
		Toggle,
		Button,
		InlineLoading,
		InlineNotification,
		Tag,
		ContentSwitcher,
		Switch,
		Grid,
		Row,
		Column,
		Tabs,
		Tab,
		TabContent,
		Tooltip
	} from 'carbon-components-svelte';
	import { Upload, Reset, MachineLearning } from 'carbon-icons-svelte';
	import { createEventDispatcher, onMount } from 'svelte';
	import { Utils } from '$lib/utils';
	import { API } from '$lib/api';
	import { appState, datasets, datasetTypes } from '$lib/app';
	import type {
		Dataset,
		DatasetForm,
		DatasetFormatType,
		ColumnMetadata,
		ParsedDataRow,
		ColumnMapping,
		AiMappingSuggestion,
		TuningGoal
	} from '$lib/app-types';
	import Table from '$lib/components/Table.svelte';

	const dispatch = createEventDispatcher();
	const api = new API();

	// Props from parent wizard
	export let uploadedFile: File | null = null;
	export let parsedData: ParsedDataRow[] = [];
	export let columnMetadata: ColumnMetadata[] = [];
	export let detectedFormat: DatasetFormatType = 'unknown';
	export let datasetForm: DatasetForm = {
		name: '',
		description: '',
		train_file: null,
		validation_file: null
	};
	export let totalRecords: number = 0;
	export let existingDatasetId: string | null = null;
	export let splitRatio: number = 80;
	export let validationFile: File | null = null;
	export let selectedAlgorithm: string = 'lora';
	export let selectedGoal: TuningGoal | null = null;
	export let columnMapping: ColumnMapping = {};
	export let isDatasetCompatible: boolean = true;
	export let keepAsParquet: boolean = true;

	// Detect if uploaded file is parquet
	$: isParquetFile = uploadedFile?.name?.endsWith('.parquet') ?? false;

	// A Parquet file over the client preview cap is parsed for its column names
	// only, and `parsedData` is then a single blank placeholder row: the column
	// mapping below is real, the preview cells are not. Say so rather than showing
	// an empty table as if that were the data.
	$: isPreviewDeferred = Utils.isPreviewDeferred(uploadedFile);

	// Internal state
	let isProcessing = false;
	let processingProgress = '';
	let error = '';
	let previewRows: Record<string, any>[] = [];
	let previewHeaders: { key: string; value: string }[] = [];
	let valPreviewRows: Record<string, any>[] = [];
	let valPreviewHeaders: { key: string; value: string }[] = [];
	let validationRecordCount: number = 0;
	let activePreviewTab: number = 0;
	export let isSplitEnabled = true;
	let validationSetPercentage = 20;
	$: splitRatio = 100 - validationSetPercentage;
	let existingDatasets: Dataset[] = [];
	export let selectedExistingDataset: Dataset | null = null;
	let isLoadingDatasets = false;
	let userColumns: string[] = [];
	let dataSourceIndex = 0; // 0 = Upload, 1 = Select Existing

	// AI suggestion state
	let isAiSuggesting = false;
	let aiSuggestion: AiMappingSuggestion | null = null;
	let aiColumnConfidences: Record<string, number> = {};
	let aiSuggestedFields: Set<string> = new Set();
	let showAiReasoning = false;

	// Algorithm options for dropdown (grouped)
	const algorithmItems = Utils.ALGORITHM_OPTIONS.map((a) => ({
		id: a.id,
		text: `${a.name} (${a.category.replace('_', ' ')})`
	}));

	// Map specific algorithm IDs to grouped dropdown IDs
	const algorithmGroupMap: Record<string, string> = {
		lora: 'lora',
		sft: 'lora',
		alora: 'lora',
		lokr: 'lora',
		loha: 'lora',
		vera: 'lora',
		dpo: 'dpo',
		kto: 'kto',
		ppo: 'ppo',
		grpo: 'grpo',
		dapo: 'dapo'
	};
	$: dropdownAlgorithmId = algorithmGroupMap[selectedAlgorithm] || selectedAlgorithm;

	// All columns with metadata (required/optional, description) from API
	$: allColumns =
		Object.keys($datasetTypes).length > 0
			? Utils.getColumnsFromTypes(selectedAlgorithm, $datasetTypes)
			: [];

	// Required column names only (for validation gating)
	$: requiredColumns =
		Object.keys($datasetTypes).length > 0
			? Utils.getRequiredColumnsFromTypes(selectedAlgorithm, $datasetTypes)
			: Utils.getRequiredColumns(selectedAlgorithm);

	// All column names (required + optional) for mapping suggestions
	$: allColumnNames = allColumns.map((c) => c.name);

	// Check if all required columns are mapped (optional columns don't block)
	$: allColumnsMapped = requiredColumns.every((c) => columnMapping[c]);

	// Validate dataset format against selected goal
	$: datasetGoalWarning =
		selectedGoal && detectedFormat !== 'unknown'
			? Utils.validateDatasetForGoal(detectedFormat, selectedGoal)
			: { valid: true, message: '' };

	// Expose compatibility to parent wizard for Next button gating
	$: isDatasetCompatible = datasetGoalWarning.valid;

	// When algorithm changes, re-suggest mapping (skip if AI mapping is active or mapping already valid)
	$: if (selectedAlgorithm && userColumns.length > 0 && !aiSuggestion && !isAiSuggesting) {
		const alreadyMapped =
			requiredColumns.length > 0 && requiredColumns.every((c) => columnMapping[c]);
		if (!alreadyMapped) {
			// Suggest mappings for all columns (required + optional)
			const columnsToMap = allColumnNames.length > 0 ? allColumnNames : requiredColumns;
			columnMapping = Utils.suggestColumnMapping(userColumns, columnsToMap);
		}
	}

	// Helper: build preview table data from rows
	function buildPreviewData(data: ParsedDataRow[]): {
		headers: { key: string; value: string }[];
		rows: Record<string, any>[];
	} {
		const cols = Object.keys(data[0] || {});
		const headers = cols.map((col) => ({
			key: col,
			value: Utils.toUpperCase(col) || col
		}));
		const rows = data.slice(0, 15).map((row, i) => {
			const processedRow: Record<string, any> = { id: String(i) };
			for (const col of cols) {
				const val = row[col];
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

	// Reactive: build train preview when parsedData changes
	$: if (parsedData.length > 0) {
		userColumns = columnMetadata.map((col) => col.name);
		const result = buildPreviewData(parsedData);
		previewHeaders = result.headers;
		previewRows = result.rows;
	}

	// Reactive: build validation preview for split mode
	$: if (isSplitEnabled && parsedData.length > 0 && uploadedFile) {
		const splitIndex = Math.floor((parsedData.length * splitRatio) / 100);
		const valSlice = parsedData.slice(splitIndex);
		if (valSlice.length > 0) {
			const result = buildPreviewData(valSlice);
			valPreviewHeaders = result.headers;
			valPreviewRows = result.rows;
			validationRecordCount = totalRecords - Math.floor((totalRecords * splitRatio) / 100);
		}
	}

	// Reactive: build validation preview for manual validation file
	$: if (!isSplitEnabled && validationFile) {
		Utils.processUploadedFile(validationFile, 50).then((rawData) => {
			const valData = rawData as ParsedDataRow[];
			const result = buildPreviewData(valData);
			valPreviewHeaders = result.headers;
			valPreviewRows = result.rows;
			validationRecordCount = valData.length;
			Utils.countLinesInFile(validationFile!).then((count) => {
				validationRecordCount = count;
			});
		});
	}

	// Clear validation preview when split is toggled off and no validation file
	$: if (!isSplitEnabled && !validationFile) {
		valPreviewRows = [];
		valPreviewHeaders = [];
		validationRecordCount = 0;
	}

	// Computed: train record count for tab label
	$: trainRecordCount = existingDatasetId
		? selectedExistingDataset?.train_records || totalRecords
		: isSplitEnabled && uploadedFile
		  ? Math.floor((totalRecords * splitRatio) / 100)
		  : totalRecords;

	async function handleFileUpload(file: File) {
		isProcessing = true;
		processingProgress = '';
		error = '';
		existingDatasetId = null;
		selectedExistingDataset = null;

		try {
			uploadedFile = file;

			if (!datasetForm.name) {
				datasetForm.name = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9-_]/g, '-');
			}

			if (file.size > 5 * 1024 * 1024) {
				processingProgress = 'Processing large file...';
			}

			const rawData = await Utils.processUploadedFileAsync(file, 50);
			parsedData = rawData as ParsedDataRow[];

			const columns = Object.keys(parsedData[0] || {});
			columnMetadata = Utils.extractColumnMetadata(parsedData);
			detectedFormat = Utils.detectDatasetFormat(columns);

			// Auto-suggest algorithm based on columns, but only if compatible with selected goal
			const suggestedAlgo = Utils.suggestAlgorithm(columns);
			const suggestedDetail = Utils.ALGORITHM_DETAILS.find((a) => a.id === suggestedAlgo);
			if (!selectedGoal || (suggestedDetail && suggestedDetail.category === selectedGoal)) {
				selectedAlgorithm = suggestedAlgo;
			}
			// If mismatch, keep the algorithm from Step 0 (don't override)

			totalRecords = parsedData.length;
			Utils.countLinesInFileAsync(file).then((count) => {
				totalRecords = count;
			});

			dispatch('fileProcessed', {
				parsedData,
				columnMetadata,
				format: detectedFormat,
				totalRecords
			});
			dispatch('datasetChanged');

			// Fire AI suggestion in background (non-blocking)
			suggestMappingWithAI(parsedData, columnMetadata);
		} catch (err: any) {
			error = err.message || 'Failed to process file';
			console.error('File processing error:', err);
		} finally {
			isProcessing = false;
			processingProgress = '';
		}
	}

	async function handleExistingDatasetSelect(datasetId: string) {
		if (!datasetId) {
			selectedExistingDataset = null;
			existingDatasetId = null;
			parsedData = [];
			columnMetadata = [];
			detectedFormat = 'unknown';
			totalRecords = 0;
			userColumns = [];
			return;
		}

		isProcessing = true;
		error = '';
		dispatch('datasetChanged');

		try {
			const dataset = await api.getDataset(datasetId, true);
			selectedExistingDataset = dataset;
			existingDatasetId = dataset.id;
			totalRecords = (dataset.train_records || 0) + (dataset.validation_records || 0);
			datasetForm.name = dataset.name;
			datasetForm.description = dataset.description;

			if (dataset.train_data && dataset.train_data.length > 0) {
				parsedData = dataset.train_data as ParsedDataRow[];
				const columns = Object.keys(parsedData[0] || {});
				columnMetadata = Utils.extractColumnMetadata(parsedData);
				detectedFormat = Utils.detectDatasetFormat(columns);
				// Only change algorithm if compatible with selected goal
				const suggestedAlgo = Utils.suggestAlgorithm(columns);
				const suggestedDetail = Utils.ALGORITHM_DETAILS.find((a) => a.id === suggestedAlgo);
				if (!selectedGoal || (suggestedDetail && suggestedDetail.category === selectedGoal)) {
					selectedAlgorithm = suggestedAlgo;
				}
			} else {
				parsedData = [];
				columnMetadata = [];
				detectedFormat = 'unknown';
			}

			// Build validation preview if available
			if (dataset.validation_data && dataset.validation_data.length > 0) {
				const result = buildPreviewData(dataset.validation_data as ParsedDataRow[]);
				valPreviewHeaders = result.headers;
				valPreviewRows = result.rows;
				validationRecordCount = dataset.validation_records || dataset.validation_data.length;
			} else {
				valPreviewRows = [];
				valPreviewHeaders = [];
				validationRecordCount = 0;
			}
		} catch (err: any) {
			error = err.message || 'Failed to load dataset';
			console.error('Dataset load error:', err);
		} finally {
			isProcessing = false;
		}
	}

	function clearTrainFile() {
		uploadedFile = null;
		parsedData = [];
		columnMetadata = [];
		detectedFormat = 'unknown';
		datasetForm.name = '';
		totalRecords = 0;
		userColumns = [];
		previewRows = [];
		previewHeaders = [];
		valPreviewRows = [];
		valPreviewHeaders = [];
		validationRecordCount = 0;
		validationFile = null;
		columnMapping = {};
		selectedAlgorithm = selectedGoal ? Utils.getDefaultAlgorithmForGoal(selectedGoal) : 'lora';
		activePreviewTab = 0;
		aiSuggestion = null;
		aiColumnConfidences = {};
		aiSuggestedFields = new Set();
	}

	function updateColumnMapping(requiredCol: string, userCol: string) {
		columnMapping = { ...columnMapping, [requiredCol]: userCol };
		// Remove AI indicator when user manually changes a mapping
		aiSuggestedFields.delete(requiredCol);
		aiSuggestedFields = aiSuggestedFields;
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

			// Pass the user's chosen dataset type so AI maps to it even if data looks like a different format
			const targetType = Utils.ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm] || undefined;

			const result: AiMappingSuggestion = await api.suggestColumnMapping(
				data.slice(0, 8),
				colNames,
				colSamples,
				targetType
			);

			aiSuggestion = result;

			// Apply suggested algorithm only if it matches the selected goal
			if (result.algorithm) {
				const aiAlgoDetail = Utils.ALGORITHM_DETAILS.find((a) => a.id === result.algorithm);
				if (!selectedGoal || (aiAlgoDetail && aiAlgoDetail.category === selectedGoal)) {
					selectedAlgorithm = result.algorithm;
				}
				// If mismatch, keep the algorithm from Step 0
			}

			// Apply column mapping from AI
			if (result.column_mapping) {
				const newMapping: ColumnMapping = {};
				const newConfidences: Record<string, number> = {};
				const newSuggested = new Set<string>();

				// Ensure datasetTypes is available (race condition: AI may respond before onMount loads types)
				let types = $datasetTypes;
				if (Object.keys(types).length === 0) {
					try {
						types = await api.getAutotuneDatasetTypes();
						datasetTypes.set(types);
					} catch (e) {
						console.warn('Failed to fetch dataset types inline, using fallback:', e);
					}
				}

				// Always use the CURRENT algorithm's columns (respects goal from Step 0),
				// not the AI-suggested algorithm which may belong to a different goal
				const algo = selectedAlgorithm;
				const aiAllCols =
					Object.keys(types).length > 0
						? Utils.getColumnsFromTypes(algo, types).map((c) => c.name)
						: Utils.getRequiredColumns(algo);

				// Build dict key -> column name lookup from backend types
				// e.g. "reward_model_col" -> "reward", "prompt_col" -> "prompt"
				const typeKey = Utils.ALGORITHM_TO_DATASET_TYPE[algo];
				const columnsDict = types[typeKey]?.columns || {};
				const dictKeyToName: Record<string, string> = {};
				for (const [key, col] of Object.entries(columnsDict)) {
					dictKeyToName[key] = (col as any).name;
				}

				for (const [aiKey, mapping] of Object.entries(result.column_mapping)) {
					if (!mapping.source_column || !colNames.includes(mapping.source_column)) continue;

					// Primary: direct lookup by backend dict key (most reliable)
					let matchedCol = dictKeyToName[aiKey];

					// Fallback: normalize AI key and search all columns (required + optional)
					if (!matchedCol) {
						const normalized = aiKey.replace(/_col$/, '');
						matchedCol =
							aiAllCols.find(
								(rc) => rc === aiKey || rc === normalized || rc === mapping.source_column
							) || '';
					}

					if (matchedCol && aiAllCols.includes(matchedCol)) {
						newMapping[matchedCol] = mapping.source_column;
						newConfidences[matchedCol] = mapping.confidence;
						newSuggested.add(matchedCol);
					}
				}

				console.log('AI mapping applied:', { aiAllCols, newMapping, raw: result.column_mapping });
				columnMapping = newMapping;
				aiColumnConfidences = newConfidences;
				aiSuggestedFields = newSuggested;
			}
		} catch (err) {
			console.warn('AI suggestion failed, using heuristic fallback:', err);
			// Heuristic already applied by reactive block, no action needed
		} finally {
			isAiSuggesting = false;
		}
	}

	function resetForm() {
		uploadedFile = null;
		parsedData = [];
		columnMetadata = [];
		detectedFormat = 'unknown';
		datasetForm = { name: '', description: '', train_file: null, validation_file: null };
		totalRecords = 0;
		existingDatasetId = null;
		selectedExistingDataset = null;
		validationFile = null;
		selectedAlgorithm = selectedGoal ? Utils.getDefaultAlgorithmForGoal(selectedGoal) : 'lora';
		columnMapping = {};
		userColumns = [];
		previewRows = [];
		previewHeaders = [];
		valPreviewRows = [];
		valPreviewHeaders = [];
		validationRecordCount = 0;
		activePreviewTab = 0;
		error = '';
		isSplitEnabled = true;
		aiSuggestion = null;
		aiColumnConfidences = {};
		aiSuggestedFields = new Set();
		dispatch('datasetChanged');
	}

	onMount(async () => {
		// Restore existing dataset state on remount
		if (existingDatasetId) {
			try {
				const dataset = await api.getDataset(existingDatasetId);
				selectedExistingDataset = dataset;
			} catch (err) {
				console.error('Error restoring dataset:', err);
			}
		}

		try {
			isLoadingDatasets = true;
			if (!$appState.isDatasetsLoaded) {
				const data = await api.getDatasets();
				datasets.set(data);
				appState.update((prev) => ({ ...prev, isDatasetsLoaded: true }));
			}
			existingDatasets = $datasets || [];
		} catch (err) {
			console.error('Error loading datasets:', err);
		} finally {
			isLoadingDatasets = false;
		}

		// Fetch dataset types from backend (for dynamic required columns)
		if (Object.keys($datasetTypes).length === 0) {
			try {
				const types = await api.getAutotuneDatasetTypes();
				datasetTypes.set(types);
			} catch (err) {
				console.warn('Failed to fetch dataset types, using hardcoded fallback:', err);
			}
		}
	});
</script>

<Grid noGutter fullWidth>
	<Row>
		<!-- Left column -->
		<Column sm={4} md={4} lg={6}>
			<Tile style="padding: 1.25rem;">
				<h6 class="tile-heading">Dataset Settings</h6>

				<!-- DATASET NAME -->
				{#if uploadedFile && !existingDatasetId}
					<div class="name-field">
						<TextInput
							labelText="Dataset Name"
							bind:value={datasetForm.name}
							placeholder="my-dataset"
						/>
					</div>
					<div class="split-toggle-row">
						<div class="toggle-label-with-tooltip">
							<span class="toggle-label-text">Split dataset</span>
							<Tooltip
								>Automatically splits your uploaded file into training and validation sets using the
								specified ratio. Disable this to upload separate files for each.</Tooltip
							>
						</div>
						<Toggle labelText="" hideLabel bind:toggled={isSplitEnabled} size="sm" />
					</div>
					<!-- {#if isParquetFile}
						<div class="split-toggle-row">
							<Toggle
								labelText="Keep as Parquet"
								bind:toggled={keepAsParquet}
								size="sm"
							/>
							<p class="helper-text" style="margin: 0; font-size: 0.75rem; color: var(--cds-text-helper, #6f6f6f);">
								{keepAsParquet ? 'File will be uploaded as .parquet (column mapping applied server-side)' : 'File will be converted to .jsonl before upload'}
							</p>
						</div>
					{/if} -->
				{/if}

				{#if !uploadedFile && !existingDatasetId}
					{#if existingDatasets.length > 0}
						<ContentSwitcher bind:selectedIndex={dataSourceIndex} style="margin-bottom: 0.75rem;">
							<Switch text="Upload" />
							<Switch text="Select Existing" />
						</ContentSwitcher>
					{/if}

					{#if dataSourceIndex === 0 || existingDatasets.length === 0}
						<div class="drop-zone">
							<FileUploaderDropContainer
								labelText="Drag and drop a file here or click to upload"
								accept={['.jsonl', '.json', '.csv', '.parquet']}
								on:change={async (e) => {
									if (e.detail && e.detail.length > 0) {
										await handleFileUpload(e.detail[0]);
									}
								}}
							/>
							<p class="drop-zone-hint">Accepted formats: .jsonl, .json, .csv, .parquet</p>
						</div>
					{:else}
						<Select
							labelText=""
							selected=""
							on:change={(e) => {
								const target = e.target;
								if (target instanceof HTMLSelectElement) handleExistingDatasetSelect(target.value);
							}}
						>
							<SelectItem value="" text="Choose a dataset..." />
							{#each existingDatasets as ds}
								<SelectItem
									value={ds.id}
									text="{ds.name} ({(ds.train_records || 0) +
										(ds.validation_records || 0)} records)"
								/>
							{/each}
						</Select>
					{/if}
				{:else if existingDatasetId}
					<FileUploaderItem
						name="{selectedExistingDataset?.name || ''} ({totalRecords.toLocaleString()} records)"
						status="edit"
						on:delete={resetForm}
					/>
				{:else if uploadedFile && !isSplitEnabled}
					<div class="file-row">
						<span class="file-label"
							>Train file <Tooltip
								>The main dataset used to train the model. This is where the model learns patterns
								from your data.</Tooltip
							></span
						>
						<FileUploaderItem name={uploadedFile.name} status="edit" on:delete={clearTrainFile} />
					</div>
				{/if}

				{#if isProcessing}
					<InlineLoading
						description={processingProgress || 'Processing...'}
						style="margin-top: var(--cds-spacing-03, 0.5rem);"
					/>
				{/if}

				{#if error}
					<InlineNotification
						kind="error"
						title="Error"
						subtitle={error}
						style="margin-top: var(--cds-spacing-03, 0.5rem);"
					/>
				{/if}

				{#if isPreviewDeferred && !existingDatasetId}
					<InlineNotification
						kind="info"
						lowContrast
						hideCloseButton
						title="Preview skipped"
						subtitle="This Parquet file is too large to preview in the browser, so only its column names were read. The rows below are blank placeholders — the file itself uploads and is validated in full on the server."
						style="margin-top: var(--cds-spacing-03, 0.5rem);"
					/>
				{/if}

				<!-- VALIDATION FILE (when split disabled) -->
				{#if uploadedFile && !existingDatasetId && !isSplitEnabled}
					<div style="margin-top: 0.75rem;">
						{#if !validationFile}
							<FileUploaderButton
								labelText="Upload Validation File"
								kind="secondary"
								size="small"
								accept={['.jsonl', '.json', '.csv', '.parquet']}
								on:change={async (e) => {
									if (e.detail && e.detail.length > 0) {
										validationFile = e.detail[0];
									}
								}}
							/>
						{:else}
							<div class="file-row">
								<span class="file-label"
									>Validation file <Tooltip
										>A separate dataset used to evaluate model performance during training. Helps
										detect overfitting.</Tooltip
									></span
								>
								<FileUploaderItem
									name={validationFile.name}
									status="edit"
									on:delete={() => {
										validationFile = null;
									}}
								/>
							</div>
						{/if}
					</div>
				{/if}

				<!-- ALGORITHM & COLUMN MAPPING (new uploads with parsed data) -->
				{#if uploadedFile && parsedData.length > 0 && userColumns.length > 0 && !existingDatasetId}
					<!-- {#if !datasetGoalWarning.valid}
						<InlineNotification
							kind="warning"
							title="Dataset Mismatch"
							subtitle={datasetGoalWarning.message}
							hideCloseButton
							lowContrast
							style="margin-bottom: 0.75rem;"
						/>
					{/if} -->

					<!-- {#if !isAiSuggesting && selectedGoal}
						<hr class="section-divider" />
						<div style="margin-top: var(--cds-spacing-03, 0.5rem);">
							<Dropdown
								labelText="Selected Algorithm"
								size="sm"
								selectedId={selectedAlgorithm}
								items={Utils.getAlgorithmsForGoal(selectedGoal).map((a) => ({ id: a.id, text: a.name }))}
								on:select={(e) => {
									selectedAlgorithm = e.detail.selectedItem.id;
								}}
							/>
						</div>
					{/if} -->

					<!-- Column Mapping -->
					<hr class="section-divider" />
					<div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
						<p class="section-title" style="margin: 0;">Column Mapping</p>
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
								AI Suggested ({Math.round(aiSuggestion.confidence * 100)}%) {showAiReasoning
									? '▴'
									: '▾'}
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
											<Tooltip>{colInfo.desc}</Tooltip>
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
							<!-- Fallback: dataset types not loaded, show requiredColumns without metadata -->
							{#each requiredColumns as reqCol}
								<div class="mapping-row">
									<div class="mapping-label">
										{Utils.toUpperCase(reqCol)}
										<span class="required-marker">*</span>
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

				<!-- RESET -->
				{#if uploadedFile || existingDatasetId}
					<!-- <hr class="section-divider" /> -->
					<Button
						kind="tertiary"
						size="small"
						style="margin-top: 0.5rem;"
						icon={Reset}
						on:click={resetForm}
					>
						Reset All
					</Button>
				{/if}
			</Tile>
		</Column>

		<!-- Right column: Preview -->
		<Column sm={4} md={4} lg={10}>
			{#if previewRows.length > 0 && previewHeaders.length > 0}
				<Tile class="preview-tile">
					{#if valPreviewRows.length > 0}
						<!-- Tabbed preview: Train + Validation -->
						<Tabs bind:selected={activePreviewTab}>
							<Tab label="Train ({trainRecordCount.toLocaleString()})" />
							<Tab label="Validation ({validationRecordCount.toLocaleString()})" />
							<svelte:fragment slot="content">
								<TabContent style="padding: 0.5rem 0;">
									<div style="overflow-x: auto;">
										<Table
											showActionButton={false}
											batchSelection={false}
											expandable={false}
											selectable={false}
											headers={previewHeaders}
											rows={previewRows}
											size="short"
											sortable={false}
										/>
									</div>
								</TabContent>
								<TabContent style="padding: 0.5rem 0;">
									<div style="overflow-x: auto;">
										<Table
											showActionButton={false}
											batchSelection={false}
											expandable={false}
											selectable={false}
											headers={valPreviewHeaders}
											rows={valPreviewRows}
											size="short"
											sortable={false}
										/>
									</div>
								</TabContent>
							</svelte:fragment>
						</Tabs>
					{:else}
						<!-- Train only (no validation available) -->
						<div
							style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;"
						>
							<h6 class="tile-heading">Data Preview</h6>
							<span class="helper-text-inline">
								{previewRows.length} of {totalRecords.toLocaleString()} records
							</span>
						</div>
						<div style="overflow-x: auto;">
							<DataTable
								headers={previewHeaders}
								rows={previewRows}
								size="short"
								sortable={false}
							/>
						</div>
					{/if}
				</Tile>
			{:else if !uploadedFile && !selectedExistingDataset}
				{@const examples =
					Object.keys($datasetTypes).length > 0
						? Utils.getDatasetExamplesFromTypes(selectedAlgorithm, $datasetTypes)
						: Utils.getDatasetExamples(selectedAlgorithm)}
				{@const algoDetail = Utils.ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm)}
				{@const typeKey = Utils.ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm]}
				{@const typeDesc = $datasetTypes[typeKey]?.desc}
				<Tile style="padding: 1.25rem;">
					<div
						style="display: flex; align-items: center; gap: var(--cds-spacing-03, 0.5rem); margin-bottom: 1rem;"
					>
						<h6 class="tile-heading" style="margin: 0;">Expected Dataset Format</h6>
						<!-- {#if algoDetail}
							<Tag type="blue" size="sm">{algoDetail.name}</Tag>
						{/if} -->
					</div>
					{#if typeDesc}
						<p class="helper-text-inline" style="margin-bottom: 0.75rem;">{typeDesc}</p>
					{/if}

					{@const typeColumns = $datasetTypes[typeKey]?.columns}
					{@const formats = typeColumns
						? Utils.generateFormatExamples(Object.values(typeColumns))
						: null}
					<Tabs>
						<Tab label="JSON" />
						<Tab label="JSONL" />
						<Tab label="CSV" />
						<Tab label="Parquet" />
						<svelte:fragment slot="content">
							<TabContent style="padding: 0.5rem 0;">
								<pre class="example-code">{formats?.json || JSON.stringify(examples, null, 2)}</pre>
							</TabContent>
							<TabContent style="padding: 0.5rem 0;">
								<pre class="example-code">{formats?.jsonl ||
										examples.map((ex) => JSON.stringify(ex)).join('\n')}</pre>
							</TabContent>
							<TabContent style="padding: 0.5rem 0;">
								<pre class="example-code">{formats?.csv ||
										examples.map((ex) => Object.values(ex).join(', ')).join('\n')}</pre>
							</TabContent>
							<TabContent style="padding: 0.5rem 0;">
								<pre
									class="example-code">Apache Parquet is a columnar storage format.{'\n'}Use the same column structure as CSV/JSON.{'\n'}Generate with: df.to_parquet("data.parquet")</pre>
							</TabContent>
						</svelte:fragment>
					</Tabs>

					<p class="empty-state-hint">Upload a dataset or select an existing one to get started.</p>
				</Tile>
			{/if}
		</Column>
	</Row>
</Grid>

<style>
	.tile-heading {
		font-weight: 600;
		margin: 0 0 1.25rem 0;
	}

	.section-title {
		font-size: 0.75rem;
		margin-bottom: 0.5rem;
		font-weight: 600;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		margin: 0 0 var(--cds-spacing-03, 0.5rem) 0;
	}

	.name-field {
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

	.algo-info-card {
		background: var(--cds-background-inverse, #262626);
		color: var(--cds-layer, #f4f4f4);
		padding: 0.75rem 1rem;
		border-radius: 4px;
	}

	.mapping-header {
		display: grid;
		grid-template-columns: 160px 1fr;
		gap: 0.75rem;
		font-size: 0.75rem;
		font-weight: 400;
		/* text-transform: uppercase; */
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		padding-bottom: 0.375rem;
		/* border-bottom: 1px solid var(--cds-border-subtle, #e0e0e0); */
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

	.example-label {
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		margin: 0 0 0.375rem 0;
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

	/* Ensure form inputs have visible background on white tiles */
	.name-field :global(.bx--text-input) {
		background-color: #f4f4f4 !important;
	}
	:global(.bx--select-input) {
		background-color: #f4f4f4 !important;
	}
	:global(.bx--list-box) {
		background-color: #f4f4f4 !important;
	}

	:global(.bx--tooltip .bx--tooltip__content) {
		max-width: 20rem;
		white-space: normal;
		font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
		font-size: 0.875rem;
		font-weight: 400;
		line-height: 1.4;
		letter-spacing: 0.16px;
	}
</style>
