<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Utils } from '$lib/utils';
	import { API } from '$lib/api';
	import {
		Form,
		Grid,
		Row,
		Column,
		TextInput,
		TextArea,
		FileUploader,
		ProgressIndicator,
		ProgressStep,
		ProgressBar,
		Slider,
		InlineLoading,
		Button,
		ButtonSet,
		CodeSnippet,
		Tile,
		InlineNotification
	} from 'carbon-components-svelte';
	import Table from '../Table.svelte';
	import { createEventDispatcher, onMount } from 'svelte';
	import type { DatasetForm, DatasetType } from '$lib/app-types';

	const api = new API();

	export let isUploading = false;
	export let uploadProgress = 0;
	export let dataset: DatasetForm = {
		name: '',
		description: '',
		train_file: null,
		validation_file: null
	};
	export let selectedTabId = 0; // 0: upload, 1: analyze, 2: split, 3: finalize

	type ProcessedResult = {
		train: DatasetType[];
		validation: DatasetType[];
	};

	// File upload state
	let uploadedFile: File | null = null;
	let rawFileContent: string = '';
	let fileFormat: string = '';
	let fileInfo = { size: 0, lines: 0 };

	// Analysis state
	let isAnalyzing = false;
	let parsingStrategy: any = null;
	let parsingError: string = '';
	let needsParsing = false;
	let allData: DatasetType[] = [];
	let isEditingStrategy = false;
	let editedInputPattern = '';
	let editedOutputPattern = '';
	let editedInputField = '';
	let editedOutputField = '';
	let sampleData: string | any[] = '';
	let isEditingSample = false;
	let editedSample = '';
	let customPrompt = '';
	let isEditingPrompt = false;
	let editedPrompt = '';

	// Field selection for structured data (JSON/CSV)
	let availableFields: string[] = [];
	let isSelectingFields = false;

	// Split state
	let splitMode: 'auto' | 'manual' = 'auto'; // auto = split single file, manual = upload separate validation
	let splitRatio = 80;
	let trainCount = 0;
	let validationCount = 0;
	let validationFile: File | null = null;
	let isProcessingValidation = false;
	let validationError = '';
	const processedResult: ProcessedResult = {
		train: [],
		validation: []
	};
	const dispatch = createEventDispatcher();

	const headers = [
		{ key: 'input', value: 'Input', width: '60%' },
		{ key: 'output', value: 'Output' }
	];

	onMount(() => {
		selectedTabId = 0;
	});

	// Convert data to JSONL string
	function dataToJsonl(data: DatasetType[]): string {
		return data.map((item) => JSON.stringify(item)).join('\n');
	}

	// Create a File object from data
	function createFileFromData(data: DatasetType[], filename: string): File {
		const jsonlContent = dataToJsonl(data);
		const blob = new Blob([jsonlContent], { type: 'application/jsonl' });
		return new File([blob], filename, { type: 'application/jsonl' });
	}

	// Split the data based on the ratio
	function splitData(data: DatasetType[], ratio: number) {
		const trainSize = Math.floor((data.length * ratio) / 100);
		const trainData = data.slice(0, trainSize);
		const validationData = data.slice(trainSize);

		processedResult.train = trainData;
		processedResult.validation = validationData;
		trainCount = trainData.length;
		validationCount = validationData.length;

		// Create actual File objects with split data
		const originalFilename = uploadedFile?.name.replace(/\.[^/.]+$/, '') || 'dataset';
		dataset.train_file = createFileFromData(trainData, `${originalFilename}_train.jsonl`);
		dataset.validation_file = createFileFromData(
			validationData,
			`${originalFilename}_validation.jsonl`
		);
	}

	// Handle file upload and intelligent parsing
	async function handleFileUpload(file: File) {
		try {
			uploadedFile = file;
			rawFileContent = await file.text();

			// Get file info
			fileInfo.size = file.size;
			fileInfo.lines = rawFileContent.split('\n').length;

			// Detect format
			const extension = file.name.split('.').pop()?.toLowerCase() || 'txt';
			fileFormat = extension === 'txt' ? 'text' : extension;

			// Auto-detect format from content if needed
			if (fileFormat === 'text' || !fileFormat) {
				fileFormat = Utils.detectFormatFromContent(rawFileContent);
			}

			// Auto-generate dataset name from filename
			if (!dataset.name) {
				dataset.name = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9-_]/g, '-');
			}

			// Extract and show sample immediately for user to review
			const sample = Utils.extractSample(rawFileContent, fileFormat);
			sampleData = sample;

			// Move to analyze step
			selectedTabId = 1;
		} catch (error: any) {
			parsingError = `Failed to read file: ${error.message}`;
			console.error('File reading error:', error);
		}
	}

	// Analyze file and generate parsing strategy
	async function analyzeFile() {
		if (!rawFileContent) return;

		isAnalyzing = true;
		parsingError = '';
		parsingStrategy = null;
		needsParsing = false;

		try {
			// Process file to check structure
			const { data, format } = await Utils.processAnyFile(uploadedFile!);
			fileFormat = format;

			// Check if data already has input-output structure
			if (Utils.hasInputOutputStructure(data)) {
				// Data is already in the right format
				allData = Utils.normalizeInputOutput(data);
				needsParsing = false;
				isAnalyzing = false;

				if (allData.length > 0) {
					splitData(allData, splitRatio);
					// Don't auto-advance - let user review the preview
				} else {
					parsingError = 'No valid data found in file';
				}
			} else {
				// Data needs parsing - get strategy from LLM
				needsParsing = true;

				// Use the sample that was already extracted or edited by user
				const sample = sampleData || Utils.extractSample(rawFileContent, format);
				if (!sampleData) {
					sampleData = sample; // Store if not already set
				}

				// Call LLM API (with custom prompt if provided)
				const strategy = await api.generateParsingStrategy(
					sample,
					format,
					customPrompt || undefined
				);
				parsingStrategy = strategy;

				// Apply the strategy
				if (format === 'text') {
					allData = Utils.applyParsingStrategy(rawFileContent, strategy);
				} else {
					allData = Utils.applyParsingStrategy(data, strategy);
				}

				isAnalyzing = false;

				if (allData.length === 0) {
					parsingError =
						'Failed to parse data with generated strategy. Please check your file format.';
				} else {
					splitData(allData, splitRatio);
					// Don't auto-advance - let user review strategy and preview first
				}
			}
		} catch (error: any) {
			parsingError = `Failed to generate parsing strategy: ${error.message}`;
			isAnalyzing = false;
			console.error('Parsing error:', error);
		}
	}

	// Retry parsing with the current strategy
	async function retryParsing() {
		if (!parsingStrategy) return;

		try {
			parsingError = '';
			allData = Utils.applyParsingStrategy(rawFileContent, parsingStrategy);

			if (allData.length > 0) {
				splitData(allData, splitRatio);
				selectedTabId = 2;
			} else {
				parsingError = 'Failed to parse data. No valid input-output pairs found.';
			}
		} catch (error: any) {
			parsingError = `Parsing failed: ${error.message}`;
		}
	}

	// Test strategy with validation
	async function testStrategy() {
		if (!parsingStrategy) return;

		try {
			const sample = Utils.extractSample(rawFileContent, fileFormat);
			const result = await api.validateParsingStrategy(parsingStrategy, sample);

			if (result.success) {
				console.log('Strategy validation successful:', result);
			} else {
				parsingError = result.errors.join(', ');
			}
		} catch (error: any) {
			console.error('Strategy test failed:', error);
		}
	}

	// Regenerate strategy with AI (call LLM again)
	async function regenerateStrategy() {
		if (!rawFileContent) return;

		isAnalyzing = true;
		parsingError = '';
		parsingStrategy = null;
		allData = [];

		try {
			// Use existing sample data if available (user may have edited it), otherwise extract fresh
			const sample = sampleData || Utils.extractSample(rawFileContent, fileFormat);

			// Store the sample if not already set
			if (!sampleData) {
				sampleData = sample;
			}

			// Call LLM API again (with custom prompt if provided)
			const strategy = await api.generateParsingStrategy(
				sample,
				fileFormat,
				customPrompt || undefined
			);
			parsingStrategy = strategy;

			// Apply the strategy
			if (fileFormat === 'text') {
				allData = Utils.applyParsingStrategy(rawFileContent, strategy);
			} else {
				const { data } = await Utils.processAnyFile(uploadedFile!);
				allData = Utils.applyParsingStrategy(data, strategy);
			}

			isAnalyzing = false;

			if (allData.length === 0) {
				parsingError = 'Failed to parse data with new strategy. Please try editing manually.';
			} else {
				splitData(allData, splitRatio);
			}
		} catch (error: any) {
			parsingError = `Failed to regenerate strategy: ${error.message}`;
			isAnalyzing = false;
			console.error('Regeneration error:', error);
		}
	}

	// Start editing the strategy
	function startEditingStrategy() {
		isEditingStrategy = true;

		// Populate edit fields with current values
		if (parsingStrategy.type === 'regex') {
			editedInputPattern = parsingStrategy.input_pattern || '';
			editedOutputPattern = parsingStrategy.output_pattern || '';
		} else if (parsingStrategy.type === 'direct_mapping') {
			editedInputField = parsingStrategy.input_field || '';
			editedOutputField = parsingStrategy.output_field || '';
		}
	}

	// Cancel editing
	function cancelEditingStrategy() {
		isEditingStrategy = false;
		editedInputPattern = '';
		editedOutputPattern = '';
		editedInputField = '';
		editedOutputField = '';
	}

	// Apply edited strategy
	async function applyEditedStrategy() {
		try {
			parsingError = '';

			// Update strategy with edited values
			if (parsingStrategy.type === 'regex') {
				parsingStrategy.input_pattern = editedInputPattern;
				parsingStrategy.output_pattern = editedOutputPattern;
			} else if (parsingStrategy.type === 'direct_mapping') {
				parsingStrategy.input_field = editedInputField;
				parsingStrategy.output_field = editedOutputField;
			}

			// Apply the updated strategy
			if (fileFormat === 'text') {
				allData = Utils.applyParsingStrategy(rawFileContent, parsingStrategy);
			} else {
				const { data } = await Utils.processAnyFile(uploadedFile!);
				allData = Utils.applyParsingStrategy(data, parsingStrategy);
			}

			if (allData.length > 0) {
				splitData(allData, splitRatio);
				isEditingStrategy = false;
				// Don't auto-advance - let user review the results
			} else {
				parsingError = 'No records parsed with edited strategy. Please check your patterns.';
			}
		} catch (error: any) {
			parsingError = `Error applying edited strategy: ${error.message}`;
		}
	}

	// Sample editing functions
	function startEditingSample() {
		isEditingSample = true;
		editedSample =
			typeof sampleData === 'string' ? sampleData : JSON.stringify(sampleData, null, 2);
	}

	function cancelEditingSample() {
		isEditingSample = false;
		editedSample = '';
	}

	async function applyEditedSample() {
		if (!editedSample.trim()) {
			parsingError = 'Sample data cannot be empty';
			return;
		}

		isAnalyzing = true;
		parsingError = '';

		try {
			// Update sample data
			sampleData = editedSample;

			// Regenerate strategy with new sample (with custom prompt if provided)
			const strategy = await api.generateParsingStrategy(
				editedSample,
				fileFormat,
				customPrompt || undefined
			);
			parsingStrategy = strategy;

			// Apply the strategy
			if (fileFormat === 'text') {
				allData = Utils.applyParsingStrategy(rawFileContent, strategy);
			} else {
				const { data } = await Utils.processAnyFile(uploadedFile!);
				allData = Utils.applyParsingStrategy(data, strategy);
			}

			isAnalyzing = false;

			if (allData.length > 0) {
				splitData(allData, splitRatio);
				isEditingSample = false;
			} else {
				parsingError = 'No records parsed with edited sample. Please check your data.';
			}
		} catch (error: any) {
			isAnalyzing = false;
			parsingError = `Error regenerating with edited sample: ${error.message}`;
		}
	}

	// Prompt editing functions
	function startEditingPrompt() {
		isEditingPrompt = true;
		editedPrompt = customPrompt;
	}

	function cancelEditingPrompt() {
		isEditingPrompt = false;
		editedPrompt = '';
	}

	function applyEditedPrompt() {
		customPrompt = editedPrompt.trim();
		isEditingPrompt = false;
	}

	// Field selection functions for structured data
	function extractAvailableFields(data: any[]): string[] {
		if (!data || data.length === 0) return [];

		const fields = new Set<string>();

		// Helper to recursively extract field paths
		function extractFieldPaths(obj: any, prefix: string = '') {
			if (typeof obj !== 'object' || obj === null) return;

			for (const key in obj) {
				const fieldPath = prefix ? `${prefix}.${key}` : key;
				const value = obj[key];

				// Add this field
				fields.add(fieldPath);

				// If it's an object (but not array), recurse
				if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
					extractFieldPaths(value, fieldPath);
				}
			}
		}

		// Extract from first few items to get comprehensive field list
		data.slice(0, 10).forEach((item) => {
			if (typeof item === 'object' && item !== null) {
				extractFieldPaths(item);
			}
		});

		return Array.from(fields).sort();
	}

	function startSelectingFields() {
		isSelectingFields = true;

		// Extract available fields from sample data
		if (Array.isArray(sampleData)) {
			availableFields = extractAvailableFields(sampleData);
		} else {
			// If sampleData is string, try to parse it
			try {
				const parsed = JSON.parse(sampleData as string);
				if (Array.isArray(parsed)) {
					availableFields = extractAvailableFields(parsed);
				}
			} catch {
				availableFields = [];
			}
		}

		// Pre-populate with AI-suggested fields if available
		if (parsingStrategy?.input_field) {
			editedInputField = parsingStrategy.input_field;
		}
		if (parsingStrategy?.output_field) {
			editedOutputField = parsingStrategy.output_field;
		}
	}

	function cancelSelectingFields() {
		isSelectingFields = false;
	}

	async function applySelectedFields() {
		if (!editedInputField || !editedOutputField) {
			parsingError = 'Please select both input and output fields';
			return;
		}

		try {
			parsingError = '';

			// Update or create strategy
			if (!parsingStrategy) {
				parsingStrategy = { type: 'direct_mapping' };
			}

			parsingStrategy.input_field = editedInputField;
			parsingStrategy.output_field = editedOutputField;
			parsingStrategy.description = `Map ${editedInputField} to input and ${editedOutputField} to output`;

			// Apply the strategy
			const { data } = await Utils.processAnyFile(uploadedFile!);
			allData = Utils.applyParsingStrategy(data, parsingStrategy);

			if (allData.length > 0) {
				splitData(allData, splitRatio);
				isSelectingFields = false;
				needsParsing = true;
			} else {
				parsingError = 'No records parsed with selected fields. Please check your field names.';
			}
		} catch (error: any) {
			parsingError = `Error applying field selection: ${error.message}`;
		}
	}

	// Handle validation file upload
	async function handleValidationFileUpload(file: File) {
		isProcessingValidation = true;
		validationError = '';

		try {
			validationFile = file;
			const content = await file.text();

			// Process validation file using same strategy as training file
			const { data, format } = await Utils.processAnyFile(file);

			// Check if data has input-output structure
			if (Utils.hasInputOutputStructure(data)) {
				processedResult.validation = Utils.normalizeInputOutput(data);
			} else if (parsingStrategy) {
				// Apply same parsing strategy as training data
				if (format === 'text') {
					processedResult.validation = Utils.applyParsingStrategy(content, parsingStrategy);
				} else {
					processedResult.validation = Utils.applyParsingStrategy(data, parsingStrategy);
				}
			} else {
				validationError =
					'Cannot parse validation file without a parsing strategy. Please analyze training data first.';
				isProcessingValidation = false;
				return;
			}

			validationCount = processedResult.validation.length;

			if (validationCount === 0) {
				validationError =
					'No valid records found in validation file. Please check the file format.';
			} else {
				// Update training data to use all data (no split)
				processedResult.train = allData;
				trainCount = allData.length;

				// Create File objects
				const originalFilename = uploadedFile?.name.replace(/\.[^/.]+$/, '') || 'dataset';
				dataset.train_file = createFileFromData(
					processedResult.train,
					`${originalFilename}_train.jsonl`
				);
				dataset.validation_file = createFileFromData(
					processedResult.validation,
					`${file.name.replace(/\.[^/.]+$/, '')}_validation.jsonl`
				);
			}

			isProcessingValidation = false;
		} catch (error: any) {
			validationError = `Failed to process validation file: ${error.message}`;
			isProcessingValidation = false;
			console.error('Validation file error:', error);
		}
	}

	// Navigate between steps
	function goToStep(step: number) {
		selectedTabId = step;
	}

	// React to ratio changes (only for auto split mode)
	$: if (allData.length > 0 && splitMode === 'auto') {
		splitData(allData, splitRatio);
	}

	// Format file size
	function formatFileSize(bytes: number): string {
		if (bytes < 1024) return bytes + ' B';
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
		return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	}
</script>

<Form>
	<Grid noGutter fullWidth>
		<Row>
			<Column>
				<ProgressIndicator bind:currentIndex={selectedTabId} spaceEqually>
					<ProgressStep
						complete={uploadedFile !== null}
						label="Upload File"
						description="Choose your dataset file"
					/>
					<ProgressStep
						disabled={!uploadedFile}
						complete={allData.length > 0}
						label="AI Analysis"
						description="Parse and preview data"
					/>
					<ProgressStep
						disabled={allData.length === 0}
						complete={processedResult.train.length > 0}
						label="Configure Split"
						description="Train/validation split"
					/>
					<ProgressStep
						disabled={!dataset.name || processedResult.train.length === 0}
						label="Finalize"
						description="Name and upload"
					/>
				</ProgressIndicator>

				<div style="padding: 1rem 0;">
					<!-- Step 0: Upload File -->
					{#if selectedTabId === 0}
						<Row>
							<Column sm={4} md={8} lg={16}>
								<Tile light style="padding: 2rem; text-align: center; min-height: 400px;">
									<div style="max-width: 600px; margin: 0 auto;">
										<div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
										<h4 style="margin-bottom: 1rem;">Upload Your Dataset</h4>
										<p style="color: #525252; margin-bottom: 2rem;">
											Upload any file format - we'll use AI to understand and parse it automatically
										</p>

										<FileUploader
											files={uploadedFile ? [uploadedFile] : []}
											labelTitle=""
											buttonLabel={!uploadedFile ? 'Select File' : 'Change File'}
											kind="primary"
											labelDescription="Supports .txt, .json, .jsonl, .csv, .xml and more"
											accept={['.txt', '.jsonl', '.json', '.csv', '.tsv', '.xml', '.log', '.md']}
											status="complete"
											on:change={async (e) => {
												if (e.detail && e.detail.length > 0) {
													await handleFileUpload(e.detail[0]);
												}
											}}
										/>

										<div style="margin-top: 2rem; text-align: left;">
											<p style="font-weight: 600; margin-bottom: 0.5rem;">✓ Supported formats:</p>
											<ul style="color: #525252; padding-left: 1.5rem;">
												<li>Text files (.txt, .log)</li>
												<li>Structured data (.json, .jsonl, .csv, .tsv)</li>
												<li>Markup files (.xml, .md)</li>
												<li>Any text-based format</li>
											</ul>
										</div>
									</div>
								</Tile>
							</Column>
						</Row>
					{/if}

					<!-- Step 1: AI Analysis -->
					{#if selectedTabId === 1}
						<Row>
							<Column md={4}>
								<Tile style="margin-bottom: 1rem;">
									<h6 style="font-weight: 600; margin-bottom: 0.5rem;">📄 File Information</h6>
									<p style="font-size: 0.875rem; color: #525252; margin: 0.25rem 0;">
										<strong>Name:</strong>
										{uploadedFile?.name}
									</p>
									<p style="font-size: 0.875rem; color: #525252; margin: 0.25rem 0;">
										<strong>Size:</strong>
										{formatFileSize(fileInfo.size)}
									</p>
									<p style="font-size: 0.875rem; color: #525252; margin: 0.25rem 0;">
										<strong>Lines:</strong>
										{fileInfo.lines.toLocaleString()}
									</p>
									<p style="font-size: 0.875rem; color: #525252; margin: 0.25rem 0;">
										<strong>Format:</strong>
										{fileFormat.toUpperCase()}
									</p>
								</Tile>

								{#if sampleData && !parsingStrategy}
									<Tile style="margin-top: 1rem;">
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
											📋 Sample Data for AI Analysis {isEditingSample ? '(Editing)' : ''}
										</h6>
										<p style="font-size: 0.75rem; color: #6f6f6f; margin-bottom: 0.5rem;">
											This sample will be sent to AI to generate the parsing strategy. You can edit
											it before analyzing.
										</p>

										{#if !isEditingSample}
											<div
												style="background: #f4f4f4; padding: 0.75rem; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; margin-bottom: 0.5rem;"
											>
												{typeof sampleData === 'string'
													? sampleData
													: JSON.stringify(sampleData, null, 2)}
											</div>
											<Button
												size="small"
												kind="ghost"
												on:click={startEditingSample}
												style="width: 100%;"
											>
												✏️ Edit Sample Before Analysis
											</Button>
										{:else}
											<TextArea
												bind:value={editedSample}
												rows={8}
												placeholder="Edit the sample data that will be sent to AI..."
												style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;"
											/>
											<ButtonSet style="margin-top: 0.5rem;">
												<Button size="small" kind="ghost" on:click={cancelEditingSample}>
													Cancel
												</Button>
												<Button
													size="small"
													kind="primary"
													on:click={() => {
														sampleData = editedSample;
														isEditingSample = false;
														editedSample = '';
													}}
												>
													Apply Changes
												</Button>
											</ButtonSet>
											<p
												style="font-size: 0.75rem; color: #6f6f6f; margin-top: 0.5rem; font-style: italic;"
											>
												💡 Click "Apply Changes" then "Analyze with AI" to use this sample
											</p>
										{/if}
									</Tile>
								{/if}

								{#if !isAnalyzing}
									<Tile style="margin-top: 1rem;">
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
											🎯 Additional AI Instructions {isEditingPrompt
												? '(Editing)'
												: customPrompt
												  ? '(Active)'
												  : '(Optional)'}
										</h6>

										{#if !isEditingPrompt}
											{#if customPrompt}
												<div
													style="background: #e8f4ff; padding: 0.75rem; border-radius: 4px; font-size: 0.75rem; max-height: 150px; overflow-y: auto; white-space: pre-wrap; margin-bottom: 0.5rem; border-left: 3px solid #0f62fe;"
												>
													{customPrompt}
												</div>
												<ButtonSet style="margin-top: 0.5rem;">
													<Button size="small" kind="ghost" on:click={startEditingPrompt}>
														✏️ Edit Instructions
													</Button>
													<Button
														size="small"
														kind="danger-ghost"
														on:click={() => (customPrompt = '')}
													>
														Clear Instructions
													</Button>
												</ButtonSet>
											{:else}
												<p style="font-size: 0.75rem; color: #6f6f6f; margin-bottom: 0.5rem;">
													Add specific instructions to guide the AI (e.g., "Extract Task: as input,
													next paragraph as output"). These will be added to the system prompt.
												</p>
												<Button
													size="small"
													kind="tertiary"
													on:click={startEditingPrompt}
													style="width: 100%;"
												>
													➕ Add Custom Instructions
												</Button>
											{/if}
										{:else}
											<TextArea
												bind:value={editedPrompt}
												rows={6}
												placeholder="Enter specific instructions... e.g., 'Extract the line starting with Task: as input, and the text block after it until the next Task: as output. Ensure multiline blocks are captured.'"
												style="font-size: 0.75rem; margin-bottom: 0.5rem;"
											/>
											<ButtonSet>
												<Button size="small" kind="ghost" on:click={cancelEditingPrompt}>
													Cancel
												</Button>
												<Button size="small" kind="primary" on:click={applyEditedPrompt}>
													Apply Instructions
												</Button>
											</ButtonSet>
											<p
												style="font-size: 0.75rem; color: #6f6f6f; margin-top: 0.5rem; font-style: italic;"
											>
												💡 Your instructions will be added to the system prompt as additional
												guidance for the AI
											</p>
										{/if}
									</Tile>
								{/if}

								{#if !isAnalyzing && !parsingStrategy && sampleData}
									<Button
										kind="primary"
										size="field"
										on:click={analyzeFile}
										style="width: 100%; margin-top: 1rem;"
									>
										🤖 Analyze with AI
									</Button>
								{/if}

								{#if sampleData && parsingStrategy && !isAnalyzing}
									<Tile style="margin-top: 1rem;">
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
											📋 Sample Used for Analysis {isEditingSample ? '(Editing)' : ''}
										</h6>

										{#if !isEditingSample}
											<div
												style="background: #f4f4f4; padding: 0.75rem; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; margin-bottom: 0.5rem;"
											>
												{typeof sampleData === 'string'
													? sampleData.substring(0, 500) + (sampleData.length > 500 ? '...' : '')
													: JSON.stringify(sampleData, null, 2).substring(0, 500)}
											</div>
											<Button
												size="small"
												kind="ghost"
												on:click={startEditingSample}
												style="width: 100%;"
											>
												✏️ Edit & Regenerate
											</Button>
										{:else}
											<TextArea
												bind:value={editedSample}
												rows={8}
												placeholder="Edit the sample data and click Apply to regenerate strategy..."
												style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;"
											/>
											<ButtonSet style="margin-top: 0.5rem;">
												<Button size="small" kind="ghost" on:click={cancelEditingSample}>
													Cancel
												</Button>
												<Button
													size="small"
													kind="primary"
													on:click={applyEditedSample}
													disabled={isAnalyzing}
												>
													Apply & Regenerate
												</Button>
											</ButtonSet>
											<p
												style="font-size: 0.75rem; color: #6f6f6f; margin-top: 0.5rem; font-style: italic;"
											>
												💡 This will regenerate the parsing strategy with the edited sample
											</p>
										{/if}
									</Tile>
								{/if}

								{#if isAnalyzing}
									<Tile style="padding: 1rem; background: #f4f4f4; margin-top: 1rem;">
										<InlineLoading description="Analyzing file structure..." />
										<p style="font-size: 0.875rem; color: #525252; margin-top: 0.5rem;">
											✓ Reading file content<br />
											✓ Detecting format<br />
											⏳ Generating parsing strategy...
										</p>
									</Tile>
								{/if}

								{#if needsParsing && parsingStrategy && !isAnalyzing}
									<Tile style="margin-top: 1rem; padding: 1rem;">
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">🤖 AI Parsing Strategy</h6>
										<p style="font-size: 0.875rem; color: #525252; margin-bottom: 0.5rem;">
											<strong>Type:</strong>
											{parsingStrategy.type || 'auto'}
										</p>
										<p style="font-size: 0.875rem; color: #525252; margin-bottom: 0.5rem;">
											{parsingStrategy.description}
										</p>

										{#if parsingStrategy.confidence}
											<p style="font-size: 0.875rem; color: #525252; margin-bottom: 0.5rem;">
												<strong>Confidence:</strong>
												{Math.round(parsingStrategy.confidence * 100)}%
											</p>
										{/if}

										{#if !isEditingStrategy && !isSelectingFields}
											<!-- Display mode -->
											{#if parsingStrategy.input_field && parsingStrategy.output_field}
												<CodeSnippet
													type="multi"
													code={`Input: ${parsingStrategy.input_field}\nOutput: ${parsingStrategy.output_field}`}
												/>
											{/if}

											{#if parsingStrategy.input_pattern && parsingStrategy.output_pattern}
												<details style="margin-top: 0.5rem;" open>
													<summary style="cursor: pointer; font-size: 0.875rem; font-weight: 600;">
														View Regex Patterns
													</summary>
													<CodeSnippet
														type="multi"
														code={`Input Pattern:\n${parsingStrategy.input_pattern}\n\nOutput Pattern:\n${parsingStrategy.output_pattern}`}
													/>
												</details>
											{/if}

											<!-- Action buttons -->
											<ButtonSet style="margin-top: 1rem;">
												<Button size="small" kind="tertiary" on:click={startEditingStrategy}>
													Edit Strategy
												</Button>
												{#if parsingStrategy.type === 'direct_mapping'}
													<Button size="small" kind="tertiary" on:click={startSelectingFields}>
														📋 Select Fields
													</Button>
												{/if}
												<Button
													size="small"
													kind="ghost"
													on:click={regenerateStrategy}
													disabled={isAnalyzing}
												>
													🤖 Regenerate with AI
												</Button>
											</ButtonSet>

											<p
												style="font-size: 0.75rem; color: #6f6f6f; margin-top: 0.5rem; font-style: italic;"
											>
												💡 Not satisfied? Click "Edit Strategy" to manually adjust patterns, or
												"Regenerate with AI" to get a new strategy.
											</p>
										{:else if isSelectingFields}
											<!-- Field Selection Mode -->
											<div
												style="margin-top: 1rem; padding: 1rem; background: #e8f4ff; border-radius: 4px; border-left: 3px solid #0f62fe;"
											>
												<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
													📋 Select Input/Output Fields
												</h6>
												<p style="font-size: 0.75rem; color: #525252; margin-bottom: 1rem;">
													Choose which fields from your data should be used as input and output.
													Supports nested fields using dot notation (e.g., "data.question").
												</p>

												<div style="margin-bottom: 1rem;">
													<label
														style="font-size: 0.875rem; font-weight: 600; display: block; margin-bottom: 0.25rem;"
													>
														Input Field:
													</label>
													<select
														bind:value={editedInputField}
														style="width: 100%; padding: 0.5rem; border: 1px solid #8d8d8d; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.875rem;"
													>
														<option value="">-- Select Input Field --</option>
														{#each availableFields as field}
															<option value={field}>{field}</option>
														{/each}
													</select>
												</div>

												<div style="margin-bottom: 1rem;">
													<label
														style="font-size: 0.875rem; font-weight: 600; display: block; margin-bottom: 0.25rem;"
													>
														Output Field:
													</label>
													<select
														bind:value={editedOutputField}
														style="width: 100%; padding: 0.5rem; border: 1px solid #8d8d8d; border-radius: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 0.875rem;"
													>
														<option value="">-- Select Output Field --</option>
														{#each availableFields as field}
															<option value={field}>{field}</option>
														{/each}
													</select>
												</div>

												{#if availableFields.length === 0}
													<p style="font-size: 0.75rem; color: #da1e28; margin-top: 0.5rem;">
														⚠️ No fields detected. Make sure your data is properly formatted JSON or
														CSV.
													</p>
												{/if}

												<ButtonSet>
													<Button size="small" kind="secondary" on:click={cancelSelectingFields}>
														Cancel
													</Button>
													<Button
														size="small"
														kind="primary"
														on:click={applySelectedFields}
														disabled={!editedInputField || !editedOutputField}
													>
														Apply Fields
													</Button>
												</ButtonSet>
											</div>
										{:else if isEditingStrategy}
											<!-- Edit mode -->
											<div
												style="margin-top: 1rem; padding: 1rem; background: #f4f4f4; border-radius: 4px;"
											>
												<h6 style="font-weight: 600; margin-bottom: 0.5rem;">✏️ Edit Strategy</h6>

												{#if parsingStrategy.type === 'regex'}
													<TextArea
														labelText="Input Pattern (Regex)"
														bind:value={editedInputPattern}
														placeholder="e.g., Task:\s*(.+?)(?=\n\n@obj)"
														rows={3}
														helperText="Regex pattern to capture input text"
													/>
													<TextArea
														labelText="Output Pattern (Regex)"
														bind:value={editedOutputPattern}
														placeholder={'e.g., @obj Person \\{[^}]*\\}'}
														rows={3}
														helperText="Regex pattern to capture output text"
														style="margin-top: 0.5rem;"
													/>
												{:else if parsingStrategy.type === 'direct_mapping'}
													<TextInput
														labelText="Input Field"
														bind:value={editedInputField}
														placeholder="e.g., question, prompt, input"
														helperText="Field name for input data"
													/>
													<TextInput
														labelText="Output Field"
														bind:value={editedOutputField}
														placeholder="e.g., answer, completion, output"
														helperText="Field name for output data"
														style="margin-top: 0.5rem;"
													/>
												{/if}

												<ButtonSet style="margin-top: 1rem;">
													<Button size="small" kind="secondary" on:click={cancelEditingStrategy}>
														Cancel
													</Button>
													<Button size="small" kind="primary" on:click={applyEditedStrategy}>
														Apply Changes
													</Button>
												</ButtonSet>
											</div>
										{/if}
									</Tile>
								{/if}

								{#if parsingError}
									<InlineNotification
										kind="error"
										title="Parsing Error"
										subtitle={parsingError}
										style="margin-top: 1rem;"
									/>

									<ButtonSet style="margin-top: 1rem;">
										{#if parsingStrategy}
											<Button size="small" kind="tertiary" on:click={startEditingStrategy}>
												Edit Strategy
											</Button>
											<Button size="small" kind="tertiary" on:click={retryParsing}>
												Retry with Current
											</Button>
										{/if}
										<Button
											size="small"
											kind="primary"
											on:click={regenerateStrategy}
											disabled={isAnalyzing}
										>
											🤖 Regenerate with AI
										</Button>
									</ButtonSet>
								{/if}

								{#if allData.length > 0 && !isAnalyzing}
									<InlineNotification
										kind="success"
										title="Success!"
										subtitle={`Successfully parsed ${allData.length} records`}
										style="margin-top: 1rem;"
									/>
									<Button
										kind="primary"
										size="field"
										on:click={() => goToStep(2)}
										style="width: 100%; margin-top: 1rem;"
									>
										Next: Configure Split →
									</Button>
								{/if}
							</Column>

							<Column md={4} style="min-height: 400px; max-height: 600px; overflow-y: auto;">
								<h6 style="font-weight: 600; margin-bottom: 1rem;">👁️ Preview (First 5 Records)</h6>
								{#if allData.length > 0}
									<Table
										rows={allData.slice(0, 5).map((item, index) => ({ id: index, ...item }))}
										{headers}
										batchSelection={false}
										selectable={false}
										expandable={false}
										showActionButton={false}
									/>
								{:else if isAnalyzing}
									<Tile light style="padding: 2rem; text-align: center;">
										<p style="color: #525252;">Analyzing file...</p>
									</Tile>
								{:else}
									<Tile light style="padding: 2rem; text-align: center;">
										<p style="color: #525252;">Click "Analyze with AI" to preview your data</p>
									</Tile>
								{/if}
							</Column>
						</Row>
					{/if}

					<!-- Step 2: Configure Split -->
					{#if selectedTabId === 2}
						<Row>
							<Column md={3}>
								<Tile>
									<h6 style="font-weight: 600; margin-bottom: 1rem;">⚙️ Configure Data Split</h6>

									<!-- Split Mode Selection -->
									<div style="margin-bottom: 1.5rem;">
										<p style="font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem;">
											Split Method:
										</p>
										<div style="display: flex; gap: 0.5rem;">
											<Button
												kind={splitMode === 'auto' ? 'primary' : 'tertiary'}
												size="small"
												on:click={() => {
													splitMode = 'auto';
													validationFile = null;
													validationError = '';
													// Re-apply auto split
													splitData(allData, splitRatio);
												}}
												style="flex: 1;"
											>
												📊 Auto Split
											</Button>
											<Button
												kind={splitMode === 'manual' ? 'primary' : 'tertiary'}
												size="small"
												on:click={() => {
													splitMode = 'manual';
													// When switching to manual, use all data for training initially
													processedResult.train = allData;
													trainCount = allData.length;
													if (!validationFile) {
														processedResult.validation = [];
														validationCount = 0;
													}
												}}
												style="flex: 1;"
											>
												📁 Upload Validation
											</Button>
										</div>
									</div>

									{#if splitMode === 'auto'}
										<!-- Auto Split Mode -->
										<Slider
											labelText="Train / Validation Ratio"
											min={10}
											max={90}
											step={5}
											maxLabel="{splitRatio}:{100 - splitRatio}"
											bind:value={splitRatio}
										/>

										<div
											style="margin-top: 1rem; padding: 1rem; background: #f4f4f4; border-radius: 4px;"
										>
											<p style="font-size: 0.875rem; margin: 0.25rem 0;">
												<strong>Total Records:</strong>
												{allData.length}
											</p>
											<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #0f62fe;">
												<strong>🔵 Training:</strong>
												{trainCount} records ({splitRatio}%)
											</p>
											<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #198038;">
												<strong>🟢 Validation:</strong>
												{validationCount} records ({100 - splitRatio}%)
											</p>
										</div>

										{#if allData.length < 100}
											<InlineNotification
												kind="info"
												lowContrast
												title="Small Dataset"
												subtitle="For datasets with fewer than 100 records, we recommend a 90/10 split to maximize training data."
												style="margin-top: 1rem;"
											/>
										{/if}

										<div style="margin-top: 1.5rem;">
											<div
												style="height: 20px; background: linear-gradient(to right, #0f62fe {splitRatio}%, #198038 {splitRatio}%); border-radius: 4px;"
											/>
											<div
												style="display: flex; justify-content: space-between; margin-top: 0.25rem;"
											>
												<span style="font-size: 0.75rem; color: #0f62fe;">Train</span>
												<span style="font-size: 0.75rem; color: #198038;">Validation</span>
											</div>
										</div>
									{:else}
										<!-- Manual Upload Mode -->
										<div
											style="margin-top: 1rem; padding: 1rem; background: #e8f4ff; border-radius: 4px; border-left: 3px solid #0f62fe;"
										>
											<p style="font-size: 0.875rem; margin-bottom: 0.5rem;">
												<strong>📄 Training Data:</strong>
												{uploadedFile?.name}
											</p>
											<p style="font-size: 0.875rem; color: #0f62fe;">
												All {allData.length} records will be used for training
											</p>
										</div>

										<div style="margin-top: 1rem;">
											<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
												Upload Validation Dataset
											</h6>
											<p style="font-size: 0.75rem; color: #525252; margin-bottom: 0.5rem;">
												Upload a separate file to use as validation data. It should have the same
												format as your training file.
											</p>

											<FileUploader
												files={validationFile ? [validationFile] : []}
												labelTitle=""
												buttonLabel={!validationFile ? 'Select Validation File' : 'Change File'}
												kind="tertiary"
												labelDescription="Same format as training file"
												accept={['.txt', '.jsonl', '.json', '.csv', '.tsv', '.xml', '.log', '.md']}
												status="complete"
												on:change={async (e) => {
													if (e.detail && e.detail.length > 0) {
														await handleValidationFileUpload(e.detail[0]);
													}
												}}
											/>

											{#if isProcessingValidation}
												<InlineLoading
													description="Processing validation file..."
													style="margin-top: 0.5rem;"
												/>
											{/if}

											{#if validationError}
												<InlineNotification
													kind="error"
													title="Validation Error"
													subtitle={validationError}
													style="margin-top: 0.5rem;"
													lowContrast
												/>
											{/if}

											{#if validationFile && validationCount > 0}
												<div
													style="margin-top: 1rem; padding: 1rem; background: #d0f0e4; border-radius: 4px; border-left: 3px solid #198038;"
												>
													<p style="font-size: 0.875rem; margin-bottom: 0.25rem;">
														<strong>✓ Validation File:</strong>
														{validationFile.name}
													</p>
													<p style="font-size: 0.875rem; color: #198038;">
														{validationCount} records loaded
													</p>
												</div>
											{/if}
										</div>

										<div
											style="margin-top: 1rem; padding: 1rem; background: #f4f4f4; border-radius: 4px;"
										>
											<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #0f62fe;">
												<strong>🔵 Training:</strong>
												{trainCount} records
											</p>
											<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #198038;">
												<strong>🟢 Validation:</strong>
												{validationCount} records
											</p>
										</div>
									{/if}

									<ButtonSet style="margin-top: 2rem;">
										<Button kind="secondary" size="field" on:click={() => goToStep(1)}>
											← Back
										</Button>
										<Button
											kind="primary"
											size="field"
											on:click={() => goToStep(3)}
											disabled={!dataset.name || (splitMode === 'manual' && validationCount === 0)}
										>
											Next: Finalize →
										</Button>
									</ButtonSet>

									{#if splitMode === 'manual' && validationCount === 0}
										<p
											style="font-size: 0.75rem; color: #da1e28; margin-top: 0.5rem; font-style: italic;"
										>
											⚠️ Please upload a validation file to continue
										</p>
									{/if}
								</Tile>
							</Column>

							<Column md={5}>
								<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
									<div>
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
											🔵 Training Data ({trainCount})
										</h6>
										<div style="max-height: 500px; overflow-y: auto;">
											<Table
												rows={processedResult.train.map((item, index) => ({
													id: index,
													...item
												}))}
												{headers}
												batchSelection={false}
												selectable={false}
												expandable={false}
												showActionButton={false}
											/>
										</div>
									</div>

									<div>
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">
											🟢 Validation Data ({validationCount})
										</h6>
										<div style="max-height: 500px; overflow-y: auto;">
											<Table
												rows={processedResult.validation.map((item, index) => ({
													id: index,
													...item
												}))}
												{headers}
												batchSelection={false}
												selectable={false}
												expandable={false}
												showActionButton={false}
											/>
										</div>
									</div>
								</div>
							</Column>
						</Row>
					{/if}

					<!-- Step 3: Finalize -->
					{#if selectedTabId === 3}
						<Row>
							<Column md={4}>
								<Tile>
									<h6 style="font-weight: 600; margin-bottom: 1rem;">📝 Finalize Dataset</h6>

									<TextInput
										labelText="Dataset Name *"
										bind:value={dataset.name}
										placeholder="my-dataset"
										required
										helperText="Use lowercase letters, numbers, hyphens, and underscores"
									/>

									<TextArea
										labelText="Description (optional)"
										bind:value={dataset.description}
										placeholder="Describe your dataset and its purpose..."
										style="margin-top: 1rem;"
										rows={3}
									/>

									<Tile light style="margin-top: 1.5rem; padding: 1rem;">
										<h6 style="font-weight: 600; margin-bottom: 0.5rem;">📋 Summary</h6>
										<p style="font-size: 0.875rem; margin: 0.25rem 0;">
											<strong>Training File:</strong>
											{uploadedFile?.name} ({formatFileSize(fileInfo.size)})
										</p>
										{#if splitMode === 'manual' && validationFile}
											<p style="font-size: 0.875rem; margin: 0.25rem 0;">
												<strong>Validation File:</strong>
												{validationFile.name}
											</p>
										{/if}
										<p style="font-size: 0.875rem; margin: 0.25rem 0;">
											<strong>Format:</strong>
											{fileFormat.toUpperCase()}
										</p>
										<p style="font-size: 0.875rem; margin: 0.25rem 0;">
											<strong>Parsing:</strong>
											{parsingStrategy?.type || 'direct'}
										</p>
										<p style="font-size: 0.875rem; margin: 0.25rem 0;">
											<strong>Split Method:</strong>
											{splitMode === 'auto'
												? `Auto (${splitRatio}:${100 - splitRatio})`
												: 'Manual Upload'}
										</p>
										<p style="font-size: 0.875rem; margin: 0.25rem 0;">
											<strong>Total Records:</strong>
											{allData.length}
										</p>
										<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #0f62fe;">
											<strong>Training:</strong>
											{trainCount} records
										</p>
										<p style="font-size: 0.875rem; margin: 0.25rem 0; color: #198038;">
											<strong>Validation:</strong>
											{validationCount} records
										</p>
									</Tile>

									{#if isUploading}
										<div style="margin-top: 1.5rem;">
											<ProgressBar
												labelText="Uploading dataset..."
												value={uploadProgress}
												max={100}
												size="sm"
											/>
											<p style="text-align: center; margin-top: 0.5rem; font-size: 0.875rem;">
												{Math.round(uploadProgress)}% complete
											</p>
										</div>
									{:else}
										<ButtonSet style="margin-top: 2rem;">
											<Button kind="secondary" size="field" on:click={() => goToStep(2)}>
												← Back
											</Button>
											<Button
												kind="primary"
												size="field"
												disabled={!dataset.name || trainCount === 0}
												on:click={() => dispatch('createDataset', { dataset, processedResult })}
											>
												🚀 Create Dataset
											</Button>
										</ButtonSet>
									{/if}
								</Tile>
							</Column>

							<Column md={4}>
								<h6 style="font-weight: 600; margin-bottom: 1rem;">👁️ Final Preview</h6>
								<p style="font-size: 0.875rem; color: #525252; margin-bottom: 1rem;">
									Here's a sample of what will be uploaded:
								</p>
								<Table
									rows={[
										...processedResult.train.slice(0, 3).map((item, index) => ({
											id: `train-${index}`,
											...item,
											_type: '🔵 Train'
										})),
										...processedResult.validation.slice(0, 2).map((item, index) => ({
											id: `val-${index}`,
											...item,
											_type: '🟢 Val'
										}))
									]}
									headers={[{ key: '_type', value: 'Type' }, ...headers]}
									batchSelection={false}
									selectable={false}
									expandable={false}
									showActionButton={false}
								/>
							</Column>
						</Row>
					{/if}
				</div>
			</Column>
		</Row>
	</Grid>
</Form>
