<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		ProgressIndicator,
		ProgressStep,
		Button,
		Grid,
		Row,
		Column,
		InlineLoading,
		InlineNotification,
		Breadcrumb,
		BreadcrumbItem
	} from 'carbon-components-svelte';
	import { ArrowLeft, ArrowRight, Rocket, Close } from 'carbon-icons-svelte';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { API, DatasetUploadError } from '$lib/api';
	import { Utils } from '$lib/utils';
	import { appState, configurations, datasets, datasetTypes } from '$lib/app';
	import { userMetadata, currentUser } from '$lib/store';
	import { ModelSource } from '$lib/app-types';
	import type {
		DatasetForm,
		DatasetFormatType,
		ColumnMetadata,
		ParsedDataRow,
		Dataset,
		Configuration,
		TuningForm,
		PendingConfigData,
		PendingConfigUpdate,
		LaunchPhase,
		TuningGoal,
		Resources,
		WizardDraft,
		HuggingFaceModel
	} from '$lib/app-types';

	import Step0GetStarted from './steps/Step0GetStarted.svelte';
	import Step1DatasetUpload from './steps/Step1DatasetUpload.svelte';
	import Step2Configure from './steps/Step2Configure.svelte';
	import StepRewardFunction from './steps/StepRewardFunction.svelte';
	import Step3ReviewLaunch from './steps/Step3ReviewLaunch.svelte';

	const api = new API();
	const DRAFT_KEY = 'autotunex_wizard_draft';
	const DRAFT_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours

	// Step tracking
	let currentStep = 0;
	let completedSteps = [false, false, false, false, false]; // 5 slots (max with reward step)

	// Reward function step is shown only for Online RL
	$: hasRewardStep = selectedGoal === 'online_rl';
	$: totalSteps = hasRewardStep ? 5 : 4;
	$: lastStepIndex = totalSteps - 1;
	// Logical step mapping: 0=GetStarted, 1=Dataset, 2=Configure, 3=RewardFn(conditional), lastStepIndex=Review

	// Step 0: Get Started state
	let selectedGoal: TuningGoal | null = null;

	// Step 1: Dataset state
	let uploadedFile: File | null = null;
	let parsedData: ParsedDataRow[] = [];
	let columnMetadata: ColumnMetadata[] = [];
	let detectedFormat: DatasetFormatType = 'unknown';
	let datasetForm: DatasetForm = {
		name: '',
		description: '',
		train_file: null,
		validation_file: null
	};
	let totalRecords = 0;
	let datasetId: string | null = null;
	let existingDatasetId: string | null = null;
	let selectedExistingDataset: Dataset | null = null;
	let splitRatio = 80;
	let validationFile: File | null = null;
	let isSplitEnabled = true;
	let selectedAlgorithm = 'lora';
	let columnMapping: Record<string, string> = {};
	let isDatasetCompatible = true;
	let keepAsParquet = true;

	// Step 2: Config state
	let selectedConfigId: string | null = null;
	let selectedConfig: Configuration | null = null;
	let pendingNewConfig: PendingConfigData | null = null;
	let pendingConfigUpdate: PendingConfigUpdate | null = null;
	let isEditingConfig = false;
	let isCreatingConfig = false;

	// Step 2.5: Reward function state (Online RL only)
	let rewardFunctionCode = '';
	let rewardFunctionName = 'compute_score';
	let allTestsPassed = false;

	// Step 0: Model state (now in Get Started step)
	let selectedModel = 'ibm-granite/granite-4.0-h-micro';
	let modelSource: ModelSource = ModelSource.HuggingFace;
	let autotuneEnabled = true;

	// Step 3: Launch state
	let experimentName = '';
	let isLaunching = false;

	// Launch progress
	let transitionError = '';
	// Not an error: the dataset uploaded but is still being validated server-side
	// when our bounded client-side poll runs out. See handleLaunch.
	let datasetPendingNotice = '';
	let uploadProgress = 0;
	let launchPhase: LaunchPhase = null;

	// Idempotent retry: track resources already created on failed launch
	let createdDatasetId: string | null = null;
	let createdConfigId: string | null = null;
	let createdJobId: string | null = null;

	// Resource estimation
	let resourceEstimation: Resources | null = null;

	// Draft restore state
	let showDraftRestore = false;
	let pendingDraft: WizardDraft | null = null;

	// Pre-fetched models for Step0
	let prefetchedModels: HuggingFaceModel[] | null = null;

	// Task 8: Pre-fetch parallel API calls on wizard open
	if (typeof window !== 'undefined') {
		// Fire eagerly at script init time, before children mount
		if (!$appState.isDatasetsLoaded) {
			api
				.getDatasets()
				.then((data) => {
					datasets.set(data);
					appState.update((prev) => ({ ...prev, isDatasetsLoaded: true }));
				})
				.catch(() => {});
		}
		if (!$appState.isConfigurationsLoaded) {
			api
				.getConfigurations()
				.then((data) => {
					configurations.set(data);
					appState.update((prev) => ({ ...prev, isConfigurationsLoaded: true }));
				})
				.catch(() => {});
		}
		// Pre-fetch models for Step0
		api
			.getHFModels('ibm-granite/granite-4.0-h-micro', 20)
			.then((data) => {
				prefetchedModels = data;
			})
			.catch(() => {});
	}

	// Reset dataset & config state when tuning goal changes (user changed their mind in Step 0)
	let prevGoal: TuningGoal | null = selectedGoal;
	$: if (
		selectedGoal !== null &&
		prevGoal !== null &&
		selectedGoal !== prevGoal &&
		currentStep === 0
	) {
		// Reset Step 1: Dataset state
		uploadedFile = null;
		parsedData = [];
		columnMetadata = [];
		detectedFormat = 'unknown';
		datasetForm = { name: '', description: '', train_file: null, validation_file: null };
		totalRecords = 0;
		datasetId = null;
		existingDatasetId = null;
		splitRatio = 80;
		validationFile = null;
		isSplitEnabled = true;
		columnMapping = {};

		// Reset Step 2: Config state
		selectedConfigId = null;
		selectedConfig = null;
		pendingNewConfig = null;
		pendingConfigUpdate = null;

		// Reset Step 3: Launch state
		experimentName = '';
		resourceEstimation = null;

		// Reset reward function state
		rewardFunctionCode = '';
		rewardFunctionName = 'compute_score';
		allTestsPassed = false;

		// Reset step completion (keep step 0 as still valid)
		completedSteps = [false, false, false, false, false];

		prevGoal = selectedGoal;
	} else {
		prevGoal = selectedGoal;
	}

	// Task 5: Sync goal when algorithm changes — only on Step 0 (user picking goal).
	// On later steps, dataset validation warnings handle mismatches instead.
	$: {
		if (selectedAlgorithm && currentStep === 0) {
			const algo =
				Utils.ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm) ||
				Utils.ALGORITHM_OPTIONS.find((a) => a.id === selectedAlgorithm);
			if (algo && algo.category !== selectedGoal) selectedGoal = algo.category;
		}
	}

	// Derived: can proceed to next step?
	$: canProceed = (() => {
		switch (currentStep) {
			case 0:
				return (
					selectedGoal !== null && selectedAlgorithm !== '' && (selectedModel ?? '').trim() !== ''
				);
			case 1: {
				// Dataset ready + all columns mapped + name provided (warning shown if format mismatches goal)
				const hasDataset = existingDatasetId !== null || parsedData.length > 0;
				const hasName = datasetForm.name.trim() !== '';
				const requiredCols =
					Object.keys($datasetTypes).length > 0
						? Utils.getRequiredColumnsFromTypes(selectedAlgorithm, $datasetTypes)
						: Utils.getRequiredColumns(selectedAlgorithm);
				const allMapped = requiredCols.every((c) => columnMapping[c]);
				const hasValidation =
					existingDatasetId !== null || isSplitEnabled || validationFile !== null;
				return hasDataset && hasName && allMapped && hasValidation;
			}
			case 2:
				return selectedConfigId !== null && !isEditingConfig && !isCreatingConfig;
			case 3:
				if (hasRewardStep) {
					// This is the reward function step — require all test cases to pass
					return (
						rewardFunctionCode.trim().length > 0 &&
						rewardFunctionName.trim().length > 0 &&
						allTestsPassed
					);
				}
				// This is the review step (no reward step)
				return experimentName.trim() !== '' && !isLaunching;
			case 4:
				// Review step when reward step is present
				return experimentName.trim() !== '' && !isLaunching;
			default:
				return false;
		}
	})();

	// Breadcrumb trail: progressively shows selections from completed steps
	$: breadcrumbItems = (() => {
		const items: { label: string; step: number }[] = [];
		if (!completedSteps[0] || currentStep === 0) return items;

		// Step 0 selections: Goal > Algorithm > Model
		if (selectedGoal) {
			const goalLabels: Record<string, string> = {
				sft: 'SFT',
				offline_rl: 'Offline RL',
				online_rl: 'Online RL'
			};
			items.push({ label: goalLabels[selectedGoal] || selectedGoal, step: 0 });
		}
		// if (selectedAlgorithm) {
		// 	const algo = Utils.ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm);
		// 	items.push({ label: algo?.name || selectedAlgorithm.toUpperCase(), step: 0 });
		// }
		if (selectedModel) {
			items.push({ label: selectedModel.split('/').pop() || selectedModel, step: 0 });
		}

		// Step 1: Dataset name
		if (completedSteps[1] && currentStep > 1) {
			items.push({ label: datasetForm.name || 'Dataset', step: 1 });
		}

		// Step 2: Config name
		if (completedSteps[2] && currentStep > 2) {
			items.push({ label: selectedConfig?.name || pendingNewConfig?.name || 'Config', step: 2 });
		}

		return items;
	})();

	// Task 6: Debounced localStorage draft save
	let saveDraftTimeout: ReturnType<typeof setTimeout>;
	$: {
		// Trigger on any wizard state change
		const _trigger = [
			currentStep,
			selectedGoal,
			selectedAlgorithm,
			selectedModel,
			modelSource,
			datasetForm.name,
			datasetForm.description,
			existingDatasetId,
			splitRatio,
			selectedConfigId,
			experimentName,
			autotuneEnabled
		];
		if (typeof window !== 'undefined' && selectedGoal) {
			clearTimeout(saveDraftTimeout);
			saveDraftTimeout = setTimeout(() => {
				saveDraft();
			}, 500);
		}
	}

	function saveDraft() {
		try {
			const draft: WizardDraft = {
				savedAt: new Date().toISOString(),
				currentStep,
				completedSteps,
				selectedGoal,
				selectedAlgorithm,
				selectedModel,
				modelSource,
				datasetForm: { name: datasetForm.name, description: datasetForm.description },
				existingDatasetId,
				splitRatio,
				selectedConfigId: selectedConfigId === '__pending__' ? null : selectedConfigId,
				experimentName,
				autotuneEnabled
			};
			localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
		} catch {
			// Silently ignore storage errors
		}
	}

	function restoreDraft(draft: WizardDraft) {
		currentStep = draft.currentStep;
		completedSteps = draft.completedSteps;
		selectedGoal = draft.selectedGoal;
		selectedAlgorithm = draft.selectedAlgorithm;
		selectedModel = draft.selectedModel;
		modelSource = draft.modelSource;
		datasetForm.name = draft.datasetForm.name;
		datasetForm.description = draft.datasetForm.description;
		existingDatasetId = draft.existingDatasetId;
		splitRatio = draft.splitRatio;
		selectedConfigId = draft.selectedConfigId;
		experimentName = draft.experimentName;
		autotuneEnabled = draft.autotuneEnabled ?? true;
		showDraftRestore = false;
		pendingDraft = null;
	}

	function discardDraft() {
		try {
			localStorage.removeItem(DRAFT_KEY);
		} catch {}
		showDraftRestore = false;
		pendingDraft = null;
	}

	function clearDraft() {
		try {
			localStorage.removeItem(DRAFT_KEY);
		} catch {}
	}

	onMount(() => {
		// Task 6: Check for saved draft
		try {
			const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(DRAFT_KEY) : null;
			if (raw) {
				const draft: WizardDraft = JSON.parse(raw);
				const age = Date.now() - new Date(draft.savedAt).getTime();
				if (age < DRAFT_MAX_AGE_MS && draft.selectedGoal) {
					pendingDraft = draft;
					showDraftRestore = true;
				} else {
					localStorage.removeItem(DRAFT_KEY);
				}
			}
		} catch {
			// Ignore parse errors
		}
	});

	function goToStep(step: number) {
		if (step < 0 || step > lastStepIndex) return;
		// Allow going back to any completed step or one step forward if current is complete
		if (step <= currentStep || completedSteps[step - 1]) {
			currentStep = step;
		}
	}

	async function handleNext() {
		transitionError = '';

		if (currentStep === 0) {
			// Step 0 → Step 1: just mark complete, advance
			completedSteps[0] = true;
			currentStep = 1;
		} else if (currentStep === 1) {
			// Step 1 → Step 2: validate dataset, advance
			if (existingDatasetId) {
				datasetId = existingDatasetId;
			}
			completedSteps[1] = true;
			currentStep = 2;
		} else if (currentStep === 2) {
			// Step 2 → Step 3 (Reward Function if Online RL) or Review
			if (selectedConfigId && selectedConfigId !== '__pending__' && !selectedConfig) {
				try {
					selectedConfig = await api.getConfiguration(selectedConfigId);
				} catch (err: any) {
					transitionError = 'Failed to load configuration details.';
					return;
				}
			}
			completedSteps[2] = true;

			if (hasRewardStep) {
				// Go to reward function step (step 3)
				currentStep = 3;
			} else {
				// Go directly to review (step 3, which is the last step)
				currentStep = 3;
				prepareReviewStep();
			}
		} else if (currentStep === 3 && hasRewardStep) {
			// Step 3 (Reward Function) → Step 4 (Review)
			completedSteps[3] = true;
			currentStep = 4;
			prepareReviewStep();
		}
	}

	function prepareReviewStep() {
		if (!experimentName) {
			const modelShort = selectedModel.split('/').pop() || selectedModel;
			const configName = selectedConfig?.name || pendingNewConfig?.name || 'config';
			experimentName = `${modelShort}_${configName}`.substring(0, 50);
		}

		// Non-blocking resource estimation. A pending (unsaved) config has no id yet, so
		// send its composed config_data + tuner types inline instead of a config_id —
		// the same fields Step 2 already composed for the create-config call.
		if (selectedModel && selectedConfigId === '__pending__' && pendingNewConfig) {
			api
				.estimateUsage({
					model_name: selectedModel,
					config_data: pendingNewConfig.config_data,
					tuner_type: pendingNewConfig.tuner_type,
					rl_tuner_type: pendingNewConfig.rl_tuner_type,
					gpu_memory: 80
				})
				.then((res) => {
					resourceEstimation = res;
				})
				.catch(() => {
					resourceEstimation = null;
				});
		} else if (selectedModel && selectedConfigId && selectedConfigId !== '__pending__') {
			api
				.estimateUsage({
					model_name: selectedModel,
					config_id: selectedConfigId,
					gpu_memory: 80
				})
				.then((res) => {
					resourceEstimation = res;
				})
				.catch(() => {
					resourceEstimation = null;
				});
		}
	}

	// Handle pending config from Step 2 inline creation
	function handlePendingConfig(event: CustomEvent<PendingConfigData>) {
		pendingNewConfig = event.detail;
		createdConfigId = null;
	}

	function handlePendingConfigUpdate(event: CustomEvent<PendingConfigUpdate>) {
		pendingConfigUpdate = event.detail;
	}

	function handleClearPendingConfig() {
		pendingNewConfig = null;
		pendingConfigUpdate = null;
		createdConfigId = null;
	}

	function handleDatasetChanged() {
		// Reset Step 2: Config state
		selectedConfigId = null;
		selectedConfig = null;
		pendingNewConfig = null;
		pendingConfigUpdate = null;

		// Reset Step 3: Launch state
		experimentName = '';
		resourceEstimation = null;

		// Mark Steps 2 & 3 as incomplete
		completedSteps[2] = false;
		completedSteps[3] = false;
	}

	async function handleLaunch() {
		isLaunching = true;
		transitionError = '';
		datasetPendingNotice = '';
		uploadProgress = 0;

		try {
			// --- Phase 1: Create dataset (if new upload) ---
			let finalDatasetId = datasetId || existingDatasetId;

			if (!finalDatasetId && uploadedFile) {
				launchPhase = 'creating_dataset';

				// Use previously created ID if retrying after upload failure
				if (!createdDatasetId) {
					const resp = await api.createDataset({
						name: datasetForm.name.trim(),
						description: datasetForm.description
					});
					if (!resp?.id) {
						throw new Error('Failed to create dataset metadata.');
					}
					createdDatasetId = resp.id;
				}
				finalDatasetId = createdDatasetId;

				// --- Phase 2: Process and upload files ---
				launchPhase = 'uploading_files';

				// Stream the RAW file(s) to the backend in chunks. Column mapping and
				// the train/validation split are applied server-side, so the browser
				// never has to read or re-serialize the whole file (which crashed the
				// renderer for large datasets). Works for both JSONL and Parquet.
				let stillProcessing = false;
				try {
					await api.uploadDatasetChunked(finalDatasetId!, {
						trainFile: uploadedFile!,
						validationFile: validationFile ?? undefined,
						columnMapping,
						// Auto-split only when no manual validation file was provided.
						trainSetPercentage: validationFile ? undefined : splitRatio,
						onProgress: (percent) => {
							uploadProgress = percent;
						}
					});
				} catch (err) {
					// A dataset still being validated server-side is not a launch
					// failure — the row exists, every byte arrived, and only our bounded
					// (~2 min) client-side wait ran out, which for a multi-GB file is the
					// expected outcome rather than the exceptional one.
					if (!(err instanceof DatasetUploadError) || err.status !== 202) throw err;
					stillProcessing = true;
				}

				// Update stores
				userMetadata.update((prev) => ({
					...prev,
					number_of_datasets: (prev?.number_of_datasets || 0) + 1
				}));

				datasetId = finalDatasetId;

				if (stillProcessing) {
					// Stop here rather than launching: POST /jobs requires a `ready`
					// dataset, so submitting now would only produce a real error. `datasetId`
					// is set above, so pressing Launch again once it is ready skips straight
					// past creating and re-uploading it.
					datasetPendingNotice =
						'Your dataset finished uploading but is still being processed, which can take a while for large files. ' +
						'You can check its status in the Datasets list under Settings, then press Launch Tuning again once it is ready.';
					return;
				}
			}

			// --- Phase 3a: Update config (if edited in-place) ---
			if (pendingConfigUpdate && selectedConfigId !== '__pending__') {
				launchPhase = 'updating_config';
				Utils.normalizeTokenizerListFields(pendingConfigUpdate.config_data);
				await api.updateConfiguration(pendingConfigUpdate.configId, {
					name: pendingConfigUpdate.name,
					tuner_type: pendingConfigUpdate.tuner_type,
					rl_tuner_type: pendingConfigUpdate.rl_tuner_type || null,
					config_data: pendingConfigUpdate.config_data
				});
				const configs = await api.getConfigurations();
				configurations.set(configs);
				appState.update((prev) => ({ ...prev, isConfigurationsLoaded: true }));
			}

			// --- Phase 3b: Create config (if pending) ---
			let finalConfigId = selectedConfigId;

			if (pendingNewConfig && selectedConfigId === '__pending__') {
				launchPhase = 'creating_config';

				if (!createdConfigId) {
					Utils.normalizeTokenizerListFields(pendingNewConfig.config_data);
					const payload = {
						name: pendingNewConfig.name.trim(),
						tuner_type: pendingNewConfig.tuner_type,
						rl_tuner_type: pendingNewConfig.rl_tuner_type || null,
						config_data: pendingNewConfig.config_data
					};
					const createdConfig = await api.createConfiguration(payload);
					createdConfigId = createdConfig.id;

					// Refresh global config store
					const configs = await api.getConfigurations();
					configurations.set(configs);
					appState.update((prev) => ({ ...prev, isConfigurationsLoaded: true }));
				}
				finalConfigId = createdConfigId;
			}

			// --- Phase 4: Launch the job (idempotent on retry) ---
			launchPhase = 'launching_job';

			const tuningForm: TuningForm = {
				config_id: finalConfigId!,
				dataset_id: (datasetId || existingDatasetId)!,
				model: selectedModel,
				model_source: modelSource,
				experiment_name: experimentName.trim().replace(/\s+/g, '_'),
				autotune: autotuneEnabled,
				...(hasRewardStep && rewardFunctionCode.trim()
					? {
							reward_function_code: rewardFunctionCode,
							reward_function_name: rewardFunctionName || 'compute_score'
					  }
					: {})
			};

			if (!createdJobId) {
				const job = await api.startJob(tuningForm);
				createdJobId = job?.id ?? null;
			}
			appState.update((prev) => ({ ...prev, isTuningsLoaded: false }));
			// Mark complete, clear draft, and redirect
			completedSteps[lastStepIndex] = true;
			clearDraft();
			goto('/autotune');
		} catch (err: any) {
			// `.message` before `.title`: a problem-detail `title` is a short label
			// ("Still processing", "Dataset processing failed") while a DatasetUploadError
			// carries the sentence a user can act on in `.message`. Preferring the title
			// showed the label and threw the explanation away.
			transitionError =
				err?.detail || err?.message || err?.title || 'Launch failed. Please try again.';
		} finally {
			isLaunching = false;
			launchPhase = null;
			uploadProgress = 0;
		}
	}

	function handleBack() {
		if (currentStep > 0) {
			currentStep = currentStep - 1;
		}
	}
</script>

<div class="wizard-container">
	<Grid noGutter fullWidth>
		<Row>
			<Column>
				<div class="wizard-header">
					<div style="display: flex; justify-content: space-between; align-items: flex-start;">
						<div>
							<h3>
								Configure {selectedGoal === 'sft'
									? 'Supervised Fine-Tuning'
									: selectedGoal === 'offline_rl'
									  ? 'Preference Learning'
									  : selectedGoal === 'online_rl'
									    ? 'Reinforcement Learning'
									    : 'Tuning'}
							</h3>
							<p class="wizard-subtitle">
								Follow the steps to configure and launch your fine-tuning job
							</p>
						</div>
						<Button
							kind="ghost"
							size="small"
							icon={Close}
							iconDescription="Close wizard"
							on:click={() => goto('/autotune')}
						/>
					</div>
				</div>

				{#if breadcrumbItems.length > 0}
					<div class="wizard-breadcrumb">
						<Breadcrumb noTrailingSlash>
							{#each breadcrumbItems as item, i}
								<BreadcrumbItem
									isCurrentPage={i === breadcrumbItems.length - 1}
									on:click={() => goToStep(item.step)}
								>
									{item.label}
								</BreadcrumbItem>
							{/each}
						</Breadcrumb>
					</div>
				{/if}

				{#key hasRewardStep}
					<ProgressIndicator
						currentIndex={currentStep}
						spaceEqually
						on:change={(e) => goToStep(e.detail)}
					>
						<ProgressStep
							complete={completedSteps[0]}
							label="Get Started"
							description="Choose your approach"
						/>
						<ProgressStep
							disabled={!completedSteps[0]}
							complete={completedSteps[1]}
							label="Upload Dataset"
							description="Upload and preview your data"
						/>
						<ProgressStep
							disabled={!completedSteps[1]}
							complete={completedSteps[2]}
							label="Configure"
							description="Select or create a configuration"
						/>
						{#if hasRewardStep}
							<ProgressStep
								disabled={!completedSteps[2]}
								complete={completedSteps[3]}
								label="Reward Function"
								description="Define your reward function"
							/>
						{/if}
						<ProgressStep
							disabled={!completedSteps[hasRewardStep ? 3 : 2]}
							complete={completedSteps[hasRewardStep ? 4 : 3]}
							label="Review & Launch"
							description="Review and start tuning"
						/>
					</ProgressIndicator>
				{/key}
			</Column>
		</Row>

		<!-- Draft restore banner -->
		<!-- {#if showDraftRestore && pendingDraft}
			<Row>
				<Column>
					<InlineNotification
						kind="info"
						title="Resume previous session?"
						subtitle="You have an unsaved wizard draft. Would you like to continue where you left off?"
						hideCloseButton
						lowContrast
						style="margin-bottom: 1rem;"
					>
						<svelte:fragment slot="actions">
							<Button kind="ghost" size="small" on:click={() => pendingDraft && restoreDraft(pendingDraft)}>Resume</Button>
							<Button kind="ghost" size="small" on:click={discardDraft}>Discard</Button>
						</svelte:fragment>
					</InlineNotification>
				</Column>
			</Row>
		{/if} -->

		<!-- Step Content -->
		<Row>
			<Column>
				<div class="step-content">
					{#if currentStep === 0}
						<Step0GetStarted
							bind:selectedAlgorithm
							bind:selectedGoal
							bind:selectedModel
							bind:modelSource
							bind:autotuneEnabled
							{prefetchedModels}
						/>
					{:else if currentStep === 1}
						<Step1DatasetUpload
							bind:uploadedFile
							bind:parsedData
							bind:columnMetadata
							bind:detectedFormat
							bind:datasetForm
							bind:totalRecords
							bind:existingDatasetId
							bind:splitRatio
							bind:validationFile
							bind:isSplitEnabled
							bind:selectedAlgorithm
							bind:columnMapping
							bind:isDatasetCompatible
							bind:keepAsParquet
							bind:selectedExistingDataset
							{selectedGoal}
							on:datasetChanged={handleDatasetChanged}
						/>
					{:else if currentStep === 2}
						<Step2Configure
							{selectedAlgorithm}
							{selectedGoal}
							bind:selectedConfigId
							bind:selectedConfig
							bind:isEditingConfig
							bind:isCreatingConfig
							on:pendingConfig={handlePendingConfig}
							on:pendingConfigUpdate={handlePendingConfigUpdate}
							on:clearPendingConfig={handleClearPendingConfig}
						/>
					{:else if currentStep === 3 && hasRewardStep}
						<StepRewardFunction
							bind:rewardFunctionCode
							bind:rewardFunctionName
							bind:allTestsPassed
							datasetId={datasetId || existingDatasetId}
							parsedData={parsedData.length > 0 && !existingDatasetId
								? Utils.applyColumnMapping(parsedData, columnMapping)
								: []}
						/>
					{:else if currentStep === lastStepIndex}
						<Step3ReviewLaunch
							{uploadedFile}
							{datasetForm}
							{selectedExistingDataset}
							{selectedConfig}
							{selectedModel}
							{modelSource}
							{resourceEstimation}
							{totalRecords}
							{splitRatio}
							{isSplitEnabled}
							{validationFile}
							{autotuneEnabled}
							{columnMetadata}
							bind:experimentName
							isPendingDataset={!existingDatasetId && !datasetId && !!uploadedFile}
							isPendingConfig={selectedConfigId === '__pending__'}
							{launchPhase}
							{uploadProgress}
							on:editStep={(e) => goToStep(e.detail)}
						/>
					{/if}
				</div>
			</Column>
		</Row>

		<!-- Dataset uploaded, still being validated server-side — informational, not a failure -->
		{#if datasetPendingNotice}
			<Row>
				<Column>
					<InlineNotification
						kind="info"
						lowContrast
						title="Dataset still processing"
						subtitle={datasetPendingNotice}
						on:close={() => (datasetPendingNotice = '')}
						style="margin-bottom: 1rem;"
					/>
				</Column>
			</Row>
		{/if}

		<!-- Error display -->
		{#if transitionError}
			<Row>
				<Column>
					<InlineNotification
						kind="error"
						title="Error"
						subtitle={transitionError}
						on:close={() => (transitionError = '')}
						style="margin-bottom: 1rem;"
					/>
				</Column>
			</Row>
		{/if}

		<!-- Navigation buttons -->
		<Row>
			<Column>
				<div class="wizard-footer">
					<Button kind="tertiary" on:click={() => goto('/autotune')}>Cancel</Button>
					{#if currentStep > 0}
						<Button kind="secondary" icon={ArrowLeft} on:click={handleBack} disabled={isLaunching}>
							Back
						</Button>
					{/if}

					{#if currentStep < lastStepIndex}
						<Button kind="primary" icon={ArrowRight} on:click={handleNext} disabled={!canProceed}>
							Next
						</Button>
					{:else}
						<Button
							kind="primary"
							icon={Rocket}
							on:click={handleLaunch}
							disabled={!canProceed || isLaunching}
						>
							{#if isLaunching}
								<InlineLoading
									description={launchPhase === 'creating_dataset'
										? 'Creating dataset...'
										: launchPhase === 'uploading_files'
										  ? `Uploading files (${uploadProgress}%)...`
										  : launchPhase === 'updating_config'
										    ? 'Updating configuration...'
										    : launchPhase === 'creating_config'
										      ? 'Creating configuration...'
										      : launchPhase === 'launching_job'
										        ? 'Launching job...'
										        : 'Launching...'}
								/>
							{:else}
								Launch Tuning
							{/if}
						</Button>
					{/if}
				</div>
			</Column>
		</Row>
	</Grid>
</div>

<style>
	.wizard-container {
		--atx-accent-sft: #009d9a;
		--atx-accent-offline-rl: #8a3ffc;
		--atx-accent-online-rl: #d12771;
		max-width: 1600px;
		margin: 0 auto;
		padding: 1rem;
	}

	.wizard-header {
		margin-bottom: var(--cds-spacing-06, 1.5rem);
	}

	.wizard-header h3 {
		margin: 0;
	}

	.wizard-subtitle {
		color: var(--cds-text-02, #525252);
		margin-top: var(--cds-spacing-02, 0.25rem);
	}

	.wizard-breadcrumb {
		margin-top: var(--cds-spacing-03, 0.5rem);
		margin-bottom: var(--cds-spacing-04, 0.75rem);
	}

	.wizard-breadcrumb :global(.bx--breadcrumb) {
		font-size: 0.8125rem;
	}

	.wizard-breadcrumb :global(.bx--breadcrumb-item) {
		cursor: pointer;
	}

	.wizard-breadcrumb :global(.bx--breadcrumb-item--current) {
		color: var(--cds-text-02, #525252);
	}

	.step-content {
		padding: var(--cds-spacing-06, 1.5rem) 0;
		min-height: 400px;
	}

	.wizard-footer {
		display: flex;
		justify-content: flex-end;
		gap: var(--cds-spacing-03, 0.5rem);
		padding: 1rem 0;
		border-top: 1px solid var(--cds-border-subtle, #e0e0e0);
	}
</style>
