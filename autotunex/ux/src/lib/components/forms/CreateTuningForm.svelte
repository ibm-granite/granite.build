<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Form,
		Grid,
		Row,
		Column,
		TextInput,
		FormGroup,
		Select,
		SelectItem,
		ProgressBar,
		ComboBox,
		Tile,
		NumberInput,
		Checkbox,
		FormItem,
		Toggle,
		Dropdown,
		RadioButtonGroup,
		RadioButton,
		InlineLoading,
		InlineNotification,
		Button
	} from 'carbon-components-svelte';
	import { Edit } from 'carbon-icons-svelte';
	import {
		HF_DEFAULT_JOB,
		DMF_DEFAULT_JOB,
		CUSTOM_PATH_DEFAULT_JOB
	} from '$lib/components/forms/default_job';
	import { createEventDispatcher, onDestroy, onMount, tick } from 'svelte';
	import { API } from '$lib/api';
	import { Utils } from '$lib/utils';
	import ConfigDisplay from '../displays/ConfigDisplay.svelte';
	import CreateConfigForm from './CreateConfigForm.svelte';
	import DatasetDisplay from '../displays/DatasetDisplay.svelte';
	import DmfModelDisplay from '../displays/DmfModelDisplay.svelte';
	import { compile } from 'mdsvex';
	import type { TuningForm, Configuration, HuggingFaceModel } from '$lib/app-types';
	import { ModelSource } from '$lib/app-types';
	import TimeInput from '../TimeInput.svelte';
	import { appState, configurations, datasets } from '$lib/app';
	import { currentUser, featureFlags } from '$lib/store';

	export let tuning: TuningForm = structuredClone(HF_DEFAULT_JOB);
	export let config: Configuration | null = null;
	export let configClone: Configuration | null = null;

	let isEditingConfig: boolean = false; // Track if user is editing the config
	let editableConfig: any = null; // Hold the editable config form data
	let isSavingConfig: boolean = false;
	let saveError: string = '';
	let needsSaveAs: boolean = false;
	let newConfigName: string = '';

	const SYSTEM_USER_ID = '00000000-0000-0000-0000-000000000001';

	const api = new API();
	const dispatch = createEventDispatcher();
	let modelCard: any = null;
	let debounceTimeout: number;
	let models: HuggingFaceModel[] = [];
	let suggestions: { id: string; text: string; isOpen?: boolean }[] = [];
	let showItem: 'model' | 'config' | 'dataset' = 'model';
	let modelSource: ModelSource = ModelSource.HuggingFace;
	let previousModelSource: ModelSource = ModelSource.HuggingFace;
	let selectedModelIsOpen: boolean = true;
	let modelWarningText: string = '';
	let selectedDmfModel: { id: string; text: string; isOpen?: boolean; rawModel?: any } | null =
		null;
	let isSearching: boolean = false;
	let hasFetchedDefaultDmfModel: boolean = false;

	// Determine if config needs "Save As" when entering edit mode
	$: if (isEditingConfig && config) {
		const isSystemConfig = config.user_id === SYSTEM_USER_ID;
		const hasAssociatedJobs = config.associated_jobs && config.associated_jobs.length > 0;
		needsSaveAs = isSystemConfig || hasAssociatedJobs;
		if (needsSaveAs && !newConfigName) {
			newConfigName = `${config.name}_modified`;
		}
	} else {
		needsSaveAs = false;
	}

	// Sync modelSource with tuning.model_source and reset form when switching
	$: if (modelSource !== previousModelSource) {
		// Reset tuning object with the appropriate default
		tuning = structuredClone(
			modelSource === ModelSource.DMF
				? DMF_DEFAULT_JOB
				: modelSource === ModelSource.CustomPath
				  ? CUSTOM_PATH_DEFAULT_JOB
				  : HF_DEFAULT_JOB
		);
		selectedModelIsOpen = true;
		modelWarningText = '';
		selectedDmfModel = null;
		hasFetchedDefaultDmfModel = false;
		if (modelSource === ModelSource.DMF) {
			// Fetch default DMF model details from API
			fetchDefaultDmfModel();
		} else if (modelSource === ModelSource.CustomPath) {
			suggestions = [];
		} else {
			suggestions = models?.map((item) => {
				return { id: item.id, text: item.id };
			});
		}
		// Update previous model source after handling the change
		previousModelSource = modelSource;
	}

	// Fetch the default DMF model details from the API
	async function fetchDefaultDmfModel() {
		try {
			isSearching = true;
			const dmfModels = await api.searchDMFModels(DMF_DEFAULT_JOB.model);

			if (Array.isArray(dmfModels?.data) && dmfModels.data.length > 0) {
				// Process all models from the search results
				const uniqueModels = new Map();

				dmfModels.data.forEach((model: any) => {
					// Validate required fields
					if (!model.namespace || !model.base_model || !model.revision) {
						console.warn('⚠️ Skipping model with missing fields:', model);
						return;
					}

					// Skip models with invalid base_model
					if (model.base_model.includes('"') || model.base_model.includes('/')) {
						console.warn('⚠️ Skipping model with invalid base_model:', model.base_model);
						return;
					}

					const modelId = model.model_id;
					const label = model.model_label || model.model_id || model.base_model;
					const isOpen = model.open === true;

					if (!uniqueModels.has(modelId)) {
						uniqueModels.set(modelId, {
							id: modelId,
							text: label,
							isOpen: isOpen,
							rawModel: model
						});
					}
				});

				suggestions = Array.from(uniqueModels.values());

				// Find the exact match for the default model
				const defaultModel =
					dmfModels.data.find(
						(model: any) =>
							model.model_id === DMF_DEFAULT_JOB.model || model.base_model === DMF_DEFAULT_JOB.model
					) || dmfModels.data[0]; // Fallback to first result

				const modelId = defaultModel.model_id;
				const label = defaultModel.model_label || defaultModel.model_id || defaultModel.base_model;
				const isOpen = defaultModel.open === true;

				selectedDmfModel = {
					id: modelId,
					text: label,
					isOpen: isOpen,
					rawModel: defaultModel
				};

				tuning.model = modelId;
				tuning.additional_info = defaultModel;

				console.log('✅ Default DMF model loaded with', suggestions.length, 'suggestions');
			} else {
				// Fallback if no results found
				suggestions = [{ id: DMF_DEFAULT_JOB.model, text: DMF_DEFAULT_JOB.model }];
			}

			isSearching = false;
			hasFetchedDefaultDmfModel = true;
		} catch (error) {
			console.error('❌ Error fetching default DMF model:', error);
			suggestions = [{ id: DMF_DEFAULT_JOB.model, text: DMF_DEFAULT_JOB.model }];
			isSearching = false;
			hasFetchedDefaultDmfModel = true;
		}
	}

	// $: if (config && configClone && !isEqual(config, configClone)) {
	// 	console.log('🚀 ~ if isEqual:', isEqual(config, configClone));
	// } else {
	// 	console.log('🚀 ~ else isEqual:', isEqual(config, configClone));
	// }

	// const doFetch = async () => {
	// 	if (!tuning.model || !tuning.config_id) return;
	// 	api
	// 		.estimateUsage({
	// 			config_id: tuning.config_id,
	// 			model_name: tuning.model,
	// 			gpu_memory: 80
	// 		})
	// 		.then((data) => {
	// 			if (data.num_gpus) {
	// 				if (config) {
	// 					config.config_data['training_config']['num_gpus_per_trial'].default = data.num_gpus;
	// 				}
	// 			}
	// 		});
	// };

	// let debounceTimer: ReturnType<typeof setTimeout> | null = null;
	// function debounceFetch() {
	// 	if (debounceTimer) clearTimeout(debounceTimer);
	// 	debounceTimer = setTimeout(() => doFetch(), 200);
	// }

	// $: tuning.config_id, tuning.model, debounceFetch();

	async function fetchModelCard() {
		try {
			if (modelSource === ModelSource.DMF) {
				// No model card needed for DMF - user confirmed
				modelCard = null;
				return;
			} else {
				// Existing HuggingFace logic
				let rawContent = await api.getHFModelCard(tuning.model);

				// --- Logic to remove YAML front matter ---
				const lines = rawContent.split('\n');
				let inFrontMatter = false;
				let contentStarted = false;
				let filteredContentLines = [];

				for (const line of lines) {
					if (line.trim() === '---') {
						if (!inFrontMatter) {
							// First '---', indicates start of front matter
							inFrontMatter = true;
						} else {
							// Second '---', indicates end of front matter
							inFrontMatter = false;
							contentStarted = true; // Content starts after this
							continue; // Skip this '---' line
						}
					} else if (inFrontMatter) {
						// Skip lines inside the front matter
						continue;
					}

					if (contentStarted) {
						filteredContentLines.push(line);
					} else if (!inFrontMatter && line.trim() !== '') {
						// If not in front matter and we haven't hit the second '---' yet,
						// but we encounter a non-empty line, it means there was no
						// YAML front matter, or it was malformed. In this case,
						// we treat the whole content as markdown.
						// This ensures that model cards without explicit YAML
						// are still rendered correctly.
						contentStarted = true;
						filteredContentLines.push(line);
					}
				}

				// Join the filtered lines back into a single string
				const filtered = filteredContentLines.join('\n').trim();
				modelCard = await compile(filtered);
				showItem = 'model';
			}
		} catch (e) {
			console.error('Error fetching model card:', e);
			modelCard = null;
		}
	}

	async function applyAndSaveConfig() {
		if (!config || !editableConfig) return;

		// Validate name if Save As is required
		if (needsSaveAs && (!newConfigName || newConfigName.trim() === '')) {
			saveError = 'Please provide a name for the new configuration.';
			return;
		}

		// Check for duplicate name in Save As scenario
		if (needsSaveAs) {
			const nameExists = $configurations.some((c) => c.name === newConfigName.trim());
			if (nameExists) {
				saveError = `A configuration named "${newConfigName.trim()}" already exists. Please choose a different name.`;
				return;
			}
		}

		try {
			isSavingConfig = true;
			saveError = '';

			// Build payload from editableConfig directly (don't mutate config until API succeeds)
			const { name: _, tuner_type, rl_tuner_type, ...configSections } = editableConfig;
			Utils.normalizeTokenizerListFields(configSections);

			let newConfigId: string | null = null;

			if (needsSaveAs) {
				// Create a new configuration
				const newConfig = {
					name: newConfigName.trim(),
					tuner_type: tuner_type,
					rl_tuner_type: rl_tuner_type || null,
					config_data: configSections
				};
				const createdConfig = await api.createConfiguration(newConfig);

				// Only mutate config AFTER success
				config.id = createdConfig.id;
				config.name = newConfigName.trim();
				config.user_id = createdConfig.user_id;
				config.associated_jobs = [];
				config.tuner_type = tuner_type;
				config.rl_tuner_type = rl_tuner_type || '';
				config.config_data = configSections;
				newConfigId = createdConfig.id;
			} else {
				// Update existing configuration in place
				await api.updateConfiguration(config.id, {
					name: config.name,
					tuner_type: tuner_type,
					rl_tuner_type: rl_tuner_type || null,
					config_data: configSections
				});

				// Only mutate config AFTER success
				config.tuner_type = tuner_type;
				config.rl_tuner_type = rl_tuner_type || '';
				config.config_data = configSections;
			}

			// Refresh the configurations store
			let configs = await api.getConfigurations();
			configurations.set(configs);
			appState.update((prev) => ({
				...prev,
				isConfigurationsLoaded: true
			}));

			// Wait for Svelte to re-render Select options, then select the new config
			if (newConfigId) {
				await tick();
				tuning.config_id = newConfigId;
			}

			// Sync configClone so submission flow does NOT detect changes
			configClone = structuredClone(config);

			// Notify parent that config was saved
			dispatch('configSaved', { config, isNew: needsSaveAs });

			// Exit edit mode and reset state
			isEditingConfig = false;
			editableConfig = null;
			newConfigName = '';
			needsSaveAs = false;
		} catch (error: any) {
			console.error('Error saving configuration:', error);
			saveError = error?.detail || 'Failed to save configuration. Please try again.';
		} finally {
			isSavingConfig = false;
		}
	}

	async function fetchSuggestions(term: string) {
		// If the term is empty, handle based on model source
		if (!term?.trim()) {
			isSearching = false;
			if (modelSource === ModelSource.DMF) {
				suggestions = []; // Clear suggestions for DMF when empty
			} else {
				suggestions = models?.map((item) => {
					return { id: item.id, text: item.id };
				});
			}
			return;
		}

		try {
			if (modelSource === ModelSource.DMF) {
				// Set loading state for DMF search
				isSearching = true;

				const dmfModels = await api.searchDMFModels(term);

				// DMF returns array of model objects in data property
				// Use artifact_uri directly from DMF response (already in correct format)
				// Also filter for unique URIs to avoid duplicate key errors
				const uniqueModels = new Map();

				if (Array.isArray(dmfModels?.data)) {
					dmfModels.data.forEach((model: any) => {
						// Validate required fields
						if (!model.namespace || !model.base_model || !model.revision) {
							console.warn('⚠️ Skipping model with missing fields:', model);
							return;
						}

						// Skip models with invalid base_model (contains quotes, slashes, etc.)
						if (model.base_model.includes('"') || model.base_model.includes('/')) {
							console.warn('⚠️ Skipping model with invalid base_model:', model.base_model);
							return;
						}

						// Use model_id as the temporary ID (will be replaced by backend with proper URI)
						const modelId = model.model_id;
						const label = model.model_label || model.model_id || model.base_model;
						const isOpen = model.open === true;

						console.log(isOpen ? '✅ Open model:' : '🔒 Restricted model:', { modelId, label });

						// Only add if we haven't seen this model before
						if (!uniqueModels.has(modelId)) {
							uniqueModels.set(modelId, {
								id: modelId,
								text: label,
								isOpen: isOpen,
								rawModel: model // Store the complete DMF model object
							});
						}
					});
				} else {
					console.error('DMF data is not an array:', dmfModels?.data);
				}

				suggestions = Array.from(uniqueModels.values());

				// Preserve previously selected model in suggestions to prevent text reset
				if (selectedDmfModel && !uniqueModels.has(selectedDmfModel.id)) {
					suggestions = [selectedDmfModel, ...suggestions];
				}

				console.log('✅ Final suggestions:', suggestions);
				isSearching = false;
			} else {
				// Fetch from HuggingFace (existing logic)
				const response = await api.getHFModels(term.replace(/(\w+)[-/]\1(?=[-/])/g, '$1'));
				suggestions = response.map((model: HuggingFaceModel) => ({
					id: model.id,
					text: model.id
				}));
			}
		} catch (error) {
			console.error('❌ Error fetching suggestions:', error);
			suggestions = []; // Clear suggestions on error
			isSearching = false;
		}
	}

	onMount(async () => {
		await fetchModelCard();
		if (!$appState.isConfigurationsLoaded) {
			let configs = await api.getConfigurations();
			configurations.set(configs);
			appState.update((prev) => {
				return { ...prev, isConfigurationsLoaded: true };
			});
		}

		if (!$appState.isDatasetsLoaded) {
			let data = await api.getDatasets();
			datasets.set(data);
			appState.update((prev) => {
				return { ...prev, isDatasetsLoaded: true };
			});
		}

		models = await api.getHFModels('ibm-granite/granite-4.0-h-micro', 20);
		suggestions = models?.map((item) => {
			return { id: item.id, text: item.id };
		});
	});

	onDestroy(() => {
		if (tuning) {
			tuning = structuredClone(HF_DEFAULT_JOB);
		}
	});
</script>

{#if $configurations?.length > 0 && $datasets?.length > 0 && models.length > 0}
	<Form>
		<Grid fullWidth noGutter>
			<Row>
				<Column md={3}>
					<Row>
						<Column>
							<FormGroup>
								<RadioButtonGroup
									legendText="Model source"
									name="model_source"
									bind:selected={modelSource}
								>
									<RadioButton labelText="Huggingface" value={ModelSource.HuggingFace} />
									<RadioButton labelText="Data Model Factory" value={ModelSource.DMF} />
									{#if $featureFlags.customPathModelSource && $currentUser?.role === 'admin'}
										<RadioButton labelText="Custom Path" value={ModelSource.CustomPath} />
									{/if}
								</RadioButtonGroup>
							</FormGroup>
						</Column>
					</Row>
					<Row>
						<Column>
							<FormGroup>
								<TextInput
									labelText="Tuning name"
									bind:value={tuning.experiment_name}
									placeholder="Enter tuning name"
									on:blur={() =>
										(tuning.experiment_name = tuning.experiment_name.trim().replace(/\s+/g, '_'))}
								/>
							</FormGroup>
						</Column>
					</Row>
					<Row>
						<Column>
							<FormGroup>
								{#if modelSource === ModelSource.CustomPath}
									<TextInput
										labelText="Model path"
										placeholder="/gb-lakehouse-prod-read-only/models/..."
										bind:value={tuning.model}
										helperText="Enter the full filesystem path to the model on GB compute workers"
									/>
								{:else}
									<ComboBox
										titleText="Model"
										bind:selectedId={tuning.model}
										placeholder="Select model"
										items={suggestions}
										shouldFilterItem={(item, value) => {
											if (!value) return true;
											return (
												item.text?.toLowerCase().includes(value.toLowerCase()) ||
												item.id?.toLowerCase().includes(value.toLowerCase())
											);
										}}
										on:clear={() => {
											if (modelSource === ModelSource.DMF) {
												suggestions = [];
												tuning.model = '';
												selectedModelIsOpen = true;
												modelWarningText = '';
												selectedDmfModel = null;
											} else {
												suggestions = models?.map((item) => {
													return { id: item.id, text: item.id };
												});
												tuning.model = '';
												selectedModelIsOpen = true;
												modelWarningText = '';
											}
										}}
										on:keyup={(e) => {
											clearTimeout(debounceTimeout);
											// Set a new debounce timeout
											debounceTimeout = setTimeout(() => {
												// Use the input value for search, not tuning.model
												const inputValue = e.target?.value || '';
												if (inputValue) {
													fetchSuggestions(inputValue);
												}
											}, 500);
										}}
										on:focus={() => {
											showItem = 'model';
											// For DMF, only fetch on focus if we haven't already fetched the default model suggestions
											if (
												modelSource === ModelSource.DMF &&
												!hasFetchedDefaultDmfModel &&
												tuning.model
											) {
												fetchSuggestions(tuning.model);
											}
										}}
										on:select={(e) => {
											if (e.detail.selectedItem?.id) {
												tuning.model = e.detail.selectedItem.id;

												// Save selected DMF model to preserve it in suggestions
												if (modelSource === ModelSource.DMF) {
													selectedDmfModel = e.detail.selectedItem;

													// Store the raw DMF model object in additional_info
													tuning.additional_info = e.detail.selectedItem.rawModel;
													console.log('📦 Additional info set:', tuning.additional_info);

													// Check if model is open
													if (e.detail.selectedItem?.isOpen === false) {
														selectedModelIsOpen = false;
														modelWarningText =
															'⚠️ This model has restricted access. You may not have permission to use it for fine-tuning.';
													} else {
														selectedModelIsOpen = true;
														modelWarningText = '';
													}
												} else {
													selectedModelIsOpen = true;
													modelWarningText = '';
													tuning.additional_info = undefined;
												}

												fetchModelCard();
											}
										}}
									/>
									{#if isSearching && modelSource === ModelSource.DMF}
										<InlineLoading description="Searching DMF models..." />
									{/if}
								{/if}
							</FormGroup>
						</Column>
					</Row>
					<Row>
						<Column>
							<FormGroup>
								<Select
									labelText="Data set"
									bind:selected={tuning.dataset_id}
									on:focus={(e) => (showItem = 'dataset')}
								>
									{#each $datasets?.sort((a, b) => a.name.localeCompare(b.name)) as dataset}
										<SelectItem value={dataset.id} text={dataset.name} />
									{/each}
								</Select>
							</FormGroup>
						</Column>
					</Row>
					<Row>
						<Column>
							<FormGroup>
								<Select
									labelText="Configuration"
									bind:selected={tuning.config_id}
									on:focus={(e) => (showItem = 'config')}
								>
									{#each $configurations.sort((a, b) => a.name.localeCompare(b.name)) as config}
										<SelectItem value={config.id} text={config.name} />
									{/each}
								</Select>
								{#if $currentUser?.role === 'admin'}
									<FormItem>
										<Checkbox
											bind:checked={tuning.autotune}
											labelText="AutoTune (use hyperparameter optimization)"
											style="margin-top: 1rem;"
										/>
									</FormItem>
								{/if}
							</FormGroup>
						</Column>
					</Row>
				</Column>
				<Column md={5} style="height: 720px; overflow:scroll">
					{#if showItem === 'config' && tuning?.config_id}
						{#if isEditingConfig}
							<!-- Edit mode: Show CreateConfigForm -->
							<div style="padding: 1rem; background-color: #f4f4f4;">
								<div
									style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;"
								>
									<h5 style="margin: 0;">Edit Configuration</h5>
									<div style="display: flex; gap: 0.5rem;">
										<Button
											size="small"
											kind="secondary"
											disabled={isSavingConfig}
											on:click={() => {
												isEditingConfig = false;
												editableConfig = null;
												saveError = '';
												newConfigName = '';
											}}
										>
											Cancel
										</Button>
										<Button
											size="small"
											kind="primary"
											disabled={isSavingConfig ||
												(needsSaveAs && (!newConfigName || newConfigName.trim() === ''))}
											on:click={applyAndSaveConfig}
										>
											{#if isSavingConfig}
												Saving...
											{:else if needsSaveAs}
												Save As New
											{:else}
												Apply Changes
											{/if}
										</Button>
									</div>
								</div>

								{#if needsSaveAs}
									<div style="margin-bottom: 1rem;">
										<InlineNotification
											kind="info"
											lowContrast
											hideCloseButton
											title="Save as new:"
											subtitle={config?.user_id === SYSTEM_USER_ID
												? 'This is a system configuration and cannot be modified directly.'
												: `This configuration has ${
														config?.associated_jobs?.length || 0
												  } associated job(s) and cannot be updated directly.`}
										/>
										<div style="margin-top: 0.5rem;">
											<TextInput
												labelText="New Configuration Name"
												placeholder="Enter new configuration name"
												bind:value={newConfigName}
												disabled={isSavingConfig}
												invalid={newConfigName.trim() !== '' &&
													$configurations.some((c) => c.name === newConfigName.trim())}
												invalidText={`"${newConfigName.trim()}" already exists`}
											/>
										</div>
									</div>
								{/if}

								{#if saveError}
									<div style="margin-bottom: 1rem;">
										<InlineNotification
											kind="error"
											title="Error:"
											subtitle={saveError}
											on:close={() => (saveError = '')}
										/>
									</div>
								{/if}

								{#if isSavingConfig}
									<InlineLoading description="Saving configuration..." />
								{/if}

								<CreateConfigForm
									bind:config={editableConfig}
									configurations={$configurations}
									editMode={true}
									existingConfig={config}
									hideNameField={true}
								/>
							</div>
						{:else}
							<!-- View mode: Show ConfigDisplay with edit button -->
							<div style="position: relative;">
								<div style="position: absolute; right: 0.5rem; z-index: 10;">
									<Button
										style="min-height: 1.85rem"
										size="small"
										kind="primary"
										icon={Edit}
										on:click={() => {
											isEditingConfig = true;
											// Initialize editableConfig with current config data
											if (config) {
												editableConfig = {
													name: config.name,
													tuner_type: config.tuner_type,
													...config.config_data
												};
											}
										}}
									>
										Edit
									</Button>
								</div>
								<ConfigDisplay
									config_id={tuning.config_id}
									bind:configuration={config}
									editable={false}
									showEditButton={false}
									on:configLoaded={(e) => {
										configClone = structuredClone(e.detail.configuration);
									}}
								/>
							</div>
						{/if}
					{:else if showItem === 'dataset' && tuning.dataset_id}
						<DatasetDisplay datasetId={tuning.dataset_id} />
					{:else if showItem === 'model' && tuning.model}
						{#if modelSource === ModelSource.DMF}
							<!-- Display DMF model details -->
							<DmfModelDisplay modelData={tuning.additional_info} />
						{:else if modelSource === ModelSource.CustomPath}
							<p>Custom filesystem path: <code>{tuning.model}</code></p>
						{:else if modelCard}
							<!-- Display HuggingFace model card -->
							<div class="markdown-content">
								{@html modelCard.code}
							</div>
						{/if}
					{/if}
				</Column>
			</Row>
		</Grid>
	</Form>
{:else}
	<ProgressBar size="sm" helperText="Loading configurations and datasets details..." />
{/if}

<style>
	:global(.markdown-content h1) {
		font-size: 20px;
		font-weight: 600;
	}
	:global(.markdown-content h2) {
		margin-top: 1rem;
		font-size: 18px;
		font-weight: 600;
	}
	:global(.markdown-content h3) {
		margin-top: 1rem;
		font-size: 16px;
		font-weight: 600;
	}
	:global(.markdown-content > p) {
		padding-top: 1rem;
		padding-bottom: 1rem;
	}
	:global(.markdown-content li) {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
	}
	:global(.markdown-content pre) {
		background-color: #f4f4f4;
	}
	:global(.markdown-content img) {
		max-width: -webkit-fill-available;
	}
	:global(.markdown-content table) {
		width: 100%;
	}
</style>
