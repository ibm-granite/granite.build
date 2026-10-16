<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	// @ts-nocheck
	import { API } from '$lib/api';
	import { onMount } from 'svelte';
	import { Utils } from '$lib/utils';
	import {
		Checkbox,
		Column,
		ContentSwitcher,
		Dropdown,
		FormLabel,
		Loading,
		MultiSelect,
		NumberInput,
		RadioButton,
		RadioButtonGroup,
		Row,
		Select,
		SelectItem,
		Switch,
		TextInput,
		Toggle
	} from 'carbon-components-svelte';
	import GeneralConfigForm from './GeneralConfigForm.svelte';
	import type { ConfigData, ConfigForm, Configuration } from '$lib/app-types';
	import TimeInput from '../TimeInput.svelte';

	const api = new API();

	export let config: ConfigForm;
	export let configurations: Configuration[] = [];
	export let editMode: boolean = false; // New prop to indicate if we're editing
	export let existingConfig: Configuration | null = null; // Existing config to edit
	export let hideNameField: boolean = false; // Hide name field when editing inline
	export let presetGoal: import('$lib/app-types').TuningGoal | null = null;
	export let presetAlgorithm: string | null = null;

	let configCopy: ConfigForm; // Deep copy of the original config for selecting the values of multi-selects
	let selectedTuner: string = '';
	let selectedRlTuner: string = '';
	let selectedSection: string;
	let isLoading: boolean = true;
	let error: string | null = null;
	let sectionNames: string[] = [];
	let tuners: string[] = [];
	let rlTuners: string[] = [];
	let mode = false;
	let selectedIndex = 0;
	let errorFields = {};
	let trainingMode: string = presetGoal === 'online_rl' ? 'online_tuning' : 'offline_tuning';

	// Canonical tab order for advanced mode (ensures consistent ordering regardless of Object.keys order)
	const SECTION_ORDER = [
		'general_config',
		'tune_config',
		'training_config',
		'tokenizer_config',
		'tuners_config',
		'training_rl_config',
		'tuners_rl_config'
	];

	// Auto-set training mode and algorithm from wizard preset
	$: if (presetGoal) {
		trainingMode = presetGoal === 'online_rl' ? 'online_tuning' : 'offline_tuning';
	}

	// Only set tuner_type when in Offline Tuning mode
	$: if (config && selectedTuner && trainingMode === 'offline_tuning') {
		config.tuner_type = selectedTuner;
	}

	// Clear tuner_type in Online Tuning mode
	$: if (config && trainingMode === 'online_tuning') {
		config.tuner_type = null;
	}

	// Set rl_tuner_type when selectedRlTuner is available and not 'none'
	$: if (config && selectedRlTuner && selectedRlTuner !== 'none') {
		config.rl_tuner_type = selectedRlTuner;
	}

	// Clear rl_tuner_type when RL algorithm is set to 'none'
	$: if (config && selectedRlTuner === 'none') {
		config.rl_tuner_type = null;
	}

	// Force RL tuner to 'none' when SFT goal is selected
	$: if (presetGoal === 'sft' && config) {
		selectedRlTuner = 'none';
		config.rl_tuner_type = null;
	}

	// Track previous training mode to detect changes
	let previousTrainingMode = trainingMode;

	// Update config and default algorithms when training mode changes
	$: if (
		config &&
		config['training_config'] &&
		trainingMode &&
		trainingMode !== previousTrainingMode
	) {
		if (trainingMode === 'offline_tuning') {
			// Offline Tuning: Always requires a tuning algorithm
			// RL algorithm can be 'none' for default finetuning or set to offline RL (DPO, KTO)
			// Set default tuner to lora for offline tuning
			if (tuners.includes('lora')) {
				selectedTuner = 'lora';
			} else if (tuners.length > 0) {
				selectedTuner = tuners[0];
			}
			// Set default RL tuner to 'none' (user can change to DPO/KTO if needed)
			if (!selectedRlTuner || !availableRlTuners.includes(selectedRlTuner)) {
				selectedRlTuner = 'none';
			}
		} else if (trainingMode === 'online_tuning') {
			// Online Tuning: Only uses online RL algorithms (PPO, GRPO, DAPO)
			if (config['training_config']['tuning_algorithm']) {
				config['training_config']['tuning_algorithm'].default = 'none';
			}
			// Clear selectedTuner in Online Tuning mode (no traditional tuning algorithms needed)
			selectedTuner = '';
			// Set default RL tuner to first available online RL algorithm
			if (availableRlTuners.length > 0) {
				selectedRlTuner = availableRlTuners[0];
			}
		}
		previousTrainingMode = trainingMode;
	}

	// Filter RL algorithms based on training mode
	$: availableRlTuners =
		trainingMode === 'offline_tuning'
			? ['none', ...rlTuners.filter((t) => ['dpo', 'kto'].includes(t))]
			: rlTuners.filter((t) => ['ppo', 'grpo', 'dapo'].includes(t));

	// Update selectedRlTuner when available algorithms change
	$: if (availableRlTuners.length > 0 && !availableRlTuners.includes(selectedRlTuner)) {
		selectedRlTuner = availableRlTuners[0];
	}

	// Update sections when training mode changes
	$: if (config && trainingMode && !mode) {
		// Only update in basic mode
		const newSectionNames = ['general_config'];

		if (trainingMode === 'offline_tuning') {
			newSectionNames.push('tuners_config');
			if (presetGoal !== 'sft') {
				newSectionNames.push('tuners_rl_config');
			}
		} else if (trainingMode === 'online_tuning') {
			// Online Tuning: Only show RL tuners (no traditional tuning)
			newSectionNames.push('tuners_rl_config');
		}

		// Only update if sections actually changed
		if (JSON.stringify(sectionNames) !== JSON.stringify(newSectionNames)) {
			// Update selected section if current section is no longer available
			if (!newSectionNames.includes(selectedSection)) {
				selectedSection = newSectionNames[0];
				selectedIndex = 0;
			} else {
				// Update selectedIndex to match selectedSection's new position
				selectedIndex = newSectionNames.indexOf(selectedSection);
			}

			// Update sectionNames after selectedIndex is set
			sectionNames = newSectionNames;
		}
	}

	// Update sections when training mode changes in Advanced mode
	$: if (config && trainingMode && mode) {
		// Only update in advanced mode
		let allSections = Object.keys(config).filter(
			(key) =>
				key !== 'name' &&
				key !== 'tuner_type' &&
				key !== 'rl_tuner_type' &&
				typeof config[key] === 'object'
		);

		// Only show training_rl_config in Online Tuning mode
		if (trainingMode !== 'online_tuning') {
			allSections = allSections.filter((key) => key !== 'training_rl_config');
		}

		// Hide tuners_config in Online Tuning mode (only uses RL algorithms)
		if (trainingMode === 'online_tuning') {
			allSections = allSections.filter((key) => key !== 'tuners_config');
		}

		// Hide tuners_rl_config in SFT mode (no RL algorithms needed)
		if (presetGoal === 'sft') {
			allSections = allSections.filter((key) => key !== 'tuners_rl_config');
		}

		// Sort sections by canonical order
		allSections.sort((a, b) => {
			const ai = SECTION_ORDER.indexOf(a);
			const bi = SECTION_ORDER.indexOf(b);
			return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi);
		});

		// Only update if sections actually changed
		if (JSON.stringify(sectionNames) !== JSON.stringify(allSections)) {
			// Update selected section if current section is no longer available
			if (!allSections.includes(selectedSection)) {
				selectedSection = allSections[0];
				selectedIndex = 0;
			} else {
				// Update selectedIndex to match selectedSection's new position
				selectedIndex = allSections.indexOf(selectedSection);
			}

			// Update sectionNames after selectedIndex is set
			sectionNames = allSections;
		}
	}

	// Synchronize selectedTuner with training_config.tuning_algorithm (only when not manually changed)
	let previousTuner = '';
	$: if (
		config &&
		config['training_config'] &&
		selectedTuner &&
		trainingMode === 'offline_tuning' &&
		selectedTuner !== previousTuner
	) {
		if (
			config['training_config']['tuning_algorithm'] &&
			config['training_config']['tuning_algorithm'].default !== selectedTuner
		) {
			config['training_config']['tuning_algorithm'].default = selectedTuner;
		}
		previousTuner = selectedTuner;
	}

	// Synchronize selectedRlTuner with training_config.rl_algorithm (only when not manually changed)
	let previousRlTuner = '';
	$: if (
		config &&
		config['training_config'] &&
		selectedRlTuner &&
		selectedRlTuner !== previousRlTuner
	) {
		if (
			config['training_config']['rl_algorithm'] &&
			config['training_config']['rl_algorithm'].default !== selectedRlTuner
		) {
			config['training_config']['rl_algorithm'].default = selectedRlTuner;
		}
		previousRlTuner = selectedRlTuner;
	}

	// Filter RL algorithm values in training_config based on training mode
	$: if (config && config['training_config'] && config['training_config']['rl_algorithm']) {
		const offlineRlAlgs = ['none', 'dpo', 'kto'];
		const onlineRlAlgs = ['none', 'ppo', 'grpo', 'dapo'];

		if (trainingMode === 'offline_tuning') {
			config['training_config']['rl_algorithm'].values = offlineRlAlgs;
		} else if (trainingMode === 'online_tuning') {
			config['training_config']['rl_algorithm'].values = onlineRlAlgs;
		}
	}

	// Fetch configuration data from API
	async function fetchConfigData() {
		try {
			isLoading = true;

			// If editing existing config, use it instead of template
			if (editMode && existingConfig) {
				config = {
					name: existingConfig.name,
					tuner_type: existingConfig.tuner_type,
					rl_tuner_type: existingConfig.rl_tuner_type || '',
					...existingConfig.config_data
				};
				selectedTuner = existingConfig.tuner_type;
				if (existingConfig.rl_tuner_type) {
					selectedRlTuner = existingConfig.rl_tuner_type;
				}
			} else {
				// A failure (e.g. 503 when the autotune template backend is unavailable)
				// throws a problem-detail from handleResponse and is caught below.
				const template = await api.getConfigurationTemplate();
				error = null;
				config = {
					name: '',
					tuner_type: '',
					rl_tuner_type: '',
					...template
				};

				config['tune_config']['max_concurrent_trials'].default = Math.floor(
					config['training_config']['num_gpus_per_trial'].max_val /
						config['training_config']['num_gpus_per_trial'].default
				);
			}

			configCopy = structuredClone(config);

			// Infer training mode ONLY when editing existing config
			if (editMode && existingConfig) {
				const hasTuningAlg = config['training_config']?.['tuning_algorithm']?.default !== 'none';
				const hasRlAlg = config['training_config']?.['rl_algorithm']?.default !== 'none';
				const rlAlg = config['training_config']?.['rl_algorithm']?.default;

				// Check if RL algorithm is online (PPO, GRPO, DAPO)
				const isOnlineRlAlg = ['ppo', 'grpo', 'dapo'].includes(rlAlg?.toLowerCase() || '');

				if (isOnlineRlAlg) {
					trainingMode = 'online_tuning';
				} else {
					// Either no RL (default finetuning) or offline RL (DPO/KTO)
					// Both are handled by offline_tuning mode
					trainingMode = 'offline_tuning';
				}
			} else {
				// For new configs, respect presetGoal if provided
				if (presetGoal === 'online_rl') {
					trainingMode = 'online_tuning';
					if (config['training_config'] && config['training_config']['rl_algorithm']) {
						config['training_config']['rl_algorithm'].default = presetAlgorithm || 'grpo';
					}
				} else {
					trainingMode = 'offline_tuning';
					if (config['training_config'] && config['training_config']['rl_algorithm']) {
						config['training_config']['rl_algorithm'].default = 'none';
					}
				}
			}

			// Build section names dynamically based on training mode
			sectionNames = ['general_config'];

			if (trainingMode === 'offline_tuning') {
				sectionNames.push('tuners_config');
				if (presetGoal !== 'sft') {
					sectionNames.push('tuners_rl_config');
				}
			} else if (trainingMode === 'online_tuning') {
				// Online Tuning: only show RL algorithm section
				sectionNames.push('tuners_rl_config');
			}

			// Set initial selected section
			if (sectionNames.length > 0) {
				selectedSection = sectionNames[0];
			}
			// Get tuners if available
			tuners = config.tuners_config ? Object.keys(config.tuners_config).sort() : [];
			// Set initial selected tuner based on training mode
			if (tuners.length > 0 && !selectedTuner && trainingMode === 'offline_tuning') {
				// For offline tuning, default to 'lora'
				if (tuners.includes('lora')) {
					selectedTuner = 'lora';
				} else {
					selectedTuner = tuners[0];
				}
			}

			// Get RL tuners if available
			rlTuners = config.tuners_rl_config ? Object.keys(config.tuners_rl_config).sort() : [];
			// Set initial selected RL tuner
			if (rlTuners.length > 0 && !selectedRlTuner) {
				if (trainingMode === 'offline_tuning') {
					// For offline tuning, default to 'none' (user can enable RL if needed)
					selectedRlTuner = 'none';
				} else {
					// For online tuning, select first available online algorithm
					selectedRlTuner = availableRlTuners[0] || rlTuners[0];
				}
			}

			// Apply preset algorithm from wizard if provided
			if (presetAlgorithm) {
				const sftAlgos = ['lora', 'sft', 'alora', 'lokr', 'loha', 'vera'];
				const rlAlgos = ['dpo', 'kto', 'ppo', 'grpo', 'dapo'];
				if (sftAlgos.includes(presetAlgorithm) && tuners.includes(presetAlgorithm)) {
					selectedTuner = presetAlgorithm;
				} else if (rlAlgos.includes(presetAlgorithm) && rlTuners.includes(presetAlgorithm)) {
					selectedRlTuner = presetAlgorithm;
				}
			}

			// Sync previousTrainingMode so the training-mode-change reactive
			// doesn't override the preset selections we just applied
			previousTrainingMode = trainingMode;

			isLoading = false;
		} catch (err: any) {
			console.error('Error fetching config:', err);
			// handleResponse throws the parsed problem-detail ({title, status, detail}) on
			// failure, not an Error — so read its detail rather than a missing .message.
			error =
				err?.detail || err?.message || err?.title || 'Could not load the configuration template.';
			isLoading = false;
		}
	}

	// Load configuration when component mounts
	onMount(() => {
		fetchConfigData();
	});

	// Handle section selection
	function handleSectionChange(sectionName: string) {
		selectedSection = sectionName;
		selectedIndex = sectionNames.indexOf(sectionName);
	}

	// Reset configuration to default values
	function resetConfig() {
		fetchConfigData();
	}

	// Get current section data
	$: currentSection =
		config && selectedSection ? config[selectedSection as keyof ConfigData] : null;

	// Helper function to check if a value is an object
	function isObject(item: any) {
		return item && typeof item === 'object' && !Array.isArray(item);
	}

	function getOption(option: 'uniform' | 'loguniform' | 'choice') {
		if (option === 'uniform') {
			return 'Uniform sampling';
		} else if (option === 'loguniform') {
			return 'Logarithmic sampling';
		} else {
			return Utils.toUpperCase(option);
		}
	}

	// Decide whether a field in the generic (advanced) section view should be rendered.
	// Hides:
	//   - fields explicitly marked required: false
	//   - fields gated by search_alg that don't match the current selection
	//   - fields gated by scheduler that don't match the current selection
	function shouldShowField(section: any, value: any): boolean {
		if (!isObject(value)) return true;
		if (value.required === false) return false;
		if (Array.isArray(value.search_alg)) {
			const current = section?.search_alg?.default;
			if (!current || !value.search_alg.includes(current)) return false;
		}
		if (Array.isArray(value.scheduler)) {
			const current = section?.scheduler?.default;
			if (!current || !value.scheduler.includes(current)) return false;
		}
		return true;
	}
</script>

<div class="container" style="display: block; position: relative">
	{#if isLoading}
		<div class="loading">
			<p>Loading configuration data...</p>
			<Loading withOverlay={false} />
		</div>
	{:else if error}
		<div class="error">
			<p>Error loading configuration: {error}</p>
			<button on:click={fetchConfigData}>Retry</button>
		</div>
	{:else if config}
		<header>
			<Row style="margin-bottom: 1rem;">
				{#if !hideNameField}
					<Column md={8}>
						<TextInput
							labelText="Configuration name:"
							id="config_name"
							placeholder="Enter config name"
							bind:value={config.name}
							invalid={configurations?.map((item) => item.name).includes(config?.name ?? '')}
							invalidText={`${config?.name} already exist`}
							on:blur={() => {
								config.name = config?.name?.split(' ').join('_');
							}}
						/>
					</Column>
					<!-- {:else}
					<Column md={8}>
						<div style="padding: 0.5rem 0;">
							<strong>Editing: {config.name}</strong>
						</div>
					</Column> -->
				{/if}
			</Row>
			<Row style="margin-bottom: 1rem; align-items: flex-end;">
				{#if !presetGoal}
					<Column md={4}>
						<RadioButtonGroup
							legendText="Tuning Mode"
							bind:selected={trainingMode}
							orientation="horizontal"
							disabled={editMode}
						>
							<RadioButton value="offline_tuning" labelText="Offline Tuning" />
							<RadioButton value="online_tuning" labelText="Online Tuning" />
						</RadioButtonGroup>
					</Column>
				{/if}
				<Column md={2}>
					<Toggle
						class="mode-toggle"
						labelA="Basic"
						labelB="Advanced"
						labelText="Configuration Mode"
						bind:toggled={mode}
						size="sm"
						on:toggle={(e) => {
							if (e.detail.toggled) {
								// Advanced mode: show all relevant sections
								let allSections = Object.keys(config).filter(
									(key) =>
										key !== 'name' &&
										key !== 'tuner_type' &&
										key !== 'rl_tuner_type' &&
										typeof config[key] === 'object'
								);

								// Only show training_rl_config in Online Tuning mode
								if (trainingMode !== 'online_tuning') {
									allSections = allSections.filter((key) => key !== 'training_rl_config');
								}

								// Hide tuners_config in Online Tuning mode (only uses RL algorithms)
								if (trainingMode === 'online_tuning') {
									allSections = allSections.filter((key) => key !== 'tuners_config');
								}

								// Hide tuners_rl_config in SFT mode
								if (presetGoal === 'sft') {
									allSections = allSections.filter((key) => key !== 'tuners_rl_config');
								}

								// Sort sections by canonical order
								allSections.sort((a, b) => {
									const ai = SECTION_ORDER.indexOf(a);
									const bi = SECTION_ORDER.indexOf(b);
									return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi);
								});

								// Check if current section is still available, otherwise reset
								if (allSections.includes(selectedSection)) {
									selectedIndex = allSections.indexOf(selectedSection);
								} else {
									selectedSection = allSections[0];
									selectedIndex = 0;
								}

								// Update sectionNames after selectedIndex is set
								sectionNames = allSections;
							} else {
								// Basic mode: show sections dynamically based on training mode
								const newSections = ['general_config'];

								if (trainingMode === 'offline_tuning') {
									newSections.push('tuners_config');
									if (presetGoal !== 'sft') {
										newSections.push('tuners_rl_config');
									}
								} else if (trainingMode === 'online_tuning') {
									newSections.push('tuners_rl_config');
								}

								// Check if current section is still available, otherwise reset
								if (newSections.includes(selectedSection)) {
									selectedIndex = newSections.indexOf(selectedSection);
								} else {
									selectedSection = newSections[0];
									selectedIndex = 0;
								}

								// Update sectionNames after selectedIndex is set
								sectionNames = newSections;
							}
						}}
					></Toggle>
				</Column>
			</Row>
			<Row style="margin-bottom: 1rem;">
				<Column>
					<ContentSwitcher bind:selectedIndex>
						{#each sectionNames.map((name) => {
							const labelMap = { tuners_rl_config: 'RL Tuners', training_rl_config: 'RL Training' };
							return { id: name, text: labelMap[name] || Utils.toUpperCase(name.replace('_config', '')) };
						}) as section}
							<Switch text={section.text} on:click={() => handleSectionChange(section.id)} />
						{/each}
					</ContentSwitcher>
				</Column>
			</Row>
		</header>
		<main
			style={selectedSection != 'tuners_config'
				? 'max-height:400px; overflow-y: scroll; overflow-x: hidden'
				: null}
		>
			<Row>
				{#if selectedSection === 'general_config'}
					<GeneralConfigForm bind:config />
				{:else if selectedSection === 'tuners_rl_config' && availableRlTuners.length > 0}
					<Column>
						<Row style="margin-bottom: 1rem;">
							<Column>
								<Dropdown
									titleText="RL Algorithm type:"
									bind:selectedId={selectedRlTuner}
									items={availableRlTuners.map((name) => ({
										id: name,
										text:
											name === 'none'
												? 'No RL Algorithm (NONE)'
												: `${config.tuners_rl_config[name].description} (${Utils.toUpperCase(
														name
												  )})`
									}))}
									on:select={(e) => {
										selectedRlTuner = e.detail.selectedId;
									}}
								/>
							</Column>
						</Row>
						<div
							class="config-section"
							style="max-height: 336px; overflow-y: scroll; overflow-x: hidden"
						>
							{#if selectedRlTuner === 'none'}
								<div class="hyperparams-section" style="height: 300px; text-align: center;">
									<p style="color: #525252; font-size: 14px;">
										No RL algorithm selected. Using default finetuning approach.
									</p>
								</div>
							{:else if config.tuners_rl_config[selectedRlTuner]}
								<div class="hyperparams-section">
									<h5>RL Hyperparameter search space settings</h5>
									{#each Object.entries(config.tuners_rl_config[selectedRlTuner].hyperparams) as [paramName, paramConfig]}
										<div class="config-item" style="margin-bottom: 0.5rem; padding:1rem 0;">
											<FormLabel for={paramName}>
												<span style="font-size:14px; font-weight:600"
													>{Utils.toUpperCase(paramConfig.description)}</span
												>
											</FormLabel>

											{#if paramConfig.values && (paramConfig.type === 'int' || paramConfig.type === 'float')}
												<Row>
													<Column sm={1}>
														<Select
															id={paramName}
															labelText="Strategy"
															bind:selected={paramConfig.strategy}
															on:change={() =>
																(config.tuners_rl_config[selectedRlTuner].hyperparams[paramName] =
																	paramConfig)}
														>
															{#each paramConfig.options as option}
																<SelectItem value={option} text={getOption(option)} />
															{/each}
														</Select>
													</Column>
													<Column sm={1}>
														<NumberInput
															type="number"
															labelText="Default"
															min={paramConfig.min_val}
															max={paramConfig.max_val}
															id={paramName}
															invalidText={`Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}`}
															step={paramConfig.type === 'float' ? 0.01 : 1}
															bind:value={paramConfig.default}
														/>
													</Column>

													{#if paramConfig.strategy === 'uniform'}
														<Column sm={1}>
															<NumberInput
																labelText="Min value"
																type="number"
																id={paramName}
																step={paramConfig.type === 'float' ? 0.01 : 1}
																bind:value={paramConfig.min_val}
															/>
														</Column>
														<Column sm={1}>
															<NumberInput
																labelText="Max value"
																type="number"
																id={paramName}
																step={paramConfig.type === 'float' ? 0.01 : 1}
																bind:value={paramConfig.max_val}
															/>
														</Column>
													{:else}
														<Column sm={2}>
															<TextInput
																labelText="Values"
																type="text"
																id={paramName}
																bind:value={paramConfig.values}
																invalid={errorFields[paramName]?.error}
																invalidText={errorFields[paramName]?.message}
																on:change={(e) => {
																	errorFields[paramName] = {
																		error: e.detail
																			?.split(',')
																			.some(
																				(val) =>
																					val < paramConfig.min_val || val > paramConfig.max_val
																			),
																		message: `Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}`
																	};
																	if (errorFields[paramName]?.error) {
																		return;
																	}
																	config.tuners_rl_config[selectedRlTuner].hyperparams[
																		paramName
																	].values = e.detail
																		?.split(',')
																		.sort((a, b) => a - b)
																		.map((val) => Number(val));
																}}
															/>
														</Column>
													{/if}
												</Row>
											{:else if paramConfig.options?.length === 1 && paramConfig.strategy === 'string'}
												<Row>
													<Column>
														<TextInput id={paramName} bind:value={paramConfig.default} />
													</Column>
												</Row>
											{:else if paramConfig.options?.length === 1 && paramConfig.type === 'str'}
												<Row>
													<Column sm={1}>
														<Select
															id={paramName}
															labelText="Strategy"
															bind:selected={paramConfig.strategy}
															on:change={() =>
																(config.tuners_rl_config[selectedRlTuner].hyperparams[paramName] =
																	paramConfig)}
														>
															{#each paramConfig.options as option}
																<SelectItem value={option} text={getOption(option)} />
															{/each}
														</Select>
													</Column>
													<Column sm={1}>
														<Select
															labelText="Default"
															id={paramName}
															bind:selected={paramConfig.default}
														>
															{#each paramConfig.values as option}
																<SelectItem value={option} text={option} />
															{/each}
														</Select>
													</Column>
													<Column>
														<MultiSelect
															labelText="Values"
															id={paramName}
															on:select={(e) => {
																if (!e.detail.selectedIds.includes(paramConfig.default)) {
																	paramConfig.default = e.detail.selectedIds[0];
																}
																paramConfig.values = e.detail.selectedIds;
															}}
															items={configCopy.tuners_rl_config[selectedRlTuner].hyperparams[
																paramName
															].values.map((item) => {
																return {
																	id: item,
																	text: item,
																	disabled:
																		paramConfig.values.length === 1 && item === paramConfig.default
																};
															})}
															bind:selectedIds={paramConfig.values}
														/>
													</Column>
												</Row>
											{:else if paramConfig.type === 'bool'}
												<Row>
													<Column>
														<Checkbox bind:checked={paramConfig.default} id={paramName} />
													</Column>
												</Row>
											{/if}
										</div>
									{/each}
								</div>
							{/if}
						</div>
					</Column>
				{:else if selectedSection === 'training_rl_config' && config['training_rl_config']}
					<Column>
						<div
							class="config-section"
							style="max-height: 336px; overflow-y: scroll; overflow-x: hidden"
						>
							<Row class="standard-section">
								{#each Object.entries(config['training_rl_config']) as [key, value]}
									{#if value.type !== 'bool'}
										<Column class="config-item" md={4}>
											{#if isObject(value)}
												<div class="input-container">
													{#if value.type === 'str' && value.values?.length > 0}
														<Select
															labelText={Utils.toUpperCase(key)}
															helperText={value.description}
															bind:selected={value.default}
														>
															{#each value.values as option}
																<SelectItem value={option} text={option} />
															{/each}
														</Select>
													{:else if value.type === 'int' || value.type === 'float'}
														<NumberInput
															labelText={Utils.toUpperCase(key)}
															helperText={value.description}
															id={key}
															bind:value={value.default}
															min={value.min_val}
															max={value.max_val}
															step={value.type === 'float' ? 0.01 : 1}
														/>
													{:else}
														<TextInput
															labelText={Utils.toUpperCase(key)}
															helperText={value.description}
															placeholder={`Enter ${key}`}
															bind:value={value.default}
														/>
													{/if}
												</div>
											{/if}
										</Column>
									{/if}
								{/each}
							</Row>
						</div>
					</Column>
				{:else}
					<Column>
						{#if selectedSection === 'tuners_config' && tuners.length > 0}
							<Row style="margin-bottom: 1rem;">
								<Column>
									<Dropdown
										titleText="Tuner type:"
										bind:selectedId={selectedTuner}
										items={tuners.map((name) => ({
											id: name,
											text: `${config.tuners_config[name].description} (${Utils.toUpperCase(name)})`
										}))}
										on:select={(e) => {
											selectedTuner = e.detail.selectedId;
										}}
									/>
								</Column>
							</Row>
						{/if}
						<div
							class="config-section"
							style="max-height: 336px; overflow-y: scroll; overflow-x: hidden"
						>
							{#if selectedSection === 'tuners_config' && config.tuners_config[selectedTuner]}
								<div class="hyperparams-section">
									<h5>Hyperparameter search space settings</h5>
									{#each Object.entries(config.tuners_config[selectedTuner].hyperparams) as [paramName, paramConfig]}
										<div class="config-item" style="margin-bottom: 0.5rem; padding:1rem 0;">
											<FormLabel for={paramName}>
												<span style="font-size:14px; font-weight:600"
													>{Utils.toUpperCase(paramConfig.description)}</span
												>
											</FormLabel>

											{#if paramConfig.values && (paramConfig.type === 'int' || paramConfig.type === 'float')}
												<Row>
													<Column sm={1}>
														<Select
															id={paramName}
															labelText="Strategy"
															bind:selected={paramConfig.strategy}
															on:change={() =>
																(config.tuners_config[selectedTuner].hyperparams[paramName] =
																	paramConfig)}
														>
															{#each paramConfig.options as option}
																<SelectItem value={option} text={getOption(option)} />
															{/each}
														</Select>
													</Column>
													<Column sm={1}>
														<NumberInput
															type="number"
															labelText="Default"
															min={paramConfig.min_val}
															max={paramConfig.max_val}
															id={paramName}
															invalidText={`Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}`}
															step={paramConfig.type === 'float' ? 0.01 : 1}
															bind:value={paramConfig.default}
														/>
													</Column>

													{#if paramConfig.strategy === 'uniform'}
														<Column sm={1}>
															<NumberInput
																labelText="Min value"
																type="number"
																id={paramName}
																step={paramConfig.type === 'float' ? 0.01 : 1}
																bind:value={paramConfig.min_val}
															/>
														</Column>
														<Column sm={1}>
															<NumberInput
																labelText="Max value"
																type="number"
																id={paramName}
																step={paramConfig.type === 'float' ? 0.01 : 1}
																bind:value={paramConfig.max_val}
															/>
														</Column>
													{:else}
														<Column sm={2}>
															<TextInput
																labelText="Values"
																type="text"
																id={paramName}
																bind:value={paramConfig.values}
																invalid={errorFields[paramName]?.error}
																invalidText={errorFields[paramName]?.message}
																on:change={(e) => {
																	errorFields[paramName] = {
																		error: e.detail
																			?.split(',')
																			.some(
																				(val) =>
																					val < paramConfig.min_val || val > paramConfig.max_val
																			),
																		message: `Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}`
																	};
																	if (errorFields[paramName]?.error) {
																		return;
																	}
																	config.tuners_config[selectedTuner].hyperparams[
																		paramName
																	].values = e.detail
																		?.split(',')
																		.sort((a, b) => a - b)
																		.map((val) => Number(val));
																}}
															/>
														</Column>
													{/if}
												</Row>
											{:else if paramConfig.options?.length === 1 && paramConfig.strategy === 'string'}
												<Row>
													<Column>
														<TextInput id={paramName} bind:value={paramConfig.default} />
													</Column>
												</Row>
											{:else if paramConfig.options?.length === 1 && paramConfig.type === 'str'}
												<Row>
													<Column sm={1}>
														<Select
															id={paramName}
															labelText="Strategy"
															bind:selected={paramConfig.strategy}
															on:change={() =>
																(config.tuners_config[selectedTuner].hyperparams[paramName] =
																	paramConfig)}
														>
															{#each paramConfig.options as option}
																<SelectItem value={option} text={getOption(option)} />
															{/each}
														</Select>
													</Column>
													<Column sm={1}>
														<Select
															labelText="Default"
															id={paramName}
															bind:selected={paramConfig.default}
														>
															{#each paramConfig.values as option}
																<SelectItem value={option} text={option} />
															{/each}
														</Select>
													</Column>
													<Column>
														<MultiSelect
															labelText="Values"
															id={paramName}
															on:select={(e) => {
																if (!e.detail.selectedIds.includes(paramConfig.default)) {
																	paramConfig.default = e.detail.selectedIds[0];
																}
																paramConfig.values = e.detail.selectedIds;
															}}
															items={configCopy.tuners_config[selectedTuner].hyperparams[
																paramName
															].values.map((item) => {
																return {
																	id: item,
																	text: item,
																	disabled:
																		paramConfig.values.length === 1 && item === paramConfig.default
																};
															})}
															bind:selectedIds={paramConfig.values}
														/>
													</Column>
												</Row>
											{:else if paramConfig.type === 'bool'}
												<Row>
													<Column>
														<Checkbox bind:checked={paramConfig.default} id={paramName} />
													</Column>
												</Row>
											{/if}
										</div>
									{/each}
								</div>
							{:else if currentSection && mode}
								<Row class="standard-section">
									{#each Object.entries(currentSection) as [key, value]}
										{#if value.type !== 'bool' && shouldShowField(currentSection, value)}
											<!-- Remove if statement to show boolean fields -->
											{#if key !== 'resource_name'}
												<Column class="config-item" md={4}>
													{#if isObject(value)}
														<div class="input-container">
															{#if value.type === 'bool'}
																<Checkbox labelText={key} bind:checked={value.default} id={key} />
															{:else if key === 'max_concurrent_trials'}
																<NumberInput
																	labelText={Utils.toUpperCase(key)}
																	helperText={value.description}
																	id={key}
																	bind:value={value.default}
																	min={value.min_val}
																	max={Math.floor(
																		config['training_config']['num_gpus_per_trial'].max_val /
																			config['training_config']['num_gpus_per_trial'].default
																	)}
																	step={value.type === 'float' ? 0.01 : 1}
																/>
															{:else if key === 'num_gpus_per_trial'}
																<NumberInput
																	labelText={Utils.toUpperCase(key)}
																	helperText={value.description}
																	id={key}
																	bind:value={value.default}
																	min={value.min_val}
																	max={value.max_val}
																	step={value.type === 'float' ? 0.01 : 1}
																	on:change={() => {
																		config['tune_config']['max_concurrent_trials'].default =
																			Math.floor(value.max_val / value.default);
																	}}
																/>
															{:else if key === 'time_budget_s'}
																<TimeInput label={key} bind:value />
															{:else if value.type === 'int' || value.type === 'float'}
																<NumberInput
																	labelText={Utils.toUpperCase(key)}
																	helperText={value.description}
																	id={key}
																	bind:value={value.default}
																	min={value.min_val}
																	max={value.max_val}
																	step={value.type === 'float' ? 0.01 : 1}
																/>
															{:else if value.type === 'str' && value.values?.length > 0}
																<Select
																	labelText={Utils.toUpperCase(key)}
																	helperText={value.description}
																	bind:selected={value.default}
																	on:change={() => {
																		config[selectedSection][key].default = value.default;
																	}}
																>
																	{#each value.values as option}
																		<SelectItem value={option} text={option} />
																	{/each}
																</Select>
															{:else if value.type === 'list'}
																<TextInput
																	labelText={Utils.toUpperCase(key)}
																	helperText={`${value.description} (comma-separated)`}
																	placeholder="tok_a, tok_b, tok_c"
																	value={Array.isArray(value.default)
																		? value.default.join(', ')
																		: value.default ?? ''}
																	on:input={(e) => {
																		value.default = e.detail;
																	}}
																	on:blur={() => {
																		value.default = Utils.parseCommaList(value.default);
																	}}
																/>
															{:else}
																<TextInput
																	labelText={Utils.toUpperCase(key)}
																	helperText={value.description}
																	placeholder={`Enter ${key}`}
																	bind:value={value.default}
																/>
															{/if}
														</div>
													{:else}
														<div class="input-container">
															<TextInput
																labelText={Utils.toUpperCase(key)}
																type="text"
																id={key}
																bind:value={currentSection[key]}
															/>
														</div>
													{/if}
												</Column>
											{/if}
										{/if}
									{/each}
								</Row>
							{/if}
						</div>
					</Column>
				{/if}
			</Row>
		</main>
	{/if}
</div>

<style>
	.config-section {
		background-color: #fff;
		border-radius: 5px;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
	}
	.hyperparams-section {
		padding: 1rem;
	}
	.input-container {
		padding: 1rem;
	}

	.loading {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 40px;
		background-color: #f9f9f9;
		border-radius: 5px;
	}

	.error {
		background-color: #ffebee;
		color: #c62828;
		padding: 20px;
		border-radius: 5px;
		margin-bottom: 20px;
		text-align: center;
	}

	.error button {
		margin-top: 15px;
		padding: 8px 16px;
		background-color: #c62828;
		color: white;
		border: none;
		border-radius: 4px;
		cursor: pointer;
	}

	:global(.bx--number input[type='number']) {
		padding: 1px 0px 1px 16px;
	}

	:global(.mode-toggle .bx--toggle__switch) {
		margin-top: 0.5rem;
	}
</style>
