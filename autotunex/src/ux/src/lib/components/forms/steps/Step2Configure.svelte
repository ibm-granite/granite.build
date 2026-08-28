<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Tile,
		Select,
		SelectItem,
		Button,
		TextInput,
		Tag,
		InlineLoading,
		InlineNotification,
		Grid,
		Row,
		Column
	} from 'carbon-components-svelte';
	import { Add, Settings, Compare, Edit } from 'carbon-icons-svelte';
	import { onMount, createEventDispatcher, tick } from 'svelte';
	import { API } from '$lib/api';
	import { appState, configurations } from '$lib/app';
	import { Utils } from '$lib/utils';
	import ConfigDisplay from '../../displays/ConfigDisplay.svelte';
	import CreateConfigForm from '../CreateConfigForm.svelte';
	import CompareDialog from '../../CompareDialog.svelte';
	import type {
		Configuration,
		ConfigForm,
		ConfigData,
		PendingConfigData,
		PendingConfigUpdate,
		TuningGoal
	} from '$lib/app-types';

	const dispatch = createEventDispatcher();

	const api = new API();

	export let selectedAlgorithm: string = 'lora';
	export let selectedGoal: TuningGoal | null = null;
	export let selectedConfigId: string | null = null;
	export let selectedConfig: Configuration | null = null;

	// SFT algorithms use tuner_type, RL algorithms use rl_tuner_type
	const sftAlgorithms = ['lora', 'sft', 'alora', 'lokr', 'loha', 'vera'];
	const rlAlgorithms = ['dpo', 'kto', 'ppo', 'grpo', 'dapo'];
	const onlineRlAlgorithms = ['ppo', 'grpo', 'dapo'];

	const SYSTEM_USER_ID = '00000000-0000-0000-0000-000000000001';

	let allConfigs: Configuration[] = $appState.isConfigurationsLoaded ? $configurations || [] : [];
	let suggestedConfigs: Configuration[] =
		allConfigs.length > 0 ? suggestConfigs(allConfigs, selectedGoal, selectedAlgorithm) : [];
	let isLoading = !$appState.isConfigurationsLoaded;
	let prevAlgorithm: string = selectedAlgorithm;
	let prevGoal: TuningGoal | null = selectedGoal;

	// Derive goal from selected algorithm for preset config form
	$: presetGoal = (Utils.ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm)?.category ||
		null) as TuningGoal | null;

	// Config creation state
	let newConfigForm: ConfigForm;
	let newConfigName = '';
	let saveError = '';
	let configTemplateLoaded = false;

	// Config comparison state
	let compareMode = false;
	let compareSelections: Configuration[] = [];
	let showCompareDialog = false;
	let compareRows: Record<string, any>[] = [];

	// Inline config creation state (exported so wizard can disable Next)
	export let isCreatingConfig = false;
	let isLoadingCreateConfig = false;

	// Inline config editing state (exported so wizard can disable Next)
	export let isEditingConfig = false;
	let editableConfig: any = null;
	let editSaveError = '';
	let needsSaveAs = false;
	let editConfigName = '';

	// Determine if editing requires "Save As" (system config or config with jobs)
	$: if (isEditingConfig && selectedConfig) {
		const isSystemConfig = selectedConfig.user_id === SYSTEM_USER_ID;
		const hasAssociatedJobs =
			selectedConfig.associated_jobs && selectedConfig.associated_jobs.length > 0;
		needsSaveAs = isSystemConfig || hasAssociatedJobs;
		if (needsSaveAs && !editConfigName) {
			editConfigName = `${selectedConfig.name}_modified`;
		}
	} else {
		needsSaveAs = false;
	}

	function suggestConfigs(
		configs: Configuration[],
		goal: TuningGoal | null,
		algorithm: string
	): Configuration[] {
		const filtered = configs.filter((config) => {
			const rlTuner = config.rl_tuner_type?.toLowerCase();
			const tunerType = config.tuner_type?.toLowerCase();

			if (goal === 'sft') {
				// SFT goal: show ALL SFT configs (any SFT tuner_type, no RL tuner)
				return sftAlgorithms.includes(tunerType || '') && (!rlTuner || rlTuner === 'none');
			} else if (goal === 'online_rl') {
				// Online RL goal: show ALL online RL configs (ppo, grpo, dapo)
				return onlineRlAlgorithms.includes(rlTuner || '');
			} else if (goal === 'offline_rl') {
				// Offline RL: filter by specific algorithm (dpo vs kto have different data requirements)
				return rlTuner === algorithm;
			}

			// Fallback if goal is null: use algorithm-level matching
			if (sftAlgorithms.includes(algorithm)) {
				return sftAlgorithms.includes(tunerType || '') && (!rlTuner || rlTuner === 'none');
			} else if (rlAlgorithms.includes(algorithm)) {
				return rlTuner === algorithm;
			}
			return true;
		});

		// System configs first, then alphabetical by name
		return filtered.sort((a, b) => {
			const aIsSystem = a.user_id === SYSTEM_USER_ID;
			const bIsSystem = b.user_id === SYSTEM_USER_ID;
			if (aIsSystem !== bIsSystem) return aIsSystem ? -1 : 1;
			return (a.name || '').localeCompare(b.name || '');
		});
	}

	let isLoadingEditConfig = false;

	async function enterEditMode() {
		if (!selectedConfig) return;
		if (isCreatingConfig) cancelCreateMode();
		isEditingConfig = true;
		editSaveError = '';
		editConfigName = selectedConfig.name;
		isLoadingEditConfig = true;

		try {
			// Fetch full config data (list endpoint may have incomplete config_data)
			const fullConfig = await api.getConfiguration(selectedConfig.id);
			selectedConfig = fullConfig;
			editableConfig = {
				name: fullConfig.name,
				tuner_type: fullConfig.tuner_type,
				rl_tuner_type: fullConfig.rl_tuner_type || '',
				...fullConfig.config_data
			};
		} catch (err: any) {
			editSaveError = 'Failed to load configuration details for editing.';
			console.error('Error loading config for edit:', err);
		} finally {
			isLoadingEditConfig = false;
		}
	}

	function cancelEditMode() {
		isEditingConfig = false;
		editableConfig = null;
		editSaveError = '';
		editConfigName = '';
		needsSaveAs = false;
	}

	function confirmConfigEdit() {
		if (!selectedConfig || !editableConfig) return;

		// Extract structured config data from the flat editableConfig
		const tuner_type = editableConfig.tuner_type || selectedConfig.tuner_type;
		const rl_tuner_type = editableConfig.rl_tuner_type || selectedConfig.rl_tuner_type || null;
		const configData: ConfigData = {
			tune_config: editableConfig.tune_config,
			tuners_config: editableConfig.tuners_config,
			training_config: editableConfig.training_config,
			...(editableConfig.training_rl_config
				? { training_rl_config: editableConfig.training_rl_config }
				: {}),
			...(editableConfig.tuners_rl_config
				? { tuners_rl_config: editableConfig.tuners_rl_config }
				: {})
		};

		if (needsSaveAs) {
			// --- Save As New: reuse the existing pendingConfig flow ---
			if (!editConfigName || !editConfigName.trim()) {
				editSaveError = 'Please provide a name for the new configuration.';
				return;
			}
			if (allConfigs.some((c) => c.name === editConfigName.trim())) {
				editSaveError = `A configuration named "${editConfigName.trim()}" already exists.`;
				return;
			}
			editSaveError = '';

			const pendingData: PendingConfigData = {
				name: editConfigName.trim(),
				tuner_type: tuner_type || null,
				rl_tuner_type: rl_tuner_type,
				config_data: configData
			};

			const virtualConfig = {
				id: '__pending__',
				user_id: '',
				name: pendingData.name,
				tuner_type: pendingData.tuner_type,
				rl_tuner_type: pendingData.rl_tuner_type,
				artifact_id: '',
				artifact_url: '',
				config_data: pendingData.config_data,
				associated_jobs: [],
				created_at: new Date(),
				updated_at: new Date()
			} as Configuration;

			selectedConfigId = '__pending__';
			selectedConfig = virtualConfig;
			dispatch('pendingConfig', pendingData);
		} else {
			// --- In-place update: dispatch deferred update to wizard ---
			if (!editConfigName || !editConfigName.trim()) {
				editSaveError = 'Please provide a configuration name.';
				return;
			}
			// Check for duplicate name (skip if name unchanged)
			if (
				editConfigName.trim() !== selectedConfig.name &&
				allConfigs.some((c) => c.name === editConfigName.trim())
			) {
				editSaveError = `A configuration named "${editConfigName.trim()}" already exists.`;
				return;
			}
			editSaveError = '';

			const updatedName = editConfigName.trim();
			const pendingUpdate: PendingConfigUpdate = {
				configId: selectedConfig.id,
				name: updatedName,
				tuner_type: tuner_type || null,
				rl_tuner_type: rl_tuner_type,
				config_data: configData
			};

			// Update selectedConfig locally so preview and Review step show edits
			selectedConfig = {
				...selectedConfig,
				name: updatedName,
				tuner_type: tuner_type,
				rl_tuner_type: rl_tuner_type,
				config_data: configData
			};

			dispatch('pendingConfigUpdate', pendingUpdate);

			// Update local config lists so the dropdown reflects the new name
			const idx = suggestedConfigs.findIndex((c) => c.id === selectedConfig!.id);
			if (idx >= 0) {
				suggestedConfigs[idx] = { ...suggestedConfigs[idx], name: updatedName };
				suggestedConfigs = suggestedConfigs;
			}
			const allIdx = allConfigs.findIndex((c) => c.id === selectedConfig!.id);
			if (allIdx >= 0) {
				allConfigs[allIdx] = { ...allConfigs[allIdx], name: updatedName };
				allConfigs = allConfigs;
			}
		}

		cancelEditMode();
	}

	async function selectConfig(config: Configuration) {
		if (compareMode) {
			toggleCompareSelection(config);
			return;
		}
		if (isEditingConfig) cancelEditMode();
		if (isCreatingConfig) cancelCreateMode();
		selectedConfigId = config.id;
		selectedConfig = config;
		// Notify parent to clear any pending config when switching to existing
		dispatch('clearPendingConfig');

		// Eagerly fetch full config_data if not already loaded (list endpoint omits it)
		if (!config.config_data && config.id) {
			try {
				const fullConfig = await api.getConfiguration(config.id);
				selectedConfig = fullConfig;
			} catch (err) {
				console.error('Failed to load full config data:', err);
			}
		}
	}

	function toggleCompareSelection(config: Configuration) {
		const idx = compareSelections.findIndex((c) => c.id === config.id);
		if (idx >= 0) {
			compareSelections = compareSelections.filter((c) => c.id !== config.id);
		} else if (compareSelections.length < 2) {
			compareSelections = [...compareSelections, config];
		}
	}

	function openCompare() {
		if (compareSelections.length === 2) {
			compareRows = buildCompareRows(compareSelections);
			showCompareDialog = true;
		}
	}

	function buildCompareRows(configs: Configuration[]): Record<string, any>[] {
		return configs.map((config) => {
			const row: Record<string, any> = { id: config.name };
			row['tuner_type'] = config.tuner_type || 'N/A';
			if (config.rl_tuner_type && config.rl_tuner_type !== 'none') {
				row['rl_tuner_type'] = config.rl_tuner_type;
			}
			// Flatten config_data hyperparams
			if (config.config_data) {
				const tunerKey = config.tuner_type || 'lora';
				const tunerConfig = config.config_data.tuners_config?.[tunerKey];
				if (tunerConfig?.hyperparams) {
					for (const [key, val] of Object.entries(tunerConfig.hyperparams)) {
						if (val && typeof val === 'object' && 'default' in val) {
							row[key] = (val as any).default;
						}
					}
				}
				// Training config defaults
				if (config.config_data.training_config) {
					for (const [key, val] of Object.entries(config.config_data.training_config)) {
						if (val && typeof val === 'object' && 'default' in val) {
							row[`training_${key}`] = (val as any).default;
						}
					}
				}
			}
			return row;
		});
	}

	async function openCreateForm() {
		if (isEditingConfig) cancelEditMode();
		isCreatingConfig = true;
		saveError = '';
		// Pre-populate name if editing existing pending config
		if (selectedConfigId === '__pending__' && selectedConfig) {
			newConfigName = selectedConfig.name;
		} else {
			newConfigName = '';
		}

		// Load config template if not already loaded
		if (!configTemplateLoaded) {
			isLoadingCreateConfig = true;
			try {
				const template = await api.getConfigurationTemplate();
				newConfigForm = template;
				configTemplateLoaded = true;
			} catch (err: any) {
				saveError = 'Failed to load configuration template.';
				console.error('Error loading config template:', err);
			} finally {
				isLoadingCreateConfig = false;
			}
		}
	}

	function cancelCreateMode() {
		isCreatingConfig = false;
		saveError = '';
		newConfigName = '';
	}

	function confirmNewConfig() {
		if (!newConfigName.trim()) {
			saveError = 'Please enter a configuration name.';
			return;
		}

		// Check duplicate name among existing configs
		if (allConfigs.some((c) => c.name === newConfigName.trim())) {
			saveError = `A configuration named "${newConfigName.trim()}" already exists.`;
			return;
		}

		saveError = '';

		// Extract config sections from the form data
		const { name: _, tuner_type, rl_tuner_type, ...configSections } = newConfigForm;

		const pendingData: PendingConfigData = {
			name: newConfigName.trim(),
			tuner_type: presetGoal === 'online_rl' ? null : tuner_type || 'lora',
			rl_tuner_type: rl_tuner_type || null,
			config_data: configSections as ConfigData
		};

		// Create a virtual Configuration object for display in later steps
		const virtualConfig = {
			id: '__pending__',
			user_id: '',
			name: pendingData.name,
			tuner_type: pendingData.tuner_type,
			rl_tuner_type: pendingData.rl_tuner_type,
			artifact_id: '',
			artifact_url: '',
			config_data: pendingData.config_data,
			associated_jobs: [],
			created_at: new Date(),
			updated_at: new Date()
		} as Configuration;

		selectedConfigId = '__pending__';
		selectedConfig = virtualConfig;
		isCreatingConfig = false;

		// Dispatch to parent wizard for deferred creation at launch
		dispatch('pendingConfig', pendingData);
	}

	onMount(async () => {
		try {
			if (!$appState.isConfigurationsLoaded) {
				const configs = await api.getConfigurations();
				configurations.set(configs);
				appState.update((prev) => ({ ...prev, isConfigurationsLoaded: true }));
			}

			allConfigs = $configurations || [];
			suggestedConfigs = suggestConfigs(allConfigs, selectedGoal, selectedAlgorithm);

			// Auto-select first suggestion
			if (!selectedConfigId && suggestedConfigs.length > 0) {
				selectConfig(suggestedConfigs[0]);
			}
		} catch (err) {
			console.error('Error loading configurations:', err);
		} finally {
			isLoading = false;
		}
	});

	// Reactively update suggestions and reset selection when algorithm or goal changes
	$: if (
		allConfigs.length > 0 &&
		selectedAlgorithm &&
		(selectedAlgorithm !== prevAlgorithm || selectedGoal !== prevGoal)
	) {
		if (isEditingConfig) cancelEditMode();
		if (isCreatingConfig) cancelCreateMode();
		suggestedConfigs = suggestConfigs(allConfigs, selectedGoal, selectedAlgorithm);
		if (suggestedConfigs.length > 0) {
			selectConfig(suggestedConfigs[0]);
		} else {
			selectedConfigId = null;
			selectedConfig = null;
		}
		prevAlgorithm = selectedAlgorithm;
		prevGoal = selectedGoal;
	}
</script>

<Grid noGutter fullWidth>
	{#if isLoading}
		<Row>
			<Column>
				<InlineLoading description="Loading configurations..." />
			</Column>
		</Row>
	{:else}
		{#key selectedAlgorithm}
			{#if isEditingConfig || isCreatingConfig}
				<!-- Edit / Create layout: left sidebar for name + actions, right for config form -->
				<Row>
					<Column sm={4} md={4} lg={6}>
						<Tile style="padding: 1.25rem;">
							{#if isEditingConfig}
								<h5 class="panel-title" style="margin-bottom: 1rem;">Edit Configuration</h5>

								{#if needsSaveAs}
									<InlineNotification
										kind="info"
										lowContrast
										hideCloseButton
										title="Save as new"
										subtitle={selectedConfig?.user_id === SYSTEM_USER_ID
											? 'System config — cannot modify directly.'
											: `Has ${
													selectedConfig?.associated_jobs?.length || 0
											  } job(s) — cannot modify directly.`}
										style="margin-bottom: 0.75rem;"
									/>
									<TextInput
										labelText="New Configuration Name"
										placeholder="Enter new configuration name"
										bind:value={editConfigName}
										invalid={editConfigName.trim() !== '' &&
											allConfigs.some((c) => c.name === editConfigName.trim())}
										invalidText={`"${editConfigName.trim()}" already exists`}
									/>
								{:else}
									<TextInput
										labelText="Configuration Name"
										bind:value={editConfigName}
										invalid={editConfigName.trim() !== '' &&
											editConfigName.trim() !== selectedConfig?.name &&
											allConfigs.some((c) => c.name === editConfigName.trim())}
										invalidText={`"${editConfigName.trim()}" already exists`}
									/>
								{/if}

								{#if editSaveError}
									<InlineNotification
										kind="error"
										title="Error:"
										subtitle={editSaveError}
										on:close={() => (editSaveError = '')}
										style="margin-top: 0.75rem;"
									/>
								{/if}

								<div class="left-panel-actions">
									<Button size="small" kind="ghost" on:click={cancelEditMode}>Cancel</Button>
									<Button
										size="small"
										kind="primary"
										disabled={isLoadingEditConfig ||
											(needsSaveAs && (!editConfigName || editConfigName.trim() === ''))}
										on:click={confirmConfigEdit}
									>
										{needsSaveAs ? 'Confirm as New' : 'Confirm'}
									</Button>
								</div>
							{:else}
								<!-- Create mode left panel -->
								<h5 class="panel-title" style="margin-bottom: 1rem;">New Configuration</h5>
								<TextInput
									labelText="Configuration Name"
									bind:value={newConfigName}
									placeholder="my-config"
									invalid={saveError !== '' && !newConfigName.trim()}
									invalidText="Name is required"
								/>

								{#if saveError}
									<InlineNotification
										kind="error"
										title="Error"
										subtitle={saveError}
										on:close={() => (saveError = '')}
										style="margin-top: 0.75rem;"
									/>
								{/if}

								<div class="left-panel-actions">
									<Button size="small" kind="ghost" on:click={cancelCreateMode}>Cancel</Button>
									<Button
										size="small"
										kind="primary"
										disabled={!newConfigName.trim()}
										on:click={confirmNewConfig}
									>
										Confirm
									</Button>
								</div>
							{/if}
						</Tile>
					</Column>

					<Column sm={4} md={4} lg={10}>
						<Tile class="config-preview-tile">
							{#if isEditingConfig}
								{#if isLoadingEditConfig}
									<InlineLoading description="Loading configuration..." />
								{:else if editableConfig}
									<CreateConfigForm
										bind:config={editableConfig}
										configurations={allConfigs}
										editMode={true}
										existingConfig={selectedConfig}
										hideNameField={true}
										{presetGoal}
										presetAlgorithm={selectedAlgorithm}
									/>
								{/if}
							{:else if isLoadingCreateConfig}
								<InlineLoading description="Loading configuration template..." />
							{:else if configTemplateLoaded && newConfigForm}
								{#key `${presetGoal}-${selectedAlgorithm}`}
									<CreateConfigForm
										bind:config={newConfigForm}
										configurations={allConfigs}
										hideNameField={true}
										{presetGoal}
										presetAlgorithm={selectedAlgorithm}
									/>
								{/key}
							{/if}
						</Tile>
					</Column>
				</Row>
			{:else}
				<!-- View layout: left for selection, right for preview -->
				<Row>
					<Column sm={4} md={4} lg={6}>
						<Tile style="padding: 1.25rem;">
							{#if selectedConfigId === '__pending__' && selectedConfig}
								<!-- Created config: show name + actions -->
								<div class="created-config-card">
									<TextInput bind:value={selectedConfig.name} labelText="Configuration Name" />

									<div class="created-config-actions">
										<Button kind="tertiary" icon={Edit} size="small" on:click={openCreateForm}>
											Edit
										</Button>
										{#if suggestedConfigs.length > 0}
											<Button
												kind="ghost"
												size="small"
												on:click={() => {
													selectedConfigId = null;
													selectedConfig = null;
													dispatch('clearPendingConfig');
												}}
											>
												Choose Existing
											</Button>
										{/if}
									</div>
								</div>
							{:else if suggestedConfigs.length === 0}
								<!-- No matching configs -->
								<p class="suggestion-text">
									{allConfigs.length === 0
										? 'No configurations yet. Create one to get started.'
										: `No matching configurations for ${
												selectedGoal === 'sft'
													? 'SFT'
													: selectedGoal === 'online_rl'
													  ? 'Online RL'
													  : selectedAlgorithm.toUpperCase()
										  }.`}
								</p>
								<Button kind="tertiary" icon={Add} size="small" on:click={openCreateForm}>
									Create New Configuration
								</Button>
							{:else}
								<!-- Configs dropdown (filtered by algorithm) -->
								<Select
									labelText="Configurations"
									selected={selectedConfigId || ''}
									on:change={(e) => {
										const target = e.target;
										if (target instanceof HTMLSelectElement) {
											const found = suggestedConfigs.find((c) => c.id === target.value);
											if (found) selectConfig(found);
										}
									}}
								>
									<SelectItem value="" text="Choose a configuration..." />
									{#each suggestedConfigs as config}
										<SelectItem value={config.id} text={config.name} />
									{/each}
								</Select>
								<div style="margin-top: 0.75rem;">
									<Button kind="tertiary" icon={Add} size="small" on:click={openCreateForm}>
										Create New Configuration
									</Button>
								</div>
							{/if}
						</Tile>
					</Column>

					<Column sm={4} md={4} lg={10}>
						{#if selectedConfig}
							<Tile class="config-preview-tile">
								<div
									style="display: flex; align-items: center; gap: var(--cds-spacing-03, 0.5rem); margin-bottom: 0.75rem;"
								>
									<h6 class="tile-heading">Configuration Preview</h6>
									{#if selectedConfig.id === '__pending__'}
										<Tag type="cyan" size="sm">New</Tag>
									{/if}
									{#if selectedConfig.id !== '__pending__'}
										<div style="margin-left: auto;">
											<Button size="small" kind="primary" icon={Edit} on:click={enterEditMode}>
												Edit
											</Button>
										</div>
									{/if}
								</div>
								<ConfigDisplay
									config_id={selectedConfig.id}
									configuration={selectedConfig}
									showEditButton={false}
								/>
							</Tile>
						{:else}
							<Tile
								style="padding: 2rem; text-align: center; color: var(--cds-text-02, #525252); min-height: 400px; display: flex; align-items: center; justify-content: center;"
							>
								<div>
									<Settings
										size={32}
										style="margin-bottom: var(--cds-spacing-03, 0.5rem); opacity: 0.5;"
									/>
									<p>Select a configuration to preview</p>
								</div>
							</Tile>
						{/if}
					</Column>
				</Row>
			{/if}
		{/key}
	{/if}
</Grid>

<!-- Config comparison dialog -->
<CompareDialog
	bind:open={showCompareDialog}
	entities="Configurations"
	rows={compareRows}
	on:submit={() => {
		showCompareDialog = false;
		compareMode = false;
		compareSelections = [];
	}}
	on:close={() => {
		showCompareDialog = false;
	}}
/>

<style>
	.tile-heading {
		font-weight: 600;
		margin: 0;
	}

	.suggestion-text {
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
		margin-bottom: 0.75rem;
	}

	.suggestion-text.warning {
		color: var(--cds-support-error, #da1e28);
	}

	.preview-row {
		display: flex;
		align-items: center;
		gap: var(--cds-spacing-03, 0.5rem);
		margin-bottom: var(--cds-spacing-03, 0.5rem);
	}

	.preview-label {
		color: var(--cds-text-02, #525252);
		font-weight: 500;
		min-width: 100px;
	}

	.pending-config-hint {
		color: var(--cds-text-02, #525252);
		margin-top: 1rem;
		font-size: 0.8125rem;
	}

	.created-config-card {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.created-config-header {
		margin-bottom: 0.25rem;
	}

	.created-config-name {
		font-weight: 600;
		font-size: 1.125rem;
		color: var(--cds-text-01, #161616);
		margin: 0;
		word-break: break-word;
	}

	.created-config-hint {
		font-size: 0.75rem;
		color: var(--cds-text-helper, #6f6f6f);
		margin: 0.25rem 0 0 0;
	}

	.created-config-actions {
		display: flex;
		gap: 0.25rem;
		margin-top: 0.75rem;
		padding-top: 0.75rem;
	}

	:global(.config-preview-tile) {
		padding: 1.25rem;
		max-height: calc(100vh - 16rem);
		overflow: auto;
	}

	/* Left sidebar panel in edit/create mode */
	.panel-title {
		font-size: 0.9375rem;
		font-weight: 600;
		margin: 0;
		color: var(--cds-text-01, #161616);
		letter-spacing: 0.01em;
	}

	.left-panel-actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 1.5rem;
		padding-top: 1rem;
		border-top: 1px solid var(--cds-border-subtle, #e0e0e0);
	}
</style>
