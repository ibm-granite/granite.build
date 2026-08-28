<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Tile,
		TextInput,
		Tag,
		Button,
		ProgressBar,
		Grid,
		Row,
		Column
	} from 'carbon-components-svelte';
	import { DataBase, Settings, ModelTuned, Checkmark, Edit } from 'carbon-icons-svelte';
	import { createEventDispatcher } from 'svelte';
	import { Utils } from '$lib/utils';
	import { ModelSource } from '$lib/app-types';
	import type {
		DatasetForm,
		Dataset,
		Configuration,
		LaunchPhase,
		Resources,
		ColumnMetadata
	} from '$lib/app-types';

	const dispatch = createEventDispatcher();

	export let uploadedFile: File | null = null;
	export let datasetForm: DatasetForm;
	export let selectedExistingDataset: Dataset | null = null;
	export let selectedConfig: Configuration | null = null;
	export let selectedModel: string = '';
	export let modelSource: ModelSource = ModelSource.HuggingFace;
	export let experimentName: string = '';
	export let isPendingDataset: boolean = false;
	export let isPendingConfig: boolean = false;
	export let launchPhase: LaunchPhase = null;
	export let uploadProgress: number = 0;
	export let resourceEstimation: Resources | null = null;
	export let totalRecords: number = 0;
	export let splitRatio: number = 80;
	export let isSplitEnabled: boolean = true;
	export let validationFile: File | null = null;
	export let autotuneEnabled: boolean = true;
	export let columnMetadata: ColumnMetadata[] = [];

	// Derived dataset values — works for both new and existing datasets
	$: isExisting = !!selectedExistingDataset;
	$: trainFileName = isExisting ? selectedExistingDataset!.train_file : uploadedFile?.name ?? null;
	$: valFileName = isExisting
		? selectedExistingDataset!.validation_file
		: validationFile?.name ?? (isSplitEnabled ? 'Auto-split from train' : null);
	$: trainRecords = isExisting
		? selectedExistingDataset!.train_records
		: isSplitEnabled
		  ? Math.round((totalRecords * splitRatio) / 100)
		  : totalRecords;
	// null = unknown (separate validation file uploaded, not parsed)
	$: valRecords = isExisting
		? selectedExistingDataset!.validation_records || 0
		: isSplitEnabled
		  ? Math.round((totalRecords * (100 - splitRatio)) / 100)
		  : null;
	$: trainFileSize = isExisting
		? selectedExistingDataset!.train_file_size
		: uploadedFile?.size ?? 0;
	$: valFileSize = isExisting
		? selectedExistingDataset!.validation_file_size
		: validationFile?.size ?? 0;

	// Shorthand accessors for config_data sections
	$: tuneConfig = selectedConfig?.config_data?.tune_config ?? null;
	$: trainingConfig = selectedConfig?.config_data?.training_config ?? null;

	// Safely extract dynamic training_config fields (index-signature keys not on the TS interface)
	$: numGpusPerTrial = (trainingConfig as any)?.num_gpus_per_trial?.default as number | undefined;
	$: maxLength = (trainingConfig as any)?.max_length?.default as number | undefined;
	$: trainImpl = (trainingConfig as any)?.train_implementation?.default as string | undefined;
	$: dsStrategy = (trainingConfig as any)?.ds_strategy?.default as string | undefined;
	$: fsdpStrategy = (trainingConfig as any)?.fsdp_strategy?.default as string | undefined;

	function formatFileSize(bytes: number): string {
		if (!bytes || bytes <= 0) return '';
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatTimeBudget(seconds: number): string {
		if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
		const h = Math.floor(seconds / 3600);
		const m = Math.round((seconds % 3600) / 60);
		return m > 0 ? `${h}h ${m}m` : `${h}h`;
	}

	function formatPercent(val: number): string {
		return `${(val * 100).toFixed(0)}%`;
	}
</script>

<Grid noGutter fullWidth>
	<!-- Experiment name -->
	<Row>
		<Column sm={4} md={4} lg={8}>
			<TextInput
				labelText="Experiment Name"
				bind:value={experimentName}
				placeholder="Enter a unique name for this tuning job"
				on:blur={() => (experimentName = experimentName.trim().replace(/\s+/g, '_'))}
			/>
		</Column>
	</Row>

	<!-- Summary cards -->
	<Row style="margin-top: var(--cds-spacing-06, 1.5rem);">
		<!-- Model card -->
		<Column sm={4} md={4} lg={5}>
			<Tile class="review-card">
				<div class="card-header">
					<ModelTuned size={20} class="card-icon" />
					<h6 class="card-heading">Model</h6>
					<div style="margin-left: auto;">
						<Button
							kind="ghost"
							size="small"
							icon={Edit}
							iconDescription="Edit model"
							on:click={() => dispatch('editStep', 0)}
						/>
					</div>
				</div>
				<div class="card-body">
					<div class="card-row">
						<span class="card-label">Model</span>
						<span class="card-value">{selectedModel}</span>
					</div>
					<div class="card-row">
						<span class="card-label">Source</span>
						<span class="card-value"
							>{modelSource === ModelSource.CustomPath ? 'Custom Path' : 'HuggingFace'}</span
						>
					</div>
					<div class="card-row">
						<span class="card-label">AutoTune</span>
						<span class="card-value">
							<Tag size="sm" type={autotuneEnabled ? 'green' : 'cool-gray'}
								>{autotuneEnabled ? 'Enabled' : 'Disabled'}</Tag
							>
						</span>
					</div>
				</div>
			</Tile>
		</Column>

		<!-- Dataset card -->
		<Column sm={4} md={4} lg={5}>
			<Tile class="review-card">
				<div class="card-header">
					<DataBase size={20} class="card-icon" />
					<h6 class="card-heading">Dataset</h6>
					{#if isPendingDataset}
						<Tag type="cyan" size="sm">New</Tag>
					{/if}
					<div style="margin-left: auto;">
						<Button
							kind="ghost"
							size="small"
							icon={Edit}
							iconDescription="Edit dataset"
							on:click={() => dispatch('editStep', 1)}
						/>
					</div>
				</div>
				<div class="card-body">
					<div class="card-row">
						<span class="card-label">Name</span>
						<span class="card-value">{datasetForm.name}</span>
					</div>
					{#if datasetForm.description}
						<div class="card-row">
							<span class="card-label">Description</span>
							<span class="card-value card-value-truncate" title={datasetForm.description}
								>{datasetForm.description}</span
							>
						</div>
					{/if}

					<!-- Train file section -->
					{#if trainFileName}
						<div class="card-section-divider" />
						<span class="card-section-label">Training</span>
						<div class="card-row">
							<span class="card-label">File</span>
							<span class="card-value card-value-truncate" title={trainFileName}
								>{trainFileName}</span
							>
						</div>
						{#if trainRecords > 0}
							<div class="card-row">
								<span class="card-label">Records</span>
								<span class="card-value">{trainRecords.toLocaleString()}</span>
							</div>
						{/if}
						{#if trainFileSize > 0}
							<div class="card-row">
								<span class="card-label">Size</span>
								<span class="card-value">{formatFileSize(trainFileSize)}</span>
							</div>
						{/if}
					{/if}

					<!-- Validation file section -->
					{#if valFileName}
						<div class="card-section-divider" />
						<span class="card-section-label">Validation</span>
						<div class="card-row">
							<span class="card-label">File</span>
							<span class="card-value card-value-truncate" title={valFileName}>{valFileName}</span>
						</div>
						<div class="card-row">
							<span class="card-label">Records</span>
							<span class="card-value"
								>{valRecords != null ? valRecords.toLocaleString() : '—'}</span
							>
						</div>
						{#if valFileSize > 0}
							<div class="card-row">
								<span class="card-label">Size</span>
								<span class="card-value">{formatFileSize(valFileSize)}</span>
							</div>
						{/if}
					{/if}

					{#if columnMetadata.length > 0}
						<div class="card-section-divider" />
						<div class="card-row" style="align-items: flex-start;">
							<span class="card-label">Columns</span>
							<span class="card-value">
								<div class="column-tags">
									{#each columnMetadata as col}
										<Tag size="sm" type="cool-gray">{col.name}</Tag>
									{/each}
								</div>
							</span>
						</div>
					{/if}
				</div>
			</Tile>
		</Column>

		<!-- Config card -->
		<Column sm={4} md={4} lg={6}>
			<Tile class="review-card">
				<div class="card-header">
					<Settings size={20} class="card-icon" />
					<h6 class="card-heading">Configuration</h6>
					{#if isPendingConfig}
						<Tag type="cyan" size="sm">New</Tag>
					{/if}
					<div style="margin-left: auto;">
						<Button
							kind="ghost"
							size="small"
							icon={Edit}
							iconDescription="Edit configuration"
							on:click={() => dispatch('editStep', 2)}
						/>
					</div>
				</div>
				{#if selectedConfig}
					<div class="card-body">
						<div class="card-row">
							<span class="card-label">Name</span>
							<span class="card-value">{selectedConfig.name}</span>
						</div>
						<div class="card-row">
							<span class="card-label">Algorithm</span>
							<span class="card-value">{Utils.getConfigSummary(selectedConfig)}</span>
						</div>

						<!-- Operational config details -->
						{#if tuneConfig || trainingConfig}
							<div class="card-section-divider" />
							<span class="card-section-label">HPO Settings</span>
							<div class="config-grid">
								{#if tuneConfig?.num_samples}
									<div class="config-field">
										<span class="config-key">Trials</span>
										<span class="config-val">{tuneConfig.num_samples.default}</span>
									</div>
								{/if}
								{#if tuneConfig?.max_concurrent_trials}
									<div class="config-field">
										<span class="config-key">Concurrent</span>
										<span class="config-val">{tuneConfig.max_concurrent_trials.default}</span>
									</div>
								{/if}
								{#if numGpusPerTrial != null}
									<div class="config-field">
										<span class="config-key">GPUs / Trial</span>
										<span class="config-val">{numGpusPerTrial}</span>
									</div>
								{/if}
								{#if tuneConfig?.time_budget_s?.default != null}
									<div class="config-field">
										<span class="config-key">Time Budget</span>
										<span class="config-val"
											>{formatTimeBudget(tuneConfig.time_budget_s.default)}</span
										>
									</div>
								{/if}
								{#if trainingConfig?.hpo_dataset_percentage}
									<div class="config-field">
										<span class="config-key">HPO Data %</span>
										<span class="config-val"
											>{formatPercent(Number(trainingConfig.hpo_dataset_percentage.default))}</span
										>
									</div>
								{/if}
								{#if maxLength != null}
									<div class="config-field">
										<span class="config-key">Max Length</span>
										<span class="config-val">{maxLength.toLocaleString()}</span>
									</div>
								{/if}
							</div>

							{#if trainImpl}
								<div class="card-section-divider" />
								<span class="card-section-label">Distribution</span>
								<div class="config-grid">
									<div class="config-field">
										<span class="config-key">Train Impl</span>
										<span class="config-val">{trainImpl}</span>
									</div>
									{#if trainImpl?.toLowerCase() === 'deepspeed' && dsStrategy}
										<div class="config-field">
											<span class="config-key">DS Strategy</span>
											<span class="config-val">{dsStrategy}</span>
										</div>
									{/if}
									{#if trainImpl?.toLowerCase() === 'fsdp' && fsdpStrategy}
										<div class="config-field">
											<span class="config-key">FSDP Strategy</span>
											<span class="config-val">{fsdpStrategy}</span>
										</div>
									{/if}
								</div>
							{/if}
						{/if}
					</div>
				{:else}
					<p class="empty-hint">No configuration selected</p>
				{/if}
			</Tile>
		</Column>
	</Row>

	<!-- Resource estimation -->
	{#if resourceEstimation}
		<Row style="margin-top: var(--cds-spacing-05, 1rem);">
			<Column>
				<Tile class="review-card">
					<h6 class="card-heading" style="margin-bottom: 0.75rem;">Estimated Resources</h6>
					<div class="resource-row">
						<div class="resource-item">
							<span class="config-key">Model Size</span>
							<span class="config-val"
								>{resourceEstimation.model_size_billion_params.toFixed(1)}B params</span
							>
						</div>
						<div class="resource-item">
							<span class="config-key">GPU Memory</span>
							<span class="config-val">{resourceEstimation.gpu_memory_gb.toFixed(1)} GB</span>
						</div>
						<div class="resource-item">
							<span class="config-key">GPUs Required</span>
							<span class="config-val">{resourceEstimation.num_gpus}</span>
						</div>
						<div class="resource-item">
							<span class="config-key">CPU Memory</span>
							<span class="config-val">{resourceEstimation.cpu_memory_gb.toFixed(1)} GB</span>
						</div>
					</div>
				</Tile>
			</Column>
		</Row>
	{/if}

	<!-- Launch progress banner -->
	{#if launchPhase}
		<Row style="margin-top: var(--cds-spacing-06, 1.5rem);">
			<Column>
				<Tile class="review-card">
					<h6 class="card-heading" style="margin-bottom: 0.75rem;">Launching...</h6>
					<div class="launch-steps">
						{#if isPendingDataset}
							<div
								class="launch-step"
								class:active={launchPhase === 'creating_dataset'}
								class:done={['uploading_files', 'creating_config', 'launching_job'].includes(
									launchPhase || ''
								)}
							>
								{#if ['uploading_files', 'creating_config', 'launching_job'].includes(launchPhase || '')}
									<Checkmark size={16} />
								{/if}
								<span>Create dataset</span>
							</div>
							<div
								class="launch-step"
								class:active={launchPhase === 'uploading_files'}
								class:done={['creating_config', 'launching_job'].includes(launchPhase || '')}
							>
								{#if ['creating_config', 'launching_job'].includes(launchPhase || '')}
									<Checkmark size={16} />
								{/if}
								<span>Upload files</span>
								{#if launchPhase === 'uploading_files' && uploadProgress > 0}
									<div style="flex: 1; max-width: 200px;">
										<ProgressBar value={uploadProgress} max={100} size="sm" />
									</div>
									<span class="progress-label">{uploadProgress}%</span>
								{/if}
							</div>
						{/if}
						{#if isPendingConfig}
							<div
								class="launch-step"
								class:active={launchPhase === 'creating_config'}
								class:done={launchPhase === 'launching_job'}
							>
								{#if launchPhase === 'launching_job'}
									<Checkmark size={16} />
								{/if}
								<span>Create configuration</span>
							</div>
						{/if}
						<div class="launch-step" class:active={launchPhase === 'launching_job'}>
							<span>Launch job</span>
						</div>
					</div>
				</Tile>
			</Column>
		</Row>
	{/if}
</Grid>

<style>
	/* ── Card shell ── */
	:global(.review-card) {
		padding: 1.25rem !important;
		height: 100%;
	}

	.card-header {
		display: flex;
		align-items: center;
		gap: var(--cds-spacing-03, 0.5rem);
		margin-bottom: 1rem;
		padding-bottom: 0.625rem;
		border-bottom: 2px solid var(--cds-border-subtle, #e0e0e0);
	}

	:global(.card-icon) {
		color: var(--cds-interactive, #0f62fe);
		flex-shrink: 0;
	}

	.card-heading {
		font-size: 0.875rem;
		font-weight: 600;
		letter-spacing: 0.01em;
		margin: 0;
		color: var(--cds-text-01, #161616);
	}

	/* ── Card body & rows ── */
	.card-body {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
	}

	.card-row {
		display: flex;
		align-items: center;
		gap: var(--cds-spacing-03, 0.5rem);
		font-size: 0.8125rem;
		line-height: 1.4;
	}

	.card-label {
		color: var(--cds-text-02, #525252);
		font-weight: 500;
		min-width: 76px;
		flex-shrink: 0;
	}

	.card-value {
		color: var(--cds-text-01, #161616);
		word-break: break-word;
	}

	.card-value-truncate {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 220px;
	}

	/* ── Section dividers & labels ── */
	.card-section-divider {
		border-top: 1px solid var(--cds-border-subtle, #e0e0e0);
		margin: 0.5rem 0 0.25rem;
	}

	.card-section-label {
		display: block;
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--cds-text-02, #525252);
		margin-bottom: 0.25rem;
	}

	/* ── Config grid (2-col key/value pairs) ── */
	.config-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.5rem 1.25rem;
	}

	.config-field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.config-key {
		font-size: 0.75rem;
		font-weight: 500;
		color: var(--cds-text-02, #525252);
		letter-spacing: 0.02em;
	}

	.config-val {
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--cds-text-01, #161616);
	}

	/* ── Resource estimation row ── */
	.resource-row {
		display: flex;
		gap: 2rem;
		flex-wrap: wrap;
	}

	.resource-item {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	/* ── Column tags ── */
	.column-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}

	/* ── Empty state ── */
	.empty-hint {
		color: var(--cds-text-02, #525252);
		font-size: 0.8125rem;
	}

	/* ── Launch progress ── */
	.launch-steps {
		display: flex;
		flex-direction: column;
		gap: var(--cds-spacing-03, 0.5rem);
	}

	.launch-step {
		display: flex;
		align-items: center;
		gap: var(--cds-spacing-03, 0.5rem);
		padding: 0.375rem var(--cds-spacing-03, 0.5rem);
		border-radius: 4px;
		font-size: 0.875rem;
		color: var(--cds-text-02, #525252);
	}

	.launch-step.active {
		background: var(--cds-highlight, #edf5ff);
		color: var(--cds-interactive, #0f62fe);
		font-weight: 500;
	}

	.launch-step.done {
		color: var(--cds-support-success, #198038);
	}

	.progress-label {
		font-size: 0.75rem;
		color: var(--cds-text-02, #525252);
	}

	/* White background for TextInput against g10 page */
	:global(.bx--text-input) {
		background-color: #ffffff !important;
	}
</style>
