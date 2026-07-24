<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { DataTableSkeleton, Link, ProgressBar, Tag, Button } from 'carbon-components-svelte';
	import { Download } from 'carbon-icons-svelte';
	import yaml from 'js-yaml';

	import Table from '../Table.svelte';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';
	import CreateConfigForm from '../forms/CreateConfigForm.svelte';
	import ImportConfigsModal from '../forms/ImportConfigsModal.svelte';
	import ExportConfigsModal from '../forms/ExportConfigsModal.svelte';
	import { Utils } from '$lib/utils';
	import TuningDisplay from '../displays/TuningDisplay.svelte';
	import ConfigDisplay from '../displays/ConfigDisplay.svelte';
	import { showLoader, userMetadata, currentUser } from '$lib/store';
	import type { ConfigForm, Job, ImportPreviewRow } from '$lib/app-types';
	import { appState, configurations, notifications } from '$lib/app';

	let api = new API();

	let config: ConfigForm;
	let openView: boolean = false;
	let openCreateConfig: boolean = false;
	let openImport: boolean = false;
	let openExport: boolean = false;
	let entityName: string = 'configuration';
	let selectedId: string[];
	let selectedTuning: Job | null;

	async function handleExport(e: CustomEvent<{ ids: string[]; format: 'json' | 'yaml' }>) {
		const { ids, format } = e.detail;
		if (ids.length === 0) return;

		try {
			const fullConfigs = await Promise.all(ids.map((id) => api.getConfiguration(id)));

			const exportData = fullConfigs.map((c) => ({
				name: c.name,
				tuner_type: c.tuner_type,
				rl_tuner_type: c.rl_tuner_type || null,
				config_data: c.config_data
			}));

			const dataToExport = exportData.length === 1 ? exportData[0] : exportData;

			let content: string;
			let mimeType: string;
			let extension: string;

			if (format === 'json') {
				content = JSON.stringify(dataToExport, null, 2);
				mimeType = 'application/json';
				extension = 'json';
			} else {
				content = yaml.dump(dataToExport, { indent: 2, sortKeys: false });
				mimeType = 'text/yaml';
				extension = 'yaml';
			}

			const filename =
				exportData.length === 1
					? `${exportData[0].name.replace(/\s+/g, '_')}.${extension}`
					: `configurations_export.${extension}`;

			const blob = new Blob([content], { type: mimeType });
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			openExport = false;
		} catch (error) {
			notifications.set({
				show: true,
				kind: 'error',
				title: 'Export failed',
				subtitle: 'Could not export configuration(s)',
				timeout: 5000
			});
		}
	}

	async function handleImportSubmit(
		e: CustomEvent<{
			rowsToImport: ImportPreviewRow[];
			onRowFailed: (rowId: string, message: string) => void;
			onFinished: (okCount: number, failed: boolean) => void;
		}>
	) {
		const { rowsToImport, onRowFailed, onFinished } = e.detail;
		let okCount = 0;
		let failed = false;

		showLoader.set(true);
		try {
			for (const row of rowsToImport) {
				try {
					await api.createConfiguration({
						name: row.editedName,
						tuner_type: row.tunerType ?? '',
						rl_tuner_type: row.rlTunerType ?? null,
						config_data: row.configData
					});
					okCount++;
				} catch (err) {
					failed = true;
					const msg = err instanceof Error ? err.message : 'Unknown error';
					onRowFailed(row.rowId, msg);
					break;
				}
			}

			if (okCount > 0) {
				appState.update((prev) => ({ ...prev, isConfigurationsLoaded: false }));
				await fetchConfigurations();
				userMetadata.update((prev) => ({
					...prev,
					number_of_configurations: prev.number_of_configurations + okCount
				}));
			}

			if (!failed) {
				notifications.set({
					show: true,
					kind: 'success',
					title: 'Import successful',
					subtitle: `${okCount} configuration(s) imported`,
					timeout: 5000
				});
			} else {
				notifications.set({
					show: true,
					kind: 'error',
					title: 'Import partially failed',
					subtitle: `Imported ${okCount} of ${rowsToImport.length} — resolve errors and retry`,
					timeout: 5000
				});
			}
		} finally {
			showLoader.set(false);
			onFinished(okCount, failed);
		}
	}

	let configHeaders = [
		{ key: 'name', value: 'Name' },
		{ key: 'associated_jobs', value: 'Tunings' },
		{
			key: 'created_at',
			value: 'Created on',
			display: (date: Date) => new Date(date).toLocaleString()
		}
	];

	const fetchConfigurations = async () => {
		try {
			if ($appState.isConfigurationsLoaded) {
				return;
			}
			let configurationsData = await api.getConfigurations();
			// configurations.set(configurationsData);
			configurations.update((prev) => {
				// guard prev in case it's undefined
				const prevArr = Array.isArray(prev) ? prev : [];

				// Start with existing configs in a map by id (keeps insertion order of prev)
				const map = new Map(prevArr.map((job) => [job.id, job]));

				for (const config of configurationsData) {
					const oldConfig = map.get(config.id);
					// merge: keep UI-only fields from oldConfig, let API fields override
					map.set(config.id, oldConfig ? { ...oldConfig, ...config } : config);
				}
				return Array.from(map.values());
			});
			appState.update((prev) => {
				return { ...prev, isConfigurationsLoaded: true };
			});
			return configurationsData;
		} catch (error) {}
	};

	onMount(async () => {
		await fetchConfigurations();
	});

	// reset tuning id when modal is closed
	$: if (!openView) {
		selectedTuning = null;
		entityName = 'configuration';
	}
</script>

{#if $appState.isConfigurationsLoaded}
	<Table
		title="Configurations"
		entity={entityName}
		entities="configurations"
		actionButtonText="Create New Configuration"
		description="Shows your configurations."
		headers={configHeaders}
		rows={$configurations}
		expandable={false}
		disableDeleteButton={$configurations
			?.filter((conf) => selectedId?.includes(conf.id))
			.map((item) => item.user_id)
			?.includes('00000000-0000-0000-0000-000000000001') &&
			$currentUser?.email?.toLowerCase() !== 'system@example.com'}
		bind:openView
		bind:selectedRowIds={selectedId}
		submitBtnDisable={config?.name === '' ||
			$configurations?.map((item) => item.name).includes(config?.name ?? '')}
		bind:openNew={openCreateConfig}
		on:new={async () => {
			showLoader.set(true);

			// Build config_data dynamically based on training mode
			const config_data = { ...config };

			// Infer training mode from tuner_type and rl_tuner_type
			const hasTunerType = config.tuner_type !== null && config.tuner_type !== '';
			const hasRlTunerType = config.rl_tuner_type !== null && config.rl_tuner_type !== '';

			let trainingMode = 'offline_tuning';
			if (hasRlTunerType) {
				// Check if RL algorithm is online type
				const onlineRlTypes = ['ppo', 'grpo', 'dapo'];
				const isOnline = onlineRlTypes.includes(config.rl_tuner_type?.toLowerCase() || '');

				if (isOnline) {
					trainingMode = 'online_tuning';
				} else {
					// Offline RL (DPO/KTO) paired with tuning algorithm
					trainingMode = 'offline_tuning';
				}
			} else {
				trainingMode = 'offline_tuning';
			}

			// Filter tuners_config only if tuner_type is set
			if (config.tuner_type && config['tuners_config']) {
				config_data.tuners_config = Utils.filterObject(
					config['tuners_config'],
					(key) => key === config.tuner_type
				);
			}

			// Filter tuners_rl_config only if rl_tuner_type is set
			if (config.rl_tuner_type && config['tuners_rl_config']) {
				config_data.tuners_rl_config = Utils.filterObject(
					config['tuners_rl_config'],
					(key) => key === config.rl_tuner_type
				);
			}

			// Remove top-level fields that should not be in config_data
			delete config_data.name;
			delete config_data.tuner_type;
			delete config_data.rl_tuner_type;

			Utils.normalizeTokenizerListFields(config_data);

			// // Remove unnecessary sections based on training mode
			// if (trainingMode === 'default_finetuning') {
			// 	// Default Finetuning doesn't need any RL-related configs
			// 	config_data.training_rl_config = {};
			// 	config_data.tuners_rl_config = {};
			// } else if (trainingMode === 'offline_rl') {
			// 	// Offline RL doesn't need training_rl_config (no rollout generation)
			// 	delete config_data.training_rl_config;
			// } else if (trainingMode === 'online_rl') {
			// 	// Online RL doesn't need tuners_config (no traditional tuning algorithms)
			// 	config_data.tuners_config = {};
			// }

			let config_payload = {
				name: config.name,
				tuner_type: config.tuner_type,
				rl_tuner_type: config.rl_tuner_type,
				config_data: config_data
			};
			if (!config_payload.name || config_payload.name === '') {
				return;
			}
			try {
				await api.createConfiguration(config_payload);
				appState.update((prev) => {
					return { ...prev, isConfigurationsLoaded: false };
				});
				await fetchConfigurations();
				openCreateConfig = false;
				showLoader.set(false);
				userMetadata.update((prev) => {
					return { ...prev, number_of_configurations: prev.number_of_configurations + 1 };
				});
			} catch (e) {
				showLoader.set(false);
				const body = await Promise.resolve(e).catch(() => null);
				const subtitle =
					(body && typeof body === 'object' && 'detail' in body && String(body.detail)) ||
					'Could not create configuration';
				notifications.set({
					show: true,
					kind: 'error',
					title: 'Create configuration failed',
					subtitle,
					timeout: 5000
				});
			}
		}}
		on:delete={async (e) => {
			for (let id of e.detail) {
				await api.deleteConfiguration(id);
				let updatedConfigs = $configurations.filter((config) => !e.detail?.includes(config.id));
				configurations.set(updatedConfigs);
				userMetadata.update((prev) => {
					return { ...prev, number_of_configurations: prev.number_of_configurations - 1 };
				});
			}
		}}
		on:view={(e) => {
			entityName = e.detail.row.name;
		}}
	>
		<svelte:fragment slot="batch-actions">
			<Button
				icon={Download}
				on:click={() => {
					openExport = true;
				}}
			>
				Export
			</Button>
		</svelte:fragment>
		<svelte:fragment slot="toolbar-actions">
			<Button kind="tertiary" on:click={() => (openImport = true)}>Import</Button>
		</svelte:fragment>
		<svelte:fragment slot="cell" let:cell let:row>
			{#if cell.key === 'name'}
				<Link
					on:click={() => {
						selectedId = [row.id];
						entityName = row.name;
						openView = true;
					}}
					href="#">{cell.value}</Link
				>
			{:else if cell.key === 'associated_jobs'}
				{#each cell.value as tuning}
					<Tag
						style="cursor: pointer;"
						on:click={() => {
							selectedTuning = tuning;
							entityName = tuning?.experiment_name;
							openView = true;
						}}>{tuning.experiment_name}</Tag
					>
				{/each}
			{:else}
				{cell.display ? cell.display(cell.value, row) : cell.value}
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="create">
			<CreateConfigForm bind:config configurations={$configurations} />
		</svelte:fragment>
		<svelte:fragment slot="expanded-row" let:row>
			{#if !row.detail}
				<ProgressBar size="sm" helperText="Loading details..." />
			{:else}
				<code>
					<pre>{JSON.stringify(row.detail, null, 2)}</pre>
				</code>
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="view" let:selectedRows>
			{#if selectedTuning}
				<TuningDisplay tuning_id={selectedTuning?.id} />
			{:else}
				<ConfigDisplay config_id={selectedRows[0].id} />
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="delete" let:selectedRows>
			{#if selectedRows.some((row) => row.associated_jobs.length > 0)}
				<p>
					The selected configuration has associated jobs. Please delete the jobs before proceeding.
				</p>
				<div style="padding-top: 1rem;">
					{#each selectedRows.filter((row) => row?.associated_jobs?.length > 0) as row}
						{#each row.associated_jobs?.map((item) => item?.experiment_name) as job, index}
							<p>{job}</p>
						{/each}
					{/each}
				</div>
			{:else}
				<p>This is a permanent action and cannot be undone.</p>
			{/if}
		</svelte:fragment>
	</Table>

	<ImportConfigsModal
		bind:open={openImport}
		configurations={$configurations}
		on:submit={handleImportSubmit}
	/>

	<ExportConfigsModal
		bind:open={openExport}
		configurations={$configurations ?? []}
		selectedIds={selectedId ?? []}
		on:submit={handleExport}
	/>
{:else}
	<DataTableSkeleton />
{/if}
