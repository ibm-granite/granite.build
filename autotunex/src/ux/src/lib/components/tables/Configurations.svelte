<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { DataTableSkeleton, Link, ProgressBar, Tag, Button } from 'carbon-components-svelte';
	import { Download } from 'carbon-icons-svelte';
	import yaml from 'js-yaml';

	import Table from '../Table.svelte';
	import { API } from '$lib/api';
	import CreateConfigForm from '../forms/CreateConfigForm.svelte';
	import ImportConfigsModal from '../forms/ImportConfigsModal.svelte';
	import ExportConfigsModal from '../forms/ExportConfigsModal.svelte';
	import { Utils } from '$lib/utils';
	import TuningDisplay from '../displays/TuningDisplay.svelte';
	import ConfigDisplay from '../displays/ConfigDisplay.svelte';
	import { showLoader, userMetadata, currentUser } from '$lib/store';
	import type { ConfigForm, Configuration, Job, ImportPreviewRow } from '$lib/app-types';
	import { appState, notifications } from '$lib/app';

	// The seed/system account that owns configurations shipped out of the box. Stable
	// across deployments (unlike an email domain literal), so it gates the delete
	// button both for "is this row owned by the system account" and "is the viewer
	// the system account".
	const SYSTEM_USER_ID = '00000000-0000-0000-0000-000000000001';

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
				// Deliberately invalidates the tuning-wizard/form cache — CreateTuningForm,
				// StartTuningWizard and Step2Configure check this flag on mount and only
				// refetch $configurations when it is false. Not a dead write.
				appState.update((prev) => ({ ...prev, isConfigurationsLoaded: false }));
				// Reset to page 1 and pre-sync prevKey (synchronously, before the explicit
				// fetchPage below) so the reactive page/pageSize/q watcher doesn't also fire
				// a second fetch once its flush runs.
				page = 1;
				prevKey = `${page} ${pageSize} ${q}`;
				await fetchPage();
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

	let rows: Configuration[] = [];
	let total = 0;
	let page = 1;
	let pageSize = 10;
	let q = '';
	let loaded = false;
	// True while a page/page-size change is refetching — drives the table skeleton
	// (see fetchPage). Distinct from `loaded`, which only gates the very first load.
	let fetching = false;
	// The full (all-pages) configuration list, fetched only when the import modal
	// opens — it needs every existing name to detect collisions, not just the
	// current page (see the Import button handler below).
	let allConfigsForImport: Configuration[] = [];

	// `showSkeleton` swaps the table for a loading skeleton while this fetch runs.
	// Callers pass it for page navigation only; create/delete/import refetches leave
	// it false (they already show the global loader), and search leaves it false so
	// the toolbar search input stays mounted and keeps focus.
	const fetchPage = async (showSkeleton = false) => {
		if (showSkeleton) fetching = true;
		try {
			const res = await api.getConfigurationsPage({
				limit: pageSize,
				offset: (page - 1) * pageSize,
				q
			});
			rows = res.items;
			total = res.total;
		} catch (error) {
			notifications.set({
				show: true,
				kind: 'error',
				title: 'Error',
				subtitle: 'Could not load configurations',
				timeout: 5000
			});
		} finally {
			loaded = true;
			fetching = false;
		}
	};

	function onSearch(value: string) {
		q = value;
		page = 1;
	}

	// Fetch exactly once per distinct (page, pageSize, q) — `loaded` must not be
	// part of the key: fetchPage itself flips it false -> true, and referencing it
	// here would re-trigger this block and double-fetch.
	let prevKey: string | null = null;
	$: {
		const key = `${page} ${pageSize} ${q}`;
		if (key !== prevKey) {
			// Show the skeleton only for a genuine page turn: the page/page-size
			// portion changed while the search term did not (see fetchPage). A search
			// always changes `q` — and also resets the page to 1 — so skeletoning then
			// would unmount the toolbar search input mid-type and drop focus. Both
			// parts are read back off `prevKey` (format "<page> <pageSize> <q>", and
			// `q` may itself contain spaces) so this stays correct through the manual
			// prevKey pre-syncs in the create/import/delete handlers.
			const [prevPage, prevSize, ...prevQParts] = (prevKey ?? '').split(' ');
			const pageChanged =
				prevKey !== null &&
				prevQParts.join(' ') === q &&
				`${prevPage} ${prevSize}` !== `${page} ${pageSize}`;
			prevKey = key;
			fetchPage(pageChanged);
		}
	}

	// reset tuning id when modal is closed
	$: if (!openView) {
		selectedTuning = null;
		entityName = 'configuration';
	}
</script>

{#if loaded}
	<Table
		title="Configurations"
		entity={entityName}
		entities="configurations"
		actionButtonText="Create New Configuration"
		description="Shows your configurations."
		headers={configHeaders}
		{rows}
		expandable={false}
		serverSide
		loading={fetching}
		{total}
		bind:page
		bind:pageSize
		on:search={(e) => onSearch(e.detail)}
		disableDeleteButton={rows
			?.filter((conf) => selectedId?.includes(conf.id))
			.map((item) => item.user_id)
			?.includes(SYSTEM_USER_ID) && $currentUser?.user_id !== SYSTEM_USER_ID}
		bind:openView
		bind:selectedRowIds={selectedId}
		submitBtnDisable={config?.name === '' ||
			rows?.map((item) => item.name).includes(config?.name ?? '')}
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
				// Deliberately invalidates the tuning-wizard/form cache — see the comment on
				// the same call in handleImportSubmit above. Not a dead write.
				appState.update((prev) => {
					return { ...prev, isConfigurationsLoaded: false };
				});
				// See the presync comment in handleImportSubmit above.
				page = 1;
				prevKey = `${page} ${pageSize} ${q}`;
				await fetchPage();
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
				userMetadata.update((prev) => {
					return { ...prev, number_of_configurations: prev.number_of_configurations - 1 };
				});
			}
			// Deliberately invalidates the tuning-wizard/form cache — see the comment on
			// the same call in handleImportSubmit above. Not a dead write.
			appState.update((prev) => ({ ...prev, isConfigurationsLoaded: false }));
			await fetchPage();
			// Deleting the last row(s) on a page beyond the first can strand the user on
			// an empty page — step back one page and refetch rather than leave them there.
			if (rows.length === 0 && page > 1) {
				page -= 1;
				prevKey = `${page} ${pageSize} ${q}`;
				await fetchPage();
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
			<Button
				kind="tertiary"
				on:click={async () => {
					// Fetch the complete (all-pages) list right before opening the modal so its
					// name-collision / auto-rename check sees every existing name, not just the
					// current page.
					try {
						allConfigsForImport = await api.getConfigurations();
						openImport = true;
					} catch (error) {
						notifications.set({
							show: true,
							kind: 'error',
							title: 'Could not load configurations',
							subtitle: 'Failed to load the full configuration list needed for import.',
							timeout: 5000
						});
					}
				}}>Import</Button
			>
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
			<!-- Deliberately page-scoped: this inline check only sees the current page's
			     names, not the full list. The backend's UNIQUE(user_id, name) constraint
			     (surfaced as a 409) is the real backstop against a same-name collision that
			     spans pages. -->
			<CreateConfigForm bind:config configurations={rows} />
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
		configurations={allConfigsForImport}
		on:submit={handleImportSubmit}
	/>

	<ExportConfigsModal
		bind:open={openExport}
		configurations={rows ?? []}
		selectedIds={selectedId ?? []}
		on:submit={handleExport}
	/>
{:else}
	<DataTableSkeleton headers={configHeaders} rows={pageSize} zebra />
{/if}
