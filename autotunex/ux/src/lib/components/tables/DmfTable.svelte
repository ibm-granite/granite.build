<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import Table from '../Table.svelte';
	import { API } from '$lib/api';
	import {
		Checkbox,
		DataTableSkeleton,
		InlineLoading,
		InlineNotification,
		Link,
		NotificationActionButton
	} from 'carbon-components-svelte';
	import TuningDisplay from '../displays/TuningDisplay.svelte';
	import DmfDisplay from '../displays/DmfDisplay.svelte';
	import { Launch } from 'carbon-icons-svelte';
	import { currentUser } from '$lib/store';
	import { DMF_UI_URL, buildDmfUrl } from '$lib/constants';
	const api = new API();
	let models;
	let isLoading = false;
	let entityName = 'published models';
	let showAllModels: boolean;
	$: showAllModels = false;
	let dmfHeaders = [
		{ key: 'name', value: 'Tuning' },
		{ key: 'base_model', value: 'Model' },
		{ key: 'user', value: 'Email' },
		// { key: 'revision', value: 'Tuning id' },
		{
			key: 'open',
			value: 'Visibility',
			display: (item) => (item === true ? 'IBM Public' : 'Private')
		},
		{ key: 'created', value: 'Published on' },
		{ key: 'action', empty: true }
	];

	const fetchModels = async () => {
		isLoading = true;
		models = await api.getPublishedModels();
		isLoading = false;
	};

	const fetchAllModels = async () => {
		isLoading = true;
		models = await api.getAllModels();
		isLoading = false;
	};

	$: if (showAllModels) {
		fetchAllModels();
	} else {
		fetchModels();
	}

	let selectedTuning;
	let openView = false;

	// reset tuning id when modal is closed
	$: if (!openView) {
		selectedTuning = null;
	}
</script>

{#key showAllModels}
	{#if models?.detail}
		<InlineNotification
			kind="error"
			title="DMF issue:"
			hideCloseButton
			subtitle="Error occured while fetching models"
		>
			<svelte:fragment slot="actions">
				<NotificationActionButton on:click={fetchModels} bind:disabled={isLoading}>
					{#if !isLoading}
						Retry
					{:else}
						<InlineLoading />
					{/if}
				</NotificationActionButton>
			</svelte:fragment>
		</InlineNotification>
	{/if}
	{#if !isLoading && !models?.detail && models?.length >= 0}
		<Table
			title="Published Models"
			description="List of models published to Data Model Factory"
			headers={dmfHeaders}
			selectable={false}
			batchSelection={false}
			bind:entity={entityName}
			bind:openView
			on:new={() => {
				showAllModels = !showAllModels;
			}}
			actionButtonText={!showAllModels ? 'Show all models' : 'Show my models'}
			customAction={true}
			expandable={false}
			showActionButton={$currentUser?.role === 'admin' ? true : false}
			rows={models.map((item) => {
				let name = item?.model_label.replace('autotunex.', '');
				return { id: item.revision, name, ...item };
			})}
			on:delete={async (e) => {
				for (let id of e.detail) {
					await api.deleteModel(id);
				}
			}}
			on:view={(e) => {
				entityName = `${e.detail.row.name} details`;
			}}
		>
			<!-- <svelte:fragment slot="title">
				<div style="display: flex; align-items:center; justify-content:space-between">
					<div>Published Models</div>
					<div style="padding-right: 1rem;">
						{#if $currentUser?.role === 'admin'}
							<Checkbox
								style="margin-bottom: 0.5rem"
								labelText="Show all models"
								bind:checked={showAllModels}
							/>
						{/if}
					</div>
				</div>
			</svelte:fragment> -->
			<svelte:fragment slot="cell" let:cell let:row>
				{#if cell.key === 'name'}
					<Link
						href="#"
						on:click={() => {
							selectedTuning = row;
							openView = true;
							entityName = selectedTuning?.name;
						}}>{cell.value}</Link
					>
				{:else if cell.key === 'created'}
					{#if row.files}
						{new Date(row.files[0]?.created)?.toLocaleString()}
					{:else}
						{new Date(row?.updated_at)?.toLocaleString()}
					{/if}
				{:else if cell.key === 'action'}
					{#if row?.dmf_url || DMF_UI_URL}
						<Link
							icon={Launch}
							href={row?.dmf_url || buildDmfUrl(row.model_label, row.id)}
							target="_blank">Open in DMF</Link
						>
					{/if}
				{:else}
					{cell.display ? cell.display(cell.value) : cell.value}
				{/if}
			</svelte:fragment>
			<svelte:fragment slot="view" let:selectedRows>
				{#if selectedTuning}
					<TuningDisplay tuning_id={selectedTuning.revision} selectedTabId={2} />
				{:else}
					<DmfDisplay data={selectedRows} />
				{/if}
			</svelte:fragment>
		</Table>
	{:else}
		<DataTableSkeleton />
	{/if}
{/key}
