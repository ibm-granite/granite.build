<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Button,
		DataTable,
		Toolbar,
		ToolbarContent,
		ToolbarSearch,
		Pagination,
		ToolbarBatchActions
	} from 'carbon-components-svelte';
	import { Compare, View, TrashCan } from 'carbon-icons-svelte';
	import ViewDialog from '$lib/components/ViewDialog.svelte';
	import CreateDialog from '$lib/components/CreateDialog.svelte';
	import CompareDialog from '$lib/components/CompareDialog.svelte';
	import { createEventDispatcher } from 'svelte';
	import DeleteDialog from './DeleteDialog.svelte';
	import type {
		DataTableHeader,
		DataTableRow
	} from 'carbon-components-svelte/src/DataTable/DataTable.svelte';

	const dispatch = createEventDispatcher();

	export let disableActionButton = false;
	export let title: string | undefined = undefined;
	export let description: string | undefined = undefined;
	export let entity: string | undefined = undefined;
	export let entities: string | undefined = undefined;
	export let rows: DataTableRow[];
	export let headers: DataTableHeader[];
	export let sortKey: string = '';
	export let sortDirection: 'ascending' | 'descending' = 'descending';
	export let expandable: boolean = true;
	export let openView: boolean = false;
	export let selectable: boolean = true;
	export let showSearch: boolean = true;
	export let batchSelection: boolean = true;
	export let showActionButton: boolean = true;
	export let submitBtnDisable: boolean = false;
	export let disableDeleteButton: boolean = false;
	export let primaryButtonText = 'OK';
	export let secondaryButtonText = 'Cancel';
	export let passiveCreateModal = false;
	export let actionButtonText: string | null = null;
	export let customAction: boolean = false;
	export let size: 'compact' | 'short' | 'medium' | 'tall' = 'medium';

	export let openNew = false;
	let openDelete = false;
	let openCompare = false;

	export let selectedRowIds: string[] = [];
	let filteredRowIds: string[] = [];

	$: selectedRows = rows?.filter(
		(row) => selectedRowIds.filter((r_id) => r_id === row.id).length > 0
	);

	let pageSize = 10;
	let page = 1;
</script>

<DataTable
	{size}
	{batchSelection}
	{selectable}
	{expandable}
	bind:selectedRowIds
	sortable
	zebra
	{sortKey}
	{sortDirection}
	{title}
	{description}
	{headers}
	{pageSize}
	{page}
	{rows}
	on:click:row--expand={(e) => {
		dispatch('row-expanded', e.detail);
	}}
>
	<svelte:fragment slot="expanded-row" let:row>
		<slot name="expanded-row" {row}>
			<code>
				<pre>{JSON.stringify(row, null, 2)}</pre>
			</code>
		</slot>
	</svelte:fragment>
	<svelte:fragment slot="cell" let:cell let:row>
		<slot name="cell" {cell} {row}>
			{cell.display ? cell.display(cell.value, row) : cell.value}
		</slot>
	</svelte:fragment>
	<!-- <svelte:fragment slot="title">
		<slot name="title">
			{title ?? ''}
		</slot>
	</svelte:fragment> -->
	<Toolbar>
		<ToolbarBatchActions>
			{#if selectedRowIds.length > 1}
				<Button
					icon={Compare}
					on:click={() => {
						openCompare = true;
						dispatch('compare', selectedRows);
					}}
				>
					Compare
				</Button>
			{:else}
				<Button
					icon={View}
					on:click={() => {
						openView = true;
						dispatch('view', { row: selectedRows[0] });
					}}
				>
					View
				</Button>
			{/if}
			<Button
				icon={TrashCan}
				disabled={disableDeleteButton}
				on:click={(e) => {
					openDelete = true;
				}}
			>
				Delete
			</Button>
			<slot name="batch-actions" {selectedRows} />
		</ToolbarBatchActions>
		{#if showSearch || showActionButton}
			<ToolbarContent>
				{#if showSearch}
					<ToolbarSearch persistent shouldFilterRows bind:filteredRowIds />
				{/if}
				<slot name="toolbar-actions" />
				{#if showActionButton}
					<Button
						on:click={() => {
							if (customAction) {
								dispatch('new');
							} else {
								openNew = true;
							}
						}}
						disabled={disableActionButton}
					>
						{#if actionButtonText}
							{actionButtonText}
						{:else}
							Create new {entity}
						{/if}
					</Button>
				{/if}
			</ToolbarContent>
		{/if}
	</Toolbar>
</DataTable>
<Pagination bind:pageSize bind:page totalItems={filteredRowIds.length} pageSizeInputDisabled />

<CreateDialog
	bind:submitBtnDisable
	bind:primaryButtonText
	bind:secondaryButtonText
	bind:open={openNew}
	{passiveCreateModal}
	{entity}
	on:submit={() => {
		dispatch('new');
		// openNew = false;
	}}
>
	<slot name="create" />
</CreateDialog>

<ViewDialog bind:open={openView} {entity}>
	<slot name="view" {selectedRows}>
		<code>
			<pre>{JSON.stringify(selectedRows[0], null, 2)}</pre>
		</code>
	</slot>
</ViewDialog>

<CompareDialog {entities} bind:open={openCompare} bind:rows={selectedRows} />

<DeleteDialog
	bind:open={openDelete}
	primaryButtonDisabled={selectedRows?.some((row) => row?.is_published && !row?.github_pr_url) ||
		selectedRows.some((row) => row?.associated_jobs?.length > 0)}
	{entity}
	on:submit={(e) => {
		dispatch('delete', selectedRowIds);
		rows = rows.filter((row) => !selectedRowIds.includes(row.id));
		selectedRowIds = [];
		openDelete = false;
	}}
>
	<slot name="delete" {selectedRows}>
		<p>This is a permanent action and cannot be undone.</p>
	</slot>
</DeleteDialog>
