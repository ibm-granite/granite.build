<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Button,
		DataTable,
		DataTableSkeleton,
		InlineNotification,
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
	/**
	 * Why the current selection cannot be deleted, or `null` when it can.
	 *
	 * Deliberately a *message* rather than another boolean: `disableDeleteButton`
	 * greys the toolbar button out, which makes the delete dialog unreachable and
	 * so leaves the user with no way to find out why. Setting this instead keeps
	 * the button clickable and blocks the dialog's own Delete, with the reason
	 * rendered where the user is already looking. The caller owns the wording, so
	 * `Table` needs no knowledge of what makes a given row protected.
	 */
	export let deleteBlockedMessage: string | null = null;
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

	export let serverSide = false;
	export let total = 0;
	export let page = 1;
	export let pageSize = 10;
	export let searchValue = '';

	// When true, the DataTable is swapped for a matching skeleton to signal that a
	// new page is loading. The parent sets this only for page/page-size changes
	// (never for search — the search input lives in the toolbar the skeleton would
	// replace, so skeleton-on-keystroke would drop typing focus). The Pagination
	// below stays mounted throughout, so the control the user just clicked keeps
	// focus while the rows reload.
	export let loading = false;

	// DataTableSkeleton only accepts compact/short/tall (not the DataTable's
	// "medium" default) — fall back to its own default for anything else.
	$: skeletonSize = size === 'compact' || size === 'short' || size === 'tall' ? size : undefined;

	let searchDebounce: ReturnType<typeof setTimeout> | undefined;
	function onSearchInput(value: string) {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => {
			searchValue = value;
			// A new search term can only match a different set of rows than whatever
			// is currently selected (server mode) — clear the stale selection so the
			// delete/compare/view actions can't act on rows the user can no longer see.
			selectedRowIds = [];
			dispatch('search', value);
		}, 300);
	}

	// Carbon's ToolbarSearch forwards the underlying <input>'s native DOM `input`
	// event (not a custom dispatch with `.detail`), so the value is read off the
	// target. The cast lives here, not inline in the markup, to avoid tripping the
	// Svelte template parser on `as` type assertions (see Trials.svelte for the
	// same convention).
	function handleSearchInput(e: Event) {
		onSearchInput((e.target as HTMLInputElement)?.value ?? '');
	}

	// The clear button (the ✕) and the Escape key reset only ToolbarSearch's own
	// internal value and dispatch `clear` — they never fire the `input` event
	// handleSearchInput listens for. Without handling `clear`, `searchValue` would
	// stay set and the table would remain filtered instead of returning to its
	// original state. Reset immediately (no debounce) and cancel any pending
	// debounced input so a trailing keystroke can't re-apply the just-cleared term.
	function onSearchClear() {
		clearTimeout(searchDebounce);
		searchValue = '';
		selectedRowIds = [];
		dispatch('search', '');
	}

	// Server mode: the server already returns exactly one page, so the DataTable
	// must render all given rows (no internal slicing). Client mode is unchanged.
	$: dtPage = serverSide ? 1 : page;
	$: dtPageSize = serverSide ? Math.max(rows?.length ?? 0, 1) : pageSize;

	// In server mode, `rows` is only the current page. A selection made on one
	// page/pageSize is meaningless once the page changes — the previously selected
	// ids may not even be present anymore. Clear it so the delete confirmation
	// (which renders only the current-page `selectedRows`) can never diverge from
	// what actually gets deleted (the full `selectedRowIds` dispatched on submit).
	// Guarded against firing on the initial render (before any real change).
	let prevPage = page;
	let prevPageSize = pageSize;
	$: if (serverSide && (page !== prevPage || pageSize !== prevPageSize)) {
		prevPage = page;
		prevPageSize = pageSize;
		selectedRowIds = [];
	}
</script>

{#if loading}
	<DataTableSkeleton {headers} rows={pageSize} zebra size={skeletonSize} />
{:else}
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
		pageSize={dtPageSize}
		page={dtPage}
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
						{#if serverSide}
							<ToolbarSearch
								persistent
								value={searchValue}
								on:input={handleSearchInput}
								on:clear={onSearchClear}
							/>
						{:else}
							<ToolbarSearch persistent shouldFilterRows bind:filteredRowIds />
						{/if}
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
{/if}
{#if serverSide}
	<Pagination bind:page bind:pageSize totalItems={total} pageSizeInputDisabled />
{:else}
	<Pagination bind:page bind:pageSize totalItems={filteredRowIds.length} pageSizeInputDisabled />
{/if}

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
	primaryButtonDisabled={deleteBlockedMessage !== null ||
		selectedRows?.some((row) => row?.is_published && !row?.github_pr_url) ||
		selectedRows.some((row) => row?.associated_jobs?.length > 0)}
	{entity}
	on:submit={(e) => {
		dispatch('delete', selectedRowIds);
		rows = rows.filter((row) => !selectedRowIds.includes(row.id));
		selectedRowIds = [];
		openDelete = false;
	}}
>
	{#if deleteBlockedMessage}
		<InlineNotification
			kind="info"
			lowContrast
			hideCloseButton
			title="Cannot delete"
			subtitle={deleteBlockedMessage}
		/>
	{:else}
		<slot name="delete" {selectedRows}>
			<p>This is a permanent action and cannot be undone.</p>
		</slot>
	{/if}
</DeleteDialog>
