<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {
		ComposedModal,
		ModalHeader,
		ModalBody,
		ModalFooter,
		FileUploaderDropContainer,
		TextInput,
		Button,
		Tag
	} from 'carbon-components-svelte';
	import { TrashCan, Add } from 'carbon-icons-svelte';
	import {
		parseFilesToRows,
		applyNameAutoSuggestions,
		validateRows,
		existingNameSet,
		importableCount
	} from '$lib/config-import';
	import type { Configuration, ImportPreviewRow } from '$lib/app-types';

	export let open = false;
	export let configurations: Configuration[] | undefined = [];

	const dispatch = createEventDispatcher<{
		submit: {
			rowsToImport: ImportPreviewRow[];
			onRowFailed: (rowId: string, message: string) => void;
			onFinished: (okCount: number, failed: boolean) => void;
		};
		close: void;
	}>();

	let rows: ImportPreviewRow[] = [];
	let isParsing = false;
	let isSubmitting = false;

	$: hasPreview = rows.length > 0;
	$: importable = importableCount(rows);
	$: primaryLabel = hasPreview ? `Import ${importable} configs` : 'Import';
	$: primaryDisabled =
		isSubmitting ||
		!hasPreview ||
		importable === 0 ||
		rows.some((r) => !r.skipped && r.status !== 'ready');

	async function ingestFiles(files: File[]) {
		if (!files || files.length === 0) return;
		isParsing = true;
		try {
			const parsed = await parseFilesToRows(files);
			const existing = existingNameSet(configurations);
			applyNameAutoSuggestions(parsed, existing);
			rows = [...rows, ...parsed];
			validateRows(rows, existing);
		} finally {
			isParsing = false;
		}
	}

	function handleDropAdd(e: CustomEvent<readonly File[]>) {
		ingestFiles(Array.from(e.detail ?? []));
	}

	function resetAll() {
		rows = [];
	}

	function closeAndReset() {
		rows = [];
		open = false;
		dispatch('close');
	}

	function onPrimary() {
		const toImport = rows.filter((r) => !r.skipped && r.status === 'ready');
		if (toImport.length === 0) return;
		isSubmitting = true;
		dispatch('submit', {
			rowsToImport: toImport,
			onRowFailed: (rowId, message) => {
				const r = rows.find((x) => x.rowId === rowId);
				if (!r) return;
				r.status = 'import_failed';
				r.errorMessage = `Import failed: ${message}`;
				rows = rows;
			},
			onFinished: (okCount, failed) => {
				isSubmitting = false;
				if (!failed) {
					closeAndReset();
				} else {
					// Keep modal open; drop successfully-imported rows from preview
					rows = rows.filter((r) => r.status !== 'ready' || r.skipped);
					// Re-validate against the now-updated existing names (parent refreshes store first)
					validateRows(rows, existingNameSet(configurations));
					rows = rows;
				}
			}
		});
	}

	function revalidate() {
		validateRows(rows, existingNameSet(configurations));
		rows = rows; // trigger reactivity
	}

	function onNameInput(row: ImportPreviewRow, value: string | number | null) {
		row.editedName = String(value ?? '');
		row.edited = row.editedName !== row.originalName;
		revalidate();
	}

	function removeRow(row: ImportPreviewRow) {
		rows = rows.filter((r) => r.rowId !== row.rowId);
		revalidate();
	}

	function addMoreFiles() {
		const input = document.createElement('input');
		input.type = 'file';
		input.multiple = true;
		input.accept = '.json,.yaml,.yml';
		input.onchange = () => {
			if (input.files) ingestFiles(Array.from(input.files));
		};
		input.click();
	}

	function statusBadgeKind(status: ImportPreviewRow['status']): 'green' | 'red' | 'gray' {
		if (status === 'ready') return 'green';
		return 'red';
	}

	function statusBadgeLabel(row: ImportPreviewRow): string {
		switch (row.status) {
			case 'ready':
				return 'Ready';
			case 'name_required':
				return 'Name required';
			case 'name_exists':
				return 'Name exists';
			case 'duplicate_in_batch':
				return 'Duplicate';
			case 'invalid_missing_name':
				return 'Invalid';
			case 'invalid_missing_config_data':
				return 'Invalid';
			case 'parse_error':
				return 'Parse error';
			case 'import_failed':
				return 'Import failed';
			default:
				return row.status;
		}
	}

	function noteText(row: ImportPreviewRow): string {
		if (row.status === 'ready' && row.edited && row.originalName !== '') return 'Auto-renamed';
		if (row.status === 'name_exists') return 'Rename or remove';
		if (row.status === 'name_required') return 'Enter a name';
		if (row.status === 'duplicate_in_batch') return 'Duplicate in batch';
		if (row.status === 'invalid_missing_name') return 'Missing name in file';
		if (row.status === 'invalid_missing_config_data') return 'Missing config_data';
		if (row.status === 'parse_error') return row.errorMessage ?? 'Parse error';
		if (row.status === 'import_failed') return row.errorMessage ?? 'Import failed';
		return '';
	}

	$: uniqueFiles = Array.from(new Set(rows.map((r) => r.sourceFile)));
	$: needsAttention = rows.filter((r) => !r.skipped && r.status !== 'ready').length;
</script>

<ComposedModal bind:open on:close={closeAndReset} on:click:button--primary={onPrimary} size="lg">
	<ModalHeader title="Import configuration" />
	<ModalBody hasForm>
		{#if !hasPreview}
			<p style="margin-bottom: 1rem;">
				Upload one or more <code>.json</code>, <code>.yaml</code>, or
				<code>.yml</code> files. Each file may contain a single configuration or an array of configurations.
			</p>
			<div class="drop-zone-wrapper">
				<FileUploaderDropContainer
					labelText="Drag and drop files here or click to upload"
					accept={['.json', '.yaml', '.yml']}
					multiple
					disabled={isParsing}
					on:add={handleDropAdd}
				/>
			</div>
		{:else}
			<div class="summary-bar">
				<span class="summary-text">
					<strong>{uniqueFiles.length}</strong>
					file{uniqueFiles.length === 1 ? '' : 's'} selected ·
					<strong>{rows.length}</strong>
					config{rows.length === 1 ? '' : 's'} found
				</span>
				<span class="summary-actions">
					<Button kind="ghost" size="small" icon={Add} on:click={addMoreFiles}>Add more</Button>
					<Button kind="ghost" size="small" on:click={resetAll}>Clear all</Button>
				</span>
			</div>

			<div class="preview-table" role="table">
				<div class="preview-row preview-header" role="row">
					<span>Name</span>
					<span>Tuner</span>
					<span>RL</span>
					<span>Status</span>
					<span>Notes</span>
					<span aria-hidden="true"></span>
				</div>
				{#each rows as row (row.rowId)}
					<div class="preview-row" role="row">
						<div class="cell-name">
							{#if row.status === 'parse_error' || row.status === 'invalid_missing_name' || row.status === 'invalid_missing_config_data'}
								<span class="source-only">{row.sourceFile}</span>
							{:else}
								<TextInput
									labelText=""
									hideLabel
									size="sm"
									value={row.editedName}
									invalid={['name_required', 'name_exists', 'duplicate_in_batch'].includes(
										row.status
									)}
									class={row.edited &&
									!['name_required', 'name_exists', 'duplicate_in_batch'].includes(row.status)
										? 'name-edited'
										: ''}
									on:input={(e) => onNameInput(row, e.detail ?? '')}
								/>
								<small class="source-caption">from {row.sourceFile}</small>
							{/if}
						</div>
						<div class="cell-meta">{row.tunerType ?? '—'}</div>
						<div class="cell-meta">{row.rlTunerType ?? '—'}</div>
						<div class="cell-status">
							<span class="tag-wrap">
								<Tag type={statusBadgeKind(row.status)}>{statusBadgeLabel(row)}</Tag>
							</span>
						</div>
						<div
							class="cell-notes"
							class:cell-notes-error={row.status !== 'ready' && noteText(row) !== ''}
						>
							{noteText(row) || '—'}
						</div>
						<div class="cell-action">
							<Button
								kind="ghost"
								size="small"
								iconDescription="Remove from import"
								icon={TrashCan}
								on:click={() => removeRow(row)}
							/>
						</div>
					</div>
				{/each}
			</div>

			<p style="margin-top:0.75rem;font-size:0.875rem;color:#525252;">
				{importable} of {rows.length} ready to import
				{#if needsAttention > 0}
					· <span style="color:#da1e28;">{needsAttention} needs attention</span>
				{/if}
			</p>
		{/if}
	</ModalBody>
	<ModalFooter
		primaryButtonText={primaryLabel}
		primaryButtonDisabled={primaryDisabled}
		secondaryButtonText="Cancel"
		on:click:button--secondary={closeAndReset}
	/>
</ComposedModal>

<style>
	:global(.bx--form-item.name-edited .bx--text-input) {
		border: 1px solid var(--cds-interactive-01, #0f62fe);
	}

	/* Drop zone — full width, comfortable target height */
	.drop-zone-wrapper :global(.bx--file-browse-btn) {
		max-width: 100%;
		width: 100%;
	}
	.drop-zone-wrapper :global(.bx--file__drop-container) {
		min-height: 14rem;
		padding: 2rem 1.5rem;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1rem;
	}

	/* Summary bar above the preview table */
	.summary-bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 0.75rem 1rem;
		background: var(--cds-layer-01, #f4f4f4);
		border: 1px solid var(--cds-border-subtle-01, #e0e0e0);
		margin-bottom: 1rem;
		font-size: 0.875rem;
		color: var(--cds-text-secondary, #525252);
	}
	.summary-text strong {
		color: var(--cds-text-primary, #161616);
	}
	.summary-actions {
		display: flex;
		gap: 0.25rem;
		flex-shrink: 0;
	}

	/* Preview table */
	.preview-table {
		border: 1px solid var(--cds-border-subtle-01, #e0e0e0);
		border-radius: 2px;
		overflow: hidden;
	}
	.preview-row {
		display: grid;
		/* Name | Tuner | RL | Status | Notes | action */
		grid-template-columns:
			minmax(0, 2fr)
			minmax(0, 0.8fr)
			minmax(0, 0.8fr)
			minmax(0, 1fr)
			minmax(0, 1.2fr)
			2.5rem;
		gap: 1rem;
		padding: 0.875rem 1rem;
		align-items: start;
	}
	.preview-row + .preview-row {
		border-top: 1px solid var(--cds-border-subtle-01, #e0e0e0);
	}
	.preview-header {
		background: var(--cds-layer-01, #f4f4f4);
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.02em;
		color: var(--cds-text-secondary, #525252);
		padding-top: 0.625rem;
		padding-bottom: 0.625rem;
		align-items: center;
	}

	.cell-name {
		display: flex;
		flex-direction: column;
		gap: 0.375rem;
		min-width: 0;
	}
	/* Carbon TextInput stretches to its container; cap it so it stays readable. */
	.cell-name :global(.bx--text-input) {
		max-width: 22rem;
	}
	.source-caption,
	.source-only {
		font-size: 0.75rem;
		line-height: 1.3;
		color: var(--cds-text-secondary, #525252);
	}
	.source-only {
		font-style: italic;
		padding-top: 0.375rem;
	}
	.cell-meta {
		font-size: 0.875rem;
		color: var(--cds-text-primary, #161616);
		padding-top: 0.5rem; /* align with input baseline */
		word-break: break-word;
	}
	.cell-status {
		padding-top: 0.375rem;
		min-width: 0;
	}
	/* Keep Carbon Tag at content width instead of stretching across the cell. */
	.tag-wrap {
		display: inline-flex;
	}
	.tag-wrap :global(.bx--tag) {
		margin: 0;
	}
	.cell-notes {
		font-size: 0.8125rem;
		line-height: 1.3;
		color: var(--cds-text-secondary, #525252);
		padding-top: 0.5rem;
		word-break: break-word;
	}
	.cell-notes-error {
		color: var(--cds-support-error, #da1e28);
	}
	.cell-action {
		display: flex;
		justify-content: flex-end;
		padding-top: 0.125rem;
	}
</style>
