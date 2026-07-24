<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import {
		ComposedModal,
		ModalHeader,
		ModalBody,
		ModalFooter,
		RadioButtonGroup,
		RadioButton,
		Button
	} from 'carbon-components-svelte';
	import { Close } from 'carbon-icons-svelte';
	import type { Configuration, ExportPreviewRow } from '$lib/app-types';

	export let open = false;
	export let configurations: Configuration[] = [];
	export let selectedIds: string[] = [];

	const dispatch = createEventDispatcher<{
		submit: { ids: string[]; format: 'json' | 'yaml' };
		close: void;
	}>();

	let format: 'json' | 'yaml' = 'json';
	let rows: ExportPreviewRow[] = [];

	// Rebuild rows whenever the selection or configurations change while the modal is open
	$: if (open) {
		rows = configurations
			.filter((c) => selectedIds.includes(c.id))
			.map((c) => ({
				rowId: c.id,
				name: c.name,
				tunerType: c.tuner_type,
				rlTunerType: c.rl_tuner_type ?? null,
				skipped: false
			}));
	}

	$: activeRows = rows.filter((r) => !r.skipped);
	$: primaryLabel = `Export ${activeRows.length} config${activeRows.length === 1 ? '' : 's'}`;
	$: primaryDisabled = activeRows.length === 0;

	function toggleSkip(row: ExportPreviewRow) {
		row.skipped = !row.skipped;
		rows = rows;
	}

	function onPrimary() {
		dispatch('submit', {
			ids: activeRows.map((r) => r.rowId),
			format
		});
	}

	function closeModal() {
		open = false;
		dispatch('close');
	}
</script>

<ComposedModal bind:open on:close={closeModal} on:click:button--primary={onPrimary} size="lg">
	<ModalHeader title="Export configurations" />
	<ModalBody hasForm>
		<p class="lead">Review configurations before exporting.</p>

		<div class="preview-table" role="table">
			<div class="preview-row preview-header" role="row">
				<span>Name</span>
				<span>Tuner</span>
				<span>RL</span>
				<span aria-hidden="true"></span>
			</div>
			{#each rows as row (row.rowId)}
				<div class="preview-row" class:row-skipped={row.skipped} role="row">
					<span class="cell-name">{row.name}</span>
					<span class="cell-meta">{row.tunerType || '—'}</span>
					<span class="cell-meta">{row.rlTunerType ?? '—'}</span>
					<div class="cell-action">
						<Button
							kind="ghost"
							size="small"
							iconDescription={row.skipped ? 'Include' : 'Skip'}
							icon={Close}
							on:click={() => toggleSkip(row)}
						/>
					</div>
				</div>
			{/each}
		</div>

		<RadioButtonGroup bind:selected={format} legendText="Format">
			<RadioButton labelText="JSON" value="json" />
			<RadioButton labelText="YAML" value="yaml" />
		</RadioButtonGroup>
	</ModalBody>
	<ModalFooter
		primaryButtonText={primaryLabel}
		primaryButtonDisabled={primaryDisabled}
		secondaryButtonText="Cancel"
		on:click:button--secondary={closeModal}
	/>
</ComposedModal>

<style>
	.lead {
		margin-bottom: 1rem;
		color: var(--cds-text-secondary, #525252);
		font-size: 0.875rem;
	}
	.preview-table {
		border: 1px solid var(--cds-border-subtle-01, #e0e0e0);
		border-radius: 2px;
		margin-bottom: 1.5rem;
		overflow: hidden;
	}
	.preview-row {
		display: grid;
		grid-template-columns: minmax(0, 2.2fr) minmax(0, 1fr) minmax(0, 1fr) 2.5rem;
		gap: 1rem;
		padding: 0.875rem 1rem;
		align-items: center;
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
	}
	.row-skipped {
		opacity: 0.5;
	}
	.cell-name {
		font-size: 0.875rem;
		color: var(--cds-text-primary, #161616);
		word-break: break-word;
	}
	.cell-meta {
		font-size: 0.875rem;
		color: var(--cds-text-primary, #161616);
		word-break: break-word;
	}
	.cell-action {
		display: flex;
		justify-content: flex-end;
	}
</style>
