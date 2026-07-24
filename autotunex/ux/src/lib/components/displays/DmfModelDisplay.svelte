<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Tile, Tag } from 'carbon-components-svelte';
	import { API } from '$lib/api';
	import { compile } from 'mdsvex';

	export let modelData: any = null;
	export let showModelCard: boolean = true;

	const api = new API();
	let modelCard: any = null;
	let isLoadingCard: boolean = false;
	let cardError: string | null = null;
	let lastFetchedKey: string | null = null;

	$: formattedDate = modelData?.created
		? new Date(modelData.created).toLocaleString('en-US', {
				year: 'numeric',
				month: 'long',
				day: 'numeric',
				hour: '2-digit',
				minute: '2-digit'
		  })
		: 'N/A';

	async function fetchModelCard() {
		if (!modelData || !showModelCard) return;

		// Only fetch if we have required fields
		if (!modelData.namespace || !modelData.base_model || !modelData.revision) {
			console.warn('Missing required fields for model card fetch');
			return;
		}

		// Create a unique key for this model to prevent duplicate fetches
		const modelKey = `${modelData.namespace}:${modelData.base_model}:${modelData.revision}`;

		// Skip if we already fetched this exact model
		if (lastFetchedKey === modelKey) {
			return;
		}

		try {
			isLoadingCard = true;
			cardError = null;
			modelCard = null;
			lastFetchedKey = modelKey;

			const response = await api.getDmfModelCard({
				namespace: modelData.namespace,
				table: 'model_shared',
				model_label: modelData.base_model,
				revision: modelData.revision
			});

			// DMF API returns format: {"readme": "markdown content", "yaml": "metadata"}
			const markdownContent = response.readme || response.content || response;

			if (typeof markdownContent === 'string' && markdownContent.trim().length > 0) {
				modelCard = await compile(markdownContent);
			} else {
				modelCard = null;
				cardError = 'Model card has no documentation content';
			}
		} catch (error) {
			console.error('Failed to fetch model card:', error);
			cardError = 'Failed to load model documentation. The model card may not be available.';
			modelCard = null;
		} finally {
			isLoadingCard = false;
		}
	}

	// Fetch model card when the specific model identity changes
	$: modelIdentity = modelData
		? `${modelData.namespace}:${modelData.base_model}:${modelData.revision}`
		: null;

	$: if (modelIdentity && showModelCard) {
		fetchModelCard();
	}
</script>

{#if modelData}
	<div class="dmf-model-display">
		<!-- <h4 class="model-title">{modelData.model_label || modelData.model_id}</h4> -->

		<!-- <div class="tags-section">
			{#if modelData.namespace}
				<Tag type="high-contrast">{modelData.namespace}</Tag>
			{/if}
			{#if modelData.revision}
				<Tag type="teal">{modelData.revision}</Tag>
			{/if}

			{#if modelData.model_type}
				<Tag type="blue">{modelData.model_type}</Tag>
			{/if}

			{#if modelData.size}
				<Tag type="purple">{modelData.size}</Tag>
			{/if}

			{#if modelData.variant}
				<Tag type="cyan">{modelData.variant}</Tag>
			{/if}
			{#if modelData.base_model}
				<Tag type="warm-gray">{modelData.base_model}</Tag>
			{/if}
		</div> -->
		{#if (!modelCard && isLoadingCard) || cardError}
			<Tile light style="margin-top: 1rem;">
				<div class="info-grid">
					<div class="info-row">
						<span class="label">Model ID:</span>
						<span class="value">{modelData.model_id}</span>
					</div>

					<div class="info-row">
						<span class="label">Base Model:</span>
						<span class="value">{modelData.base_model}</span>
					</div>

					<div class="info-row">
						<span class="label">Namespace:</span>
						<span class="value">{modelData.namespace}</span>
					</div>

					<div class="info-row">
						<span class="label">Revision:</span>
						<span class="value code">{modelData.revision}</span>
					</div>

					{#if modelData.product_name}
						<div class="info-row">
							<span class="label">Product:</span>
							<span class="value">{modelData.product_name}</span>
						</div>
					{/if}

					<div class="info-row">
						<span class="label">Created:</span>
						<span class="value">{formattedDate}</span>
					</div>

					{#if modelData.comments}
						<div class="info-row full-width">
							<span class="label">Link:</span>
							<span class="value">
								<a href={modelData.comments} target="_blank" rel="noopener noreferrer">
									{modelData.comments}
								</a>
							</span>
						</div>
					{/if}
				</div>
			</Tile>
		{/if}

		{#if showModelCard}
			{#if isLoadingCard}
				<Tile light style="margin-top: 1rem;">
					<div style="text-align: center; padding: 2rem; color: #8d8d8d;">
						<p>Loading model documentation...</p>
					</div>
				</Tile>
			{:else if cardError}
				<Tile light style="margin-top: 1rem; border-left: 4px solid #da1e28;">
					<div style="padding: 1rem;">
						<h5 style="margin: 0 0 0.5rem 0; color: #da1e28; font-size: 14px; font-weight: 600;">
							⚠️ Documentation Unavailable
						</h5>
						<p style="margin: 0; font-size: 14px; color: #525252;">
							{cardError}
						</p>
					</div>
				</Tile>
			{:else if modelCard}
				<Tile light style="margin-top: 1rem;">
					<div class="markdown-content">
						{@html modelCard.code}
					</div>
				</Tile>
			{/if}
		{/if}
	</div>
{:else}
	<div class="no-data">
		<p>No model selected</p>
	</div>
{/if}

<style>
	.dmf-model-display {
		padding: 0.5rem 0;
	}

	.model-title {
		font-size: 20px;
		font-weight: 600;
		margin-bottom: 1rem;
		color: #161616;
	}

	.tags-section {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
		margin-bottom: 1rem;
	}

	.info-grid {
		display: grid;
		grid-template-columns: 1fr;
		gap: 0.75rem;
	}

	.info-row {
		display: grid;
		grid-template-columns: 140px 1fr;
		gap: 1rem;
		align-items: start;
	}

	.info-row.full-width {
		grid-column: 1 / -1;
	}

	.label {
		font-weight: 600;
		color: #525252;
		font-size: 14px;
	}

	.value {
		color: #161616;
		font-size: 14px;
		word-break: break-word;
	}

	.value.code {
		font-family: 'IBM Plex Mono', 'Menlo', 'DejaVu Sans Mono', 'Bitstream Vera Sans Mono', monospace;
		font-size: 13px;
		background-color: #f4f4f4;
		padding: 2px 6px;
		border-radius: 3px;
	}

	.value a {
		color: #0f62fe;
		text-decoration: none;
	}

	.value a:hover {
		text-decoration: underline;
	}

	.warning-message {
		margin-top: 1rem;
		padding: 1rem;
		background-color: #fff1f1;
		border-left: 4px solid #da1e28;
		border-radius: 4px;
	}

	.warning-message p {
		margin: 0;
		font-size: 14px;
		color: #161616;
	}

	.warning-message strong {
		font-weight: 600;
	}

	.no-data {
		padding: 2rem;
		text-align: center;
		color: #8d8d8d;
	}

	.no-data p {
		margin: 0;
		font-size: 14px;
	}

	:global(.markdown-content h1) {
		font-size: 18px;
		font-weight: 600;
		margin-top: 1rem;
	}

	:global(.markdown-content h2) {
		font-size: 16px;
		font-weight: 600;
		margin-top: 1rem;
	}

	:global(.markdown-content h3) {
		font-size: 14px;
		font-weight: 600;
		margin-top: 0.75rem;
	}

	:global(.markdown-content p) {
		margin: 0.5rem 0;
		line-height: 1.5;
		font-size: 14px;
	}

	:global(.markdown-content ul, .markdown-content ol) {
		margin: 0.5rem 0;
		padding-left: 1.5rem;
	}

	:global(.markdown-content li) {
		margin: 0.25rem 0;
		font-size: 14px;
	}

	:global(.markdown-content pre) {
		background-color: #f4f4f4;
		padding: 1rem;
		border-radius: 4px;
		overflow-x: auto;
		margin: 0.5rem 0;
	}

	:global(.markdown-content code) {
		font-family: 'IBM Plex Mono', monospace;
		font-size: 13px;
	}

	:global(.markdown-content table) {
		width: 100%;
		border-collapse: collapse;
		margin: 0.5rem 0;
	}

	:global(.markdown-content th, .markdown-content td) {
		border: 1px solid #e0e0e0;
		padding: 0.5rem;
		font-size: 14px;
	}

	:global(.markdown-content th) {
		background-color: #f4f4f4;
		font-weight: 600;
	}
</style>
