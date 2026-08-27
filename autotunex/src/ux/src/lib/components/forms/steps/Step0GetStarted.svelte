<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		SelectableTile,
		Tag,
		InlineLoading,
		FormGroup,
		RadioButtonGroup,
		RadioButton,
		ComboBox,
		TextInput,
		Modal,
		Button,
		Checkbox,
		Grid,
		Row,
		Column,
		Accordion,
		AccordionItem,
		Tile,
		Link
	} from 'carbon-components-svelte';
	import { Education, Compare, Growth, View, Information, Launch } from 'carbon-icons-svelte';
	import { onMount } from 'svelte';
	import { compile } from 'mdsvex';
	import { sanitizeModelCard } from '$lib/sanitize';
	import { Utils } from '$lib/utils';
	import { API } from '$lib/api';
	import { currentUser, featureFlags } from '$lib/store';
	import { ModelSource } from '$lib/app-types';
	import type { TuningGoal, HuggingFaceModel } from '$lib/app-types';

	const api = new API();

	export let selectedAlgorithm: string = 'lora';
	export let selectedGoal: TuningGoal | null = null;
	export let selectedModel: string = 'ibm-granite/granite-4.0-h-micro';
	export let modelSource: ModelSource = ModelSource.HuggingFace;
	export let prefetchedModels: HuggingFaceModel[] | null = null;
	export let autotuneEnabled: boolean = true;

	// Goal icon mapping
	const goalIcons = {
		sft: Education,
		offline_rl: Compare,
		online_rl: Growth
	};

	const goalCategoryLabels: Record<string, string> = {
		sft: 'SFT',
		offline_rl: 'Offline RL',
		online_rl: 'Online RL'
	};

	function selectGoal(goal: TuningGoal) {
		selectedGoal = goal;
		selectedAlgorithm = Utils.getDefaultAlgorithmForGoal(goal);
	}

	// --- Model selection state (moved from Step3ModelSelection) ---
	let models: HuggingFaceModel[] = [];
	let suggestions: { id: string; text: string; isOpen?: boolean; rawModel?: any }[] = [];
	let modelCard: any = null;
	let debounceTimeout: number;
	let previousModelSource: ModelSource = modelSource;
	let comboBoxReady = false;
	let showModelCardModal = false;

	$: if (modelSource !== previousModelSource) {
		if (modelSource === ModelSource.CustomPath) {
			selectedModel = '';
			suggestions = [];
		} else {
			selectedModel = 'ibm-granite/granite-4.0-h-micro';
			suggestions = models?.map((item) => ({ id: item.id, text: item.id })) || [];
			fetchModelCard();
		}
		previousModelSource = modelSource;
	}

	async function fetchSuggestions(term: string) {
		if (!term?.trim()) {
			suggestions = models?.map((item) => ({ id: item.id, text: item.id })) || [];
			return;
		}
		try {
			const response = await api.getHFModels(term.replace(/(\w+)[-/]\1(?=[-/])/g, '$1'));
			suggestions = response.map((model: HuggingFaceModel) => ({
				id: model.id,
				text: model.id
			}));
		} catch (error) {
			console.error('Error fetching suggestions:', error);
			suggestions = [];
		}
	}

	async function fetchModelCard() {
		try {
			if (!selectedModel) {
				modelCard = null;
				return;
			}
			let rawContent = await api.getHFModelCard(selectedModel);
			const lines = rawContent.split('\n');
			let inFrontMatter = false;
			let contentStarted = false;
			let filteredContentLines: string[] = [];
			for (const line of lines) {
				if (line.trim() === '---') {
					if (!inFrontMatter) {
						inFrontMatter = true;
					} else {
						inFrontMatter = false;
						contentStarted = true;
						continue;
					}
				} else if (inFrontMatter) {
					continue;
				}
				if (contentStarted) {
					filteredContentLines.push(line);
				} else if (!inFrontMatter && line.trim() !== '') {
					contentStarted = true;
					filteredContentLines.push(line);
				}
			}
			const filtered = filteredContentLines.join('\n').trim();
			modelCard = await compile(filtered);
		} catch (e) {
			console.error('Error fetching model card:', e);
			modelCard = null;
		}
	}

	onMount(() => {
		// If prefetched models were provided, use them directly
		if (prefetchedModels && prefetchedModels.length > 0) {
			models = prefetchedModels;
			suggestions = models.map((item) => ({ id: item.id, text: item.id }));
			if (selectedModel && !suggestions.some((s) => s.id === selectedModel)) {
				suggestions = [{ id: selectedModel, text: selectedModel }, ...suggestions];
			}
			comboBoxReady = true;
			return;
		}

		// Timeout fallback: ensure ComboBox renders within 5 seconds even if fetch fails
		const fallbackTimeout = setTimeout(() => {
			if (!comboBoxReady) {
				if (selectedModel && !suggestions.some((s) => s.id === selectedModel)) {
					suggestions = [{ id: selectedModel, text: selectedModel }];
				}
				comboBoxReady = true;
			}
		}, 5000);

		(async () => {
			try {
				models = await api.getHFModels('ibm-granite/granite-4.0-h-micro', 20);
				suggestions = models?.map((item) => ({ id: item.id, text: item.id })) || [];
			} catch (err) {
				console.error('Error fetching initial models:', err);
				suggestions = [];
			}
			if (selectedModel && !suggestions.some((s) => s.id === selectedModel)) {
				suggestions = [{ id: selectedModel, text: selectedModel }, ...suggestions];
			}
			comboBoxReady = true;
			clearTimeout(fallbackTimeout);
		})();

		return () => clearTimeout(fallbackTimeout);
	});
</script>

<Grid noGutter fullWidth>
	<!-- Header -->
	<Row>
		<Column>
			<div class="intro">
				<h4 class="intro-title">Choose Your Fine-Tuning Approach</h4>
				<p class="intro-subtitle">Select what you want to achieve and choose a base model.</p>
			</div>
		</Column>
	</Row>

	<!-- Question 1: Goal Selection -->
	<Row>
		<Column>
			<div class="section-label">
				<span class="section-number">1</span> What do you want to achieve?
			</div>
		</Column>
	</Row>
	<Row role="group" aria-label="Select tuning goal">
		{#each Utils.GOAL_OPTIONS as goal}
			<Column lg={5} md={4} sm={4}>
				<SelectableTile
					selected={selectedGoal === goal.id}
					on:select={() => selectGoal(goal.id)}
					style="--accent: var(--atx-accent-{goal.id.replace(
						'_',
						'-'
					)}); min-height: 120px; padding: 1.25rem;"
				>
					<div class="goal-tile-layout">
						<div class="goal-icon" class:selected={selectedGoal === goal.id}>
							<svelte:component this={goalIcons[goal.id]} size={32} />
						</div>
						<div class="goal-tile-content">
							<div
								style="display: flex; flex-direction: column; gap: 0.375rem; margin-bottom: var(--cds-spacing-03, 0.5rem);"
							>
								<h6 class="tile-heading" style="margin: 0; display:block">{goal.title}</h6>
								<p class="tile-subtitle" style="margin: 0; display:block">{goal.sub_title}</p>
							</div>
							<p class="helper-text">{goal.description}</p>
							<div class="goal-footer">
								<span class="goal-footer-label">Data:</span>
								<span class="goal-footer-desc">{goal.dataDescription}</span>
							</div>
						</div>
					</div>
				</SelectableTile>
			</Column>
		{/each}
	</Row>

	{#if selectedGoal}
		<!-- Resource Panels (library info per goal) -->
		{#if selectedGoal === 'sft'}
			<Row class="section-row">
				<Column lg={15} md={8} sm={4}>
					<Tile class="resource-panel" style="--panel-accent: var(--atx-accent-sft)">
						<div class="resource-panel-intro">
							AutoTuneX uses <strong>PEFT</strong> (Parameter-Efficient Fine-Tuning) by HuggingFace under
							the hood for all SFT training. Understanding PEFT's adapters and data format will help
							you prepare your datasets and get the most out of your fine-tuning runs.
						</div>
						<Accordion size="sm">
							<AccordionItem title="What is PEFT?" open>
								<p class="resource-panel-section-text">
									A library for efficiently adapting large language models to downstream tasks by
									training only a small number of extra parameters. It supports methods like LoRA,
									QLoRA, Prefix Tuning, P-Tuning, Prompt Tuning, IA3, and more — enabling
									fine-tuning with significantly less memory and compute than full model training.
								</p>
							</AccordionItem>
							<AccordionItem title="Data Format">
								<p class="resource-panel-section-text">
									PEFT expects standard instruction-following datasets in <strong>JSON</strong>,
									<strong>JSONL</strong>, or <strong>CSV</strong> format with input/output pairs:
								</p>
								<div class="resource-panel-data-example">
									{`{
  "input": "Summarize the following article: ...",
  "output": "The article discusses ..."
}`}
								</div>
								<p class="resource-panel-data-note">
									Each row should contain an <code>input</code> field (the instruction or prompt)
									and an <code>output</code> field (the expected response). Multi-turn conversations
									can use a <code>messages</code> list in chat-template format.
								</p>
							</AccordionItem>
							<AccordionItem title="Documentation & Resources">
								<div class="resource-panel-links">
									<Link icon={Launch} href="https://huggingface.co/docs/peft" target="_blank"
										>PEFT Documentation</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/peft/quicktour"
										target="_blank">Quickstart Guide</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/peft/conceptual_guides/lora"
										target="_blank">LoRA Conceptual Guide</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/peft/conceptual_guides/adapter"
										target="_blank">Supported Methods</Link
									>
									<Link icon={Launch} href="https://github.com/huggingface/peft" target="_blank"
										>GitHub Repository</Link
									>
								</div>
							</AccordionItem>
						</Accordion>
					</Tile>
				</Column>
			</Row>
		{:else if selectedGoal === 'offline_rl'}
			<Row class="section-row">
				<Column lg={15} md={8} sm={4}>
					<Tile class="resource-panel" style="--panel-accent: var(--atx-accent-offline-rl)">
						<div class="resource-panel-intro">
							AutoTuneX uses <strong>TRL</strong> (Transformer Reinforcement Learning) by HuggingFace
							under the hood for all Offline RL training. Understanding TRL's preference learning methods
							and data format will help you prepare your datasets and get the most out of your training
							runs.
						</div>
						<Accordion size="sm">
							<AccordionItem title="What is TRL?" open>
								<p class="resource-panel-section-text">
									A library for training language models with reinforcement learning techniques,
									including Direct Preference Optimization (DPO), Kahneman-Tversky Optimization
									(KTO), and other preference-based methods. It enables aligning models with human
									preferences without requiring online reward model inference.
								</p>
							</AccordionItem>
							<AccordionItem title="Data Format">
								<p class="resource-panel-section-text">
									TRL expects preference datasets with chosen/rejected response pairs in <strong
										>JSON</strong
									>
									or <strong>JSONL</strong> format:
								</p>
								<div class="resource-panel-data-example">
									{`{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses quantum bits ...",
  "rejected": "Quantum computing is a type of ..."
}`}
								</div>
								<p class="resource-panel-data-note">
									Each row should contain a <code>prompt</code>, a <code>chosen</code> field (the
									preferred response), and a <code>rejected</code> field (the less preferred
									response). For KTO, use <code>completion</code> and <code>label</code> (true/false)
									instead.
								</p>
							</AccordionItem>
							<AccordionItem title="Documentation & Resources">
								<div class="resource-panel-links">
									<Link icon={Launch} href="https://huggingface.co/docs/trl" target="_blank"
										>TRL Documentation</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/trl/dpo_trainer"
										target="_blank">DPO Trainer Guide</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/trl/dataset_formats"
										target="_blank">Dataset Format Guide</Link
									>
									<Link
										icon={Launch}
										href="https://huggingface.co/docs/trl/kto_trainer"
										target="_blank">KTO Trainer Guide</Link
									>
									<Link icon={Launch} href="https://github.com/huggingface/trl" target="_blank"
										>GitHub Repository</Link
									>
								</div>
							</AccordionItem>
						</Accordion>
					</Tile>
				</Column>
			</Row>
		{:else if selectedGoal === 'online_rl'}
			<Row class="section-row">
				<Column lg={15} md={8} sm={4}>
					<Tile class="resource-panel" style="--panel-accent: var(--atx-accent-online-rl)">
						<div class="resource-panel-intro">
							AutoTuneX uses <strong>VERL</strong> (Volcano Engine RL) under the hood for all Online
							RL training. Understanding VERL's data format and configuration will help you prepare your
							datasets and get the most out of your training runs.
						</div>
						<Accordion size="sm">
							<AccordionItem title="What is VERL?" open>
								<p class="resource-panel-section-text">
									A flexible, efficient, and production-ready reinforcement learning training
									library for large language models. It supports a range of algorithms and
									distributed backends for scalable post-training.
								</p>
							</AccordionItem>
							<AccordionItem title="Data Format">
								<p class="resource-panel-section-text">
									VERL expects datasets in <strong>Parquet</strong> format with the following fields:
								</p>
								<div class="resource-panel-data-example">
									{`{
  "data_source": "gsm8k",
  "prompt": [
    { "role": "user", "content": "What is 2 + 2?" }
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "4"
  },
  "extra_info": { "split": "train", "index": 0 }
}`}
								</div>
								<p class="resource-panel-data-note">
									The <code>prompt</code> field uses HuggingFace chat template format — a list of
									<code>role</code>/<code>content</code> dictionaries.
								</p>
							</AccordionItem>
							<AccordionItem title="Documentation & Resources">
								<div class="resource-panel-links">
									<Link
										icon={Launch}
										href="https://verl.readthedocs.io/en/latest/index.html"
										target="_blank">VERL Documentation</Link
									>
									<Link
										icon={Launch}
										href="https://verl.readthedocs.io/en/latest/start/quickstart.html"
										target="_blank">Quickstart Guide</Link
									>
									<Link
										icon={Launch}
										href="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html"
										target="_blank">Data Preparation</Link
									>
									<Link
										icon={Launch}
										href="https://verl.readthedocs.io/en/latest/preparation/reward_function.html"
										target="_blank">Reward Functions</Link
									>
									<Link
										icon={Launch}
										href="https://verl.readthedocs.io/en/latest/examples/config.html"
										target="_blank">Configuration Guide</Link
									>
									<Link icon={Launch} href="https://github.com/verl-project/verl" target="_blank"
										>GitHub Repository</Link
									>
								</div>
							</AccordionItem>
						</Accordion>
					</Tile>
				</Column>
			</Row>
		{/if}

		<!-- Question 2: Model Selection -->
		<Row class="section-row">
			<Column>
				<div class="section-label">
					<span class="section-number">2</span> Select a base model
				</div>
			</Column>
		</Row>
		<Row>
			<Column lg={8} md={6} sm={4}>
				<FormGroup>
					<RadioButtonGroup
						legendText="Model source"
						name="model_source_wizard"
						bind:selected={modelSource}
					>
						<RadioButton labelText="Huggingface" value={ModelSource.HuggingFace} />
						{#if $featureFlags.customPathModelSource && $currentUser?.role === 'admin'}
							<RadioButton labelText="Custom Path" value={ModelSource.CustomPath} />
						{/if}
					</RadioButtonGroup>
				</FormGroup>

				<FormGroup>
					{#if modelSource === ModelSource.CustomPath}
						<TextInput
							labelText="Model path"
							placeholder="/gb-lakehouse-prod-read-only/models/..."
							bind:value={selectedModel}
							helperText="Enter the full filesystem path to the model on GB compute workers"
						/>
					{:else}
						<div class="model-combo-row">
							<div class="model-combo-field">
								{#if comboBoxReady}
									<ComboBox
										titleText="Model"
										bind:selectedId={selectedModel}
										placeholder="Search for a model..."
										items={suggestions}
										shouldFilterItem={() => true}
										on:clear={() => {
											suggestions = models?.map((item) => ({ id: item.id, text: item.id })) || [];
											selectedModel = '';
										}}
										on:keyup={(e) => {
											clearTimeout(debounceTimeout);
											debounceTimeout = setTimeout(() => {
												const inputValue = e.target?.value || '';
												if (inputValue) fetchSuggestions(inputValue);
											}, 500);
										}}
										on:select={(e) => {
											if (e.detail.selectedItem?.id) {
												selectedModel = e.detail.selectedItem.id;
												fetchModelCard();
											}
										}}
									/>
								{:else}
									<InlineLoading description="Loading models..." />
								{/if}
							</div>
							{#if selectedModel}
								<Button
									kind="ghost"
									size="field"
									icon={View}
									iconDescription="View model details"
									on:click={() => {
										if (modelSource === ModelSource.HuggingFace && !modelCard) fetchModelCard();
										showModelCardModal = true;
									}}
								>
									View details
								</Button>
							{/if}
						</div>
					{/if}
				</FormGroup>
			</Column>
		</Row>

		{#if $currentUser?.role === 'admin'}
			<Row>
				<Column lg={8} md={6} sm={4}>
					<Checkbox
						bind:checked={autotuneEnabled}
						labelText="AutoTune (use hyperparameter optimization)"
					/>
				</Column>
			</Row>
		{/if}
	{/if}
</Grid>

<!-- Model card modal -->
<Modal bind:open={showModelCardModal} modalHeading="Model Card" passiveModal size="lg">
	{#if modelCard?.code}
		<div class="model-card-content">
			{@html sanitizeModelCard(modelCard.code)}
		</div>
	{:else}
		<InlineLoading description="Loading model card..." />
	{/if}
</Modal>

<style>
	.intro {
		margin-bottom: var(--cds-spacing-06, 1.5rem);
	}

	.intro-title {
		margin: 0 0 0.375rem 0;
		font-weight: 600;
		letter-spacing: -0.01em;
	}

	.intro-subtitle {
		color: var(--cds-text-02, #525252);
		font-size: 0.875rem;
		margin: 0;
		max-width: 600px;
		line-height: 1.4;
	}

	.section-label {
		display: flex;
		align-items: center;
		gap: var(--cds-spacing-03, 0.5rem);
		font-size: 0.75rem;
		font-weight: 600;
		letter-spacing: 0.32px;
		color: var(--cds-text-02, #525252);
		margin-bottom: 0.75rem;
	}

	.section-number {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 1.25rem;
		height: 1.25rem;
		border-radius: 50%;
		background: var(--cds-interactive, #0f62fe);
		color: var(--cds-text-on-color, #fff);
		font-size: 0.6875rem;
		font-weight: 700;
	}

	:global(.section-row) {
		margin-top: var(--cds-spacing-07, 2rem);
	}

	/* SelectableTile overrides */
	:global(.bx--tile--selectable) {
		display: flex;
		flex-direction: column;
		width: 100%;
		border-radius: 4px;
		transition:
			border-color 0.15s,
			background 0.15s,
			box-shadow 0.15s;
	}

	:global(.bx--tile--selectable:hover) {
		background: var(--cds-layer-hover, #e8e8e8);
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
	}

	:global(.bx--tile--selectable[aria-checked='true']) {
		border-color: var(--accent, var(--cds-interactive, #0f62fe));
		background: var(--cds-layer, #fff);
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
	}

	/* Goal card elements */
	.goal-tile-layout {
		display: flex;
		align-items: flex-start;
		gap: 1rem;
	}

	.goal-tile-content {
		flex: 1;
		min-width: 0;
	}

	.goal-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 48px;
		height: 48px;
		border-radius: 8px;
		background: var(--cds-border-subtle, #e0e0e0);
		color: var(--cds-text-01, #161616);
		flex-shrink: 0;
		transition:
			background 0.15s,
			color 0.15s;
	}

	.goal-icon.selected {
		background: var(--accent, var(--cds-interactive, #0f62fe));
		color: var(--cds-text-on-color, #fff);
	}

	.tile-heading {
		font-weight: 600;
		font-size: 0.875rem;
		color: var(--cds-text-01, #161616);
		margin: 0 0 var(--cds-spacing-03, 0.5rem) 0;
		line-height: 1.3;
	}

	.helper-text {
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
		line-height: 1.45;
		margin: 0 0 0.75rem 0;
		flex: 1;
	}

	.goal-footer {
		margin-top: auto;
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding-top: 0.625rem;
		border-top: 1px solid
			color-mix(in srgb, var(--accent, var(--cds-interactive, #0f62fe)) 20%, transparent);
		font-size: 0.75rem;
		line-height: 1.3;
		color: var(--cds-text-02, #525252);
	}

	.goal-footer :global(svg) {
		flex-shrink: 0;
		fill: var(--accent, var(--cds-interactive, #0f62fe));
		opacity: 0.7;
	}

	.goal-footer-label {
		font-weight: 600;
		color: var(--cds-text-01, #161616);
		white-space: nowrap;
	}

	.goal-footer-desc {
		color: var(--cds-text-02, #525252);
	}

	.model-combo-row {
		display: flex;
		align-items: flex-end;
		gap: 0.5rem;
	}

	.model-combo-field {
		flex: 1;
		min-width: 0;
	}

	/* Model card modal content */
	.model-card-content {
		font-size: 0.875rem;
		line-height: 1.5;
	}

	.model-card-content :global(h1) {
		font-size: 20px;
		font-weight: 600;
	}
	.model-card-content :global(h2) {
		margin-top: 1rem;
		font-size: 18px;
		font-weight: 600;
	}
	.model-card-content :global(h3) {
		margin-top: 1rem;
		font-size: 16px;
		font-weight: 600;
	}
	.model-card-content :global(> p) {
		padding-top: 1rem;
		padding-bottom: 1rem;
	}
	.model-card-content :global(li) {
		margin-top: 0.5rem;
		margin-bottom: 0.5rem;
	}
	.model-card-content :global(pre) {
		background-color: var(--cds-layer, #f4f4f4);
	}
	.model-card-content :global(img) {
		max-width: -webkit-fill-available;
	}
	.model-card-content :global(table) {
		width: 100%;
	}

	/* Resource Panel (shared by PEFT, TRL, VERL) */
	:global(.resource-panel) {
		border-left: 3px solid var(--panel-accent, var(--cds-interactive, #0f62fe));
		padding: var(--cds-spacing-05, 1rem) !important;
	}

	.resource-panel-intro {
		font-size: 14px;
		color: var(--cds-text-02, #525252);
		line-height: 1.45;
		margin: 0 0 var(--cds-spacing-04, 0.75rem) 0;
	}

	.resource-panel-section-text {
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
		line-height: 1.45;
		margin: 0;
	}

	.resource-panel-data-example {
		background: var(--cds-layer, #f4f4f4);
		border-radius: 4px;
		padding: var(--cds-spacing-04, 0.75rem);
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.75rem;
		line-height: 1.5;
		overflow-x: auto;
		margin: var(--cds-spacing-03, 0.5rem) 0;
		white-space: pre;
		color: var(--cds-text-01, #161616);
	}

	.resource-panel-data-note {
		font-size: 0.75rem;
		color: var(--cds-text-02, #525252);
		margin-top: var(--cds-spacing-03, 0.5rem);
		line-height: 1.4;
	}

	.resource-panel-data-note code {
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.75rem;
		background: var(--cds-layer, #f4f4f4);
		padding: 0.0625rem 0.25rem;
		border-radius: 2px;
	}

	.resource-panel-links {
		display: flex;
		flex-direction: column;
		gap: var(--cds-spacing-03, 0.5rem);
	}

	:global(.resource-panel .bx--accordion) {
		border-top: none;
	}

	:global(.resource-panel .bx--accordion__content) {
		padding-right: var(--cds-spacing-05, 1rem);
		font-size: 0.8125rem;
		line-height: 1.45;
	}

	/* Ensure ComboBox has visible background against g10 page (#f4f4f4) */
	:global(.bx--list-box) {
		background-color: #ffffff !important;
	}
	:global(.bx--list-box input.bx--text-input) {
		background-color: #ffffff !important;
	}
</style>
