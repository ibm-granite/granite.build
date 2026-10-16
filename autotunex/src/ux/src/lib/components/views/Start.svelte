<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Button,
		Grid,
		Row,
		Column,
		ButtonSet,
		Tab,
		Tabs,
		TabContent
	} from 'carbon-components-svelte';
	import { TechnicalOwner } from 'carbon-pictograms-svelte';
	import { Login, ModelTuned, Settings, UserAvatar } from 'carbon-icons-svelte';
	import Resources from '$lib/components/views/Resources.svelte';
	import { isAuthenticated, currentUser } from '$lib/store';
	import { API } from '$lib/api';

	export let view;
	const api = new API();
	$: innerWidth = 0;
	$: innerHeight = 0;
	let selectedTab = 0;
</script>

<Tabs bind:selected={selectedTab}>
	<Tab label="Welcome" />
	<Tab label="Resources" />
	<svelte:fragment slot="content">
		<TabContent>
			<Grid fullWidth padding>
				<Row>
					<Column>
						<TechnicalOwner width="120" height="120" />
						<div style="position: relative; margin-left: 160px; margin-top: -110px;">
							<h1>Welcome to AutoTuneX</h1>
							<h4>Interactive Automated Fine-Tuning for Large Language Models</h4>
						</div>
						<p style="margin-top: 30px; margin-bottom: 20px; max-width: 900px;">
							AutoTune is an end‑to‑end distributed training stack for fine‑tuning large language
							models at scale. Built on Ray, it automates hyperparameter optimization and
							orchestrates training across multiple GPUs and nodes, supporting supervised
							fine‑tuning (full‑model and parameter‑efficient methods such as LoRA and aLoRA),
							offline preference learning with DPO or KTO, and online reinforcement learning with
							PPO, GRPO, and DAPO. Beyond individual training methods, AutoTune manages the entire
							lifecycle of a run - from automated hyperparameter search and large‑scale distributed
							execution with DeepSpeed or FSDP, to robust checkpointing and model saving. The system
							is model‑agnostic and works out of the box with any HuggingFace‑compatible
							decoder‑only (causal) language model, providing a unified, scalable foundation for
							experimentation, alignment, and production‑ready fine‑tuning.
						</p>
						<div style="display: flex; align-items: center; gap: 8px;">
							<UserAvatar size={32} />
							<span>Built by the AutoTuneX team</span>
						</div>
						<ButtonSet style="margin-top: 60px;">
							{#if $isAuthenticated}
								<Button
									icon={ModelTuned}
									kind="primary"
									on:click={() => {
										view = 'tunings';
									}}
								>
									Tunings
								</Button>
								<Button
									icon={Settings}
									kind="ghost"
									on:click={() => {
										view = 'settings';
									}}
								>
									Settings
								</Button>
							{:else}
								<Button
									icon={Login}
									kind="primary"
									on:click={async () => {
										// Redirects the browser to the backend OIDC entry point (or is a
										// no-op in standalone, where the caller is already authenticated).
										await api.login();
									}}>Login</Button
								>
							{/if}
						</ButtonSet>
					</Column>
				</Row>
			</Grid>
		</TabContent>
		<TabContent>
			<Resources />
		</TabContent>
	</svelte:fragment>
</Tabs>

<svelte:window bind:innerWidth bind:innerHeight />

{#if innerWidth > 800 && innerHeight - 800 > 0}
	<div
		style={`font-family: sans-serif; position: absolute; left: 0px; bottom: 0px; width: 100%; line-height: ${
			innerHeight - 800
		}px; height: ${
			innerHeight - 800
		}px; background: #060606; color: #e0e0e0; text-align: center; padding-right: 30px; font-size: 12px;`}
	>
		Copyright &copy; 2025 IBM Corp. Licensed under the Apache License 2.0.
	</div>
{/if}
