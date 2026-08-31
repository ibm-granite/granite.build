<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Column, NumberInput, Row, TextInput, Tile } from 'carbon-components-svelte';
	import TimeInput from '$lib/components/TimeInput.svelte';
	import type { ConfigData } from '$lib/app-types';

	export let config: ConfigData;
	export let isTuning = false;
</script>

<Column>
	<Tile>
		<Row style="margin-bottom: 2rem;">
			<Column>
				<NumberInput
					labelText="Num GPUs per trial"
					helperText={config['training_config']['num_gpus_per_trial'].description}
					id="num_gpus_per_trial"
					bind:value={config['training_config']['num_gpus_per_trial'].default}
					min={config['training_config']['num_gpus_per_trial'].min_val}
					max={config['training_config']['num_gpus_per_trial'].max_val}
					invalidText={`Value must be between ${config['training_config']['num_gpus_per_trial'].min_val} and ${config['training_config']['num_gpus_per_trial'].max_val}`}
					step={config['training_config']['num_gpus_per_trial'].type === 'float' ? 0.01 : 1}
					on:change={() => {
						config['tune_config']['max_concurrent_trials'].default = Math.floor(
							config['training_config']['num_gpus_per_trial'].max_val /
								config['training_config']['num_gpus_per_trial'].default
						);
					}}
				/>
			</Column>
			<Column>
				<NumberInput
					labelText="Max concurrent trials"
					helperText={config['tune_config']['max_concurrent_trials'].description}
					id="max_concurrent_trials"
					bind:value={config['tune_config']['max_concurrent_trials'].default}
					min={config['tune_config']['max_concurrent_trials'].min_val}
					max={Math.floor(
						config['training_config']['num_gpus_per_trial'].max_val /
							config['training_config']['num_gpus_per_trial'].default
					)}
					invalidText={`Value must be between ${
						config['tune_config']['max_concurrent_trials'].min_val
					} and ${Math.floor(
						config['training_config']['num_gpus_per_trial'].max_val /
							config['training_config']['num_gpus_per_trial'].default
					)}`}
					step={config['tune_config']['max_concurrent_trials'].type === 'float' ? 0.01 : 1}
				/>
			</Column>
		</Row>

		<Row style="margin-bottom: 2rem;">
			<Column>
				<NumberInput
					labelText="Num samples (trials)"
					helperText={config['tune_config']['num_samples'].description}
					id="num_samples"
					bind:value={config['tune_config']['num_samples'].default}
					min={config['tune_config']['num_samples'].min_val}
					max={config['tune_config']['num_samples'].max_val}
					invalidText={`Value must be between ${config['tune_config']['num_samples'].min_val} and ${config['tune_config']['num_samples'].max_val}`}
					step={config['tune_config']['num_samples'].type === 'float' ? 0.01 : 1}
				/>
			</Column>
			{#if config?.tune_config?.['time_budget_s']}
				<Column>
					<TimeInput label="Time Budget" bind:value={config.tune_config.time_budget_s} />
				</Column>
			{/if}
		</Row>
		{#if isTuning}
			<Row>
				<Column>
					{#if config.tuners_config?.['alora']}
						<TextInput
							labelText="Invocation string"
							helperText={config.tuners_config?.['alora']?.hyperparams?.invocation_string
								.description}
							placeholder={`Enter invocation string`}
							bind:value={config.tuners_config['alora'].hyperparams.invocation_string.default}
						/>
					{/if}
				</Column>
			</Row>
		{/if}
	</Tile>
</Column>
