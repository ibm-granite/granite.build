<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Column, FormGroup, ProgressBar, Row } from 'carbon-components-svelte';
	import { TextInput } from 'carbon-components-svelte';
	export let data;
</script>

{#if data}
	<Row>
		<Column>
			<FormGroup>
				<TextInput
					labelText="Suffix"
					bind:value={data.suffix}
					placeholder="Enter deployment suffix"
					helperText="byom-gb-xx-<suffix>"
					maxlength={10}
				/>
			</FormGroup>
		</Column>
		<Column>
			<FormGroup>
				<TextInput
					labelText="Base model"
					bind:value={data.base_model_if_lora}
					placeholder="Enter base model with revision"
					helperText="granite-3.3-8b-instruct/r250409a"
				/>
			</FormGroup>
		</Column>
		<Column>
			<FormGroup>
				<TextInput
					labelText="Deployment reference"
					bind:value={data.rits_deployment_reference}
					placeholder="Enter deployment reference"
					helperText="granite-3-1-8b-instruct"
				/>
			</FormGroup>
		</Column>
		<!-- <Column sm={1}>
			<FormGroup>
				<TextInput
					labelText="TTL (Minutes)"
					bind:value={data.ttl}
					placeholder="Enter Time to live in minutes"
					maxlength={5}
					helperText="max value: 1440m"
				/>
			</FormGroup>
		</Column> -->
	</Row>
{:else}
	<ProgressBar size="sm" helperText="Loading details..." />
{/if}
