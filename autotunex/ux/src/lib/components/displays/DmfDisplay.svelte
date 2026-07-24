<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Utils } from '$lib/utils';
	import { Column, FormLabel, Grid, ProgressBar, Row } from 'carbon-components-svelte';

	export let data;
	let non_included_keys = ['files', 'model_id'];
</script>

{#if data}
	<Grid noGutter>
		<Row>
			{#each Object.entries(data[0]) as [key, value]}
				{#if !non_included_keys.includes(key)}
					{#if key === 'open'}
						<Column md={2} padding>
							<Column>
								<FormLabel>Visibility</FormLabel>
							</Column>
							<Column>
								<span style="font-family: monospace;">{value ? 'IBM Public' : 'Private'}</span>
							</Column>
						</Column>
					{:else}
						<Column md={2} padding>
							<Column>
								<FormLabel>{Utils.toUpperCase(key)}</FormLabel>
							</Column>
							<Column>
								<span style="font-family: monospace;">{value}</span>
							</Column>
						</Column>
					{/if}
				{/if}
			{/each}
		</Row>
	</Grid>
{:else}
	<ProgressBar size="sm" helperText="Loading model details from dmf..." />
{/if}
