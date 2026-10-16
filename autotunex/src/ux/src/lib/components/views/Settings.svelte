<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script type="ts">
	import { Grid, Row, Column } from 'carbon-components-svelte';
	import Configurations from '../tables/Configurations.svelte';
	import DatasetNotifier from '../DatasetNotifier.svelte';
	import Datasets from '../tables/Datasets.svelte';
	import { userMetadata } from '$lib/store';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';

	let api = new API();
	let updated = 0;

	onMount(async () => {
		if (!$userMetadata) {
			let metadata = await api.getUserMetadata();
			userMetadata.set(metadata);
		}
	});
</script>

{#if $userMetadata && $userMetadata?.number_of_datasets === 0}
	<DatasetNotifier
		on:create={async () => {
			userMetadata.update((prev) => {
				return { ...prev, number_of_datasets: prev.number_of_datasets + 1 };
			});
			updated = updated + 1;
		}}
	/>
{/if}
<Grid noGutter fullWidth>
	<Row>
		<Column>
			<Configurations />
		</Column>
		<Column>
			{#key updated}
				<Datasets />
			{/key}
		</Column>
	</Row>
</Grid>
