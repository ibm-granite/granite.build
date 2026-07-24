<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Column, FormGroup, Row, TextInput } from 'carbon-components-svelte';

	export let data;
</script>

{#if data}
	<Row>
		<!-- <Column sm={1}>
			<FormGroup>
				<TextInput labelText="Namespace" value="autotunex" readonly />
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Table" value="model_shared" readonly />
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Revision" value={data?.id} readonly />
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Base Model" value={data?.model} readonly />
			</FormGroup>
		</Column> -->
		<Column sm={1}>
			<FormGroup>
				<TextInput
					labelText="Label"
					bind:value={data.label}
					helperText="This will be automatically prefixed with 'autotunex.' in DMF."
				/>
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Variant" bind:value={data.variant} />
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Type" bind:value={data.type} />
			</FormGroup>
		</Column>
		<Column sm={1}>
			<FormGroup>
				<TextInput labelText="Size" bind:value={data.size} />
			</FormGroup>
		</Column>
	</Row>
{/if}
