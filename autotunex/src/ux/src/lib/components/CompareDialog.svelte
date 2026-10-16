<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import {
		Modal,
		StructuredListBody,
		CodeSnippet,
		Tag,
		StructuredList,
		StructuredListHead,
		StructuredListRow,
		StructuredListCell
	} from 'carbon-components-svelte';
	import { Utils } from '$lib/utils';

	let IGNORE_KEYS = ['id', 'detail'];

	export let open = false;
	export let entities;
	export let rows;
</script>

<Modal
	bind:open
	primaryButtonText="OK"
	secondaryButtonText="Cancel"
	preventCloseOnClickOutside
	modalHeading={`Compare ${entities}`}
	on:click:button--secondary={() => (open = false)}
	on:open
	on:close
	on:submit
	size="lg"
>
	{#if rows && rows.length > 0}
		<StructuredList condensed flush>
			<StructuredListHead>
				<StructuredListRow head>
					<StructuredListCell head></StructuredListCell>
					{#each rows as c}
						<StructuredListCell head>{c.id}</StructuredListCell>
					{/each}
				</StructuredListRow>
			</StructuredListHead>
			<StructuredListBody style="overflow: scroll;">
				{#each Object.keys(rows[0]).filter((key) => !IGNORE_KEYS.includes(key)) as key}
					<StructuredListRow>
						<StructuredListCell><strong>{Utils.toUpperCase(key)}</strong></StructuredListCell>
						{#each rows as c}
							<StructuredListCell>
								{#if key === 'logs' && c[key]}
									<CodeSnippet
										type="multi"
										code={c[key]
											.map(
												(log) =>
													`${new Date(log.timestamp).toLocaleString()} ${log.level} -- ${
														log.filename
													} -- ${log.message}`
											)
											.join('\n')}
										wrapText
										style="max-width: 100%;"
									/>
								{:else if c[key] instanceof Array}
									{#each c[key] as element}
										{#if typeof element === 'string' || element instanceof String}
											<Tag>{element}</Tag>
										{:else if element.experiment_name}
											<Tag>{element.experiment_name}</Tag>
										{:else if element.name}
											<Tag>{element.name}</Tag>
										{:else}
											<Tag>No name attribute</Tag>
										{/if}
									{/each}
								{:else}
									{c[key]}
								{/if}
							</StructuredListCell>
						{/each}
					</StructuredListRow>
				{/each}
			</StructuredListBody>
		</StructuredList>
	{/if}
</Modal>
