<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
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
	let RESULT_KEYS = ['loss', 'train_loss', 'total_time'];
	export let rows: Record<string, any>[];

	type ValueAnalysis = {
		values: Record<string, number[]>;
		hasDifferences: boolean;
	};

	type AnalysisResult = {
		[key: string]: ValueAnalysis;
	};

	type OddOnesOut = {
		[key: string]: Record<string, number[]>;
	};

	// Function to find keys with different values across objects
	function findDifferentValues(
		objectsArray: Array<Record<string, any>>,
		differKeys: string[]
	): AnalysisResult {
		const result: AnalysisResult = {};

		// Initialize result structure
		differKeys.forEach((key) => {
			result[key] = {
				values: {},
				hasDifferences: false
			};
		});

		// Count occurrences of each value for each key
		objectsArray.forEach((obj, index) => {
			differKeys.forEach((key) => {
				if (obj.hasOwnProperty(key)) {
					const value = obj[key];

					if (!result[key].values[value]) {
						result[key].values[value] = [];
					}
					result[key].values[value].push(index);
				}
			});
		});

		// Determine which keys have differences
		Object.keys(result).forEach((key) => {
			const valueKeys = Object.keys(result[key].values);
			result[key].hasDifferences = valueKeys.length > 1;
		});

		return result;
	}

	// Function to get objects with minority values (odd ones out)
	function getOddOnesOut(
		objectsArray: Array<Record<string, any>>,
		differKeys: string[]
	): OddOnesOut {
		const analysis = findDifferentValues(objectsArray, differKeys);
		const oddOnesOut: OddOnesOut = {};

		Object.keys(analysis).forEach((key) => {
			if (analysis[key].hasDifferences) {
				const values = analysis[key].values;
				const valueEntries = Object.entries(values);

				// Sort by count to find minority values
				valueEntries.sort((a, b) => (a[1] as number[]).length - (b[1] as number[]).length);

				// If there are multiple values, the ones with fewer occurrences are odd ones out
				if (valueEntries.length > 1) {
					const minCount = (valueEntries[0][1] as number[]).length;
					const maxCount = (valueEntries[valueEntries.length - 1][1] as number[]).length;

					// If counts are different, mark minority values as odd ones out
					if (minCount !== maxCount) {
						oddOnesOut[key] = {};

						valueEntries.forEach(([value, indices]) => {
							if ((indices as number[]).length === minCount) {
								oddOnesOut[key][value] = indices as number[];
							}
						});
					}
				}
			}
		});

		return oddOnesOut;
	}

	const restrictedKeys = ['id', 'loss', 'total_time', 'train_loss'];
	const minValues = getOddOnesOut(
		rows,
		Array.from(Utils.findDifferingKeys(rows)).filter((item) => !restrictedKeys.includes(item))
	);
</script>

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
			{#each Object.keys(rows[0]).filter((key) => RESULT_KEYS.includes(key) && Array.from(Utils.findDifferingKeys(rows)).includes(key)) as key, index}
				<StructuredListRow>
					<StructuredListCell><strong>{Utils.toUpperCase(key)}</strong></StructuredListCell>
					{#each rows as c}
						<StructuredListCell>
							{#if c[key] instanceof Array}
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
							{:else if minValues[key] && Object.keys(minValues[key]).includes(c[key]?.toString())}
								<strong>{c[key]}</strong>
							{:else}
								{c[key]}
							{/if}
						</StructuredListCell>
					{/each}
				</StructuredListRow>
			{/each}
			<StructuredListRow
				style={`border: ${
					Array.from(Utils.findDifferingKeys(rows)).length > 0 && '2px solid black'
				}`}
			></StructuredListRow>
			{#each Object.keys(rows[0]).filter((key) => !IGNORE_KEYS.includes(key) && !RESULT_KEYS.includes(key) && Array.from(Utils.findDifferingKeys(rows)).includes(key)) as key, index}
				<StructuredListRow>
					<StructuredListCell><strong>{Utils.toUpperCase(key)}</strong></StructuredListCell>
					{#each rows as c}
						<StructuredListCell>
							{#if c[key] instanceof Array}
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
							{:else if minValues[key] && Object.keys(minValues[key]).includes(c[key]?.toString())}
								<strong>{c[key]}</strong>
							{:else}
								{c[key]}
							{/if}
						</StructuredListCell>
					{/each}
				</StructuredListRow>
			{/each}
			<StructuredListRow
				style={`border: ${
					Array.from(Utils.findDifferingKeys(rows)).length > 0 && '2px solid black'
				}`}
			></StructuredListRow>
			{#each Object.keys(rows[0]).filter((key) => !IGNORE_KEYS.includes(key) && !Array.from(Utils.findDifferingKeys(rows)).includes(key)) as key}
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
