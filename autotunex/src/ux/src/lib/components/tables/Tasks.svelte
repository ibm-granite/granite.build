<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { API } from '$lib/api';
	import { onMount } from 'svelte';
	import Table from '../Table.svelte';
	import { InlineNotification, ProgressBar } from 'carbon-components-svelte';
	import ShowStatus from '../ShowStatus.svelte';
	import { Utils } from '$lib/utils';
	import type { Task } from '$lib/app-types';

	const api = new API();

	export let job_id;

	let isLoading = false;
	let tasks: Task[];

	let taskHeaders = [
		{ key: 'id', value: 'ID' },
		// { key: 'build_id', value: 'Build ID' },
		{
			key: 'started_at',
			value: 'Created on',
			display: (date: Date) => new Date(date).toLocaleString()
		},
		{ key: 'status', value: 'Status' },
		{ key: 'type', value: 'Type' },

		{ key: 'total_time', value: 'Total time' }
	];

	onMount(async () => {
		if (job_id) {
			try {
				isLoading = true;
				tasks = await api.getAllTaskByJob(job_id);
			} catch (error) {
				console.error(error);
			} finally {
				isLoading = false;
			}
		}
	});
</script>

{#if isLoading}
	<ProgressBar size="sm" helperText="Loading details..." />
{:else if !isLoading && tasks?.length > 0}
	<Table
		batchSelection={false}
		showActionButton={false}
		headers={taskHeaders}
		rows={tasks}
		sortKey="started_at"
	>
		<svelte:fragment slot="cell" let:cell let:row>
			{#if cell.key === 'id'}
				{row?.id.split('-')[0]}
			{:else if cell.key === 'status'}
				<ShowStatus status={row?.status} />
			{:else if cell.key === 'total_time'}
				{Utils.getTimeElapsed(row?.started_at, row?.updated_at)}
			{:else}
				{cell.display ? cell.display(cell.value, row) : cell.value}
			{/if}
		</svelte:fragment>
	</Table>
{:else}
	<InlineNotification title="No Tasks Found" kind="info" />
{/if}
