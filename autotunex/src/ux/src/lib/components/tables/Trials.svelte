<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Button,
		CodeSnippet,
		DataTable,
		InlineLoading,
		Pagination,
		ProgressBar,
		Tab,
		TabContent,
		Tabs,
		Toolbar,
		ToolbarBatchActions
	} from 'carbon-components-svelte';
	import Compare from '../Compare.svelte';
	import TrialMetricsChart from '../displays/TrialMetricsChart.svelte';
	import { Utils } from '$lib/utils';
	import { onDestroy } from 'svelte';
	import type { Trial } from '$lib/app-types';
	import {
		trialLogsStore,
		fetchAndCacheTrialLogs,
		loadOlderTrialLogs,
		startTrialLogPoll,
		stopTrialLogPoll
	} from '$lib/app';

	export let trials: Trial[] = [];
	export let jobId: string;
	export let selectedRows: Trial[] = [];
	export let showCompare: boolean = false;

	let page = 1;
	let pageSize = 10;
	let selectedRowIds: string[] = [];

	const expandedTrials = new Set<string>();

	// Per-row expanded-tab index (Logs=0, Configuration=1, Training Curves=2). Bound on
	// the inner Tabs so each expanded row tracks its own selection independently, and so
	// the chart mounts only when its tab is visible (a LineChart mis-measures in a hidden
	// 0-size TabContent). Initialized to 0 on expand so Logs still shows by default.
	let trialTab: Record<string, number> = {};

	const flattenObject = (obj: Record<string, any>, parentKey = '', sep = '.') => {
		const flatObj: Record<string, any> = {};

		for (const key in obj) {
			if (Object.prototype.hasOwnProperty.call(obj, key)) {
				const newKey = parentKey ? parentKey + sep + key : key;

				if (typeof obj[key] === 'object' && obj[key] !== null) {
					Object.assign(flatObj, flattenObject(obj[key], newKey, sep));
				} else {
					flatObj[newKey] = obj[key];
				}
			}
		}
		return flatObj;
	};

	const handleTrialScroll = (e: Event, trialId: string) => {
		const el = e.target as HTMLElement;
		if (el.scrollTop + el.clientHeight >= el.scrollHeight - 50) {
			loadOlderTrialLogs(jobId, trialId);
		}
	};

	onDestroy(() => {
		expandedTrials.forEach((id) => stopTrialLogPoll(id));
	});

	$: selectedRows = trials?.filter(
		(trial) => selectedRowIds.filter((r_id) => r_id === trial.id).length > 0
	);
</script>

{#if showCompare && selectedRows.length > 0}
	<span
		class="back_navigation"
		on:click={() => {
			showCompare = false;
		}}
	>
		&lt; Back to trials
	</span>
	<Compare
		rows={selectedRows.map((row) => {
			delete row.config.training_config.output_dir;
			delete row.config.training_config.train_file;
			delete row.config.training_config.test_file;
			delete row.config.training_config.validation_file;
			delete row.config.training_config.resource_name;

			const config = flattenObject(row.config);
			const metrics = structuredClone(row.score.metrics);
			if (metrics?.loss) {
				metrics.loss = +metrics.loss?.toFixed(5);
			}
			if (metrics?.train_loss) {
				metrics.train_loss = +metrics.train_loss?.toFixed(5);
			}
			if (metrics?.total_time) {
				metrics.total_time = Utils.formatTime(+metrics?.total_time);
			}
			return { id: row.id, ...metrics, ...config };
		})}
	/>
{:else}
	<DataTable
		sortable
		selectable
		expandable
		on:click:row--expand={async (e) => {
			const trialId = e.detail.row.id;
			if (e.detail.expanded) {
				trialTab = { ...trialTab, [trialId]: trialTab[trialId] ?? 0 };
				expandedTrials.add(trialId);
				if (['SUBMITTED', 'PENDING', 'RUNNING'].includes(e.detail.row.status)) {
					startTrialLogPoll(jobId, trialId);
				} else {
					fetchAndCacheTrialLogs(jobId, trialId, { status: e.detail.row.status });
				}
			} else {
				expandedTrials.delete(trialId);
				stopTrialLogPoll(trialId);
			}
		}}
		batchSelection={trials.some((trial) => trial.status === 'COMPLETED')}
		bind:selectedRowIds
		nonSelectableRowIds={trials
			.filter((trial) => trial.status === 'RUNNING')
			.map((trial) => trial.id)}
		on:click:row--expand
		size="short"
		sortKey={trials?.some((trial) => trial.status === 'RUNNING') ? 'status' : 'score'}
		sortDirection="ascending"
		{pageSize}
		{page}
		headers={[
			{
				key: 'created_at',
				value: 'Created on',
				display: (date) => new Date(date).toLocaleString()
			},
			{ key: 'id', value: 'Trial id' },
			{ key: 'status', value: 'Status' },
			{ key: 'score', value: 'Loss' },
			{ key: 'total_time', value: 'Total time' }
		]}
		rows={trials.map((trial) => {
			const metric = trial?.score?.metric;
			const score = trial?.score?.metrics?.[metric];
			const total_time = trial?.score?.metrics?.total_time;
			return {
				...trial,
				metric: metric,
				score: score,
				total_time: total_time
				// logs: trialsLog[trial.id] ?? []
			};
		})}
	>
		<svelte:fragment slot="cell" let:row let:cell>
			{#if cell.key === 'id' && cell.value}
				{cell.value}
			{:else if cell.key === 'status' && cell.value === 'RUNNING'}
				<InlineLoading status="active" description="Running..." />
			{:else if cell.key === 'status' && cell.value === 'PENDING'}
				<InlineLoading status="active" description="Pending..." />
			{:else if cell.key === 'status' && cell.value === 'PAUSED'}
				<InlineLoading status="active" description="Paused" />
			{:else if cell.key === 'status' && cell.value === 'TERMINATED'}
				<InlineLoading status="error" description="Terminated" />
			{:else if cell.key === 'status' && cell.value === 'ERROR'}
				<InlineLoading status="error" description="Error" />
			{:else if cell.key === 'status' && cell.value === 'COMPLETED'}
				<InlineLoading status="finished" description="Completed" />
			{:else if cell.key === 'total_time'}
				{`${
					cell.value
						? Utils.formatTime(cell.value)
						: Utils.getTimeElapsed(row.created_at, new Date(), true)
				}`}
			{:else if cell.key === 'score'}
				{cell.value ? cell.value?.toFixed(4) : ''}
			{:else}
				{cell.display ? cell.display(cell.value, row) : Utils.toUpperCase(cell.value)}
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="expanded-row" let:row>
			{@const trialLogEntry = $trialLogsStore[row.id]}
			<Tabs type="container" bind:selected={trialTab[row.id]}>
				<Tab label="Logs" />
				<Tab label="Configuration" />
				<Tab label="Training Curves" />
				<svelte:fragment slot="content">
					<TabContent>
						{#if !trialLogEntry}
							<ProgressBar size="sm" helperText="Loading details..." />
						{:else if trialLogEntry.logs.length === 0}
							<div class="log-viewer">
								<div class="log-line" style="color: #a8a8a8;">No logs available</div>
							</div>
						{:else}
							<div class="log-viewer" on:scroll={(e) => handleTrialScroll(e, row.id)}>
								{#each trialLogEntry.logs as log}
									<div class="log-line">
										{new Date(log.timestamp).toLocaleString()}
										{log.level} -- {log.filename} -- {log.message}
									</div>
								{/each}
								{#if trialLogEntry.hasMore}
									<div style="padding: 0.5rem 1rem;">
										<InlineLoading description="Loading older logs..." style="color: #f4f4f4" />
									</div>
								{/if}
							</div>
						{/if}
					</TabContent>
					<TabContent>
						<CodeSnippet
							type="multi"
							code={JSON.stringify(row.config, null, 2)}
							wrapText
							style="max-width: 100%; word-break: break-word;"
						/>
					</TabContent>
					<TabContent>
						{#if trialTab[row.id] === 2}
							<TrialMetricsChart {jobId} trialId={row.id} status={row.status} />
						{/if}
					</TabContent>
				</svelte:fragment>
			</Tabs>
		</svelte:fragment>
		<Toolbar size="sm">
			<ToolbarBatchActions
				on:cancel={(e) => {
					e.preventDefault();
					selectedRowIds = [];
					showCompare = false;
				}}
			>
				{#if selectedRows.length > 1}
					<Button on:click={() => (showCompare = true)}>Compare</Button>
				{:else}
					<Button on:click={() => (showCompare = true)}>View</Button>
				{/if}
			</ToolbarBatchActions>
		</Toolbar>
	</DataTable>
	<Pagination bind:pageSize bind:page totalItems={trials.length} pageSizeInputDisabled />
{/if}

<style>
	.back_navigation {
		cursor: pointer;
		width: fit-content;
		font-size: 20px;
		display: block;
		font-weight: 300;
		color: #504f4f;
	}

	.log-viewer {
		background: #161616;
		color: #f4f4f4;
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.75rem;
		line-height: 1.4;
		padding: 1rem;
		max-height: 300px;
		overflow-y: auto;
		white-space: pre-wrap;
		word-break: break-word;
	}

	.log-line {
		padding: 1px 0;
	}
</style>
