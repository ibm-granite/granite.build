<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Grid, Row, Column, FormLabel, OutboundLink, Tag, Tile } from 'carbon-components-svelte';
	import { Utils } from '$lib/utils';
	import ShowStatus from '../ShowStatus.svelte';
	import { currentUser } from '$lib/store';
	import { ModelSource } from '$lib/app-types';

	export let dict;

	let keys = [
		'status',
		'model',
		'model_source',
		'tuning_type',
		'config_name',
		'dataset',
		'seed',
		'precision',
		'ray_address',
		'cleanup',
		'id',
		'experiment_name',
		'autotune',
		'created_at',
		'updated_at'
	];
	let non_include_keys = [
		'id',
		'experiment_name',
		'cleanup',
		'ray_address',
		'seed',
		'autotune',
		'updated_at',
		'precision'
	];
</script>

<Grid noGutter fullWidth>
	<Row>
		{#each keys.filter((item) => !non_include_keys.includes(item)) as key}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>
						{#if key === 'config_name'}
							Configuration
						{:else}
							{Utils.toUpperCase(key)}
						{/if}
					</FormLabel>
				</Column>
				<Column noGutter>
					<span class="dict-item" style="font-family: monospace">
						{#if key === 'id'}
							{`${dict[key].split('-')[0]}-${dict[key].split('-')[1]}`}
						{:else if key === 'status'}
							<ShowStatus status={dict[key]} />
						{:else if key === 'tuning_type'}
							{#if dict['tuning_type'] && dict['rl_tuner_type']}
								{`Offline RL - ${dict['rl_tuner_type']} with ${dict['tuning_type']}`}
							{:else if !dict['tuning_type'] && dict['rl_tuner_type']}
								{`Online RL - ${dict['rl_tuner_type']}`}
							{:else}
								{dict['tuning_type']}
							{/if}
						{:else if key === 'created_at'}
							{new Date(dict[key])?.toLocaleString()}
						{:else if key === 'model' && dict[key]?.startsWith('/')}
							<span title={dict[key]}>{dict[key].split('/').slice(-2).join('/')}</span>
						{:else if key === 'model_source'}
							<Tag style="margin:0" type={dict[key] === ModelSource.CustomPath ? 'purple' : 'cyan'}
								>{dict[key]}</Tag
							>
						{:else}
							{dict[key]}
						{/if}
					</span>
				</Column>
			</Column>
		{/each}
		<!-- Total time = job start -> end. A running job ticks live to now; a finished
		     job uses finished_at (the latest gb_tasks.updated_at). Jobs with no build
		     task (e.g. local-backend) have no end -> show a dash. NOT updated_at: it is
		     ON UPDATE CURRENT_TIMESTAMP, so any later write bumps it and inflates the
		     duration. Mirrors the Tunings table's total_time. -->
		{#if dict['created_at']}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>Total time</FormLabel>
				</Column>
				<Column noGutter>
					<span style="font-family: monospace">
						{dict['status'] === 'RUNNING'
							? Utils.getTimeElapsed(dict['created_at'], new Date(), true)
							: dict['finished_at']
							  ? Utils.getTimeElapsed(dict['created_at'], dict['finished_at'])
							  : '—'}
					</span>
				</Column>
			</Column>
		{/if}
		{#if $currentUser?.role === 'admin' && dict['github_pr_url']}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>Granite.Build</FormLabel>
				</Column>
				<Column noGutter>
					<OutboundLink href={dict['github_pr_url']}>Open Dashboard</OutboundLink>
				</Column>
			</Column>
		{/if}
	</Row>
</Grid>

<style type="text/css">
	:global(.dict-item > .bx--inline-loading) {
		min-height: 20px;
	}
</style>
