<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		Grid,
		Row,
		Column,
		FormLabel,
		OutboundLink,
		Button,
		InlineLoading,
		Tag,
		Tile
	} from 'carbon-components-svelte';
	import { Utils } from '$lib/utils';
	import { RITS_UI_URL, buildRitsUrl } from '$lib/constants';
	import ShowStatus from '../ShowStatus.svelte';
	import { currentUser } from '$lib/store';
	import { showLoader } from '$lib/store';
	import { API } from '$lib/api';
	import { createEventDispatcher } from 'svelte';
	import { ModelSource } from '$lib/app-types';

	export let dict;
	export let rits;
	const api = new API();

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
		'updated_at'
	];
	const dispatch = createEventDispatcher();
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
							<Tag
								style="margin:0"
								type={dict[key] === ModelSource.DMF
									? 'blue'
									: dict[key] === ModelSource.CustomPath
									  ? 'purple'
									  : 'cyan'}>{dict[key]}</Tag
							>
						{:else}
							{dict[key]}
						{/if}
					</span>
				</Column>
			</Column>
		{/each}
		{#if dict['created_at'] && dict['updated_at']}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>Total time</FormLabel>
				</Column>
				<Column noGutter>
					<span style="font-family: monospace">
						{Utils.getTimeElapsed(
							dict['created_at'],
							dict['updated_at'],
							dict['status'] === 'RUNNING'
						)}
					</span>
				</Column>
			</Column>
		{/if}
		{#if $currentUser?.role === 'admin' && dict['github_pr_url']}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>LLM.Build PR</FormLabel>
				</Column>
				<Column noGutter>
					<OutboundLink href={dict['github_pr_url']}>Open GitHub</OutboundLink>
				</Column>
			</Column>
		{/if}
		{#if $currentUser?.role === 'admin' && dict['dmf_url']}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>Lineage</FormLabel>
				</Column>
				<Column noGutter>
					<OutboundLink href={dict['dmf_url']}>Open DMF</OutboundLink>
				</Column>
			</Column>
		{/if}
		{#if dict?.status === 'COMPLETED' && dict?.dmf_url}
			<Column sm={1} padding>
				<!-- {#if rits && rits?.status === 'COMPLETED' && rits?.rits_url} -->
				<Column noGutter>
					<FormLabel>Hosted inference</FormLabel>
				</Column>
				<!-- {/if} -->
				<Column noGutter>
					{#if rits === null || rits?.status === 'TERMINATED'}
						<Button
							kind="primary"
							on:click={async () => {
								try {
									showLoader.set(true);
									await api.pushToRits(dict?.id);
									dispatch('deploy', dict.id);
								} catch (error) {
									console.error(await error);
								} finally {
									showLoader.set(false);
								}
							}}
							size="small">Deploy now</Button
						>
					{:else if rits?.status === 'PENDING'}
						<InlineLoading status="active" description={'Deploying to RITS...'} />
					{:else if rits?.status === 'RUNNING'}
						<InlineLoading status="active" description={'Deploying to RITS...'} />
					{:else if rits?.status === 'COMPLETED'}
						{#if RITS_UI_URL}
							<OutboundLink href={buildRitsUrl(Utils.getNameFromUrl(rits?.rits_url))}
								>Open RITS</OutboundLink
							>
						{/if}
						<InlineLoading
							status="finished"
							description={`Deployed on ${new Date(rits?.updated_at).toLocaleString()}`}
						/>
					{:else}
						<InlineLoading status="active" description={'Fetching status in RITS...'} />
					{/if}
				</Column>
			</Column>
		{/if}
		{#if rits && rits?.status === 'COMPLETED' && rits?.rits_url}
			<Column sm={1} padding>
				<Column noGutter>
					<FormLabel>Hosted model name</FormLabel>
				</Column>
				<Column noGutter style="display:flex;">
					<a
						href="#"
						style="font-family: monospace; cursor:pointer; text-decoration: none; color: #161616"
						on:click={async () => {
							try {
								await navigator.clipboard.writeText(rits?.rits_url);
							} catch (e) {
								console.error(e);
							}
						}}>{Utils.getNameFromUrl(rits?.rits_url)}</a
					>
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
