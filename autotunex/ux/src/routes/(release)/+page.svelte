<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Column, ContentSwitcher, Grid, Loading, Row, Switch } from 'carbon-components-svelte';
	import {
		Settings as SettingsIcon,
		ModelTuned,
		IbmCloudVirtualServerClassic,
		UserMultiple
	} from 'carbon-icons-svelte';
	import {
		display_conversation,
		forceUpdate,
		isAuthenticated,
		currentUser,
		showDmf,
		showLoader
	} from '$lib/store';
	import Settings from '$lib/components/views/Settings.svelte';
	import Tunings from '$lib/components/views/Tunings.svelte';
	import Start from '$lib/components/views/Start.svelte';
	import ChatBox from '$lib/components/ChatBox.svelte';
	import DmfTable from '$lib/components/tables/DmfTable.svelte';
	import Users from '$lib/components/views/Users.svelte';
	import { onMount } from 'svelte';
	export let view;

	let views = ['tunings', 'settings', 'dmf', 'user'];

	$: if (views.includes(view)) {
		localStorage.setItem('view', view);
	}

	$: if (!$showDmf && view === 'dmf') {
		view = 'tunings';
	}

	$: if (!$isAuthenticated) {
		view = null;
	}

	onMount(() => {
		view = localStorage.getItem('view');
	});
</script>

<Grid>
	<Row>
		{#if views.includes(view)}
			<ChatBox />
		{/if}
		<Column>
			{#if views.includes(view)}
				<ContentSwitcher selectedIndex={views.indexOf(view)} style="margin-bottom: 30px;">
					<Switch on:click={() => (view = 'tunings')}>
						<div style="display: flex; align-items: center;">
							<ModelTuned style="margin-right: 0.5rem;" />
							Tunings
						</div>
					</Switch>
					<Switch on:click={() => (view = 'settings')}>
						<div style="display: flex; align-items: center;">
							<SettingsIcon style="margin-right: 0.5rem;" />
							Settings
						</div>
					</Switch>
					{#if $showDmf}
						<Switch on:click={() => (view = 'dmf')}>
							<div style="display: flex; align-items: center;">
								<IbmCloudVirtualServerClassic style="margin-right: 0.5rem;" />
								Published Models
							</div>
						</Switch>
					{/if}
					{#if $currentUser?.role === 'admin'}
						<Switch on:click={() => (view = 'user')}>
							<div style="display: flex; align-items: center;">
								<UserMultiple style="margin-right: 0.5rem;" />
								Users
							</div>
						</Switch>
					{/if}
				</ContentSwitcher>
			{/if}
			<Loading style="z-index: 10000;" active={$showLoader} />
			{#if view === 'tunings'}
				{#key $forceUpdate}
					<Tunings />
				{/key}
			{:else if view === 'settings'}
				<Settings />
			{:else if view === 'dmf'}
				<DmfTable />
			{:else if view === 'user'}
				<Users />
			{:else}
				<Start bind:view />
			{/if}
		</Column>
	</Row>
</Grid>
