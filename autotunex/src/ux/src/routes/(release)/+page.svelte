<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { Column, ContentSwitcher, Grid, Loading, Row, Switch } from 'carbon-components-svelte';
	import { Settings as SettingsIcon, ModelTuned, UserMultiple } from 'carbon-icons-svelte';
	import {
		display_conversation,
		forceUpdate,
		homeView,
		HOME_VIEWS,
		isAuthenticated,
		currentUser,
		showLoader
	} from '$lib/store';
	import Settings from '$lib/components/views/Settings.svelte';
	import Tunings from '$lib/components/views/Tunings.svelte';
	import Start from '$lib/components/views/Start.svelte';
	import ChatBox from '$lib/components/ChatBox.svelte';
	import Users from '$lib/components/views/Users.svelte';

	// `homeView` owns both the live value and its localStorage mirror, so losing
	// authentication only has to clear the store.
	$: if (!$isAuthenticated) {
		homeView.set(null);
	}
</script>

<Grid>
	<Row>
		{#if $homeView !== null}
			<ChatBox />
		{/if}
		<Column>
			{#if $homeView !== null}
				<ContentSwitcher selectedIndex={HOME_VIEWS.indexOf($homeView)} style="margin-bottom: 30px;">
					<Switch on:click={() => homeView.set('tunings')}>
						<div style="display: flex; align-items: center;">
							<ModelTuned style="margin-right: 0.5rem;" />
							Tunings
						</div>
					</Switch>
					<Switch on:click={() => homeView.set('settings')}>
						<div style="display: flex; align-items: center;">
							<SettingsIcon style="margin-right: 0.5rem;" />
							Settings
						</div>
					</Switch>
					{#if $currentUser?.role === 'admin'}
						<Switch on:click={() => homeView.set('user')}>
							<div style="display: flex; align-items: center;">
								<UserMultiple style="margin-right: 0.5rem;" />
								Users
							</div>
						</Switch>
					{/if}
				</ContentSwitcher>
			{/if}
			<Loading style="z-index: 10000;" active={$showLoader} />
			{#if $homeView === 'tunings'}
				{#key $forceUpdate}
					<Tunings />
				{/key}
			{:else if $homeView === 'settings'}
				<Settings />
			{:else if $homeView === 'user'}
				<Users />
			{:else}
				<Start bind:view={$homeView} />
			{/if}
		</Column>
	</Row>
</Grid>
