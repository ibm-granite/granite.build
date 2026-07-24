<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import './styles.scss';
	import {
		Header,
		SkipToContent,
		Content,
		HeaderUtilities,
		HeaderAction,
		HeaderPanelLinks,
		HeaderPanelLink,
		HeaderNavItem,
		HeaderNav,
		ToastNotification,
		Loading
	} from 'carbon-components-svelte';
	import { SettingsAdjust } from 'carbon-icons-svelte';
	import { display_conversation, isAuthenticated, currentUser, featureFlags } from '$lib/store';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { API } from '$lib/api';
	import { notifications } from '$lib/app';

	let isSideNavOpen = false;
	let isSettingsOpen = false;
	let authChecked = false;

	let theme;

	$: if (document) {
		document.documentElement.setAttribute('theme', theme);
	}
	const api = new API();
	onMount(async () => {
		theme = localStorage.getItem('theme') ? localStorage.getItem('theme') : 'g10';

		try {
			const authData = await api.me();
			if (authData.authenticated) {
				isAuthenticated.set(true);
				currentUser.set(authData.user);
			} else {
				isAuthenticated.set(false);
				currentUser.set(null);
				localStorage.clear();
				const path = $page.url.pathname;
				if (path !== '/autotune' && path !== '/autotune/') {
					await goto('/autotune');
				}
			}
		} catch {
			isAuthenticated.set(false);
			currentUser.set(null);
		}
		authChecked = true;
	});
</script>

<svelte:head>
	<title>AutoTune</title>
</svelte:head>

<Header
	href="/autotune"
	company="IBM Research"
	platformName="AutoTuneX"
	on:click={() => {
		localStorage.removeItem('view');
	}}
	bind:isSideNavOpen
>
	<svelte:fragment slot="skip-to-content">
		<SkipToContent />
	</svelte:fragment>
	<HeaderUtilities>
		{#if $isAuthenticated && $currentUser}
			<HeaderNav>
				<HeaderNavItem href="#" text={$currentUser.email} />
			</HeaderNav>
		{/if}
		<HeaderAction aria-label="Settings" icon={SettingsAdjust} bind:isOpen={isSettingsOpen}>
			<HeaderPanelLinks>
				<HeaderPanelLink
					on:click={() => {
						localStorage.removeItem('view');
						goto('/autotune');
					}}>About</HeaderPanelLink
				>
				<HeaderPanelLink
					on:click={() =>
						featureFlags.update((flags) => ({
							...flags,
							quickCreateTuning: !flags.quickCreateTuning
						}))}
					>{$featureFlags.quickCreateTuning
						? 'Hide quick-create tuning'
						: 'Show quick-create tuning'}</HeaderPanelLink
				>
				{#if $currentUser?.role === 'admin'}
					<HeaderPanelLink
						on:click={() =>
							display_conversation.update((value) => {
								localStorage.setItem('showChatWindow', `${!value}`);
								return !value;
							})}>{$display_conversation ? 'Hide chat window' : 'Show chat window'}</HeaderPanelLink
					>
					<HeaderPanelLink
						on:click={() =>
							featureFlags.update((flags) => ({
								...flags,
								customPathModelSource: !flags.customPathModelSource
							}))}
						>{$featureFlags.customPathModelSource
							? 'Hide custom model path'
							: 'Show custom model path'}</HeaderPanelLink
					>
				{/if}
				{#if $currentUser?.impersonating}
					<HeaderPanelLink
						on:click={async () => {
							let result = await api.unassumeUser();
							if (result.detail.success) {
								localStorage.removeItem('view');
								window.location.reload();
							}
						}}>{'Exit Impersonation'}</HeaderPanelLink
					>
				{/if}
				<HeaderPanelLink
					on:click={async () => {
						await api.logout();
						isAuthenticated.set(false);
						currentUser.set(null);
						localStorage.clear();
						window.location.reload();
					}}>{$isAuthenticated ? 'Logout' : 'Login'}</HeaderPanelLink
				>
			</HeaderPanelLinks>
		</HeaderAction>
	</HeaderUtilities>
</Header>
{#if $notifications?.show}
	<ToastNotification
		timeout={$notifications.timeout}
		kind={$notifications.kind}
		style="position: fixed;  right: 1rem; z-index: 1;"
		title={$notifications.title}
		subtitle={$notifications.subtitle}
		caption={$notifications.caption}
		on:close={() =>
			notifications.update((prev) => {
				return { ...prev, show: false };
			})}
	/>
{/if}
<Content>
	{#if authChecked}
		<slot />
	{:else}
		<Loading />
	{/if}
</Content>
