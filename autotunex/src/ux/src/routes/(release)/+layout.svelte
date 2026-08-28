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
		InlineNotification,
		NotificationActionButton,
		Loading,
		Grid,
		Row,
		Column
	} from 'carbon-components-svelte';
	import { SettingsAdjust } from 'carbon-icons-svelte';
	import {
		display_conversation,
		isAuthenticated,
		currentUser,
		featureFlags,
		authMode,
		appConfig
	} from '$lib/store';
	import { get } from 'svelte/store';
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

		// Fire-and-forget: non-sensitive, non-blocking. Consumers (utils.ts,
		// api.ts) fall back to safe client-side defaults if this hasn't
		// resolved yet or fails.
		api
			.getAppConfig()
			.then((config) => appConfig.set(config))
			.catch(() => {
				/* leave appConfig null — consumers use their own defaults */
			});

		try {
			const authData = await api.me();
			if (authData.authenticated) {
				isAuthenticated.set(true);
				currentUser.set(authData.user ?? null);
			} else {
				isAuthenticated.set(false);
				currentUser.set(null);
				localStorage.clear();
				// A real provider (session mode) wants a login — send the browser to the
				// backend's OIDC entry point. Standalone never reaches this branch.
				if (get(authMode) === 'session') {
					await api.login();
					return;
				}
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
				<HeaderPanelLink
					on:click={async () => {
						if ($isAuthenticated) {
							const data = await api.logout();
							isAuthenticated.set(false);
							currentUser.set(null);
							localStorage.clear();
							if (data?.end_session_endpoint) window.location.href = data.end_session_endpoint;
							else window.location.reload();
						} else {
							await api.login();
						}
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
		style="position: fixed; right: 1rem; z-index: 9000;"
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
	{#if $currentUser?.impersonator}
		<Grid>
			<Row>
				<Column>
					<InlineNotification
						kind="info"
						hideCloseButton
						title="Impersonating"
						subtitle={`Acting as ${$currentUser.email} (as ${$currentUser.impersonator}).`}
					>
						<svelte:fragment slot="actions">
							<NotificationActionButton
								on:click={async () => {
									await api.unassumeUser();
									window.location.reload();
								}}
							>
								Exit impersonation
							</NotificationActionButton>
						</svelte:fragment>
					</InlineNotification>
				</Column>
			</Row>
		</Grid>
	{/if}
	{#if authChecked}
		<slot />
	{:else}
		<Loading />
	{/if}
</Content>
