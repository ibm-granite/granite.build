<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Modal } from 'carbon-components-svelte';

	export let open = false;
	export let entity;
</script>

<Modal
	bind:open
	passiveModal
	preventCloseOnClickOutside
	modalHeading={`View ${entity}`}
	on:click:button--secondary={() => (open = false)}
	on:open
	on:close={() => (open = false)}
	on:submit
	size="lg"
>
	{#if open}
		<slot />
	{/if}
</Modal>
