<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Modal } from 'carbon-components-svelte';

	export let open = false;
	export let entity;
	export let submitBtnDisable = false;
	export let primaryButtonText = 'OK';
	export let secondaryButtonText = 'Cancel';
	export let passiveCreateModal = false;
</script>

<Modal
	passiveModal={passiveCreateModal}
	bind:open
	preventCloseOnClickOutside
	modalHeading={`Create new ${entity}`}
	{primaryButtonText}
	{secondaryButtonText}
	primaryButtonDisabled={submitBtnDisable}
	on:click:button--secondary={() => (open = false)}
	on:open
	on:close
	on:submit
	size="lg"
>
	{#if open}
		<slot />
	{/if}
</Modal>
