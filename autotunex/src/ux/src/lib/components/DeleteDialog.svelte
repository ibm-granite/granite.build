<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script>
	import { Modal } from 'carbon-components-svelte';

	export let open = false;
	export let entity;
	export let primaryButtonDisabled = false;
</script>

<Modal
	danger
	bind:open
	preventCloseOnClickOutside
	modalHeading={`Delete ${entity}`}
	primaryButtonText="Delete"
	secondaryButtonText="Cancel"
	{primaryButtonDisabled}
	on:click:button--secondary={() => (open = false)}
	on:open
	on:close
	on:submit
>
	<slot />
</Modal>
