<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { InlineLoading } from 'carbon-components-svelte';
	export let status;
</script>

{#if status === 'PENDING'}
	<InlineLoading status="active" description="Pending..." />
{:else if status === 'PAUSED'}
	<InlineLoading status="active" description="Paused" />
{:else if status === 'TERMINATED'}
	<InlineLoading status="error" description="Terminated" />
{:else if status === 'ERROR'}
	<InlineLoading status="error" description="Error" />
{:else if status === 'COMPLETED'}
	<InlineLoading status="finished" description="Completed" />
{:else}
	<InlineLoading status="active" description="Running..." />
{/if}
