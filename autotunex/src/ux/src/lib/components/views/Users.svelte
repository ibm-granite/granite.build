<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { onMount } from 'svelte';
	import Table from '../Table.svelte';

	import { API } from '$lib/api';
	import { Button, DataTableSkeleton, Modal, Tag } from 'carbon-components-svelte';
	import { currentUser, capabilities } from '$lib/store';
	import { notifications } from '$lib/app';
	import type { User } from '$lib/app-types';

	const api = new API();

	let users: User[] = [];
	let loaded = false;

	// Role-change confirm modal state.
	let modalOpen = false;
	let roleSaving = false;
	let pending: { id: string; email: string; from: string; to: 'admin' | 'user' } | null = null;

	onMount(async () => {
		loaded = false;
		if (!capabilities.users) {
			loaded = true;
			return;
		}
		users = await api.getUsers();
		loaded = true;
	});

	const userHeaders = [
		{ key: 'email', value: 'Email' },
		{ key: 'role', value: 'Role' },
		{ key: 'id', value: 'User ID' },
		{
			key: 'created_at',
			value: 'Created on',
			display: (date: Date) => new Date(date).toLocaleString()
		},
		{
			key: 'last_login_at',
			value: 'Last login on',
			// Null when the backend holds no record of a login for this user at all.
			// An em dash, not a fabricated date: `new Date(null)` is the 1970 epoch,
			// which would read as a real (and very wrong) login.
			display: (date: Date | null) => (date ? new Date(date).toLocaleString() : '—')
		},
		{ key: 'action', empty: true }
	];

	const openRoleModal = (row: any) => {
		pending = {
			id: row.id,
			email: row.email,
			from: row.role,
			to: row.role === 'admin' ? 'user' : 'admin'
		};
		modalOpen = true;
	};

	const confirmRoleChange = async () => {
		if (!pending) return;
		roleSaving = true;
		try {
			const updated = await api.setUserRole(pending.id, pending.to);
			users = users.map((u) => (u.id === updated.id ? { ...u, role: updated.role } : u));
			notifications.set({
				show: true,
				kind: 'success',
				title: 'Role updated',
				subtitle: `${updated.email} is now ${updated.role}.`,
				timeout: 5000
			});
		} catch (err: any) {
			notifications.set({
				show: true,
				kind: 'error',
				title: 'Could not change role',
				subtitle: err?.detail || err?.title || 'Request failed.',
				timeout: 7000
			});
		} finally {
			roleSaving = false;
			modalOpen = false;
			pending = null;
		}
	};
</script>

{#if !capabilities.users}
	<div style="padding: 2rem;">
		<Tag type="cool-gray">Coming soon</Tag>
		<p>User management is not available yet.</p>
	</div>
{:else if loaded}
	<Table
		title="Users"
		entity="User"
		entities="Users"
		description="Manage users and their roles."
		headers={userHeaders}
		rows={users}
		expandable={false}
		selectable={false}
		batchSelection={false}
		showActionButton={false}
	>
		<svelte:fragment slot="cell" let:cell let:row>
			{#if cell.key === 'action'}
				{#if row.email === $currentUser?.email}
					<Tag type="cool-gray">You</Tag>
				{:else}
					<Button kind="ghost" size="small" on:click={() => openRoleModal(row)}>
						{row.role === 'admin' ? 'Make user' : 'Make admin'}
					</Button>
					{#if capabilities.impersonation}
						<Button
							kind="ghost"
							size="small"
							disabled={row.id === $currentUser?.user_id}
							on:click={async () => {
								await api.assumeUser(row.id);
								window.location.reload();
							}}
						>
							Assume
						</Button>
					{/if}
				{/if}
			{:else}
				{cell.display ? cell.display(cell.value, row) : cell.value}
			{/if}
		</svelte:fragment>
	</Table>

	<Modal
		bind:open={modalOpen}
		modalHeading="Change role"
		primaryButtonText="Change role"
		secondaryButtonText="Cancel"
		primaryButtonDisabled={roleSaving}
		on:click:button--secondary={() => (modalOpen = false)}
		on:submit={confirmRoleChange}
	>
		{#if pending}
			<p>
				Change <strong>{pending.email}</strong> from <strong>{pending.from}</strong> to
				<strong>{pending.to}</strong>?
			</p>
		{/if}
	</Modal>
{:else}
	<DataTableSkeleton />
{/if}
