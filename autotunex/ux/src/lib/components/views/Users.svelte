<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { onMount } from 'svelte';
	import Table from '../Table.svelte';

	import { API } from '$lib/api';
	import {
		Button,
		DataTableSkeleton,
		ProgressBar,
		Tab,
		TabContent,
		Tabs,
		Tag
	} from 'carbon-components-svelte';
	import ShowStatus from '../ShowStatus.svelte';
	import { currentUser } from '$lib/store';
	import type { UserData, User } from '$lib/app-types';

	const api = new API();

	let users: User[];
	let userData: UserData;
	let loaded: boolean = false;
	let isUserDataLoading: boolean = false;

	onMount(async () => {
		loaded = false;
		// users = (await api.getUsers())?.filter(item=>item.email !== Cookies.get("email"));
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
			key: 'updated_at',
			value: 'Last login on',
			display: (date: Date) => new Date(date).toLocaleString()
		},
		{ key: 'action', empty: true }
	];
</script>

{#if loaded}
	<Table
		title="Users"
		entity="User"
		entities="Users"
		description="Shows all users."
		headers={userHeaders}
		rows={users}
		expandable={false}
		showActionButton={false}
		on:view={async (e) => {
			if (e.detail.row.id) {
				isUserDataLoading = true;
				userData = await api.getUserData(e.detail.row.id);
				if (userData.jobs.length > 0) {
					userData.publishedModels = await api.getModelsByUserId(e.detail.row.id);
				} else {
					userData.publishedModels = [];
				}
				isUserDataLoading = false;
			}
		}}
	>
		<svelte:fragment slot="cell" let:cell let:row>
			{#if cell.key === 'action'}
				<Button
					kind="ghost"
					on:click={async () => {
						let assume_user = await api.assumeUser(row.id);
						if (assume_user.detail.success) {
							localStorage.removeItem('view');
							window.location.reload();
						}
					}}
					disabled={row.email === $currentUser?.email}>Assume</Button
				>
			{:else}
				{cell.display ? cell.display(cell.value, row) : cell.value}
			{/if}
		</svelte:fragment>
		<svelte:fragment slot="view" let:selectedRows>
			{#if !isUserDataLoading}
				<Tabs>
					<Tab label="Jobs" />
					<Tab label="Configs" />
					<Tab label="Datasets" />
					<Tab label="Published" />
					<svelte:fragment slot="content">
						<TabContent>
							<Table
								rows={userData.jobs}
								headers={[
									{ key: 'experiment_name', value: 'Experiment name' },
									{ key: 'status', value: 'Status' },
									{ key: 'model', value: 'Model' },
									{ key: 'config_name', value: 'Configuration' },
									{ key: 'dataset', value: 'Dataset' },
									{
										key: 'created_at',
										value: 'Created on',
										display: (date) => new Date(date).toLocaleString()
									}
								]}
								batchSelection={false}
								selectable={false}
								expandable={false}
								showActionButton={false}
							>
								<svelte:fragment slot="cell" let:cell let:row>
									{#if cell.key === 'status'}
										<ShowStatus status={cell.value} />
									{:else}
										{cell.display ? cell.display(cell.value, row) : cell.value}
									{/if}
								</svelte:fragment>
							</Table>
						</TabContent>
						<TabContent>
							<Table
								rows={userData.configs}
								headers={[
									{ key: 'name', value: 'Name' },
									{ key: 'tuner_type', value: 'Tuner type' },
									{ key: 'associated_jobs', value: 'Associated jobs' },
									{
										key: 'created_at',
										value: 'Created on',
										display: (date) => new Date(date).toLocaleString()
									}
								]}
								batchSelection={false}
								selectable={false}
								expandable={false}
								showActionButton={false}
							>
								<svelte:fragment slot="cell" let:cell let:row>
									{#if cell.key === 'associated_jobs'}
										{#each cell.value as tuning}
											<Tag>{tuning.experiment_name}</Tag>
										{/each}
									{:else}
										{cell.display ? cell.display(cell.value, row) : cell.value}
									{/if}
								</svelte:fragment>
							</Table>
						</TabContent>
						<TabContent>
							<Table
								rows={userData.datasets}
								headers={[
									{ key: 'name', value: 'Name' },
									{ key: 'train_records', value: 'Training samples' },
									{ key: 'validation_records', value: 'Validation samples' },
									{ key: 'associated_jobs', value: 'Associated jobs' },
									{
										key: 'created_at',
										value: 'Created on',
										display: (date) => new Date(date).toLocaleString()
									}
								]}
								sortKey="created_at"
								batchSelection={false}
								selectable={false}
								expandable={false}
								showActionButton={false}
							>
								<svelte:fragment slot="cell" let:cell let:row>
									{#if cell.key === 'associated_jobs'}
										{#each cell.value as tuning}
											<Tag>{tuning.experiment_name}</Tag>
										{/each}
									{:else}
										{cell.display ? cell.display(cell.value, row) : cell.value}
									{/if}
								</svelte:fragment>
							</Table>
						</TabContent>
						<TabContent>
							<Table
								rows={userData.publishedModels.map((model) => {
									return { id: model.model_id, ...model };
								})}
								headers={[
									{ key: 'model_label', value: 'Tuning' },
									{ key: 'model_id', value: 'Checkpoint' },
									{ key: 'base_model', value: 'Model' },
									{
										key: 'open',
										value: 'Visibility',
										display: (item) => (item === true ? 'IBM Public' : 'Private')
									},
									{ key: 'created', value: 'Published on' }
								]}
								batchSelection={false}
								selectable={false}
								expandable={false}
								showActionButton={false}
							>
								<svelte:fragment slot="cell" let:cell let:row>
									{#if cell.key === 'created'}
										{#if row.files}
											{new Date(row.files[0]?.created)?.toLocaleString()}
										{:else}
											{new Date(row?.updated_at)?.toLocaleString()}
										{/if}
									{:else}
										{cell.display ? cell.display(cell.value, row) : cell.value}
									{/if}
								</svelte:fragment>
							</Table>
						</TabContent>
					</svelte:fragment>
				</Tabs>
			{:else}
				<ProgressBar size="sm" helperText="Loading user details..." />
			{/if}
		</svelte:fragment>
	</Table>
{:else}
	<DataTableSkeleton />
{/if}
