<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { API } from '$lib/api';
	import { Utils } from '$lib/utils';

	function displayListValue(val: unknown): string {
		if (Array.isArray(val)) return val.length > 0 ? val.join(', ') : 'N/A';
		if (val === null || val === undefined || val === '') return 'N/A';
		return String(val);
	}
	import {
		Column,
		FormLabel,
		Grid,
		InlineNotification,
		ProgressBar,
		Row,
		Tooltip,
		NumberInput,
		TextInput,
		Select,
		SelectItem,
		Button
	} from 'carbon-components-svelte';
	import { Edit, View } from 'carbon-icons-svelte';
	import type { Configuration } from '$lib/app-types';
	import { createEventDispatcher } from 'svelte';
	import { configurations, updateConfig } from '$lib/app';
	import { currentUser } from '$lib/store';

	const dispatch = createEventDispatcher();
	const api = new API();

	export let config_id: string;
	export let configuration: Configuration | null = null;
	export let editable: boolean = false;
	export let showEditButton: boolean = true;
	export let isStale: boolean = false;

	let isEditMode: boolean = false;

	let isLoading = false;
	let generalConfigs = [
		'num_gpus_per_trial',
		'num_cpus_per_worker',
		'num_train_epochs',
		'hpo_num_epochs',
		'precision',
		'num_samples',
		'hpo_dataset_percentage'
	];

	const fetchConfig = async () => {
		isLoading = true;
		if (!$configurations) {
			configurations.set([]);
		}
		let config = $configurations.find((config) => config.id === config_id)!;
		if (!config || (config && !config?.config_data)) {
			config = await api.getConfiguration(config_id);
			if (config) {
				updateConfig(config);
			}
		}
		dispatch('configLoaded', { configuration: config });
		configuration = config;
		isLoading = false;
	};

	$: if (
		config_id &&
		(!configuration || !configuration.config_data || configuration.id !== config_id)
	) {
		fetchConfig();
	}

	let non_included_keys = [
		'description',
		'tuner_name',
		'resource_name',
		'seed',
		'reward_function_name',
		'reward_function_path',
		'reward_model_path'
	];

	$: tokenizerEntries = Object.entries(
		((configuration?.config_data as any)?.tokenizer_config ?? {}) as Record<string, any>
	) as [string, any][];
	$: hasTokenizerConfig = tokenizerEntries.length > 0;

	function setTokenizerField(key: string, raw: unknown) {
		const section = (configuration as any)?.config_data?.tokenizer_config;
		if (!section?.[key]) return;
		section[key].default = raw;
		configuration = configuration;
	}

	function commitTokenizerField(key: string) {
		const section = (configuration as any)?.config_data?.tokenizer_config;
		if (!section?.[key]) return;
		if (section[key].type === 'list') {
			section[key].default = Utils.parseCommaList(section[key].default);
			configuration = configuration;
		}
	}
</script>

{#if !isLoading && configuration}
	{#if isStale}
		<InlineNotification
			kind="warning"
			title="Outdated snapshot"
			subtitle="This configuration has been updated since this job was created. You are viewing the original version used for this run."
			hideCloseButton
			lowContrast
		/>
	{/if}
	<Grid noGutter>
		{@const isRlConfig = configuration.rl_tuner_type && configuration.rl_tuner_type !== 'none'}
		{@const tuner = isRlConfig
			? configuration.config_data?.tuners_rl_config?.[configuration.rl_tuner_type]
			: configuration.config_data?.tuners_config?.[configuration.tuner_type]}
		{#if $currentUser?.role === 'admin'}
			{#if editable && showEditButton}
				<div style="padding: 1rem; display: flex; justify-content: flex-end; align-items: center;">
					{#if isEditMode}
						<Button size="small" kind="secondary" icon={View} on:click={() => (isEditMode = false)}>
							View Mode
						</Button>
					{:else}
						<Button size="small" kind="primary" icon={Edit} on:click={() => (isEditMode = true)}>
							Edit Configuration
						</Button>
					{/if}
				</div>
			{/if}
		{/if}
		<div style="padding-left: 1rem; padding-right: 1rem">
			<Row>
				<Column><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem">Overview</h5></Column>
			</Row>
			<Row>
				<Column md={2} padding>
					<Column style="display:flex">
						<FormLabel>Tuner type</FormLabel>
						<Tooltip>{tuner.description}</Tooltip>
					</Column>
					<Column>
						<span style="font-family: monospace;">
							{Utils.toUpperCase(tuner.title)}
						</span>
					</Column>
				</Column>
				{#each generalConfigs.filter((item) => !non_included_keys.includes(item)) as key}
					{@const trainingConfigItem = configuration.config_data.training_config[key]}
					{@const tuneConfigItem = configuration.config_data.tune_config[key]}
					{@const item = trainingConfigItem || tuneConfigItem}
					{#if item}
						<Column md={2} padding>
							<Column style="display:flex">
								<FormLabel>
									{Utils.toUpperCase(key)}
								</FormLabel>
								<Tooltip>
									{item.description}
								</Tooltip>
							</Column>
							<Column>
								{#if editable && isEditMode}
									{#if trainingConfigItem}
										{#if item.type === 'int' || item.type === 'float'}
											<NumberInput
												size="sm"
												hideSteppers
												bind:value={configuration.config_data.training_config[key].default}
												min={item.min_val}
												max={item.max_val}
												step={item.type === 'float' ? 0.01 : 1}
											/>
										{:else if item.type === 'str' && item.values && item.values.length > 0}
											<Select
												size="sm"
												bind:selected={configuration.config_data.training_config[key].default}
											>
												{#each item.values as value}
													<SelectItem {value} text={value} />
												{/each}
											</Select>
										{:else}
											<TextInput
												size="sm"
												bind:value={configuration.config_data.training_config[key].default}
											/>
										{/if}
									{:else if item.type === 'int' || item.type === 'float'}
										<NumberInput
											size="sm"
											hideSteppers
											bind:value={configuration.config_data.tune_config[key].default}
											min={item.min_val}
											max={item.max_val}
											step={item.type === 'float' ? 0.01 : 1}
										/>
									{:else if item.type === 'str' && item.values && item.values.length > 0}
										<Select
											size="sm"
											bind:selected={configuration.config_data.tune_config[key].default}
										>
											{#each item.values as value}
												<SelectItem {value} text={value} />
											{/each}
										</Select>
									{:else}
										<TextInput
											size="sm"
											bind:value={configuration.config_data.tune_config[key].default}
										/>
									{/if}
								{:else}
									<span style="font-family: monospace;">
										{item.default}
									</span>
								{/if}
							</Column>
						</Column>
					{/if}
				{/each}
			</Row>
			<Row>
				<Column
					><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem">
						Training config
					</h5></Column
				>
			</Row>
			<Row>
				{#each Object.entries(configuration['config_data']['training_config']) as [key, value]}
					{#if value.type !== 'bool'}
						{#if !non_included_keys.includes(key) && !generalConfigs.includes(key)}
							<Column md={2} padding>
								<Column style="display: flex">
									<FormLabel>{Utils.toUpperCase(key)}</FormLabel>
									<Tooltip>{value.description}</Tooltip>
								</Column>
								<Column>
									{#if editable && isEditMode}
										{#if value.type === 'int' || value.type === 'float'}
											<NumberInput
												size="sm"
												hideSteppers
												bind:value={configuration.config_data.training_config[key].default}
												min={value.min_val}
												max={value.max_val}
												step={value.type === 'float' ? 0.01 : 1}
											/>
										{:else if value.type === 'str' && value.values && value.values.length > 0}
											<Select
												size="sm"
												bind:selected={configuration.config_data.training_config[key].default}
											>
												{#each value.values as val}
													<SelectItem value={val} text={val} />
												{/each}
											</Select>
										{:else}
											<TextInput
												size="sm"
												bind:value={configuration.config_data.training_config[key].default}
											/>
										{/if}
									{:else}
										<span style="font-family: monospace;">{value.default}</span>
									{/if}
								</Column>
							</Column>
						{/if}
					{/if}
				{/each}
			</Row>
		</div>
		{#if isRlConfig && configuration['config_data']['training_rl_config']}
			<div style="padding-left: 1rem; padding-right: 1rem">
				<Row>
					<Column
						><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem">
							Training RL config
						</h5></Column
					>
				</Row>
				<Row>
					{#each Object.entries(configuration['config_data']['training_rl_config']) as [key, value]}
						{@const hasType = 'type' in value}
						{@const hasValues = 'values' in value && value.values && Array.isArray(value.values)}
						{#if hasType && value.type !== 'bool'}
							{#if !non_included_keys.includes(key)}
								<Column md={2} padding>
									<Column style="display: flex">
										<FormLabel>{Utils.toUpperCase(key)}</FormLabel>
										<Tooltip>{value.description}</Tooltip>
									</Column>
									<Column>
										{#if editable && isEditMode}
											{#if value.type === 'int' || value.type === 'float'}
												<NumberInput
													size="sm"
													hideSteppers
													bind:value={configuration.config_data.training_rl_config[key].default}
													min={value.min_val ?? undefined}
													max={value.max_val ?? undefined}
													step={value.type === 'float' ? 0.01 : 1}
												/>
											{:else if value.type === 'str' && hasValues && value.values && value.values.length > 0}
												<Select
													size="sm"
													bind:selected={configuration.config_data.training_rl_config[key].default}
												>
													{#each value.values as val}
														<SelectItem value={String(val)} text={String(val)} />
													{/each}
												</Select>
											{:else}
												<TextInput
													size="sm"
													bind:value={configuration.config_data.training_rl_config[key].default}
												/>
											{/if}
										{:else}
											<span style="font-family: monospace;">{value.default || 'N/A'}</span>
										{/if}
									</Column>
								</Column>
							{/if}
						{/if}
					{/each}
				</Row>
			</div>
		{/if}
		<div style="padding-left: 1rem; padding-right: 1rem">
			<Row>
				<Column
					><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem">Tune config</h5></Column
				>
			</Row>
			<Row>
				{#each Object.entries(configuration['config_data']['tune_config']) as [key, value]}
					{#if !non_included_keys.includes(key) && !generalConfigs.includes(key)}
						<Column md={2} padding>
							<Column style="display: flex">
								<FormLabel>{Utils.toUpperCase(key)}</FormLabel>
								<Tooltip>{value.description}</Tooltip>
							</Column>
							<Column>
								{#if editable && isEditMode}
									{#if value.type === 'int' || value.type === 'float'}
										<NumberInput
											size="sm"
											hideSteppers
											bind:value={configuration.config_data.tune_config[key].default}
											min={value.min_val}
											max={value.max_val}
											step={value.type === 'float' ? 0.01 : 1}
										/>
									{:else if value.type === 'str' && value.values && value.values.length > 0}
										<Select
											size="sm"
											bind:selected={configuration.config_data.tune_config[key].default}
										>
											{#each value.values as val}
												<SelectItem value={val} text={val} />
											{/each}
										</Select>
									{:else}
										<TextInput
											size="sm"
											bind:value={configuration.config_data.tune_config[key].default}
										/>
									{/if}
								{:else}
									<span style="font-family: monospace;">{value.default}</span>
								{/if}
							</Column>
						</Column>
					{/if}
				{/each}
			</Row>
		</div>
		{#if hasTokenizerConfig}
			<div style="padding-left: 1rem; padding-right: 1rem">
				<Row>
					<Column
						><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem">
							Tokenizer config
						</h5></Column
					>
				</Row>
				<Row>
					{#each tokenizerEntries as [key, field]}
						{#if Utils.isObject(field)}
							<Column md={2} padding>
								<Column style="display: flex">
									<FormLabel>{Utils.toUpperCase(key)}</FormLabel>
									<Tooltip>{field.description}</Tooltip>
								</Column>
								<Column>
									{#if editable && isEditMode}
										<TextInput
											size="sm"
											placeholder={field.type === 'list' ? 'tok_a, tok_b, tok_c' : ''}
											value={Array.isArray(field.default)
												? field.default.join(', ')
												: field.default ?? ''}
											on:input={(e) => setTokenizerField(key, e.detail)}
											on:blur={() => commitTokenizerField(key)}
										/>
									{:else if field.type === 'list'}
										<span style="font-family: monospace;">{displayListValue(field.default)}</span>
									{:else}
										<span style="font-family: monospace;">{field.default ?? 'N/A'}</span>
									{/if}
								</Column>
							</Column>
						{/if}
					{/each}
				</Row>
			</div>
		{/if}
		<div style="padding-left: 1rem; padding-right: 1rem">
			<Row>
				<Column
					><h5 style="background-color: #f4f4f4; padding:0.25rem 0.5rem; margin-bottom:0.5rem">
						{tuner.title} Configuration
					</h5></Column
				>
			</Row>
			<Row>
				{@const tunerConfig = isRlConfig
					? configuration['config_data']['tuners_rl_config']?.[configuration.rl_tuner_type]
					: configuration['config_data']['tuners_config'][configuration.tuner_type]}
				{@const hyperparams = tunerConfig?.['hyperparams']}
				{#each Object.entries(hyperparams || {}) as [key, value]}
					{#if !Utils.isObject(value)}
						{#if !non_included_keys.includes(key)}
							<Column padding>
								<Column style="display:flex">
									<FormLabel
										><span style="font-size:14px; font-weight:600">{Utils.toUpperCase(key)}</span
										></FormLabel
									>
									<Tooltip>{value.description}</Tooltip>
								</Column>
								<Column>
									<span style="font-family: monospace;">{value}</span>
								</Column>
							</Column>
						{/if}
					{:else}
						<Column sm={4} style="margin-bottom:1rem;">
							<div style="background-color: #f1f1f140; padding: 0.5rem 0;">
								<Column>
									<FormLabel
										><span style="font-size:14px; font-weight:600"
											>{Utils.toUpperCase(value.description)}</span
										></FormLabel
									>
								</Column>
								<Column>
									<Row>
										{#if value.type === 'str' && value.values?.length === 1}
											<Column>
												{#if editable && isEditMode}
													{#if isRlConfig}
														<TextInput
															size="sm"
															bind:value={configuration.config_data.tuners_rl_config[
																configuration.rl_tuner_type
															].hyperparams[key].default}
														/>
													{:else}
														<TextInput
															size="sm"
															bind:value={configuration.config_data.tuners_config[
																configuration.tuner_type
															].hyperparams[key].default}
														/>
													{/if}
												{:else}
													<span style="font-family: monospace;">{value.default}</span>
												{/if}
											</Column>
										{:else if editable && isEditMode && value.strategy === 'choice'}
											<Column>
												<FormLabel style="display:block">Strategy</FormLabel>
												<span style="font-family: monospace;">{value.strategy}</span>
											</Column>
											<Column>
												<FormLabel style="display:block">Default</FormLabel>
												{#if isRlConfig}
													<Select
														size="sm"
														bind:selected={configuration.config_data.tuners_rl_config[
															configuration.rl_tuner_type
														].hyperparams[key].default}
													>
														{#each value.values as val}
															<SelectItem value={val} text={val} />
														{/each}
													</Select>
												{:else}
													<Select
														size="sm"
														bind:selected={configuration.config_data.tuners_config[
															configuration.tuner_type
														].hyperparams[key].default}
													>
														{#each value.values as val}
															<SelectItem value={val} text={val} />
														{/each}
													</Select>
												{/if}
											</Column>
											<Column sm={2}>
												<FormLabel style="display:block">Values</FormLabel>
												<span style="font-family: monospace;"
													>{value.values?.length > 1
														? value.values?.join(', ')
														: value.values}</span
												>
											</Column>
										{:else if editable && isEditMode && value.strategy === 'uniform'}
											<Column>
												<FormLabel style="display:block">Strategy</FormLabel>
												<span style="font-family: monospace;">{value.strategy}</span>
											</Column>
											<Column>
												<FormLabel style="display:block">Default</FormLabel>
												{#if isRlConfig}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_rl_config[
															configuration.rl_tuner_type
														].hyperparams[key].default}
														min={value.min_val}
														max={value.max_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{:else}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_config[
															configuration.tuner_type
														].hyperparams[key].default}
														min={value.min_val}
														max={value.max_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{/if}
											</Column>
											<Column sm={1}>
												<FormLabel style="display:block">Min value</FormLabel>
												{#if isRlConfig}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_rl_config[
															configuration.rl_tuner_type
														].hyperparams[key].min_val}
														max={value.max_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{:else}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_config[
															configuration.tuner_type
														].hyperparams[key].min_val}
														max={value.max_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{/if}
											</Column>
											<Column sm={1}>
												<FormLabel style="display:block">Max value</FormLabel>
												{#if isRlConfig}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_rl_config[
															configuration.rl_tuner_type
														].hyperparams[key].max_val}
														min={value.min_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{:else}
													<NumberInput
														size="sm"
														hideSteppers
														bind:value={configuration.config_data.tuners_config[
															configuration.tuner_type
														].hyperparams[key].max_val}
														min={value.min_val}
														step={value.type === 'float' ? 0.01 : 1}
													/>
												{/if}
											</Column>
										{:else}
											<Column>
												<FormLabel style="display:block">Strategy</FormLabel>
												<span style="font-family: monospace;">{value.strategy}</span>
											</Column>
											<Column>
												<FormLabel style="display:block">Default</FormLabel>
												<span style="font-family: monospace;">{value.default}</span>
											</Column>
											{#if value.strategy === 'uniform'}
												<Column sm={1}>
													<FormLabel style="display:block">Min value</FormLabel>
													<span style="font-family: monospace;">{value.min_val}</span>
												</Column>
												<Column sm={1}>
													<FormLabel style="display:block">Max value</FormLabel>
													<span style="font-family: monospace;">{value.max_val}</span>
												</Column>
											{:else}
												<Column sm={2}>
													<FormLabel style="display:block">Values</FormLabel>
													<span style="font-family: monospace;"
														>{value.values?.length > 1
															? value.values?.join(', ')
															: value.values}</span
													>
												</Column>
											{/if}
										{/if}
									</Row>
								</Column>
							</div>
						</Column>
					{/if}
				{/each}
			</Row>
		</div>
	</Grid>
{:else}
	<ProgressBar size="sm" helperText="Loading configuration details..." />
{/if}
