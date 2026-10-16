<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import type { NumberInputColumn } from '$lib/app-types';
	import { Utils } from '$lib/utils';
	import { NumberInput, Select, SelectItem } from 'carbon-components-svelte';

	let selectedUnit: 'seconds' | 'minutes' | 'hours' | 'days' = 'hours';
	let displayedTimeBudget: number | null = null; // Allow null for unset values
	let previousUnit: string = 'hours'; // Start with hours to match default selectedUnit
	let initialized = false;

	export let label: string = 'Time Budget';
	export let value: NumberInputColumn;

	// Conversion factors to seconds
	const conversions = {
		seconds: 1,
		minutes: 60,
		hours: 3600,
		days: 86400
	};

	// Define unit-specific constraints
	const unitConstraints = {
		seconds: { min: 60, max: 1209600 },
		minutes: { min: 1, max: 20160 },
		hours: { min: 1, max: 336 },
		days: { min: 1, max: 14 }
	};

	// Convert input to seconds
	function convertToSeconds(value: number, unit: 'seconds' | 'minutes' | 'hours' | 'days'): number {
		return value * conversions[unit];
	}

	// Convert seconds to selected unit
	function convertFromSeconds(
		seconds: number,
		unit: 'seconds' | 'minutes' | 'hours' | 'days'
	): number {
		return seconds / conversions[unit];
	}

	// Get min/max for current unit
	$: currentMin = unitConstraints[selectedUnit].min;
	$: currentMax = unitConstraints[selectedUnit].max;

	// Pick the best unit so the displayed value is within range
	function bestUnit(seconds: number): 'seconds' | 'minutes' | 'hours' | 'days' {
		const days = seconds / 86400;
		if (days >= 1 && days <= 14) return 'days';
		const hours = seconds / 3600;
		if (hours >= 1 && hours <= 336) return 'hours';
		const minutes = seconds / 60;
		if (minutes >= 1 && minutes <= 20160) return 'minutes';
		return 'seconds';
	}

	// Initialize displayedTimeBudget when component mounts
	$: if (!initialized) {
		if (value?.default !== null && value?.default !== undefined) {
			selectedUnit = bestUnit(value.default);
			displayedTimeBudget = convertFromSeconds(value.default, selectedUnit);
		}
		// null default: leave blank (displayedTimeBudget stays null)
		previousUnit = selectedUnit;
		initialized = true;
	} else if (selectedUnit !== previousUnit) {
		if (value?.default !== null && value?.default !== undefined) {
			displayedTimeBudget = convertFromSeconds(value.default, selectedUnit);
		}
		previousUnit = selectedUnit;
	}

	// Handle input changes
	function handleChange(e: CustomEvent<number | null>) {
		if (e.detail !== null && e.detail !== undefined) {
			displayedTimeBudget = e.detail;
			value.default = convertToSeconds(e.detail, selectedUnit);
		} else {
			// If cleared, set back to null
			displayedTimeBudget = null;
			value.default = null;
		}
	}

	// Check if current value is valid
	$: isInvalid =
		displayedTimeBudget !== null &&
		(displayedTimeBudget < currentMin || displayedTimeBudget > currentMax);
</script>

<div style="display: flex; gap: 0.25rem; align-items: flex-start;">
	<div style="flex: 1;">
		<NumberInput
			id={label}
			hideSteppers
			labelText={Utils.toUpperCase(label)}
			helperText={value.description?.replace('seconds', selectedUnit)}
			invalid={isInvalid}
			invalidText={selectedUnit !== 'days'
				? `Value must be between ${currentMin} and ${currentMax} ${selectedUnit}`
				: `Value must be ${currentMax} ${selectedUnit}`}
			min={displayedTimeBudget !== null ? currentMin : undefined}
			max={displayedTimeBudget !== null ? currentMax : undefined}
			step={selectedUnit === 'hours' || selectedUnit === 'days' ? 0.01 : 1}
			value={displayedTimeBudget ?? ''}
			on:change={handleChange}
		/>
	</div>
	<div style="width: 130px; padding-top: 1.475rem;">
		<Select hideLabel labelText="Unit" bind:selected={selectedUnit}>
			<SelectItem value="seconds" text="seconds" />
			<SelectItem value="minutes" text="minutes" />
			<SelectItem value="hours" text="hours" />
			<SelectItem value="days" text="days" />
		</Select>
	</div>
</div>
