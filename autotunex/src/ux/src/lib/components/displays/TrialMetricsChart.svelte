<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import { LineChart } from '@carbon/charts-svelte';
	import {
		Dropdown,
		Slider,
		Tooltip,
		InlineLoading,
		InlineNotification
	} from 'carbon-components-svelte';
	import { onMount } from 'svelte';
	import { API } from '$lib/api';
	import type { MetricPoint, Status } from '$lib/app-types';
	import {
		buildLossPanel,
		buildLrPanel,
		buildGradPanel,
		epochBoundaries,
		median,
		hasUsableMetrics,
		TRAIN_LOSS_SMOOTH,
		TRAIN_LOSS_RAW,
		VAL_LOSS,
		type XField
	} from '$lib/metrics-shaping';

	export let jobId: string;
	export let trialId: string;
	export let status: Status;

	const api = new API();

	// A trial with no run yet cannot have metrics — empty state, no request. 'PAUSED' isn't in
	// the Status union (a pre-existing gap; see api-mappers.ts) so cast rather than widen it.
	const CAN_HAVE_METRICS: Status[] = [
		'RUNNING',
		'PAUSED' as Status,
		'COMPLETED',
		'ERROR',
		'TERMINATED'
	];

	let points: MetricPoint[] = [];
	let loading = true;
	let errored = false;

	const X_ITEMS = [
		{ id: 'epoch', text: 'Epoch' },
		{ id: 'global_step', text: 'Step' }
	];
	// Epoch by default: it is the unit users actually set (`num_train_epochs`) and think
	// in. Step stays available because it is the only axis guaranteed to plot —
	// `training_metrics.global_step` is NOT NULL while `epoch` is nullable, and a null x
	// is silently dropped — and because neither axis is fair across a whole sweep: with
	// batch size varying, one epoch is the same data seen; with `hpo_dataset_percentage`
	// varying, it is not, and step tracks compute more closely.
	// Bound as a plain string because Carbon's Dropdown types `selectedId` that way, and
	// a TS cast is not parseable inside a Svelte template expression. Narrowed here.
	let xId = 'epoch';
	$: xField = xId as XField;
	let weight = 0.7; // EMA smoothing weight; 0 disables smoothing (and the ghost with it)

	$: usable = hasUsableMetrics(points);

	$: lossData = buildLossPanel(points, xField, weight);
	$: lrData = buildLrPanel(points, xField);
	$: gradData = buildGradPanel(points, xField);
	$: epochs = epochBoundaries(points, xField);
	$: gradMedian = median(gradData.map((d) => d.value));
	$: xMax = Math.max(
		0,
		...lossData.map((d) => d.key),
		...lrData.map((d) => d.key),
		...gradData.map((d) => d.key)
	);

	// The app persists the Carbon theme on <html theme="…"> (default g10); those names are also
	// @carbon/charts' theme names (theme?: ChartTheme | string), so pass the string straight through.
	const chartTheme = (): string =>
		(typeof document !== 'undefined' && document.documentElement.getAttribute('theme')) || 'g10';

	const DARK_THEMES = new Set(['g90', 'g100']);
	/**
	 * Explicit loss-panel colours, so the raw ghost can share the smoothed line's hue
	 * instead of being handed its own categorical slot (it is the same series, and a
	 * third hue read as a third measurement next to validation loss).
	 *
	 * Both pairs were checked with a palette validator against both surfaces rather than
	 * by eye, which is how the dark case got caught: Carbon magenta-70 (#9f1853) passes
	 * on the light surface but falls below the lightness band and under 3:1 contrast on
	 * g100, so dark selects magenta-50 instead. Cyan-50 passes on both, so only the
	 * magenta step is theme-dependent. Worst adjacent CVD separation is ΔE 23.3 (light)
	 * and 13.1 (dark), both clear of the floor.
	 */
	const lossColorScale = (theme: string): Record<string, string> => {
		const train = '#1192e8'; // Carbon cyan-50
		return {
			[TRAIN_LOSS_SMOOTH]: train,
			// Same hue at 30% alpha — recessive by construction and immune to the surface,
			// where a hard-coded light tint would need its own per-theme step.
			[TRAIN_LOSS_RAW]: `${train}4d`,
			[VAL_LOSS]: DARK_THEMES.has(theme) ? '#ee5396' : '#9f1853'
		};
	};
	const expFmt = (tick: number | Date): string => Number(tick).toExponential(1);

	// The three panels' left-axis gutters are auto-sized by @carbon/charts from their widest tick
	// label, so differing label widths (loss "15" vs LR "4.0e-6" vs grad "4") push the plots out of
	// vertical alignment. Fix: give every axis a fixed-shape "NN.N" label. U+2007 (figure space)
	// renders at digit width — even in a proportional font — so padding the integer part to two
	// places makes all labels the same rendered width, hence the same gutter, hence aligned plots.
	const FIG = ' ';
	const padInt = (s: string, intWidth: number): string => {
		const intLen = (s.startsWith('-') ? s.slice(1) : s).split('.')[0].length;
		return FIG.repeat(Math.max(0, intWidth - intLen)) + s;
	};
	const fmtY = (tick: number | Date): string => padInt(Number(tick).toFixed(1), 2);
	// learning_rate is ~1e-7; label the axis in ×10⁻⁶ units so its ticks are short like the others
	// (real values still show in the tooltip via expFmt).
	const fmtLr = (tick: number | Date): string => padInt((Number(tick) * 1e6).toFixed(1), 2);
	// Show just the epoch/median label on a threshold callout, not the raw axis value ("E2: 45.333").
	const noValue = (): string => '';

	// Shared bottom axis: same [0, xMax] domain on every panel so they line up, with vertical
	// epoch threshold lines. Only the bottom-most (grad) panel carries the axis title.
	$: bottomAxis = (withTitle: boolean) => ({
		title: withTitle ? (xField === 'global_step' ? 'Step' : 'Epoch') : undefined,
		mapsTo: 'key',
		domain: [0, xMax],
		thresholds: epochs.map((e) => ({ value: e.value, label: e.label, valueFormatter: noValue }))
	});

	$: lossOptions = {
		theme: chartTheme(),
		color: { scale: lossColorScale(chartTheme()) },
		title: 'Loss',
		axes: {
			bottom: bottomAxis(false),
			left: { title: 'Loss', mapsTo: 'value', ticks: { formatter: fmtY } }
		},
		curve: 'curveMonotoneX',
		legend: { enabled: true },
		points: { enabled: true, radius: 3 },
		toolbar: { numberOfIcons: 2 }
	};
	$: lrOptions = {
		theme: chartTheme(),
		title: 'Learning rate',
		axes: {
			bottom: bottomAxis(false),
			left: { title: 'LR (×10⁻⁶)', mapsTo: 'value', ticks: { formatter: fmtLr } }
		},
		curve: 'curveMonotoneX',
		legend: { enabled: false },
		points: { enabled: true, radius: 3 },
		toolbar: { enabled: false },
		tooltip: { valueFormatter: expFmt }
	};
	$: gradOptions = {
		theme: chartTheme(),
		title: 'Gradient norm',
		axes: {
			bottom: bottomAxis(true),
			left: {
				title: 'Grad norm',
				mapsTo: 'value',
				includeZero: true,
				ticks: { formatter: fmtY },
				thresholds:
					gradMedian == null
						? []
						: [{ value: gradMedian, label: 'median', valueFormatter: noValue }]
			}
		},
		curve: 'curveMonotoneX',
		legend: { enabled: false },
		points: { enabled: true, radius: 3 },
		toolbar: { enabled: false }
	};

	onMount(async () => {
		if (!CAN_HAVE_METRICS.includes(status)) {
			loading = false;
			return;
		}
		try {
			points = await api.getTrialMetrics(jobId, trialId);
		} catch (e) {
			errored = true;
			console.error('Failed to load trial metrics', e);
		} finally {
			loading = false;
		}
	});
</script>

{#if loading}
	<InlineLoading description="Loading metrics…" />
{:else if errored}
	<InlineNotification
		kind="error"
		lowContrast
		hideCloseButton
		title="Couldn't load metrics"
		subtitle="Try reopening this tab."
	/>
{:else if !usable}
	<InlineNotification
		kind="info"
		lowContrast
		hideCloseButton
		title="No metrics yet"
		subtitle="This trial hasn't reported training metrics yet."
	/>
{:else}
	<div class="metrics-controls">
		<div class="control control-smoothing">
			<!-- Own label rather than the Slider's `labelText`, so the current value and the
			     help tooltip can sit on the label line. "Loss" is in the name because these
			     controls render above all three panels but only affect this one — the axis
			     dropdown beside it scopes everything, this does not. -->
			<span class="control-label">
				<span class="control-label-text">Loss smoothing · {weight.toFixed(2)}</span>
				<Tooltip align="start" direction="bottom">
					<p>
						Averages the training loss over nearby steps so the trend is readable when individual
						steps are noisy. The faint line behind it is the real, unsmoothed data. At 0.00 nothing
						is smoothed and only that line is drawn.
					</p>
				</Tooltip>
			</span>
			<Slider
				labelText=""
				hideLabel
				min={0}
				max={0.95}
				step={0.05}
				bind:value={weight}
				hideTextInput
			/>
		</div>
		<div class="control control-axis">
			<Dropdown size="sm" labelText="X-axis" items={X_ITEMS} bind:selectedId={xId} />
		</div>
	</div>
	<div class="panel panel-loss"><LineChart data={lossData} options={lossOptions} /></div>
	<div class="panel panel-sm"><LineChart data={lrData} options={lrOptions} /></div>
	<div class="panel panel-sm"><LineChart data={gradData} options={gradOptions} /></div>
{/if}

<style>
	.metrics-controls {
		display: flex;
		align-items: flex-end;
		gap: 1.5rem;
		flex-wrap: wrap;
		margin-bottom: 1rem;
	}
	.control-smoothing {
		min-width: 220px;
	}
	/* Carbon ships `.bx--slider { margin: 0 1rem }` to hold the track clear of the
	   min/max range labels either side of it. This slider has none — they render as empty
	   spans — so that 1rem was indenting the track past the label above it for no reason.
	   Zeroing both keeps the track flush with the label and lets it use the full width. */
	.control-smoothing :global(.bx--slider) {
		margin-left: 0;
		margin-right: 0;
	}
	/* Pushed right so the row reads as chart chrome rather than a form. The dropdown is
	   width-capped because Carbon's ContentSwitcher — what this replaced — defaults to
	   width:100%, which is why it used to swallow the whole row and force the smoothing
	   control onto a second line. */
	.control-axis {
		margin-left: auto;
		width: 10rem;
	}
	/* Flex, not `display: block` with an inline tooltip: Carbon's Tooltip trigger is a
	   block-level button, so it wrapped onto its own line and made this control three
	   lines tall. */
	.control-label {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.75rem;
		line-height: 1.34;
		letter-spacing: 0.32px;
		color: var(--cds-text-secondary, #525252);
		margin-bottom: 0.25rem;
	}
	/* Only the label text may not wrap. Putting `white-space: nowrap` on `.control-label`
	   itself inherited into the tooltip's popup, so its paragraph could not wrap and ran
	   straight out of the dark box as unreadable white-on-grey text. */
	.control-label-text {
		white-space: nowrap;
	}
	/* The tooltip trigger ships its own tall tap target, which would otherwise push this
	   label line taller than the dropdown's. */
	.control-label :global(.bx--tooltip__label),
	.control-label :global(.bx--tooltip__trigger) {
		display: inline-flex;
		align-items: center;
	}
	/* Belt-and-braces: the popup is a DOM descendant of the label, so it inherits text
	   settings from it. Pin the wrapping that its own layout depends on. */
	.control-label :global(.bx--tooltip p) {
		white-space: normal;
	}
	.panel {
		margin-bottom: 0.75rem;
	}
	.panel-loss {
		height: 280px;
	}
	.panel-sm {
		height: 170px;
	}
</style>
