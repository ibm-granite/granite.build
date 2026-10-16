// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

// Pure shaping of per-step training-metrics rows into @carbon/charts line data.
// HF Trainer emits three row kinds into one table (see the design spec):
//   - split='train' with non-null loss   -> the training curve (loss/grad_norm/learning_rate)
//   - one split='train' row with null loss -> end-of-training SUMMARY (extra.train_*); excluded here
//   - split='eval' rows                    -> validation; the metric is extra.eval_loss (loss is null)
// Kept in a plain .ts module (not the .svelte component) so the logic is isolated and
// verifiable on its own, mirroring api-mappers.ts / utils.ts.

import type { MetricPoint } from './app-types';

export type XField = 'global_step' | 'epoch';
export type MetricKind = 'loss' | 'grad_norm' | 'learning_rate';

/** One @carbon/charts line datum: `group` is the series, `key` the x, `value` the y. */
export type LinePoint = { group: string; key: number; value: number };

// Series labels — also the group key @carbon/charts colors by.
export const TRAIN_LOSS = 'Training loss';
// Short on purpose: '(smoothed)' overflowed the legend to 'Train loss (sm…'. The
// bold line is simply the train loss; '(raw)' names the faint ghost as a variant of
// it rather than a third measurement, which is also why both share one hue.
export const TRAIN_LOSS_SMOOTH = 'Train loss';
export const TRAIN_LOSS_RAW = 'Train loss (raw)';
export const VAL_LOSS = 'Validation loss';
export const GRAD_NORM = 'Grad norm';
export const LEARNING_RATE = 'Learning rate';

// A per-step training point has a non-null loss; this also excludes the summary row.
const isTrainPoint = (p: MetricPoint): boolean => p.split === 'train' && p.loss != null;
const isEvalPoint = (p: MetricPoint): boolean => p.split === 'eval';

const xOf = (p: MetricPoint, xField: XField): number | null =>
	xField === 'global_step' ? p.global_step : p.epoch;

/**
 * Build @carbon/charts line data for one metric view over the chosen x-axis.
 *
 * The end-of-training summary row (split='train', loss=null) is excluded by
 * `isTrainPoint`. Validation loss is read from `extra.eval_loss`. Points with a
 * null x or y are dropped — nulls are never plotted.
 */
export function buildLineData(
	points: MetricPoint[],
	metric: MetricKind,
	xField: XField
): LinePoint[] {
	const out: LinePoint[] = [];
	const push = (group: string, x: number | null, y: number | null | undefined): void => {
		if (x == null || y == null) return;
		out.push({ group, key: x, value: y });
	};

	if (metric === 'loss') {
		for (const p of points) if (isTrainPoint(p)) push(TRAIN_LOSS, xOf(p, xField), p.loss);
		for (const p of points)
			if (isEvalPoint(p)) push(VAL_LOSS, xOf(p, xField), p.extra?.eval_loss as number | undefined);
		return out;
	}

	const group = metric === 'grad_norm' ? GRAD_NORM : LEARNING_RATE;
	for (const p of points) if (isTrainPoint(p)) push(group, xOf(p, xField), p[metric]);
	return out;
}

/** Debiased exponential moving average — matches the sweep-readout reference's smoothing. */
export function ema(values: number[], weight: number): number[] {
	let last = 0;
	let debias = 0;
	const out: number[] = [];
	for (const v of values) {
		last = last * weight + (1 - weight) * v;
		debias = debias * weight + (1 - weight);
		out.push(debias ? last / debias : v);
	}
	return out;
}

/** Median of a numeric array (grad-norm reference line); null when empty. */
export function median(values: number[]): number | null {
	if (values.length === 0) return null;
	const a = [...values].sort((p, q) => p - q);
	const m = a.length >> 1;
	return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/**
 * Loss panel: smoothed train loss, its unsmoothed ghost, and the eval-loss overlay.
 *
 * The EMA is computed in ascending-global_step order so smoothing is stable regardless of
 * the chosen x-axis. The summary row is excluded (isTrainPoint); eval loss is extra.eval_loss.
 *
 * The raw series is emitted only when `weight > 0`, and that guard is the point rather
 * than an optimisation: `ema(values, 0)` returns its input unchanged, so at zero the
 * smoothed and raw series are identical and plotting both drew the same line twice in
 * two colours. It replaces a user-facing "Raw train loss" toggle — raw is not an
 * independent measurement, it is the same series unfiltered, so whether to draw it
 * follows from whether smoothing is doing anything. Same convention as TensorBoard
 * and W&B: a faint original behind the bold smoothed line.
 */
export function buildLossPanel(points: MetricPoint[], xField: XField, weight: number): LinePoint[] {
	const train = points
		.filter(isTrainPoint)
		.slice()
		.sort((a, b) => a.global_step - b.global_step);
	const losses = train.map((p) => p.loss as number);
	const smoothed = ema(losses, weight);
	const out: LinePoint[] = [];
	train.forEach((p, i) => {
		const x = xOf(p, xField);
		if (x == null) return;
		out.push({ group: TRAIN_LOSS_SMOOTH, key: x, value: smoothed[i] });
		if (weight > 0) out.push({ group: TRAIN_LOSS_RAW, key: x, value: losses[i] });
	});
	for (const p of points) {
		if (!isEvalPoint(p)) continue;
		const x = xOf(p, xField);
		const ev = p.extra?.eval_loss as number | undefined;
		if (x != null && ev != null) out.push({ group: VAL_LOSS, key: x, value: ev });
	}
	return out;
}

/** Learning-rate panel (single series). */
export function buildLrPanel(points: MetricPoint[], xField: XField): LinePoint[] {
	return buildLineData(points, 'learning_rate', xField);
}

/** Gradient-norm panel (single series). */
export function buildGradPanel(points: MetricPoint[], xField: XField): LinePoint[] {
	return buildLineData(points, 'grad_norm', xField);
}

/** x positions (in the chosen field) of each integer-epoch boundary, for vertical threshold lines. */
export function epochBoundaries(
	points: MetricPoint[],
	xField: XField
): { value: number; label: string }[] {
	const train = points.filter(isTrainPoint);
	if (train.length === 0) return [];
	const maxStep = Math.max(...train.map((p) => p.global_step));
	const maxEpoch = Math.max(...train.map((p) => p.epoch ?? 0));
	const epochs = Math.round(maxEpoch);
	const out: { value: number; label: string }[] = [];
	for (let e = 1; e <= epochs; e++) {
		out.push({ value: xField === 'epoch' ? e : (maxStep * e) / epochs, label: `E${e}` });
	}
	return out;
}

/** True if any train row carries a metric or any eval row carries eval_loss. */
export function hasUsableMetrics(points: MetricPoint[]): boolean {
	return points.some(
		(p) =>
			(p.split === 'train' && (p.loss != null || p.grad_norm != null || p.learning_rate != null)) ||
			(p.split === 'eval' && p.extra?.eval_loss != null)
	);
}
