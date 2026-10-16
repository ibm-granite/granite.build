// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import yaml from 'js-yaml';
import type { Configuration, ImportPreviewRow, ImportRowStatus } from './app-types';

const MAX_SUFFIX = 99;

function newRowId(): string {
	if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
		return crypto.randomUUID();
	}
	return `row-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function parseFileContents(filename: string, text: string): unknown {
	const lower = filename.toLowerCase();
	if (lower.endsWith('.yaml') || lower.endsWith('.yml')) {
		return yaml.load(text);
	}
	if (lower.endsWith('.json')) {
		return JSON.parse(text);
	}
	throw new Error('Unsupported file format. Use .json, .yaml, or .yml.');
}

function parseErrorRow(filename: string, message: string): ImportPreviewRow {
	return {
		rowId: newRowId(),
		sourceFile: filename,
		originalName: '',
		editedName: '',
		tunerType: null,
		rlTunerType: null,
		configData: {},
		status: 'parse_error',
		errorMessage: message,
		skipped: false,
		edited: false
	};
}

function toPreviewRow(filename: string, entry: any): ImportPreviewRow {
	if (entry === null || typeof entry !== 'object') {
		return parseErrorRow(filename, 'Expected an object or array of objects.');
	}
	const name = typeof entry.name === 'string' ? entry.name : '';
	const hasConfigData = entry.config_data && typeof entry.config_data === 'object';

	let status: ImportRowStatus = 'ready';
	let errorMessage: string | null = null;
	if (!name) {
		status = 'invalid_missing_name';
		errorMessage = 'Invalid: missing `name`';
	} else if (!hasConfigData) {
		status = 'invalid_missing_config_data';
		errorMessage = 'Invalid: missing `config_data`';
	}

	return {
		rowId: newRowId(),
		sourceFile: filename,
		originalName: name,
		editedName: name,
		tunerType: typeof entry.tuner_type === 'string' ? entry.tuner_type : null,
		rlTunerType: typeof entry.rl_tuner_type === 'string' ? entry.rl_tuner_type : null,
		configData: hasConfigData ? entry.config_data : {},
		status,
		errorMessage,
		skipped: false,
		edited: false
	};
}

// Exported API

export async function parseFilesToRows(files: File[]): Promise<ImportPreviewRow[]> {
	const rows: ImportPreviewRow[] = [];
	for (const file of files) {
		try {
			const text = await file.text();
			const parsed = parseFileContents(file.name, text);
			const entries = Array.isArray(parsed) ? parsed : [parsed];
			if (entries.length === 0) {
				rows.push(parseErrorRow(file.name, 'File is empty or contains no configs.'));
				continue;
			}
			for (const entry of entries) {
				rows.push(toPreviewRow(file.name, entry));
			}
		} catch (err) {
			const msg = err instanceof Error ? err.message : 'Failed to parse file';
			rows.push(parseErrorRow(file.name, `Parse error: ${msg}`));
		}
	}
	return rows;
}

/**
 * Auto-suggest a non-colliding name by appending _2, _3, … up to MAX_SUFFIX.
 * Returns the original name if no collision, or the suggested variant.
 * If all variants up to MAX_SUFFIX are taken, returns the last attempted variant.
 */
export function suggestUniqueName(desired: string, existing: Set<string>): string {
	if (!existing.has(desired)) return desired;
	for (let i = 2; i <= MAX_SUFFIX; i++) {
		const candidate = `${desired}_${i}`;
		if (!existing.has(candidate)) return candidate;
	}
	return `${desired}_${MAX_SUFFIX}`;
}

/**
 * Apply auto-suggestion to parsed rows for names that collide with `existingNames`.
 * Mutates row.editedName and sets row.edited = true when a suggestion is applied.
 * Only runs on rows whose status is 'ready' (i.e. had a valid name and config_data).
 * Does NOT re-run validation — callers should invoke validateRows afterwards.
 */
export function applyNameAutoSuggestions(
	rows: ImportPreviewRow[],
	existingNames: Set<string>
): void {
	const occupied = new Set(existingNames);
	for (const row of rows) {
		if (row.status !== 'ready') continue;
		if (!occupied.has(row.editedName)) {
			occupied.add(row.editedName);
			continue;
		}
		const suggestion = suggestUniqueName(row.editedName, occupied);
		row.editedName = suggestion;
		row.edited = true;
		occupied.add(suggestion);
	}
}

/**
 * Re-validate every non-parse-error, non-invalid-missing row against the current
 * set of existing names and the in-batch duplicates. Mutates row.status and
 * row.errorMessage. Does not touch skipped rows' data, but their status is
 * recomputed in case the user unskips them later — however skipped rows are
 * excluded from in-batch collision detection.
 */
export function validateRows(rows: ImportPreviewRow[], existingNames: Set<string>): void {
	// Count occurrences of each editedName across non-skipped rows
	const counts = new Map<string, number>();
	for (const row of rows) {
		if (row.skipped) continue;
		if (row.status === 'parse_error') continue;
		if (row.status === 'invalid_missing_name') continue;
		if (row.status === 'invalid_missing_config_data') continue;
		counts.set(row.editedName.trim(), (counts.get(row.editedName.trim()) ?? 0) + 1);
	}

	for (const row of rows) {
		// Preserve terminal / source-level errors
		if (row.status === 'parse_error') continue;
		if (row.status === 'invalid_missing_name') continue;
		if (row.status === 'invalid_missing_config_data') continue;
		if (row.status === 'import_failed') continue; // preserve server-reported failures

		const trimmed = row.editedName.trim();
		if (trimmed === '') {
			row.status = 'name_required';
			row.errorMessage = 'Name is required.';
			continue;
		}
		if (existingNames.has(trimmed)) {
			row.status = 'name_exists';
			row.errorMessage = 'A configuration with this name already exists. Rename or skip.';
			continue;
		}
		if (!row.skipped && (counts.get(trimmed) ?? 0) > 1) {
			row.status = 'duplicate_in_batch';
			row.errorMessage = 'Duplicate name in this batch.';
			continue;
		}
		row.status = 'ready';
		row.errorMessage = null;
	}
}

/** True if a row is importable: not skipped, not errored. */
export function isImportable(row: ImportPreviewRow): boolean {
	return !row.skipped && row.status === 'ready';
}

export function importableCount(rows: ImportPreviewRow[]): number {
	let n = 0;
	for (const r of rows) if (isImportable(r)) n++;
	return n;
}

export function existingNameSet(configs: Configuration[] | undefined): Set<string> {
	const s = new Set<string>();
	if (!configs) return s;
	for (const c of configs) s.add(c.name);
	return s;
}
