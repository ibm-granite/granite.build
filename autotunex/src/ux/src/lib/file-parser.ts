// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared file parsing logic used by both Utils (main thread) and the Web Worker.
 * Extracting into a standalone module avoids duplicating parsing code.
 */
import {
	parquetReadObjects,
	parquetMetadataAsync,
	parquetSchema,
	type AsyncBuffer
} from 'hyparquet';

export type ParseResult = Record<string, any>[];

/**
 * Wrap a File as a hyparquet AsyncBuffer that reads only the byte ranges
 * hyparquet actually requests (footer metadata, then the relevant row
 * groups) via File.slice(...).arrayBuffer() — the whole file is never
 * materialized in memory just to preview or count it.
 */
export function fileAsyncBuffer(file: File): AsyncBuffer {
	return {
		byteLength: file.size,
		slice: (start: number, end?: number) => file.slice(start, end).arrayBuffer()
	};
}

export function parseJsonl(content: string, maxLines?: number): ParseResult {
	const lines = content.split('\n').filter((line: string) => line.trim() !== '');
	const linesToProcess = maxLines ? lines.slice(0, maxLines) : lines;

	let skippedCount = 0;
	const jsonData = linesToProcess
		.map((line: string) => {
			try {
				return JSON.parse(line);
			} catch {
				skippedCount++;
				return null;
			}
		})
		.filter((item: any) => item !== null);

	if (jsonData.length === 0) {
		throw new Error(
			'No valid JSON objects found in file. Please check that your file is in JSONL format (one JSON object per line).'
		);
	}

	return jsonData;
}

export function parseCsv(content: string, maxLines?: number): ParseResult {
	const lines = content.trim().split('\n');
	if (lines.length === 0) return [];

	const headers = parseCsvLine(lines[0]);
	const result: ParseResult = [];
	const end = maxLines ? Math.min(lines.length, maxLines + 1) : lines.length;

	for (let i = 1; i < end; i++) {
		const values = parseCsvLine(lines[i]);
		const obj: Record<string, any> = {};
		headers.forEach((header, index) => {
			obj[header] = values[index] || '';
		});
		result.push(obj);
	}

	return result;
}

export function parseJson(
	content: string,
	maxLines?: number,
	isChunked: boolean = false
): ParseResult {
	let trimmedContent = content;

	if (isChunked) {
		const lastCompleteObject = content.lastIndexOf('},');
		if (lastCompleteObject > 0) {
			trimmedContent = content.substring(0, lastCompleteObject + 1);
			if (!trimmedContent.trim().endsWith(']')) {
				trimmedContent += '\n]';
			}
		}
	}

	let parsedData;
	try {
		parsedData = JSON.parse(trimmedContent);
	} catch {
		let fixedContent = trimmedContent.trim();
		if (fixedContent.endsWith(',')) {
			fixedContent = fixedContent.substring(0, fixedContent.length - 1) + ']';
		} else if (!fixedContent.endsWith(']')) {
			fixedContent += ']';
		}

		try {
			parsedData = JSON.parse(fixedContent);
		} catch {
			const lastOpenBrace = fixedContent.lastIndexOf('{');
			const lastCloseBrace = fixedContent.lastIndexOf('}');

			if (lastOpenBrace > lastCloseBrace) {
				fixedContent = fixedContent.substring(0, lastOpenBrace);
				if (fixedContent.trim().endsWith(',')) {
					fixedContent = fixedContent.substring(0, fixedContent.lastIndexOf(','));
				}
				fixedContent += ']';
				parsedData = JSON.parse(fixedContent);
			} else {
				throw new Error('Error parsing JSON file.');
			}
		}
	}

	if (Array.isArray(parsedData)) {
		let jsonData = parsedData.filter(
			(item) => item && typeof item === 'object' && !Array.isArray(item)
		);
		if (maxLines && jsonData.length > maxLines) {
			jsonData = jsonData.slice(0, maxLines);
		}
		if (jsonData.length === 0) {
			throw new Error('No valid entries found in JSON file.');
		}
		return jsonData;
	}

	return [parsedData];
}

export function countJsonlLines(content: string): number {
	const lines = content.split('\n').filter((line: string) => line.trim() !== '');
	let validCount = 0;
	for (const line of lines) {
		try {
			JSON.parse(line);
			validCount++;
		} catch {
			// Skip invalid lines
		}
	}
	return validCount;
}

function convertBigInts(value: any): any {
	if (typeof value === 'bigint') return Number(value);
	if (Array.isArray(value)) return value.map(convertBigInts);
	if (value !== null && typeof value === 'object') {
		const result: Record<string, any> = {};
		for (const [k, v] of Object.entries(value)) {
			result[k] = convertBigInts(v);
		}
		return result;
	}
	return value;
}

export async function parseParquet(source: AsyncBuffer, maxLines?: number): Promise<ParseResult> {
	const rows = await parquetReadObjects({
		file: source,
		rowEnd: maxLines
	});

	if (rows.length === 0) {
		throw new Error('No records found in Parquet file.');
	}

	// Deep-convert BigInt values to Number (parquet INT64 columns produce BigInt)
	return rows.map((row) => convertBigInts(row) as Record<string, any>);
}

// Footer-metadata read (num_rows) instead of materializing every row just to
// take its length — bounded regardless of file size.
export async function countParquetRows(source: AsyncBuffer): Promise<number> {
	const metadata = await parquetMetadataAsync(source);
	return Number(metadata.num_rows);
}

/**
 * Footer-metadata-only read: the file's column names, with no row decode at all.
 *
 * Used when a Parquet file is over the client preview cap. The caller still needs
 * the column list — column mapping is keyed off `Object.keys(rows[0])` — but must
 * not pull row content out of a file that may be many GB, or crafted to expand
 * pathologically once decompressed. So this resolves ONE synthetic row whose keys
 * are the real column names and whose values are empty strings; the preview layer
 * already renders an empty value as a blank cell.
 *
 * Column names come from hyparquet's own schema tree, not `metadata.schema`
 * sliced: that array is the schema flattened depth-first, so a nested column (a
 * chat `messages` LIST, say) contributes its internal `list`/`element` nodes to
 * it and slicing would report those as columns. The tree's top-level children are
 * the columns a user actually sees.
 */
export async function parquetColumnNames(source: AsyncBuffer): Promise<string[]> {
	const metadata = await parquetMetadataAsync(source);
	return parquetSchema(metadata).children.map((child) => child.element.name);
}

/**
 * A one-row {@link ParseResult} carrying only a Parquet file's column names — the
 * deferred-preview stand-in described on {@link parquetColumnNames}. Shaped as a
 * result *array* so callers can hand it straight back to whoever asked for a
 * preview, with no per-call-site wrapping.
 */
export async function parquetPlaceholderPreview(source: AsyncBuffer): Promise<ParseResult> {
	const columns = await parquetColumnNames(source);
	if (columns.length === 0) {
		throw new Error('No columns found in Parquet file metadata.');
	}
	return [Object.fromEntries(columns.map((name) => [name, '']))];
}

function parseCsvLine(line: string): string[] {
	const result: string[] = [];
	let current = '';
	let inQuotes = false;
	let i = 0;

	while (i < line.length) {
		const char = line[i];
		if (char === '"') {
			if (inQuotes && line[i + 1] === '"') {
				current += '"';
				i += 2;
			} else {
				inQuotes = !inQuotes;
				i++;
			}
		} else if (char === ',' && !inQuotes) {
			result.push(current.trim());
			current = '';
			i++;
		} else {
			current += char;
			i++;
		}
	}
	result.push(current.trim());
	return result;
}
