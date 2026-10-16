// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

// import { fromZonedTime } from 'date-fns-tz';

import { get } from 'svelte/store';
import { appConfig } from './store';
import type {
	DatasetType,
	DatasetFormatType,
	ColumnMetadata,
	ParsedDataRow,
	AlgorithmOption,
	ColumnMapping,
	TuningGoal
} from './app-types';

// Worker size threshold: files larger than this use Web Worker for processing
const WORKER_SIZE_THRESHOLD = 5 * 1024 * 1024; // 5MB

// Mirrors the backend's dataset_client_parquet_preview_max_bytes default
// (docs/operations/configuration.md) — used only if appConfig hasn't
// resolved yet or the fetch failed; the real value comes from the store.
const DEFAULT_PARQUET_PREVIEW_MAX_BYTES = 100 * 1024 * 1024;

function parquetPreviewMaxBytes(): number {
	return (
		get(appConfig)?.dataset_upload?.client_parquet_preview_max_bytes ??
		DEFAULT_PARQUET_PREVIEW_MAX_BYTES
	);
}

export class Utils {
	static toUpperCase = (text: string) => {
		if (!text) return;
		text = text.replaceAll('_', ' ').trim();
		text = text.charAt(0).toUpperCase() + text.substring(1, text.length);
		return text;
	};

	// static getTimeElapsed = (startTime: string, endTime: string, isRunning: boolean = false) => {
	// 	// let start = fromZonedTime(startTime, 'America/New_York').getTime();
	// 	// let end = isRunning
	// 	// 	? new Date().getTime()
	// 	// 	: fromZonedTime(endTime, 'America/New_York').getTime();
	// 	let start = new Date(startTime).getTime();
	// 	let end = isRunning ? new Date().getTime() : new Date(endTime).getTime();
	// 	let totalTime = (end - start) / 1000;
	// 	if (totalTime > 3600) {
	// 		return `${Math.round(totalTime / 3600)} h`;
	// 	} else if (totalTime < 3600 && totalTime > 59) {
	// 		return `${Math.round(totalTime / 60)} min`;
	// 	} else {
	// 		return `${Math.round(totalTime)} sec`;
	// 	}
	// };

	static formatTime(seconds: number) {
		// Handle negative or zero values
		if (seconds <= 0) return '0 s';

		let diff = Math.floor(seconds);

		const days = Math.floor(diff / 86400);
		diff %= 86400;

		const hours = Math.floor(diff / 3600);
		diff %= 3600;

		const minutes = Math.floor(diff / 60);
		const secs = diff % 60;

		// Format based on the largest unit
		if (days > 0) {
			if (hours > 0) {
				return `${days} d ${hours} h`;
			}
			return `${days}d`;
		}

		if (hours > 0) {
			if (minutes > 0) {
				return `${hours} h ${minutes} min`;
			}
			return `${hours}h`;
		}

		if (minutes > 0) {
			if (secs > 0) {
				return `${minutes} min ${secs} s`;
			}
			return `${minutes} min`;
		}

		return `${secs} s`;
	}

	static getTimeElapsed = (
		startTime: Date | string,
		endTime: Date | string = new Date(),
		isRunning: boolean = false
	) => {
		const start = new Date(startTime).getTime();
		const end = isRunning ? new Date().getTime() : new Date(endTime).getTime();

		// Calculate difference in seconds
		const diff = Math.floor((end - start) / 1000);
		return this.formatTime(diff);
	};

	static filterObject<T extends object>(
		obj: T,
		predicate: (key: keyof T, value: T[keyof T]) => boolean
	): Partial<T> {
		return Object.fromEntries(
			Object.entries(obj).filter(([key, value]) => predicate(key as keyof T, value))
		) as Partial<T>;
	}

	static isObject(item: any) {
		return item && typeof item === 'object' && !Array.isArray(item);
	}

	static parseCommaList(value: unknown): string[] | null {
		if (Array.isArray(value)) {
			const cleaned = value.map((v) => String(v).trim()).filter((v) => v.length > 0);
			return cleaned.length > 0 ? cleaned : null;
		}
		if (typeof value !== 'string') return null;
		const parts = value
			.split(',')
			.map((s) => s.trim())
			.filter((s) => s.length > 0);
		return parts.length > 0 ? parts : null;
	}

	static normalizeTokenizerListFields(configData: any): any {
		const section = configData?.tokenizer_config;
		if (!section || typeof section !== 'object') return configData;
		for (const key of Object.keys(section)) {
			const field = section[key];
			if (field && typeof field === 'object' && field.type === 'list') {
				field.default = Utils.parseCommaList(field.default);
			}
		}
		return configData;
	}

	static findDifferingKeys(objects: any) {
		if (!objects || objects.length === 0) {
			return [];
		}

		const firstObject = objects[0];
		const differingKeys = new Set<string>();

		// Identify the keys with differing values
		for (const key in firstObject) {
			if (firstObject.hasOwnProperty(key)) {
				const values = objects.map((obj: any) => obj[key]);
				if (!values.every((value: any) => value === values[0])) {
					differingKeys.add(key);
				}
			}
		}
		return differingKeys;
	}

	// Function to reorder keys in an object
	static reorderObjectKeys(obj: Record<string, string>, keysOrder: any) {
		const newObj: Record<string, string> = {};
		for (const key of keysOrder) {
			if (obj.hasOwnProperty(key)) {
				newObj[key] = obj[key];
			}
		}
		// Add any remaining keys that weren't in the order (to preserve all data)
		for (const key in obj) {
			if (obj.hasOwnProperty(key) && !newObj.hasOwnProperty(key)) {
				newObj[key] = obj[key];
			}
		}
		return newObj;
	}

	/**
	 * Whether this file's row preview is deferred to the server rather than read
	 * in the browser: a Parquet file over the configured preview cap resolves to
	 * ONE synthetic, value-less row carrying just its column names (see
	 * processUploadedFile), so column mapping can proceed without decoding rows
	 * out of a potentially multi-GB file.
	 *
	 * Callers use this to label that preview honestly — an explicit check against
	 * the same threshold the parser uses, rather than trying to infer it from the
	 * shape of the rows they got back.
	 */
	static isPreviewDeferred(file: File | null | undefined): boolean {
		return !!file && file.name.endsWith('.parquet') && file.size > parquetPreviewMaxBytes();
	}

	static processUploadedFile(file: File, maxLines?: number): Promise<DatasetType[]> {
		return new Promise((resolve, reject) => {
			if (!file) {
				reject(new Error('No file provided.'));
				return;
			}

			// For large files, only read a chunk instead of the whole file
			const chunkSize = maxLines ? 1024 * 1024 * 10 : file.size; // 10MB chunk for preview, full file otherwise
			const blob = file.slice(0, Math.min(chunkSize, file.size));
			const reader = new FileReader();

			reader.onload = (event: any) => {
				const content = event.target.result;
				let jsonData = [];

				if (file.name.endsWith('.jsonl')) {
					try {
						const lines = content.split('\n').filter((line: string) => line.trim() !== '');

						console.log(
							`Read ${lines.length} lines from ${(blob.size / 1024 / 1024).toFixed(2)}MB chunk`
						);

						// If maxLines is specified, only process that many lines
						const linesToProcess = maxLines ? lines.slice(0, maxLines) : lines;

						console.log(`Processing ${linesToProcess.length} lines from file`);

						// Parse each line, skipping invalid ones
						let skippedCount = 0;
						jsonData = linesToProcess
							.map((line: string, index: number) => {
								try {
									return JSON.parse(line);
								} catch (parseError) {
									skippedCount++;
									if (skippedCount <= 5) {
										// Only log first 5 errors to avoid console spam
										console.warn(
											`Skipping invalid JSON at line ${index + 1}:`,
											line.substring(0, 100)
										);
									}
									return null;
								}
							})
							.filter((item: any) => item !== null);

						if (skippedCount > 0) {
							console.log(
								`Skipped ${skippedCount} invalid lines, parsed ${jsonData.length} valid lines`
							);
						}

						if (jsonData.length === 0) {
							reject(
								new Error(
									'No valid JSON objects found in file. Please check that your file is in JSONL format (one JSON object per line).'
								)
							);
							return;
						}

						resolve(jsonData);
					} catch (error: any) {
						reject(new Error('Error parsing JSONL file: ' + error.message));
					}
				} else if (file.name.endsWith('.csv')) {
					try {
						// jsonData = this.parseCSV(content);
						const allData = this.csvToJson(content) as DatasetType[];
						jsonData = maxLines ? allData.slice(0, maxLines) : allData;
						resolve(jsonData);
					} catch (error: any) {
						reject(new Error('Error parsing CSV file: ' + error.message));
					}
				} else if (file.name.endsWith('.json')) {
					try {
						// For chunked JSON files, we need to handle incomplete objects
						// Try to find the last complete object in the chunk
						let trimmedContent = content;

						// If we're reading a chunk (not the whole file), try to trim to last complete object
						if (blob.size < file.size) {
							// Find the last complete JSON object by looking for "},\n" or "}\n]"
							const lastCompleteObject = content.lastIndexOf('},');
							if (lastCompleteObject > 0) {
								trimmedContent = content.substring(0, lastCompleteObject + 1);
								// Try to close the array properly
								if (!trimmedContent.trim().endsWith(']')) {
									trimmedContent += '\n]';
								}
							}
						}

						console.log(`Reading ${(blob.size / 1024 / 1024).toFixed(2)}MB from JSON file`);

						let parsedData;
						try {
							parsedData = JSON.parse(trimmedContent);
						} catch (parseError) {
							// If parsing fails, try to fix common issues with truncated JSON
							// Remove any incomplete object at the end
							let fixedContent = trimmedContent.trim();

							// If it ends with incomplete object, try to close the array
							if (fixedContent.endsWith(',')) {
								fixedContent = fixedContent.substring(0, fixedContent.length - 1) + ']';
							} else if (!fixedContent.endsWith(']')) {
								fixedContent += ']';
							}

							// Try parsing again
							try {
								parsedData = JSON.parse(fixedContent);
							} catch (secondError) {
								// If still fails, try removing the last incomplete entry
								const lastOpenBrace = fixedContent.lastIndexOf('{');
								const lastCloseBrace = fixedContent.lastIndexOf('}');

								if (lastOpenBrace > lastCloseBrace) {
									// There's an incomplete object
									fixedContent = fixedContent.substring(0, lastOpenBrace);
									if (fixedContent.trim().endsWith(',')) {
										fixedContent = fixedContent.substring(0, fixedContent.lastIndexOf(','));
									}
									fixedContent += ']';
									parsedData = JSON.parse(fixedContent);
								} else {
									throw secondError;
								}
							}
						}

						// If it's an array, filter out any invalid entries and limit by maxLines
						if (Array.isArray(parsedData)) {
							let skippedCount = 0;
							jsonData = parsedData.filter((item, index) => {
								// Check if item is a valid object
								if (item && typeof item === 'object' && !Array.isArray(item)) {
									return true;
								}
								skippedCount++;
								if (skippedCount <= 5) {
									console.warn(`Skipping invalid entry at index ${index}:`, item);
								}
								return false;
							});

							// Apply maxLines limit if specified
							if (maxLines && jsonData.length > maxLines) {
								console.log(
									`Limiting to first ${maxLines} entries from ${jsonData.length} parsed entries`
								);
								jsonData = jsonData.slice(0, maxLines);
							}

							if (skippedCount > 0) {
								console.log(
									`Skipped ${skippedCount} invalid entries, parsed ${jsonData.length} valid entries`
								);
							}

							if (jsonData.length === 0) {
								reject(new Error('No valid entries found in JSON file.'));
								return;
							}
						} else {
							// If it's a single object, wrap it in an array
							jsonData = [parsedData];
						}

						console.log(`Successfully parsed ${jsonData.length} entries from JSON file`);
						console.log('Parsed JSON successfully on first attempt', Object.keys(jsonData[0]));
						resolve(jsonData);
					} catch (error: any) {
						console.log('🚀 ~ Utils ~ processUploadedFile ~ error:', error);
						reject(new Error('Error parsing JSON file: ' + error.message));
					}
				} else {
					reject(
						new Error(
							'Unsupported file type. Please upload a .jsonl, .json, .csv, or .parquet file.'
						)
					);
				}
			};

			reader.onerror = (error: any) => {
				reject(new Error('Error reading file: ' + error.message));
			};

			if (file.name.endsWith('.parquet')) {
				// Parquet is binary; hyparquet reads only the byte ranges it needs
				// via a ranged AsyncBuffer over the File, so the whole file is
				// never materialized in memory just to preview it.
				(async () => {
					const { parseParquet, parquetPlaceholderPreview, fileAsyncBuffer } = await import(
						'$lib/file-parser'
					);
					const source = fileAsyncBuffer(file);

					// Over the preview cap the row preview is DEFERRED to the server,
					// not refused: reading only the footer metadata yields the column
					// names (bounded, no row decode), and one synthetic value-less row
					// built from them is enough for the callers' column mapping to
					// proceed. Rejecting here instead would block the upload outright —
					// `parsedData` would stay empty and both forms gate their submit on
					// it — for exactly the GB-scale Parquet case this path exists to
					// support. See Utils.isPreviewDeferred for the caller-side signal.
					if (file.size > parquetPreviewMaxBytes()) {
						try {
							resolve((await parquetPlaceholderPreview(source)) as DatasetType[]);
						} catch (error) {
							// A genuinely unreadable footer is a real failure (corrupt, or not a
							// Parquet file at all) — there is nothing to upload usefully.
							reject(new Error('Could not read this Parquet file: ' + (error as Error).message));
						}
						return;
					}

					try {
						const rows = await parseParquet(source, maxLines);
						resolve(rows as DatasetType[]);
					} catch (error: any) {
						reject(new Error('Error parsing Parquet file: ' + error.message));
					}
				})();
			} else {
				reader.readAsText(blob);
			}
		});
	}

	static async countLinesInFile(file: File): Promise<number> {
		if (!file) {
			throw new Error('No file provided.');
		}

		// Parquet is binary; row count comes from its metadata via hyparquet.
		if (file.name.endsWith('.parquet')) {
			try {
				const { countParquetRows, fileAsyncBuffer } = await import('$lib/file-parser');
				return await countParquetRows(fileAsyncBuffer(file));
			} catch {
				throw new Error('Error counting Parquet rows.');
			}
		}

		// Stream the file in chunks so we never hold the whole thing in memory.
		// A 1GB file decoded into a single JS string would crash the renderer
		// ("Aw, Snap!"); here we keep only one chunk plus a carried-over partial
		// line at a time.
		const isJsonl = file.name.endsWith('.jsonl');
		const decoder = new TextDecoder('utf-8');
		const reader = file.stream().getReader();
		let remainder = '';
		let count = 0;

		const tally = (line: string) => {
			if (line.trim() === '') return;
			if (isJsonl) {
				try {
					JSON.parse(line);
					count++;
				} catch {
					// Skip invalid JSON lines (matches previous behavior)
				}
			} else {
				count++;
			}
		};

		try {
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				const text = remainder + decoder.decode(value, { stream: true });
				const lines = text.split('\n');
				remainder = lines.pop() ?? ''; // last element may be a partial line
				for (const line of lines) tally(line);
			}
			remainder += decoder.decode(); // flush any multi-byte tail
			if (remainder) tally(remainder);
			return count;
		} catch {
			throw new Error('Error reading file.');
		} finally {
			reader.releaseLock();
		}
	}

	static parseCSV(csvContent: string) {
		const lines = csvContent.split('\n').filter((line: string) => line.trim() !== '');
		if (lines.length === 0) {
			return [];
		}

		const headers = lines[0].split(',').map((header) => header.trim());
		const result = [];

		for (let i = 1; i < lines.length; i++) {
			const values = lines[i].split(',').map((value) => value.trim());
			const rowObject: Record<string, any> = {};
			headers.forEach((header, index) => {
				rowObject[header] = values[index] !== undefined ? values[index] : null;
			});
			result.push(rowObject);
		}
		return result;
	}

	static extractParameterLength(modelString: string) {
		/**
		 * Extracts the parameter length (e.g., '8b', '125m') from a model identifier string.
		 *
		 * It looks for patterns like digits followed by 'b' or 'm'. If multiple are
		 * found, it assumes the last one is the main parameter size.
		 *
		 * @param {string} modelString - The input string (e.g., 'ibm-granite/granite-3.2-8b-instruct').
		 * @returns {string | null} The extracted parameter length string (like '8b', '125m'),
		 * or null if no such pattern is found.
		 */
		const pattern = /\d+[bm]/g; // 'g' flag for global search to find all matches
		const matches = [...modelString.matchAll(pattern)]; // Use matchAll to get all matches

		if (matches.length > 0) {
			// If one or more matches are found, return the last one
			return matches[matches.length - 1][0]; // [0] because matchAll returns an array of match objects
		} else {
			// If no match is found, return null
			return null;
		}
	}

	static csvToJson(csvString: string) {
		// Split the CSV string into lines
		const lines = csvString.trim().split('\n');

		if (lines.length === 0) {
			return [];
		}

		// Parse the header row
		const headers = this.parseCsvLine(lines[0]);

		// Parse the data rows
		const result = [];
		for (let i = 1; i < lines.length; i++) {
			const values = this.parseCsvLine(lines[i]);
			const obj: Record<string, any> = {};

			// Map values to headers
			headers.forEach((header, index) => {
				obj[header] = values[index] || '';
			});

			result.push(obj);
		}

		return result;
	}

	static parseCsvLine(line: string) {
		const result = [];
		let current = '';
		let inQuotes = false;
		let i = 0;

		while (i < line.length) {
			const char = line[i];

			if (char === '"') {
				if (inQuotes && line[i + 1] === '"') {
					// Handle escaped quotes
					current += '"';
					i += 2;
				} else {
					// Toggle quote state
					inQuotes = !inQuotes;
					i++;
				}
			} else if (char === ',' && !inQuotes) {
				// End of field
				result.push(current.trim());
				current = '';
				i++;
			} else {
				current += char;
				i++;
			}
		}

		// Add the last field
		result.push(current.trim());

		return result;
	}

	/**
	 * Pull a human-readable reason out of whatever `API.handleResponse` threw.
	 *
	 * It rejects with the parsed RFC 9457 problem+json body (`{title, status,
	 * detail}`), so `detail` is the server's own sentence and the best thing to
	 * show — a domain refusal like "Configuration <id> is provided by the system
	 * and cannot be deleted" explains itself far better than any generic string
	 * assembled here. Falls back to `title`, then to a real `Error.message`, then
	 * to the caller's default, so a network failure (which throws a `TypeError`,
	 * not a problem body) still reports something.
	 */
	static problemDetail(error: unknown, fallback: string): string {
		if (error && typeof error === 'object') {
			const body = error as { detail?: unknown; title?: unknown };
			if (typeof body.detail === 'string' && body.detail.trim()) return body.detail;
			if (typeof body.title === 'string' && body.title.trim()) return body.title;
		}
		if (error instanceof Error && error.message.trim()) return error.message;
		return fallback;
	}

	static getNameFromUrl(url: string) {
		try {
			const urlObj = new URL(url);
			const pathSegments = urlObj.pathname.split('/').filter((segment) => segment);
			return pathSegments[pathSegments.length - 1];
		} catch (error) {
			console.error('Invalid URL:', error);
			return null;
		}
	}

	// Function to convert data to JSONL string
	static dataToJsonl(data: DatasetType[]): string {
		return data.map((item) => JSON.stringify(item)).join('\n');
	}

	static getOption(option: 'uniform' | 'loguniform' | 'choice') {
		if (option === 'uniform') {
			return 'Uniform sampling';
		} else if (option === 'loguniform') {
			return 'Logarithmic sampling';
		} else {
			return Utils.toUpperCase(option);
		}
	}

	/**
	 * Detect dataset format based on column names.
	 * Matches column signatures to known RL/SFT dataset formats.
	 */
	static detectDatasetFormat(columns: string[]): DatasetFormatType {
		const cols = new Set(columns.map((c) => c.toLowerCase()));

		// Check preference pairs: prompt + chosen + rejected (DPO/ORPO)
		if (cols.has('prompt') && cols.has('chosen') && cols.has('rejected')) {
			return 'preference_pairs';
		}

		// Check KTO format: prompt + completion + label
		if (cols.has('prompt') && cols.has('completion') && cols.has('label')) {
			return 'kto_format';
		}

		// Check standard pairs: input+output OR prompt+response
		const hasInput =
			cols.has('input') ||
			cols.has('prompt') ||
			cols.has('question') ||
			cols.has('instruction') ||
			cols.has('text');
		const hasOutput =
			cols.has('output') ||
			cols.has('response') ||
			cols.has('answer') ||
			cols.has('completion') ||
			cols.has('target') ||
			cols.has('label');
		if (hasInput && hasOutput) {
			return 'standard_pairs';
		}

		// Check prompt-only (Online RL: PPO, GRPO, DAPO)
		if ((cols.has('prompt') || cols.has('input') || cols.has('question')) && !hasOutput) {
			return 'prompt_only';
		}

		return 'unknown';
	}

	/**
	 * Extract column metadata from parsed rows.
	 * Analyzes types, sample values, null counts, and uniqueness.
	 */
	static extractColumnMetadata(
		rows: ParsedDataRow[],
		maxSampleRows: number = 100
	): ColumnMetadata[] {
		if (!rows || rows.length === 0) return [];

		const sampleRows = rows.slice(0, maxSampleRows);
		const allColumns = new Set<string>();
		for (const row of sampleRows) {
			for (const key of Object.keys(row)) {
				allColumns.add(key);
			}
		}

		return Array.from(allColumns).map((colName) => {
			const values = sampleRows.map((row) => row[colName]);
			const nonNullValues = values.filter((v) => v !== null && v !== undefined);
			const uniqueValues = new Set(nonNullValues.map((v) => JSON.stringify(v)));

			// Detect type from first non-null value
			let detectedType: ColumnMetadata['detectedType'] = 'null';
			for (const val of nonNullValues) {
				if (Array.isArray(val)) {
					detectedType = 'array';
				} else if (typeof val === 'object') {
					detectedType = 'object';
				} else if (typeof val === 'number') {
					detectedType = 'number';
				} else if (typeof val === 'boolean') {
					detectedType = 'boolean';
				} else {
					detectedType = 'string';
				}
				break;
			}

			// Get first 3 sample values as strings
			const sampleValues = nonNullValues.slice(0, 3).map((v) => {
				const str = typeof v === 'string' ? v : JSON.stringify(v);
				return str.length > 80 ? str.substring(0, 80) + '...' : str;
			});

			return {
				name: colName,
				detectedType,
				sampleValues,
				nullCount: values.length - nonNullValues.length,
				uniqueCount: uniqueValues.size
			};
		});
	}

	/**
	 * Return human-readable list of compatible tuning methods for a dataset format.
	 */
	static getCompatibleMethods(format: DatasetFormatType): string[] {
		switch (format) {
			case 'preference_pairs':
				return ['DPO'];
			case 'kto_format':
				return ['KTO'];
			case 'standard_pairs':
				return ['SFT', 'LoRA', 'aLoRA', 'LoKR', 'LoHA', 'VeRA'];
			case 'prompt_only':
				return ['PPO', 'GRPO', 'DAPO'];
			default:
				return ['All methods'];
		}
	}

	/**
	 * Convert raw parsed rows to the format expected for dataset upload.
	 * Standard pairs are normalized to {input, output}; RL formats pass through as-is.
	 */
	static formatDatasetForUpload(rows: ParsedDataRow[], format: DatasetFormatType): any[] {
		if (format === 'standard_pairs') {
			return rows.map((row) => {
				const input = row.input ?? row.prompt ?? row.question ?? row.instruction ?? '';
				const output =
					row.output ?? row.response ?? row.answer ?? row.completion ?? row.target ?? '';
				return { input, output };
			});
		}
		// For RL formats (preference_pairs, kto_format, prompt_only), preserve original columns
		return rows;
	}

	/**
	 * Available algorithms with their required columns.
	 */
	static readonly ALGORITHM_OPTIONS: AlgorithmOption[] = [
		{
			id: 'lora',
			name: 'SFT (LoRA / aLoRA / LoKR / ...)',
			category: 'sft',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'dpo',
			name: 'DPO',
			category: 'offline_rl',
			requiredColumns: ['prompt', 'chosen', 'rejected']
		},
		{
			id: 'kto',
			name: 'KTO',
			category: 'offline_rl',
			requiredColumns: ['prompt', 'completion', 'label']
		},
		{ id: 'ppo', name: 'PPO', category: 'online_rl', requiredColumns: ['prompt'] },
		{ id: 'grpo', name: 'GRPO', category: 'online_rl', requiredColumns: ['prompt'] },
		{ id: 'dapo', name: 'DAPO', category: 'online_rl', requiredColumns: ['prompt'] }
	];

	/**
	 * Get required columns for a given algorithm.
	 */
	static getRequiredColumns(algorithmId: string): string[] {
		// Check ALGORITHM_DETAILS first (covers all individual algorithms from Step0)
		const detail = this.ALGORITHM_DETAILS.find((a) => a.id === algorithmId);
		if (detail) return detail.requiredColumns;
		// Fallback to grouped ALGORITHM_OPTIONS
		const algo = this.ALGORITHM_OPTIONS.find((a) => a.id === algorithmId);
		return algo?.requiredColumns || ['input', 'output'];
	}

	/**
	 * Maps algorithm IDs to their corresponding dataset type keys.
	 */
	static readonly ALGORITHM_TO_DATASET_TYPE: Record<string, string> = {
		sft: 'dataset_type_a',
		lora: 'dataset_type_a',
		alora: 'dataset_type_a',
		lokr: 'dataset_type_a',
		loha: 'dataset_type_a',
		vera: 'dataset_type_a',
		dpo: 'dataset_type_b',
		kto: 'dataset_type_c',
		ppo: 'dataset_type_d',
		grpo: 'dataset_type_d',
		dapo: 'dataset_type_d'
	};

	/**
	 * Get all columns (with metadata) from the backend dataset types API response.
	 */
	static getColumnsFromTypes(
		algorithmId: string,
		types: Record<string, any>
	): { name: string; desc: string; required: boolean }[] {
		const typeKey = this.ALGORITHM_TO_DATASET_TYPE[algorithmId];
		if (!typeKey || !types[typeKey]) return [];
		const columns = types[typeKey].columns || {};
		return Object.values(columns).map((col: any) => ({
			name: col.name as string,
			desc: (col.desc || '') as string,
			required: col.required !== false
		}));
	}

	/**
	 * Get required columns from the backend dataset types API response.
	 * Falls back to hardcoded values if the type is not found.
	 */
	static getRequiredColumnsFromTypes(algorithmId: string, types: Record<string, any>): string[] {
		const allCols = this.getColumnsFromTypes(algorithmId, types);
		if (allCols.length === 0) return this.getRequiredColumns(algorithmId);
		return allCols.filter((c) => c.required).map((c) => c.name);
	}

	/**
	 * Goal options for the Step 0 questionnaire.
	 */
	static readonly GOAL_OPTIONS: {
		id: TuningGoal;
		title: string;
		sub_title: string;
		description: string;
		dataDescription: string;
	}[] = [
		{
			id: 'sft',
			title: 'Supervised fine-tuning',
			sub_title: 'Teach the model a specific task or style',
			description:
				'Train your model to follow instructions, generate specific outputs, or adopt a writing style. Best for tasks with clear input/output pairs.',
			dataDescription: 'Input/output pairs (e.g., instruction + response)'
		},
		{
			id: 'offline_rl',
			title: 'Preference learning',
			sub_title: 'Align the model with human feedback',
			description:
				'Train your model to prefer better responses over worse ones using preference data. Useful for improving response quality based on human judgment.',
			dataDescription: 'Prompts with preferred and rejected responses'
		},
		{
			id: 'online_rl',
			title: 'Reinforcement learning',
			sub_title: 'Let the model learn from reward signals',
			description:
				'The model generates its own responses and improves based on automated scoring. Great when you want the model to explore and find better answers on its own.',
			dataDescription: 'Prompts only (a reward model scores the responses)'
		}
	];

	/**
	 * Algorithm details with descriptions and recommended flags for Step 0.
	 */
	static readonly ALGORITHM_DETAILS: {
		id: string;
		name: string;
		category: TuningGoal;
		recommended: boolean;
		shortDescription: string;
		requiredColumns: string[];
	}[] = [
		{
			id: 'lora',
			name: 'LoRA',
			category: 'sft',
			recommended: true,
			shortDescription:
				'Low-Rank Adaptation. Efficient fine-tuning by training small adapter weights. Best balance of quality and efficiency.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'sft',
			name: 'SFT',
			category: 'sft',
			recommended: false,
			shortDescription:
				'Full Supervised Fine-Tuning. Updates all model weights for maximum quality, but requires more compute and memory.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'alora',
			name: 'aLoRA',
			category: 'sft',
			recommended: false,
			shortDescription:
				'Adaptive LoRA. Dynamically adjusts rank allocation during training for optimal parameter efficiency.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'lokr',
			name: 'LoKR',
			category: 'sft',
			recommended: false,
			shortDescription:
				'Low-Rank adaptation with Kronecker product. Memory-efficient alternative to LoRA using decomposed weight matrices.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'loha',
			name: 'LoHA',
			category: 'sft',
			recommended: false,
			shortDescription:
				'Low-Rank adaptation with Hadamard product. Combines element-wise and low-rank factorization for expressive adapters.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'vera',
			name: 'VeRA',
			category: 'sft',
			recommended: false,
			shortDescription:
				'Vector-based Random matrix Adaptation. Uses shared frozen random matrices with small trainable scaling vectors.',
			requiredColumns: ['input', 'output']
		},
		{
			id: 'dpo',
			name: 'DPO',
			category: 'offline_rl',
			recommended: true,
			shortDescription:
				'Direct Preference Optimization. Simple, stable training from preference pairs without a separate reward model.',
			requiredColumns: ['prompt', 'chosen', 'rejected']
		},
		{
			id: 'kto',
			name: 'KTO',
			category: 'offline_rl',
			recommended: false,
			shortDescription:
				'Kahneman-Tversky Optimization. Works with binary feedback (good/bad) instead of paired preferences.',
			requiredColumns: ['prompt', 'completion', 'label']
		},
		{
			id: 'grpo',
			name: 'GRPO',
			category: 'online_rl',
			recommended: true,
			shortDescription:
				'Group Relative Policy Optimization. Learns from relative reward rankings of multiple responses. No critic model needed.',
			requiredColumns: ['prompt']
		},
		{
			id: 'ppo',
			name: 'PPO',
			category: 'online_rl',
			recommended: false,
			shortDescription:
				'Proximal Policy Optimization. Classic RL approach with a value function. Proven but requires more memory.',
			requiredColumns: ['prompt']
		},
		{
			id: 'dapo',
			name: 'DAPO',
			category: 'online_rl',
			recommended: false,
			shortDescription:
				'Decoupled Advantage Policy Optimization. Improves on PPO with decoupled clipping for more stable training.',
			requiredColumns: ['prompt']
		}
	];

	/**
	 * Example dataset rows for each format (used in Step 0 summary).
	 */
	static readonly DATASET_EXAMPLES: Record<string, Record<string, string>[]> = {
		sft: [
			{
				input: 'Summarize this article about climate change...',
				output: 'The article discusses the impact of rising temperatures...'
			},
			{ input: 'Translate to French: Hello, how are you?', output: 'Bonjour, comment allez-vous?' }
		],
		dpo: [
			{
				prompt: 'Explain quantum computing',
				chosen: 'Quantum computing leverages quantum mechanical phenomena...',
				rejected: 'I am not sure about that topic.'
			},
			{
				prompt: 'Write a haiku about spring',
				chosen: 'Cherry blossoms fall / Gentle rain on morning grass / New life awakens',
				rejected: 'Spring is nice and warm.'
			}
		],
		kto: [
			{
				prompt: 'Explain quantum computing',
				completion: 'Quantum computing leverages quantum mechanical phenomena...',
				label: 'true'
			},
			{ prompt: 'What is 2+2?', completion: 'The answer is probably 5.', label: 'false' }
		],
		online_rl: [
			{ prompt: 'Write a Python function that sorts a list using merge sort' },
			{ prompt: 'Explain the difference between TCP and UDP protocols' }
		]
	};

	/**
	 * Get algorithms for a specific goal category.
	 */
	static getAlgorithmsForGoal(goal: TuningGoal) {
		return this.ALGORITHM_DETAILS.filter((a) => a.category === goal);
	}

	/**
	 * Get the recommended (default) algorithm for a goal category.
	 */
	static getDefaultAlgorithmForGoal(goal: TuningGoal): string {
		const recommended = this.ALGORITHM_DETAILS.find((a) => a.category === goal && a.recommended);
		return recommended?.id || 'lora';
	}

	/**
	 * Get dataset examples for a specific algorithm.
	 */
	static getDatasetExamples(algorithmId: string): Record<string, string>[] {
		if (['lora', 'sft', 'alora', 'lokr', 'loha', 'vera'].includes(algorithmId))
			return this.DATASET_EXAMPLES.sft;
		if (['ppo', 'grpo', 'dapo'].includes(algorithmId)) return this.DATASET_EXAMPLES.online_rl;
		return this.DATASET_EXAMPLES[algorithmId] || this.DATASET_EXAMPLES.sft;
	}

	/**
	 * Generate dataset examples from the backend dataset types API response.
	 * Creates a single example row using column descriptions as placeholder values.
	 * Falls back to hardcoded examples if API data isn't available.
	 */
	static getDatasetExamplesFromTypes(
		algorithmId: string,
		types: Record<string, any>
	): Record<string, string>[] {
		const typeKey = this.ALGORITHM_TO_DATASET_TYPE[algorithmId];
		if (!typeKey || !types[typeKey]) return this.getDatasetExamples(algorithmId);

		const columns = types[typeKey].columns || {};
		const exampleRow: Record<string, string> = {};
		for (const col of Object.values(columns) as any[]) {
			exampleRow[col.name] = col.desc || `<${col.name}>`;
		}
		return [exampleRow];
	}

	/**
	 * Generate dataset examples in JSONL, CSV, and JSON formats from column definitions.
	 * Used to show multi-format examples in the "Expected Dataset Format" panel.
	 */
	static generateFormatExamples(columns: { name: string; desc: string; type?: string }[]): {
		jsonl: string;
		csv: string;
		json: string;
	} {
		const cols = columns.map((col) => ({
			name: col.name,
			placeholder: col.desc || `<${col.name}>`
		}));

		// JSONL: one JSON object per line
		const jsonlRow = Object.fromEntries(cols.map((c) => [c.name, c.placeholder]));
		const jsonl = [jsonlRow, jsonlRow].map((r) => JSON.stringify(r)).join('\n');

		// CSV: header row + data rows
		const header = cols.map((c) => c.name).join(', ');
		const csvRow = cols.map((c) => `"${c.placeholder}"`).join(', ');
		const csv = [header, csvRow, csvRow].join('\n');

		// JSON: array of objects
		const json = JSON.stringify([jsonlRow, jsonlRow], null, 2);

		return { jsonl, csv, json };
	}

	/**
	 * Auto-suggest column mapping based on column names and required columns.
	 * Tries exact match first, then fuzzy aliases.
	 */
	static suggestColumnMapping(userColumns: string[], requiredColumns: string[]): ColumnMapping {
		const aliases: Record<string, string[]> = {
			prompt: ['prompt', 'question', 'instruction', 'input', 'text', 'query'],
			input: ['input', 'prompt', 'question', 'instruction', 'text', 'query'],
			output: ['output', 'response', 'answer', 'completion', 'target', 'label'],
			chosen: ['chosen', 'preferred', 'accepted', 'positive', 'output'],
			rejected: ['rejected', 'dispreferred', 'negative'],
			completion: ['completion', 'output', 'response', 'answer'],
			label: ['label', 'preference', 'rating', 'score'],
			documents: ['documents', 'document', 'context', 'source', 'reference']
		};

		const mapping: ColumnMapping = {};
		const usedColumns = new Set<string>();
		const lowerUserColumns = userColumns.map((c) => c.toLowerCase());

		for (const required of requiredColumns) {
			const candidateAliases = aliases[required] || [required];

			// Try exact match (case-insensitive)
			let matched = false;
			for (const alias of candidateAliases) {
				const idx = lowerUserColumns.findIndex(
					(c) => c === alias && !usedColumns.has(userColumns[lowerUserColumns.indexOf(c)])
				);
				if (idx !== -1) {
					mapping[required] = userColumns[idx];
					usedColumns.add(userColumns[idx]);
					matched = true;
					break;
				}
			}

			// If no match, leave empty — user must map manually
			if (!matched) {
				mapping[required] = '';
			}
		}

		return mapping;
	}

	/**
	 * Suggest best algorithm based on dataset columns (auto-detect).
	 */
	static suggestAlgorithm(columns: string[]): string {
		const format = this.detectDatasetFormat(columns);
		switch (format) {
			case 'preference_pairs':
				return 'dpo';
			case 'kto_format':
				return 'kto';
			case 'standard_pairs':
				return 'lora';
			case 'prompt_only':
				return 'grpo';
			default:
				return 'lora';
		}
	}

	/**
	 * Validate that the detected dataset format is compatible with the selected tuning goal.
	 * Returns { valid: true } if compatible, or { valid: false, message: '...' } with a warning.
	 */
	static validateDatasetForGoal(
		detectedFormat: DatasetFormatType,
		goal: TuningGoal
	): { valid: boolean; message: string } {
		const goalLabels: Record<TuningGoal, string> = {
			sft: 'Supervised Fine-Tuning',
			offline_rl: 'Preference Learning (Offline RL)',
			online_rl: 'Reinforcement Learning (Online RL)'
		};
		const formatLabels: Record<DatasetFormatType, string> = {
			preference_pairs: 'preference pairs (prompt + chosen + rejected)',
			kto_format: 'KTO format (prompt + completion + label)',
			standard_pairs: 'standard input/output pairs',
			prompt_only: 'prompt-only format',
			unknown: 'unknown format'
		};

		const expected: Record<TuningGoal, DatasetFormatType[]> = {
			sft: ['standard_pairs'],
			offline_rl: ['preference_pairs', 'kto_format'],
			online_rl: ['prompt_only']
		};

		if (detectedFormat === 'unknown') {
			return { valid: true, message: '' };
		}

		if (expected[goal]?.includes(detectedFormat)) {
			return { valid: true, message: '' };
		}

		return {
			valid: false,
			message: `The uploaded dataset appears to be in ${formatLabels[detectedFormat]}, which is typically used for a different tuning approach. You selected "${goalLabels[goal]}". The AI will attempt to map columns to match your selected goal. Please verify the column mapping below.`
		};
	}

	/**
	 * Get a human-readable summary of a configuration's algorithm(s).
	 */
	static getConfigSummary(config: {
		tuner_type?: string | null;
		rl_tuner_type?: string | null;
	}): string {
		const displayNames: Record<string, string> = {
			lora: 'LoRA',
			alora: 'aLoRA',
			qlora: 'QLoRA',
			lokr: 'LoKR',
			loha: 'LoHA',
			vera: 'VeRA',
			sft: 'SFT'
		};
		const formatName = (name: string) => displayNames[name.toLowerCase()] ?? name.toUpperCase();
		const parts: string[] = [];
		if (config.tuner_type && config.tuner_type !== 'none') {
			parts.push(formatName(config.tuner_type));
		}
		if (config.rl_tuner_type && config.rl_tuner_type !== 'none') {
			parts.push(formatName(config.rl_tuner_type));
		}
		return parts.join(' + ') || 'Default';
	}

	/**
	 * Transform dataset rows using column mapping.
	 * Renames columns from user's names to required names.
	 */
	static applyColumnMapping(rows: ParsedDataRow[], mapping: ColumnMapping): ParsedDataRow[] {
		return rows.map((row) => {
			const mapped: ParsedDataRow = {};
			for (const [requiredCol, userCol] of Object.entries(mapping)) {
				if (userCol && row[userCol] !== undefined) {
					mapped[requiredCol] = row[userCol];
				}
			}
			return mapped;
		});
	}

	/**
	 * Process uploaded file asynchronously.
	 * Uses a Web Worker for files > 5MB to keep the main thread responsive.
	 * Falls back to the synchronous processUploadedFile for smaller files.
	 */
	static processUploadedFileAsync(file: File, maxLines?: number): Promise<DatasetType[]> {
		if (file.size <= WORKER_SIZE_THRESHOLD || typeof Worker === 'undefined') {
			return this.processUploadedFile(file, maxLines);
		}

		const isParquet = file.name.endsWith('.parquet');

		const spawnWorker = (
			message: Record<string, unknown>,
			resolve: (rows: DatasetType[]) => void,
			reject: (error: Error) => void
		): void => {
			try {
				const worker = new Worker(new URL('./workers/file-processor.worker.ts', import.meta.url), {
					type: 'module'
				});

				worker.onmessage = (e) => {
					worker.terminate();
					if (e.data.type === 'result') {
						resolve(e.data.data);
					} else if (e.data.type === 'error') {
						reject(new Error(e.data.message));
					}
				};

				worker.onerror = (e) => {
					worker.terminate();
					reject(new Error('Worker error: ' + (e.message || 'File processing failed')));
				};

				worker.postMessage(message);
			} catch {
				this.processUploadedFile(file, maxLines).then(resolve).catch(reject);
			}
		};

		if (isParquet) {
			// Deferred preview above the cap — see processUploadedFile for why this
			// resolves a columns-only placeholder row instead of rejecting. Done on
			// this thread deliberately: a footer-metadata read is a couple of small
			// ranged reads, so there is nothing worth handing to a worker.
			if (file.size > parquetPreviewMaxBytes()) {
				return (async () => {
					const { parquetPlaceholderPreview, fileAsyncBuffer } = await import('$lib/file-parser');
					try {
						return (await parquetPlaceholderPreview(fileAsyncBuffer(file))) as DatasetType[];
					} catch (error) {
						throw new Error('Could not read this Parquet file: ' + (error as Error).message);
					}
				})();
			}
			// Hand the File itself to the worker: hyparquet reads only the byte
			// ranges it needs there, so the whole file is never read on the main
			// thread just to spawn the worker (the previous FileReader step did).
			return new Promise((resolve, reject) => {
				spawnWorker(
					{ type: 'processFile', file, fileName: file.name, maxLines, isChunked: false },
					resolve,
					reject
				);
			});
		}

		return new Promise((resolve, reject) => {
			const chunkSize = maxLines ? 1024 * 1024 * 10 : file.size;
			const blob = file.slice(0, Math.min(chunkSize, file.size));
			const reader = new FileReader();

			reader.onload = (event: any) => {
				const content = event.target.result as string;
				spawnWorker(
					{
						type: 'processFile',
						content,
						fileName: file.name,
						maxLines,
						isChunked: blob.size < file.size
					},
					resolve,
					reject
				);
			};

			reader.onerror = () => {
				reject(new Error('Error reading file.'));
			};

			reader.readAsText(blob);
		});
	}

	/**
	 * Count lines in a file asynchronously.
	 * Uses a Web Worker for files > 5MB.
	 */
	static countLinesInFileAsync(file: File): Promise<number> {
		if (file.size <= WORKER_SIZE_THRESHOLD || typeof Worker === 'undefined') {
			return this.countLinesInFile(file);
		}

		// Hand the File itself to the worker and let it stream the content there.
		// We deliberately do NOT read the file on the main thread first — doing so
		// for a multi-hundred-MB file is what crashed the renderer.
		return new Promise((resolve, reject) => {
			let worker: Worker;
			try {
				worker = new Worker(new URL('./workers/file-processor.worker.ts', import.meta.url), {
					type: 'module'
				});
			} catch {
				// Worker unavailable (e.g. SSR / unsupported env): stream on this thread.
				this.countLinesInFile(file).then(resolve).catch(reject);
				return;
			}

			worker.onmessage = (e) => {
				worker.terminate();
				if (e.data.type === 'result') {
					resolve(e.data.data);
				} else if (e.data.type === 'error') {
					reject(new Error(e.data.message));
				}
			};

			worker.onerror = (e) => {
				worker.terminate();
				reject(new Error('Worker error: ' + (e.message || 'Line counting failed')));
			};

			worker.postMessage({
				type: 'countLinesStream' as const,
				file,
				fileName: file.name
			});
		});
	}

	/**
	 * Gzip-compress a File in the background worker. Throws if CompressionStream
	 * or Worker is unavailable, or if compression otherwise fails — callers must
	 * catch this and fall back to an uncompressed upload; a compression failure
	 * must never block the upload itself.
	 */
	static gzipFileAsync(file: File): Promise<Blob> {
		if (typeof CompressionStream === 'undefined' || typeof Worker === 'undefined') {
			return Promise.reject(new Error('Gzip compression is not supported in this browser.'));
		}

		return new Promise((resolve, reject) => {
			let worker: Worker;
			try {
				worker = new Worker(new URL('./workers/file-processor.worker.ts', import.meta.url), {
					type: 'module'
				});
			} catch {
				reject(new Error('Worker unavailable for gzip compression.'));
				return;
			}

			worker.onmessage = (e) => {
				worker.terminate();
				if (e.data.type === 'result') {
					resolve(e.data.data as Blob);
				} else if (e.data.type === 'error') {
					reject(new Error(e.data.message));
				}
			};

			worker.onerror = (e) => {
				worker.terminate();
				reject(new Error('Worker error: ' + (e.message || 'Gzip compression failed')));
			};

			worker.postMessage({ type: 'gzipFile', file });
		});
	}
}
