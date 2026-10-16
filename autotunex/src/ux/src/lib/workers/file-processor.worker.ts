// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import {
	parseJsonl,
	parseCsv,
	parseJson,
	countJsonlLines,
	parseParquet,
	countParquetRows,
	fileAsyncBuffer
} from '../file-parser';

type ProcessFileMessage = {
	type: 'processFile';
	content?: string | ArrayBuffer;
	file?: File;
	fileName: string;
	maxLines?: number;
	isChunked: boolean;
};

type CountLinesMessage = {
	type: 'countLines';
	content: string | ArrayBuffer;
	fileName: string;
};

// Streams the file inside the worker so the whole content is never held in
// memory on the main thread (the source of the large-file renderer crash).
type CountLinesStreamMessage = {
	type: 'countLinesStream';
	file: File;
	fileName: string;
};

// Compresses a File with CompressionStream inside the worker so the main
// thread never blocks on the CPU-bound work, and the whole file is streamed
// through rather than held in memory at once.
type GzipFileMessage = {
	type: 'gzipFile';
	file: File;
};

type WorkerMessage =
	| ProcessFileMessage
	| CountLinesMessage
	| CountLinesStreamMessage
	| GzipFileMessage;

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
	const msg = event.data;

	try {
		if (msg.type === 'processFile') {
			const result = await processFile(msg);
			self.postMessage({ type: 'result', data: result });
		} else if (msg.type === 'countLines') {
			const count = await countLines(msg.content, msg.fileName);
			self.postMessage({ type: 'result', data: count });
		} else if (msg.type === 'countLinesStream') {
			const count = await countLinesStream(msg.file, msg.fileName);
			self.postMessage({ type: 'result', data: count });
		} else if (msg.type === 'gzipFile') {
			const blob = await gzipFile(msg.file);
			self.postMessage({ type: 'result', data: blob });
		}
	} catch (error: any) {
		self.postMessage({ type: 'error', message: error.message || 'Worker processing failed' });
	}
};

async function processFile(msg: ProcessFileMessage): Promise<Record<string, any>[]> {
	if (msg.fileName.endsWith('.parquet')) {
		const source = msg.file ? fileAsyncBuffer(msg.file) : (msg.content as ArrayBuffer);
		return parseParquet(source, msg.maxLines);
	}
	const text = msg.content as string;
	if (msg.fileName.endsWith('.jsonl')) {
		return parseJsonl(text, msg.maxLines);
	} else if (msg.fileName.endsWith('.csv')) {
		return parseCsv(text, msg.maxLines);
	} else if (msg.fileName.endsWith('.json')) {
		return parseJson(text, msg.maxLines, msg.isChunked);
	}
	throw new Error('Unsupported file type. Please upload a .jsonl, .json, .csv, or .parquet file.');
}

async function countLines(content: string | ArrayBuffer, fileName: string): Promise<number> {
	if (fileName.endsWith('.parquet')) {
		return countParquetRows(content as ArrayBuffer);
	}
	const text = content as string;
	if (fileName.endsWith('.jsonl')) {
		return countJsonlLines(text);
	}
	// For other file types, count all non-empty lines
	return text.split('\n').filter((line: string) => line.trim() !== '').length;
}

// Stream a File chunk-by-chunk inside the worker, counting lines without ever
// materializing the entire file as one string. Carries a partial line across
// chunk boundaries.
async function countLinesStream(file: File, fileName: string): Promise<number> {
	if (fileName.endsWith('.parquet')) {
		return countParquetRows(fileAsyncBuffer(file));
	}

	const isJsonl = fileName.endsWith('.jsonl');
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
				// Skip invalid JSON lines
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
			remainder = lines.pop() ?? '';
			for (const line of lines) tally(line);
		}
		remainder += decoder.decode();
		if (remainder) tally(remainder);
		return count;
	} finally {
		reader.releaseLock();
	}
}

// Streams the file through CompressionStream so the whole content is never
// held in memory at once — the same bounded-memory goal as countLinesStream.
async function gzipFile(file: File): Promise<Blob> {
	const compressed = file.stream().pipeThrough(new CompressionStream('gzip'));
	return await new Response(compressed).blob();
}
