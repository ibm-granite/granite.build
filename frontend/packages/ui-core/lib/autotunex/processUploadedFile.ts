import { parseCsv, parseJson, parseJsonl, parseParquet } from './file-parser'

// Files larger than this are handed to the Web Worker to keep the main thread
// responsive; smaller files are parsed inline via the same file-parser
// functions the worker uses (single implementation, no drift between paths).
const WORKER_SIZE_THRESHOLD = 5 * 1024 * 1024 // 5MB

function readFile(blob: Blob, asArrayBuffer: boolean): Promise<string | ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (event) => resolve(event.target!.result as string | ArrayBuffer)
    reader.onerror = () => reject(new Error('Error reading file.'))
    if (asArrayBuffer) reader.readAsArrayBuffer(blob)
    else reader.readAsText(blob)
  })
}

async function processInline(file: File, maxLines?: number): Promise<Record<string, any>[]> {
  const isParquet = file.name.endsWith('.parquet')
  const chunkSize = maxLines && !isParquet ? 1024 * 1024 * 10 : file.size
  const blob = file.slice(0, Math.min(chunkSize, file.size))

  if (isParquet) {
    const buffer = (await readFile(file, true)) as ArrayBuffer
    return parseParquet(buffer, maxLines)
  }

  const text = (await readFile(blob, false)) as string
  const isChunked = blob.size < file.size

  if (file.name.endsWith('.jsonl')) return parseJsonl(text, maxLines)
  if (file.name.endsWith('.csv')) return parseCsv(text, maxLines)
  if (file.name.endsWith('.json')) return parseJson(text, maxLines, isChunked)
  throw new Error('Unsupported file type. Please upload a .jsonl, .json, .csv, or .parquet file.')
}

function runInWorker<T>(postMessage: (worker: Worker) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    let worker: Worker
    try {
      worker = new Worker(new URL('./file-processor.worker.ts', import.meta.url), { type: 'module' })
    } catch {
      reject(new Error('worker-unavailable'))
      return
    }

    worker.onmessage = (e) => {
      worker.terminate()
      if (e.data.type === 'result') resolve(e.data.data)
      else reject(new Error(e.data.message))
    }
    worker.onerror = (e) => {
      worker.terminate()
      reject(new Error('Worker error: ' + (e.message || 'File processing failed')))
    }

    postMessage(worker)
  })
}

/**
 * Parse an uploaded dataset file into row objects. Uses a Web Worker for
 * files above the size threshold so a large upload never blocks or crashes
 * the renderer; small files parse inline for lower latency.
 */
export async function processUploadedFileAsync(
  file: File,
  maxLines?: number
): Promise<Record<string, any>[]> {
  if (file.size <= WORKER_SIZE_THRESHOLD || typeof Worker === 'undefined') {
    return processInline(file, maxLines)
  }

  const isParquet = file.name.endsWith('.parquet')
  const chunkSize = maxLines && !isParquet ? 1024 * 1024 * 10 : file.size
  const blob = file.slice(0, Math.min(chunkSize, file.size))
  const content = await readFile(blob, isParquet)

  try {
    return await runInWorker<Record<string, any>[]>((worker) => {
      const message = {
        type: 'processFile' as const,
        content,
        fileName: file.name,
        maxLines,
        isChunked: !isParquet && blob.size < file.size,
      }
      if (isParquet && content instanceof ArrayBuffer) worker.postMessage(message, [content])
      else worker.postMessage(message)
    })
  } catch {
    return processInline(file, maxLines)
  }
}

/**
 * Count rows in an uploaded dataset file without materializing the whole
 * file on the main thread for large uploads.
 */
export async function countLinesInFileAsync(file: File): Promise<number> {
  if (file.size <= WORKER_SIZE_THRESHOLD || typeof Worker === 'undefined') {
    const rows = await processInline(file)
    return rows.length
  }

  try {
    return await runInWorker<number>((worker) => {
      worker.postMessage({ type: 'countLinesStream' as const, file, fileName: file.name })
    })
  } catch {
    const rows = await processInline(file)
    return rows.length
  }
}
