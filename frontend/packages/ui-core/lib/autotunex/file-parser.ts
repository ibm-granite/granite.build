// Shared file-parsing logic used by both the main thread (small files) and the
// Web Worker (large files) — a single implementation avoids the two paths
// silently drifting apart on edge cases.
import { parquetReadObjects } from 'hyparquet'

export type ParseResult = Record<string, any>[]

export function parseJsonl(content: string, maxLines?: number): ParseResult {
  const lines = content.split('\n').filter((line) => line.trim() !== '')
  const linesToProcess = maxLines ? lines.slice(0, maxLines) : lines

  const jsonData = linesToProcess
    .map((line) => {
      try {
        return JSON.parse(line)
      } catch {
        return null
      }
    })
    .filter((item) => item !== null)

  if (jsonData.length === 0) {
    throw new Error(
      'No valid JSON objects found in file. Please check that your file is in JSONL format (one JSON object per line).'
    )
  }

  return jsonData
}

export function parseCsvLine(line: string): string[] {
  const result: string[] = []
  let current = ''
  let inQuotes = false
  let i = 0

  while (i < line.length) {
    const char = line[i]
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"'
        i += 2
      } else {
        inQuotes = !inQuotes
        i++
      }
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim())
      current = ''
      i++
    } else {
      current += char
      i++
    }
  }
  result.push(current.trim())
  return result
}

// A CSV record can span several physical lines, because a quoted field may
// contain newlines. Record boundaries therefore have to be found with quote
// state carried across those lines: splitting the file on '\n' first shears
// such a record apart and misaligns every following field against the headers.
//
// Quote state is just the parity of the quote count, so a doubled ("") escape
// needs no lookahead — it flips parity twice and lands back inside the field.
// That also makes the scanner safe to feed arbitrary chunks, where a lookahead
// could fall off the end of one.
export type CsvScanState = { inQuotes: boolean; partial: string }

export function createCsvScanState(): CsvScanState {
  return { inQuotes: false, partial: '' }
}

/**
 * Feed one chunk of CSV text through the scanner, invoking `onRecord` for each
 * complete record. Quote state and the trailing incomplete record live in
 * `state`, so a caller streaming a file resumes by passing the same state to
 * the next chunk; after the final chunk a non-empty `state.partial` is the last
 * unterminated record.
 */
export function scanCsvChunk(
  chunk: string,
  state: CsvScanState,
  onRecord: (record: string) => void
): void {
  let start = 0
  for (let i = 0; i < chunk.length; i++) {
    const char = chunk[i]
    if (char === '"') {
      state.inQuotes = !state.inQuotes
    } else if (char === '\n' && !state.inQuotes) {
      onRecord(state.partial + chunk.slice(start, i))
      state.partial = ''
      start = i + 1
    }
  }
  state.partial += chunk.slice(start)
}

export function splitCsvRecords(content: string): string[] {
  const state = createCsvScanState()
  const records: string[] = []
  scanCsvChunk(content, state, (record) => records.push(record))
  if (state.partial !== '') records.push(state.partial)
  return records
}

export function parseCsv(content: string, maxLines?: number): ParseResult {
  const records = splitCsvRecords(content.trim())
  if (records.length === 0) return []

  const headers = parseCsvLine(records[0])
  const result: ParseResult = []
  const end = maxLines ? Math.min(records.length, maxLines + 1) : records.length

  for (let i = 1; i < end; i++) {
    const values = parseCsvLine(records[i])
    const obj: Record<string, any> = {}
    headers.forEach((header, index) => {
      obj[header] = values[index] || ''
    })
    result.push(obj)
  }

  return result
}

export function parseJson(content: string, maxLines?: number, isChunked = false): ParseResult {
  let trimmedContent = content

  if (isChunked) {
    const lastCompleteObject = content.lastIndexOf('},')
    if (lastCompleteObject > 0) {
      trimmedContent = content.substring(0, lastCompleteObject + 1)
      if (!trimmedContent.trim().endsWith(']')) trimmedContent += '\n]'
    }
  }

  let parsedData
  try {
    parsedData = JSON.parse(trimmedContent)
  } catch {
    let fixedContent = trimmedContent.trim()
    if (fixedContent.endsWith(',')) {
      fixedContent = fixedContent.substring(0, fixedContent.length - 1) + ']'
    } else if (!fixedContent.endsWith(']')) {
      fixedContent += ']'
    }

    try {
      parsedData = JSON.parse(fixedContent)
    } catch {
      const lastOpenBrace = fixedContent.lastIndexOf('{')
      const lastCloseBrace = fixedContent.lastIndexOf('}')

      if (lastOpenBrace > lastCloseBrace) {
        fixedContent = fixedContent.substring(0, lastOpenBrace)
        if (fixedContent.trim().endsWith(',')) {
          fixedContent = fixedContent.substring(0, fixedContent.lastIndexOf(','))
        }
        fixedContent += ']'
        parsedData = JSON.parse(fixedContent)
      } else {
        throw new Error('Error parsing JSON file.')
      }
    }
  }

  if (Array.isArray(parsedData)) {
    let jsonData = parsedData.filter((item) => item && typeof item === 'object' && !Array.isArray(item))
    if (maxLines && jsonData.length > maxLines) jsonData = jsonData.slice(0, maxLines)
    if (jsonData.length === 0) throw new Error('No valid entries found in JSON file.')
    return jsonData
  }

  return [parsedData]
}

export function countJsonlLines(content: string): number {
  const lines = content.split('\n').filter((line) => line.trim() !== '')
  let validCount = 0
  for (const line of lines) {
    try {
      JSON.parse(line)
      validCount++
    } catch {
      // Skip invalid lines
    }
  }
  return validCount
}

function convertBigInts(value: any): any {
  if (typeof value === 'bigint') return Number(value)
  if (Array.isArray(value)) return value.map(convertBigInts)
  if (value !== null && typeof value === 'object') {
    const result: Record<string, any> = {}
    for (const [k, v] of Object.entries(value)) result[k] = convertBigInts(v)
    return result
  }
  return value
}

export async function parseParquet(buffer: ArrayBuffer, maxLines?: number): Promise<ParseResult> {
  const rows = await parquetReadObjects({ file: buffer, rowEnd: maxLines })

  if (rows.length === 0) throw new Error('No records found in Parquet file.')

  // Deep-convert BigInt values to Number (parquet INT64 columns produce BigInt)
  return rows.map((row) => convertBigInts(row) as Record<string, any>)
}

export function countParquetRows(buffer: ArrayBuffer): Promise<number> {
  return parquetReadObjects({ file: buffer }).then((rows) => rows.length)
}

/**
 * Count dataset records by streaming `file` in chunks, so a large upload is
 * never materialized as one string. Used by the Web Worker for big files and by
 * the main thread when the worker is unavailable — one implementation, so the
 * small-file and large-file paths cannot report different totals.
 *
 * Counts the unit the parsers return: CSV data records excluding the header row
 * (honouring newlines inside quoted fields), JSONL objects that actually parse,
 * JSON array entries, Parquet rows.
 */
export async function countRecordsInBlob(file: Blob, fileName: string): Promise<number> {
  if (fileName.endsWith('.parquet')) return countParquetRows(await file.arrayBuffer())
  // A JSON array's entry count has no relationship to its line structure (a
  // pretty-printed file spans many lines per entry), so this one format must be
  // parsed to be counted; nothing incremental is possible without a streaming
  // JSON parser.
  if (fileName.endsWith('.json')) return parseJson(await file.text()).length

  const isCsv = fileName.endsWith('.csv')
  const isJsonl = fileName.endsWith('.jsonl')
  const decoder = new TextDecoder('utf-8')
  const reader = file.stream().getReader()
  const csvState = createCsvScanState()
  let remainder = ''
  let count = 0

  const tally = (record: string) => {
    if (record.trim() === '') return
    if (isJsonl) {
      try {
        JSON.parse(record)
      } catch {
        return
      }
    }
    count++
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      const text = decoder.decode(value, { stream: true })
      // Only CSV needs quote-aware boundaries. Applying the CSV scanner to
      // JSONL would be wrong: a quote inside a JSON string would toggle quote
      // state, swallow the following newline and merge two records into one.
      if (isCsv) {
        scanCsvChunk(text, csvState, tally)
      } else {
        const lines = (remainder + text).split('\n')
        remainder = lines.pop() ?? ''
        for (const line of lines) tally(line)
      }
    }

    const tail = decoder.decode()
    if (isCsv) {
      scanCsvChunk(tail, csvState, tally)
      if (csvState.partial !== '') tally(csvState.partial)
      // The first record is the header row, not data. Guarded so an empty file
      // reports 0 rather than -1.
      return count > 0 ? count - 1 : 0
    }
    remainder += tail
    if (remainder) tally(remainder)
    return count
  } finally {
    reader.releaseLock()
  }
}
