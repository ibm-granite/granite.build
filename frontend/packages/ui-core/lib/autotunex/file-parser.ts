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

export function parseCsv(content: string, maxLines?: number): ParseResult {
  const lines = content.trim().split('\n')
  if (lines.length === 0) return []

  const headers = parseCsvLine(lines[0])
  const result: ParseResult = []
  const end = maxLines ? Math.min(lines.length, maxLines + 1) : lines.length

  for (let i = 1; i < end; i++) {
    const values = parseCsvLine(lines[i])
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
