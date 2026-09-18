export interface PreviewTableHeader {
  key: string
  header: string
}

/**
 * Rendered text for one preview cell. A null/undefined value renders empty --
 * this needs saying because typeof null === 'object', which would otherwise
 * route it into JSON.stringify and print the literal text "null".
 */
export function previewCellText(value: unknown, maxCellChars?: number): string {
  const text = value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value)
  return maxCellChars != null && text.length > maxCellChars ? `${text.slice(0, maxCellChars)}...` : text
}

/**
 * Column headers derived from the union of keys across every row -- not row
 * 0's keys, which drops a column absent there on ragged data.
 */
export function derivePreviewHeaders(rows: Record<string, any>[]): PreviewTableHeader[] {
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
  return keys.map((key) => ({ key, header: key }))
}
