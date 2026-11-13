// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

export interface PreviewTableHeader {
  key: string
  header: string
}

/**
 * Rendered text for one preview cell. A null/undefined value renders empty --
 * this needs saying because typeof null === 'object', which would otherwise
 * route it into JSON.stringify and print the literal text "null".
 * cropFrom 'end' keeps the last maxCellChars characters instead of the first:
 * rows sharing a long prefix (a common system prompt) differ only at the end.
 */
export function previewCellText(value: unknown, maxCellChars?: number, cropFrom: 'start' | 'end' = 'start'): string {
  const text = value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value)
  if (maxCellChars == null || text.length <= maxCellChars) return text
  return cropFrom === 'end' ? `...${text.slice(-maxCellChars)}` : `${text.slice(0, maxCellChars)}...`
}

/**
 * Column headers derived from the union of keys across every row -- not row
 * 0's keys, which drops a column absent there on ragged data.
 */
export function derivePreviewHeaders(rows: Record<string, any>[]): PreviewTableHeader[] {
  const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
  return keys.map((key) => ({ key, header: key }))
}
