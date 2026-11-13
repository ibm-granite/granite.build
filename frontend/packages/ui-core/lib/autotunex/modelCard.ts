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

/**
 * Strips YAML front matter from a HuggingFace README so it does not render as
 * visible text.
 *
 * Only a block at the very top of the document counts. Treating every `---` line
 * as a delimiter and toggling on each one made the body between a README's
 * thematic breaks invisible -- and HF model cards use `---` freely as a section
 * rule, so a typical card silently lost everything between its third and fourth
 * one.
 *
 * An opener with no closing `---` is not front matter: the document is returned
 * whole rather than discarded entirely.
 */
export function stripFrontMatter(raw: string): string {
  const lines = raw.split('\n')

  let open = 0
  while (open < lines.length && lines[open].trim() === '') open++

  if (lines[open]?.trim() === '---') {
    const close = lines.findIndex((line, i) => i > open && line.trim() === '---')
    if (close !== -1) return lines.slice(close + 1).join('\n').trim()
  }

  return lines.join('\n').trim()
}
