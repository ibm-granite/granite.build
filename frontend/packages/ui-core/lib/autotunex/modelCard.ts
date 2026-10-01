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
