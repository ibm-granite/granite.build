'use client'

import { ContentSwitcher, Switch } from '@carbon/react'

// Rows that share a long prefix (a common system prompt) look identical when
// cropped from the start; 'end' shows where they differ.
export type CropMode = 'start' | 'end' | 'full'
const CROP_MODES: CropMode[] = ['start', 'end', 'full']
const CROP_LABELS: Record<CropMode, string> = { start: 'Start', end: 'End', full: 'Full' }
const PREVIEW_CELL_CHARS = 120

/** The PreviewTable crop props for a mode. */
export function previewCropProps(mode: CropMode): { maxCellChars?: number; cropFrom: 'start' | 'end' } {
  return {
    maxCellChars: mode === 'full' ? undefined : PREVIEW_CELL_CHARS,
    cropFrom: mode === 'end' ? 'end' : 'start',
  }
}

export function PreviewCropSwitcher({ value, onChange }: { value: CropMode; onChange: (mode: CropMode) => void }) {
  return (
    <ContentSwitcher
      size="sm"
      selectedIndex={CROP_MODES.indexOf(value)}
      onChange={({ index }) => onChange(CROP_MODES[index ?? 0])}
      style={{ width: '12rem', flexShrink: 0 }}
    >
      {CROP_MODES.map((mode) => (
        <Switch key={mode} name={mode} text={CROP_LABELS[mode]} />
      ))}
    </ContentSwitcher>
  )
}
