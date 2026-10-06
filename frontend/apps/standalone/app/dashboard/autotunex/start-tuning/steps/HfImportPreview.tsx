'use client'

import { useState } from 'react'
import type { HfImportPreview as HfPreview } from '@granite-build/ui-core/types'
import { PreviewTable } from '@granite-build/ui-core/components/autotunex/shared/PreviewTable'
import { PreviewCropSwitcher, previewCropProps, type CropMode } from './PreviewCropSwitcher'

const PREVIEW_ROWS = 10

interface HfImportPreviewProps {
  preview: HfPreview | null
}

export function HfImportPreview({ preview }: HfImportPreviewProps) {
  const [cropMode, setCropMode] = useState<CropMode>('start')

  if (!preview) return null

  // `sampled` is the number of rows the server read -- no endpoint reports the
  // split's total.
  return (
    <>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
          <h6 style={{ fontWeight: 600, margin: 0 }}>Data Preview</h6>
          <span style={{ fontSize: '0.8125rem', color: 'var(--cds-text-secondary, #525252)' }}>
            {Math.min(preview.raw_rows.length, PREVIEW_ROWS)} of{' '}
            {preview.sampled.toLocaleString()} sampled rows
          </span>
        </div>
        <PreviewCropSwitcher value={cropMode} onChange={setCropMode} />
      </div>
      <PreviewTable
        rows={preview.raw_rows}
        maxRows={PREVIEW_ROWS}
        {...previewCropProps(cropMode)}
        emptyMessage="This split returned no rows."
      />
    </>
  )
}
