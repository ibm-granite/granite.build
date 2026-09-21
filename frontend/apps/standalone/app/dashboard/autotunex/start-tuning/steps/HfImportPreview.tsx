'use client'

import { useState } from 'react'
import { Tab, TabList, TabPanel, TabPanels, Tabs } from '@carbon/react'
import type { HfImportPreview as HfPreview } from '@granite-build/ui-core/types'
import { PreviewTable } from '@granite-build/ui-core/components/autotunex/shared/PreviewTable'

const PREVIEW_ROWS = 10
const CELL_MAX = 120

interface HfImportPreviewProps {
  preview: HfPreview | null
  /**
   * The mapped preview, already checked for staleness by the hook's
   * `freshMappedPreview`. Null until every required column has a source, which is
   * exactly when the tab strip should not exist.
   */
  mappedPreview: HfPreview | null
}

export function HfImportPreview({ preview, mappedPreview }: HfImportPreviewProps) {
  // Raw is index 0 and stays selected when the strip appears -- switching to the
  // remapped view is the user's choice. Deliberately not reset when
  // `mappedPreview` arrives or changes.
  const [selectedIndex, setSelectedIndex] = useState(0)

  if (!preview) return null

  const rawTable = (
    <PreviewTable
      rows={preview.raw_rows}
      maxRows={PREVIEW_ROWS}
      maxCellChars={CELL_MAX}
      emptyMessage="This split returned no rows."
    />
  )

  // No mapped preview yet: one table, no tab strip. `sampled` is the number of rows
  // the server read -- no endpoint reports the split's total.
  if (!mappedPreview) {
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
          <h6 style={{ fontWeight: 600, margin: 0 }}>Sample rows</h6>
          <span style={{ fontSize: '0.8125rem', color: 'var(--cds-text-secondary, #525252)' }}>
            {Math.min(preview.raw_rows.length, PREVIEW_ROWS)} of{' '}
            {preview.sampled.toLocaleString()} sampled rows
          </span>
        </div>
        {rawTable}
      </>
    )
  }

  return (
    <Tabs
      selectedIndex={selectedIndex}
      onChange={({ selectedIndex: next }) => setSelectedIndex(next)}
    >
      <TabList aria-label="HuggingFace dataset preview">
        <Tab>Raw</Tab>
        <Tab>Remapped</Tab>
      </TabList>
      <TabPanels>
        <TabPanel style={{ padding: '0.5rem 0' }}>{rawTable}</TabPanel>
        <TabPanel style={{ padding: '0.5rem 0' }}>
          <PreviewTable
            rows={mappedPreview.mapped_rows}
            maxRows={PREVIEW_ROWS}
            maxCellChars={CELL_MAX}
            emptyMessage="This mapping kept no rows."
          />
        </TabPanel>
      </TabPanels>
    </Tabs>
  )
}
