'use client'

import {
  Modal,
  InlineLoading,
  FormLabel,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  InlineNotification,
} from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getDataset } from '../../../api/autotunex'
import type { Dataset } from '../../../types'
import { PreviewTable } from '../shared/PreviewTable'
import { formatBytes } from '../../../lib/autotunex/formatBytes'

interface Props {
  open: boolean
  datasetId: string | null
  onClose: () => void
  /**
   * Scope for the detail fetch. Must match the scope the caller listed with:
   * defaulting to 'own' while the caller lists `scope=all` makes an admin's
   * click on another user's dataset fail.
   */
  scope?: 'own' | 'all'
}

export function SettingsDatasetView({ open, datasetId, onClose, scope = 'own' }: Props) {
  const { data: dataset, isLoading, isError } = useQuery<Dataset>({
    queryKey: ['autotunex-dataset', datasetId, 'preview', scope],
    queryFn: () => getDataset(datasetId as string, { preview: true, previewRows: 50, scope }),
    enabled: open && datasetId != null,
  })

  return (
    <Modal
      open={open}
      passiveModal
      size="lg"
      modalHeading={dataset ? `Dataset: ${dataset.name}` : 'Dataset'}
      onRequestClose={onClose}
    >
      {isError ? (
        <InlineNotification
          kind="error"
          title="Couldn't load this dataset"
          subtitle="It may have been deleted, or you may not have access to it."
          lowContrast
          hideCloseButton
        />
      ) : isLoading || !dataset ? (
        <InlineLoading description="Loading dataset…" />
      ) : (
        <div>
          {dataset.status !== 'ready' && (
            <InlineNotification
              kind={dataset.status === 'error' ? 'error' : 'info'}
              title={dataset.status === 'error' ? 'Dataset processing failed' : 'Dataset is still processing'}
              subtitle={dataset.status_detail ?? (dataset.status === 'error' ? 'Please check the uploaded file and try again.' : 'Preview may be incomplete until processing finishes.')}
              lowContrast
              hideCloseButton
              style={{ marginBottom: '1rem' }}
            />
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '1.5rem' }}>
            {([
              ['Training samples', dataset.train_records?.toLocaleString() ?? '0'],
              ['Validation samples', dataset.validation_records?.toLocaleString() ?? '0'],
              ['Training file size', formatBytes(dataset.train_file_size)],
              ['Validation file size', formatBytes(dataset.validation_file_size)],
              ['Format', dataset.data_format ?? '—'],
              ['Created on', dataset.created_at ? new Date(dataset.created_at).toLocaleString() : '—'],
            ] as [string, string][]).map(([label, value]) => (
              <div key={label} style={{ minWidth: '10rem' }}>
                <FormLabel>{label}</FormLabel>
                <div style={{ fontFamily: 'monospace' }}>{value}</div>
              </div>
            ))}
          </div>

          <Tabs>
            <TabList aria-label="Dataset preview">
              <Tab>Train</Tab>
              <Tab>Validation</Tab>
            </TabList>
            <TabPanels>
              <TabPanel>
                <PreviewTable
                  rows={dataset.preview?.train ?? []}
                  maxRows={50}
                  emptyMessage="No preview rows available."
                />
              </TabPanel>
              <TabPanel>
                <PreviewTable
                  rows={dataset.preview?.validation ?? []}
                  maxRows={50}
                  emptyMessage="No preview rows available."
                />
              </TabPanel>
            </TabPanels>
          </Tabs>
        </div>
      )}
    </Modal>
  )
}
