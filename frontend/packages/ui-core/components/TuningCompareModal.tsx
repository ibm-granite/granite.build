'use client'

import { Modal, StructuredListWrapper, StructuredListHead, StructuredListRow, StructuredListCell, StructuredListBody } from '@carbon/react'
import type { TuningJob } from '../types'

interface Props {
  open: boolean
  jobs: TuningJob[]
  onClose: () => void
}

const COMPARE_KEYS: { key: keyof TuningJob; label: string }[] = [
  { key: 'status', label: 'Status' },
  { key: 'model', label: 'Model' },
  { key: 'config_name', label: 'Configuration' },
  { key: 'dataset', label: 'Data set' },
  { key: 'seed', label: 'Seed' },
  // 'precision' is no longer part of the job LIST payload (JobSummary) in the
  // v0.3.5 API — it is detail-only. Dropped from the compare table rather than
  // issuing a per-job detail fetch. Restore via getJob() if needed later.
  { key: 'created_at', label: 'Created on' },
]

export function TuningCompareModal({ open, jobs, onClose }: Props) {
  return (
    <Modal
      open={open}
      passiveModal
      modalHeading="Compare tunings"
      onRequestClose={onClose}
      size="lg"
    >
      {jobs.length > 0 && (
        <StructuredListWrapper>
          <StructuredListHead>
            <StructuredListRow head>
              <StructuredListCell head />
              {jobs.map((j) => (
                <StructuredListCell head key={j.id}>{j.experiment_name}</StructuredListCell>
              ))}
            </StructuredListRow>
          </StructuredListHead>
          <StructuredListBody>
            {COMPARE_KEYS.map(({ key, label }) => (
              <StructuredListRow key={key}>
                <StructuredListCell><strong>{label}</strong></StructuredListCell>
                {jobs.map((j) => (
                  <StructuredListCell key={j.id}>
                    {key === 'created_at' ? new Date(j[key] as string).toLocaleString() : String(j[key])}
                  </StructuredListCell>
                ))}
              </StructuredListRow>
            ))}
          </StructuredListBody>
        </StructuredListWrapper>
      )}
    </Modal>
  )
}
