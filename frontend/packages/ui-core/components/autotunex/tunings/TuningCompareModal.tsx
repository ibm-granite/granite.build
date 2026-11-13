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

'use client'

import { Modal, StructuredListWrapper, StructuredListHead, StructuredListRow, StructuredListCell, StructuredListBody } from '@carbon/react'
import type { TuningJob } from '../../../types'

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
