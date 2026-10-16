'use client'

import * as React from 'react'
import { useState } from 'react'
import { Link as CarbonLink, Modal, InlineLoading, InlineNotification } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getConfiguration } from '@granite-build/ui-core/api/autotunex'
import { ConfigDisplay } from '@granite-build/ui-core/components/autotunex/shared/ConfigDisplay'
import { SettingsDatasetView } from '@granite-build/ui-core/components/autotunex/settings/SettingsDatasetView'
import type { JobDetail } from '@granite-build/ui-core/types'
import styles from './DetailsPanel.module.scss'

interface DetailFieldProps {
  label: string
  column: 1 | 2
  row: number
  children: React.ReactNode
}

function DetailField({ label, column, row, children }: DetailFieldProps) {
  return (
    <div className={column === 1 ? styles.col1 : styles.col2} style={{ gridRow: row }}>
      <div className={styles.fieldLabel}>{label}</div>
      <div className={styles.fieldValue}>{children}</div>
    </div>
  )
}

interface AutoTuneXPanelProps {
  job: JobDetail
  scope: 'own' | 'all'
}

/**
 * Details for the AutoTuneX tuning job linked to a build. Rendered side-by-side
 * with DetailsPanel in the Details tab, only for builds that have a linked tuning
 * job. Configuration and Data set names open the existing view modals inline.
 *
 * BuildDetails owns the linked-job lookup (see useLinkedTuningJob) and only mounts
 * this once it holds a job, so there is no loading, error or empty state here.
 */
export function AutoTuneXPanel({ job, scope }: AutoTuneXPanelProps) {
  const [configOpen, setConfigOpen] = useState(false)
  const [datasetOpen, setDatasetOpen] = useState(false)

  // Loaded lazily when the configuration modal is opened (matches TuningDetailTabs).
  // `scope` is the one the job was resolved under and must be reused: an admin
  // viewing another user's build otherwise gets the job and then a permanently
  // spinning configuration modal.
  const { data: configuration, isError: isConfigError } = useQuery({
    queryKey: ['autotunex-config', job.config_id, scope],
    queryFn: () => getConfiguration(job.config_id, scope),
    enabled: configOpen && !!job.config_id,
  })

  return (
    <>
      <DetailField label="Experiment name" column={2} row={1}>
        <span className={styles.wordBreakAll} style={{lineHeight:'2rem'}}>{job.experiment_name}</span>
      </DetailField>
      <DetailField label="Model" column={2} row={2}>
        <span className={styles.wordBreakAll}>{job.model}</span>
      </DetailField>
      <DetailField label="Tuning type" column={2} row={3}>{job.tuning_type}</DetailField>
      <DetailField label="Configuration" column={2} row={4}>
        <CarbonLink href="#" onClick={(e) => { e.preventDefault(); setConfigOpen(true) }}>
          {job.config_name}
        </CarbonLink>
      </DetailField>
      <DetailField label="Data set" column={2} row={5}>
        <CarbonLink href="#" onClick={(e) => { e.preventDefault(); setDatasetOpen(true) }}>
          {job.dataset}
        </CarbonLink>
      </DetailField>

      <Modal
        open={configOpen}
        passiveModal
        modalHeading={`Configuration: ${job.config_name}`}
        size="lg"
        onRequestClose={() => setConfigOpen(false)}
      >
        {isConfigError ? (
          <InlineNotification
            kind="error"
            title="Couldn't load this configuration"
            subtitle="It may have been deleted, or you may not have access to it."
            lowContrast
            hideCloseButton
          />
        ) : configuration ? (
          <ConfigDisplay configuration={configuration} />
        ) : (
          <InlineLoading description="Loading configuration…" />
        )}
      </Modal>

      <SettingsDatasetView
        open={datasetOpen}
        datasetId={job.dataset_id}
        onClose={() => setDatasetOpen(false)}
        scope={scope}
      />
    </>
  )
}
