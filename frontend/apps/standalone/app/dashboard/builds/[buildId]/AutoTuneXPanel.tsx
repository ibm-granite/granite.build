'use client'

import * as React from 'react'
import { useState } from 'react'
import { Link as CarbonLink, Modal, InlineLoading, InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getConfiguration, getJobByBuildId } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import { ConfigDisplay } from '@granite-build/ui-core/components/ConfigDisplay'
import { SettingsDatasetView } from '@granite-build/ui-core/components/SettingsDatasetView'
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
  buildId: string
}

/**
 * Details for the AutoTuneX tuning job linked to a build. Rendered side-by-side
 * with DetailsPanel in the Details tab, only for builds tagged "autotunex".
 * Configuration and Data set names open the existing view modals inline.
 */
export function AutoTuneXPanel({ buildId }: AutoTuneXPanelProps) {
  const [configOpen, setConfigOpen] = useState(false)
  const [datasetOpen, setDatasetOpen] = useState(false)

  // Same "admin of at least one space" gate used by the tunings/settings
  // tables — admins get `scope=all` so this panel can resolve a job linked
  // to the build even when it doesn't belong to the viewer.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isAdmin = spaces.some((s) => s.is_admin)

  const { data: job, isLoading } = useQuery({
    queryKey: ['autotunex-job-by-build', buildId, isAdmin],
    queryFn: () => getJobByBuildId(buildId, isAdmin ? 'all' : 'own'),
  })

  // Loaded lazily when the configuration modal is opened (matches TuningDetailTabs).
  // The job above is fetched with the admin scope; the configuration must use
  // the same one or an admin viewing another user's build gets the job and then
  // a permanently-spinning configuration modal.
  const { data: configuration, isError: isConfigError } = useQuery({
    queryKey: ['autotunex-config', job?.config_id, isAdmin],
    queryFn: () => getConfiguration(job!.config_id, isAdmin ? 'all' : 'own'),
    enabled: configOpen && !!job?.config_id,
  })

  if (isLoading) {
    return (
      <div style={{ padding: '0.5rem 1rem' }}>
        <h5 style={{ marginBottom: '1rem' }}>Model Customization</h5>
        <SkeletonText paragraph lineCount={5} />
      </div>
    )
  }

  // No tuning job linked to this build — render nothing so DetailsPanel fills the row.
  if (!job) return null
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
        scope={isAdmin ? 'all' : 'own'}
      />
    </>
  )
}
