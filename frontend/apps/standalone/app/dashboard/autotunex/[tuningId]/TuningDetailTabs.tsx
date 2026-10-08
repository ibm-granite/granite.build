'use client'

import { Tabs, TabList, Tab, TabPanels, TabPanel, FormLabel, InlineNotification, Modal } from '@carbon/react'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getConfiguration } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import type { JobRead } from '@granite-build/ui-core/types'
import { TuningLogViewer } from '@granite-build/ui-core/components/autotunex/tunings/TuningLogViewer'
import { TrialsTable } from '@granite-build/ui-core/components/autotunex/trials/TrialsTable'
import { jobElapsedSeconds } from '@granite-build/ui-core/components/autotunex/trials/trialProgress'
import { formatTime } from '@granite-build/ui-core/components/autotunex/trials/trialsTableFormat'
import { TuningResultsPanel } from '@granite-build/ui-core/components/autotunex/tunings/TuningResultsPanel'
import { ConfigDisplay } from '@granite-build/ui-core/components/autotunex/shared/ConfigDisplay'
import { modelSourceLabel } from '../modelSources'
import { SettingsDatasetView } from '@granite-build/ui-core/components/autotunex/settings/SettingsDatasetView'

function DetailsPanel({ job }: { job: JobRead }) {
  const [configOpen, setConfigOpen] = useState(false)
  const [datasetOpen, setDatasetOpen] = useState(false)

  // Same "admin of at least one space" gate used by the tunings/settings
  // tables — admins get `scope=all` so the config modal can resolve a
  // configuration this viewer doesn't own.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isAdmin = spaces.some((s) => s.is_admin)
  // The scope, not `isAdmin`, keys the config query: AutoTuneXPanel on the build
  // page keys the same configuration on its scope, and two different keys for one
  // configuration meant opening the modal in both places fetched it twice.
  const scope: 'own' | 'all' = isAdmin ? 'all' : 'own'

  const { data: configuration, isError: isConfigError } = useQuery({
    queryKey: ['autotunex-config', job.config_id, scope],
    queryFn: () => getConfiguration(job.config_id, scope),
    // Same guard as AutoTuneXPanel: without it a job with no config_id fetches
    // /configurations/undefined, so the modal shows an error for what is really
    // just an absent id.
    enabled: configOpen && !!job.config_id,
  })

  // The same rule as the Hyperparameters tab's progress summary -- see jobElapsedSeconds.
  const totalTimeSeconds = jobElapsedSeconds(
    { status: job.status, createdAt: job.created_at, updatedAt: job.updated_at, finishedAt: job.finished_at },
    Date.now()
  )

  const fields: { label: string; value: React.ReactNode }[] = [
    { label: 'Build ID', value: job.tasks[0]?.build_id?.split('-')[0] as string },
    { label: 'Model', value: job.model },
    { label: 'Model source', value: modelSourceLabel(job.model_source) },
    {
      label: 'Configuration',
      value: (
        <a href="#" onClick={(e) => { e.preventDefault(); setConfigOpen(true) }}>
          {job.config_name}
        </a>
      ),
    },
    { label: 'Data set',       value: (
        <a href="#" onClick={(e) => { e.preventDefault(); setDatasetOpen(true) }}>
          {job.dataset}
        </a>
      ), },
    { label: 'Total time', value: formatTime(totalTimeSeconds) },
  ]

  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '2rem' }}>
        {fields.map((f) => (
          <div key={f.label} style={{ minWidth: '10rem' }}>
            <FormLabel style={{ marginBottom: '0.5rem' }}>{f.label}</FormLabel>
            <div style={{ fontFamily: 'monospace' }}>{f.value}</div>
          </div>
        ))}
      </div>

      <h5 style={{ marginBottom: '0.5rem' }}>Logs</h5>
      <TuningLogViewer jobId={job.id} status={job.status} scope={isAdmin ? 'all' : 'own'} />

      <Modal
        open={configOpen}
        passiveModal
        modalHeading={`Configuration: ${job.config_name}`}
        onRequestClose={() => setConfigOpen(false)}
        size="lg"
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
          <p>Loading…</p>
        )}
      </Modal>

    <SettingsDatasetView
        open={datasetOpen}
        datasetId={job.dataset_id}
        onClose={() => setDatasetOpen(false)}
        scope={isAdmin ? 'all' : 'own'}
      />
    </div>
  )
}

interface Props {
  job: JobRead
}

export function TuningDetailTabs({ job }: Props) {
  return (
    <div style={{ padding: '0 1.5rem 2rem' }}>
      <Tabs>
        <TabList aria-label="Tuning detail tabs">
          <Tab>Details</Tab>
          {job.autotune && <Tab>Hyperparameters</Tab>}
          <Tab>Results</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <DetailsPanel job={job} />
          </TabPanel>
          {job.autotune && (
            <TabPanel>
              <TrialsTable job={job} />
            </TabPanel>
          )}
          <TabPanel>
            <TuningResultsPanel jobId={job.id} jobStatus={job.status} />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </div>
  )
}
