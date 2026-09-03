'use client'

import { Tabs, TabList, Tab, TabPanels, TabPanel, FormLabel, Modal } from '@carbon/react'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getConfiguration } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import type { JobRead } from '@granite-build/ui-core/types'
import { TuningLogViewer } from '@granite-build/ui-core/components/TuningLogViewer'
import { TrialsTable } from '@granite-build/ui-core/components/TrialsTable'
import { TuningResultsPanel } from '@granite-build/ui-core/components/TuningResultsPanel'
import { ConfigDisplay } from '@granite-build/ui-core/components/ConfigDisplay'
import { modelSourceLabel } from '../modelSources'
import { SettingsDatasetView } from '@granite-build/ui-core/components/SettingsDatasetView'

function formatTime(seconds: number): string {
  if (seconds <= 0) return '0 s'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${mins}m`
  if (mins > 0) return `${mins}m ${secs}s`
  return `${secs}s`
}

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

  const { data: configuration } = useQuery({
    queryKey: ['autotunex-config', job.config_id, isAdmin],
    queryFn: () => getConfiguration(job.config_id, isAdmin ? 'all' : 'own'),
    enabled: configOpen,
  })

  const totalTimeSeconds = Math.floor(
    ((job.status === 'running' ? Date.now() : new Date(job.updated_at).getTime()) - new Date(job.created_at).getTime()) / 1000
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
      <TuningLogViewer jobId={job.id} status={job.status} />

      <Modal
        open={configOpen}
        passiveModal
        modalHeading={`Configuration: ${job.config_name}`}
        onRequestClose={() => setConfigOpen(false)}
        size="lg"
      >
        {configuration ? <ConfigDisplay configuration={configuration} /> : <p>Loading…</p>}
      </Modal>

    <SettingsDatasetView
        open={datasetOpen}
        datasetId={job.dataset_id}
        onClose={() => setDatasetOpen(false)}
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
