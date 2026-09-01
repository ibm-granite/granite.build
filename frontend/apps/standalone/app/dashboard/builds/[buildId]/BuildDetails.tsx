'use client'

import { useMemo } from 'react'
import styles from './BuildDetails.module.scss'
import {
  Tab,
  TabListVertical,
  TabPanel,
  TabPanels,
  TabsVertical,
} from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { parse as parseYaml } from 'yaml'
import { getBuildArchiveFiles } from '@granite-build/ui-core/api/gbserver'
import type { Build, BuildEvent, BuildStatusDetail, BuildTargetRun } from '@granite-build/ui-core/types'
import { DetailsPanel } from './DetailsPanel'
import { LogsPanel } from './LogsPanel'
import { TargetsPanel } from './TargetsPanel'
import { HistoryPanel } from './HistoryPanel'
import { DefinitionPanel } from './DefinitionPanel'
import { AIAnalysisPanel } from '@granite-build/ui-core/components/AIAnalysisPanel'
import LineagePanel from './LineagePanel'

interface BuildDetailsProps {
  build: Build | undefined
  status: BuildStatusDetail | undefined
  describe: Build | undefined
  events: BuildEvent[]
  loadingBuild: boolean
  loadingStatus: boolean
  statusError?: Error | null
  buildId: string
}

const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending'])

export function BuildDetails({
  build,
  status,
  describe,
  events,
  loadingBuild,
  loadingStatus,
  statusError,
  buildId,
}: BuildDetailsProps) {
  const hasLogs = build?.status === 'running'
  const logsHide = hasLogs ? undefined : 'none'
  const aiAnalysisHide = hasLogs ? 'none' : undefined
  const isActive = ACTIVE_STATUSES.has(build?.status ?? '')


  // Fetch build archive to extract planned (not-yet-run) targets from the definition
  const { data: archiveFiles } = useQuery({
    queryKey: ['build-archive', buildId],
    queryFn: () => getBuildArchiveFiles(buildId),
    enabled: isActive,
    staleTime: 60_000,
  })

  // Merge actual run targets with planned targets derived from the definition YAML
  const mergedTargets = useMemo<Record<string, BuildTargetRun> | BuildTargetRun[] | undefined>(() => {
    const actual = status?.targets ?? {}
    const yaml = archiveFiles
      ? archiveFiles['build.yaml'] ??
        archiveFiles[Object.keys(archiveFiles).find((k) => k.endsWith('.yaml') || k.endsWith('.yml')) ?? '']
      : null

    let definedNames: string[] = []
    if (yaml) {
      try {
        const def = parseYaml(yaml) as { targets?: Record<string, unknown> }
        if (def?.targets) definedNames = Object.keys(def.targets)
      } catch { /* ignore parse errors */ }
    }

    if (!definedNames.length) return Object.keys(actual).length ? actual : describe?.targets

    const result: Record<string, BuildTargetRun> = { ...actual }
    for (const name of definedNames) {
      if (!result[name]) {
        result[name] = { target_name: name, status: 'planned', steps: [], inputs: {}, outputs: {} }
      }
    }
    return result
  }, [status?.targets, archiveFiles, describe?.targets])

  return (
    <div className={styles.tabsWrapper} style={{ height: 'calc(100vh - 220px)', minHeight: '500px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Vertical tabs */}
      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0 }}>
        <TabsVertical height="100%">
          <TabListVertical aria-label="Build detail tabs">
            <Tab>Details</Tab>
            <Tab style={{ display: logsHide }}>Logs</Tab>
            <Tab>History</Tab>
            <Tab>Definition</Tab>
            <Tab style={{ display: aiAnalysisHide }}>AI Analysis</Tab>
            <Tab>Lineage</Tab>
          </TabListVertical>
          <TabPanels>
            <TabPanel style={{ overflowY: 'auto', height: '100%' }}>
              <DetailsPanel build={build} status={status} loading={loadingBuild} />
              <div style={{ borderTop: '1px solid var(--cds-border-subtle-01)', margin: '2rem 1rem' }} />
              <TargetsPanel targets={mergedTargets} />
            </TabPanel>
            <TabPanel style={{ display: logsHide, overflow: 'hidden', height: '100%', padding: 0 }}>
              <LogsPanel buildId={buildId} status={status} />
            </TabPanel>
            <TabPanel style={{ overflowY: 'auto', height: '100%' }}>
              <HistoryPanel events={events} />
            </TabPanel>
            <TabPanel style={{ padding: 0, height: '100%', overflow: 'hidden' }}>
              <DefinitionPanel buildId={buildId} />
            </TabPanel>
            <TabPanel style={{ display: aiAnalysisHide, overflowY: 'auto', height: '100%' }}>
              <AIAnalysisPanel buildId={buildId} failureReason={build?.failure_reason} />
            </TabPanel>
            <TabPanel style={{ padding: 0, height: '100%' }}>
              <LineagePanel
                build={build}
                buildStatus={status}
                describe={describe}
                loading={loadingBuild || loadingStatus}
                statusError={statusError}
              />
            </TabPanel>
          </TabPanels>
        </TabsVertical>
      </div>
    </div>
  )
}
