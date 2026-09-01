'use client'

import styles from './ArtifactDetails.module.scss'
import {
  Tab,
  TabListVertical,
  TabPanel,
  TabPanels,
  TabsVertical,
} from '@carbon/react'
import type { Artifact } from '@granite-build/ui-core/types'
import { DetailsPanel } from './DetailsPanel'
import { ContentsPanel } from './ContentsPanel'
import { ModelCardPanel } from './ModelCardPanel'
import { LineagePanel } from './LineagePanel'

interface Props {
  artifact: Artifact | undefined
  loading: boolean
  artifactId: string
}

export function ArtifactDetails({ artifact, loading, artifactId }: Props) {
  const type = artifact?.artifact_type

  // Use display:none to keep tab/panel indices aligned (same pattern as build detail)
  const contentsHide = type === 'TABLE'   ? undefined : 'none'
  const modelHide    = type === 'MODEL'   ? undefined : 'none'

  return (
    <div className={styles.tabsWrapper} style={{ height: 'calc(100vh - 220px)', minHeight: '500px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0, minWidth: "100%" }}>
        {/* Keyed on artifactId so navigating to a different artifact (e.g. via
            the lineage graph) remounts the tabs and resets to Details, instead
            of carrying over whatever tab was selected on the previous artifact. */}
        <TabsVertical key={artifactId} height="100%">
          <TabListVertical aria-label="Artifact detail tabs">
            <Tab>Details</Tab>
            <Tab style={{ display: contentsHide }}>Contents</Tab>
            <Tab style={{ display: modelHide }}>Model Card</Tab>
            <Tab>Lineage</Tab>
          </TabListVertical>
          <TabPanels>
            <TabPanel style={{ overflowY: 'auto', height: '100%' }}>
              <DetailsPanel artifact={artifact} loading={loading} />
            </TabPanel>
            <TabPanel style={{ display: contentsHide, overflowY: 'auto', height: '100%' }}>
              {type === 'TABLE' && <ContentsPanel artifactId={artifactId} />}
            </TabPanel>
            <TabPanel style={{ display: modelHide, overflowY: 'auto', height: '100%' }}>
              {type === 'MODEL' && <ModelCardPanel artifactId={artifactId} />}
            </TabPanel>
            <TabPanel style={{ overflow: 'hidden', height: '100%', padding: 0 }}>
              <LineagePanel artifact={artifact} loading={loading} />
            </TabPanel>
          </TabPanels>
        </TabsVertical>
      </div>
    </div>
  )
}
