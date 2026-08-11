'use client'

import * as React from 'react'
import { Button, InlineLoading, Loading } from '@carbon/react'
import { CenterSquare, ZoomIn, ZoomFit, ZoomOut } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import type { ElkExtendedEdge } from 'elkjs'
import type { Artifact } from '@granite-build/ui-core/types'
import { getBuild, getBuildStatus, getArtifactLineage } from '@granite-build/ui-core/api/gbserver'
import type { ArtifactRunEntry } from '@granite-build/ui-core/api/gbserver'
import BuildLineagePanel from '../../builds/[buildId]/LineagePanel'
import Graph, { type ElkNodeEx, type GraphHandle, type NodeType } from '@granite-build/ui-core/components/LineageGraph/Graph'

function artifactTypeToNodeType(artifactType: string): NodeType {
  switch (artifactType.toUpperCase()) {
    case 'MODEL':   return 'Model'
    case 'DATASET': return 'Dataset'
    case 'FILESET': return 'Fileset'
    default:        return 'Fileset'
  }
}

function buildLineageGraph(
  artifact: Artifact,
  runs: ArtifactRunEntry[],
): { nodes: ElkNodeEx[]; links: ElkExtendedEdge[] } {
  const nodes: ElkNodeEx[] = []
  const links: ElkExtendedEdge[] = []
  const seenNodes = new Set<string>()
  const seenEdges = new Set<string>()

  const addNode = (node: ElkNodeEx) => {
    if (!seenNodes.has(node.id)) {
      seenNodes.add(node.id)
      nodes.push(node)
    }
  }

  const addEdge = (edge: ElkExtendedEdge) => {
    if (!seenEdges.has(edge.id)) {
      seenEdges.add(edge.id)
      links.push(edge)
    }
  }

  addNode({
    id: artifact.uuid,
    title: artifact.name,
    type: artifactTypeToNodeType(artifact.artifact_type),
    width: 224,
    height: 64,
    labels: [{ text: artifact.name }],
  })

  for (const run of runs) {
    const runId = `run-${run.run_id}`
    addNode({
      id: runId,
      title: run.job_name || run.run_id,
      type: 'Build',
      width: 192,
      height: 64,
      labels: [{ text: run.job_name || run.run_id }],
    })

    for (const ref of run.inputs) {
      const refId = ref.uri ?? `ref-${ref.name}`
      addNode({
        id: refId,
        title: ref.name,
        type: 'Fileset',
        width: 224,
        height: 64,
        labels: [{ text: ref.name }],
      })
      addEdge({ id: `${refId}→${runId}`, sources: [`${refId}-output`], targets: [`${runId}-input`] })
    }

    for (const ref of run.outputs) {
      const refId = ref.uri ?? `ref-${ref.name}`
      addNode({
        id: refId,
        title: ref.name,
        type: 'Fileset',
        width: 224,
        height: 64,
        labels: [{ text: ref.name }],
      })
      addEdge({ id: `${runId}→${refId}`, sources: [`${runId}-output`], targets: [`${refId}-input`] })
    }
  }

  return { nodes, links }
}

// ── HF / URI-based artifact lineage panel ─────────────────────────────────────

function ArtifactLineageGraph({ artifact }: { artifact: Artifact }) {
  const graphRef = React.useRef<GraphHandle>(null)
  const [rendered, setRendered] = React.useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact-lineage', artifact.uri],
    queryFn: () => getArtifactLineage({ artifact_url: artifact.uri, direction: 'downstream', max_depth: 3 }),
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const { nodes, links } = React.useMemo(
    () => (data ? buildLineageGraph(artifact, data.runs) : { nodes: [], links: [] }),
    [artifact, data],
  )

  // The current artifact's own node is always highlighted — not click-driven.
  const currentArtifactNode = React.useMemo(
    () => nodes.find((n) => n.id === artifact.uuid),
    [nodes, artifact.uuid],
  )

  // gbserver returns 404 for every artifact when no lineage provider is
  // configured (the standalone default) — that's an expected "not available"
  // state, not a failure worth alarming the user about.
  const notAvailable = !isLoading && isAxiosError(error) && error.response?.status === 404
  const realError = !isLoading && error && !notAvailable
  const noLineage = !isLoading && !error && nodes.length === 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        height: '2rem',
        background: 'var(--cds-layer-01)',
        borderBottom: '1px solid var(--cds-border-subtle-01)',
        flexShrink: 0,
      }}>
        <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
          iconDescription="Zoom In (+10%)" renderIcon={ZoomIn}
          onClick={() => graphRef.current?.zoomIn()} />
        <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
          iconDescription="Reset Zoom" renderIcon={ZoomFit}
          onClick={() => graphRef.current?.resetZoom()} />
        <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
          iconDescription="Zoom Out (-10%)" renderIcon={ZoomOut}
          onClick={() => graphRef.current?.zoomOut()} />
        <Button size="sm" kind="ghost"
          renderIcon={CenterSquare}
          onClick={() => graphRef.current?.centerOnNode(artifact.uuid)}
        >
          Focus Node
        </Button>
      </div>

      <div style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {isLoading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
            <Loading withOverlay={false} description="Loading lineage…" />
          </div>
        )}
        {realError && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-support-error)', fontSize: '0.875rem', padding: '1rem', textAlign: 'center' }}>
            Failed to load lineage: {String(error)}
          </div>
        )}
        {!isLoading && (notAvailable || noLineage) && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-text-secondary)', fontSize: '0.875rem' }}>
            {notAvailable ? 'Lineage is not available for this artifact.' : 'No lineage data available for this artifact.'}
          </div>
        )}
        {!isLoading && !error && !noLineage && (
          <>
            {!rendered && (
              <InlineLoading
                style={{ position: 'absolute', top: '0.5rem', left: '1rem', width: 'fit-content', background: 'var(--cds-layer-01)', zIndex: 10 }}
                description="Lineage is rendering…"
              />
            )}
            <Graph
              ref={graphRef}
              nodes={nodes}
              links={links}
              allLinks={links}
              selectedNode={currentArtifactNode}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>
    </div>
  )
}

// ── Build-linked lineage panel (existing behavior) ────────────────────────────

function BuildLinkedLineage({ artifact, artifactLoading }: { artifact: Artifact; artifactLoading: boolean }) {
  const buildId = artifact.build_id!

  const { data: build, isLoading: buildLoading } = useQuery({
    queryKey: ['build', buildId],
    queryFn: () => getBuild(buildId),
    staleTime: 5 * 60 * 1000,
  })

  const { data: buildStatus, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ['build-status', buildId],
    queryFn: () => getBuildStatus(buildId),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })

  if (artifactLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <InlineLoading description="Loading lineage…" status="active" />
      </div>
    )
  }

  return (
    <BuildLineagePanel
      build={build}
      buildStatus={buildStatus}
      describe={build}
      loading={buildLoading || statusLoading}
      statusError={statusError as Error | null}
      showFocusNode
      initialFocusNodeId={artifact.uuid}
    />
  )
}

// ── Public component ──────────────────────────────────────────────────────────

export function LineagePanel({ artifact, loading }: { artifact: Artifact | undefined; loading: boolean }) {
  if (loading || !artifact) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <InlineLoading description="Loading lineage…" status="active" />
      </div>
    )
  }

  const isHf = artifact.uri?.startsWith('hf://')

  // HF artifacts: use the artifact lineage API (no build_id on these)
  if (isHf) {
    return <ArtifactLineageGraph artifact={artifact} />
  }

  // Build-produced artifacts: use the build-based lineage graph
  if (artifact.build_id) {
    return <BuildLinkedLineage artifact={artifact} artifactLoading={loading} />
  }

  // Other artifacts with a non-HF URI: try artifact lineage API as best-effort
  if (artifact.uri) {
    return <ArtifactLineageGraph artifact={artifact} />
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-text-secondary)', fontSize: '0.875rem' }}>
      No lineage data available for this artifact.
    </div>
  )
}
