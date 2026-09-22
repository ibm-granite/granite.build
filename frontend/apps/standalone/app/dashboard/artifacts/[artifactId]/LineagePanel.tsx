'use client'

import * as React from 'react'
import { Button, InlineLoading, Loading, Modal } from '@carbon/react'
import { ArrowLeft, ArrowRight, CenterSquare, ZoomIn, ZoomFit, ZoomOut } from '@carbon/icons-react'
import { useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { isAxiosError } from 'axios'
import type { ElkExtendedEdge } from 'elkjs'
import type { Artifact } from '@granite-build/ui-core/types'
import { getBuild, getBuildStatus, getArtifactLineage } from '@granite-build/ui-core/api/gbserver'
import type { ArtifactRunEntry } from '@granite-build/ui-core/api/gbserver'
import BuildLineagePanel from '../../builds/[buildId]/LineagePanel'
import Graph, { type ElkNodeEx, type GraphHandle } from '@granite-build/ui-core/components/LineageGraph/Graph'
import { buildArtifactLineageGraph } from '@granite-build/ui-core/components/LineageGraph/artifactLineageGraph'
import { getSubgraph } from '@granite-build/ui-core/components/LineageGraph/diagramUtilities'
import { ArtifactSummary } from '@granite-build/ui-core/components/ArtifactSummary'
import { useRoutes } from '@granite-build/ui-core/config/routes'

// ── HF / URI-based artifact lineage panel ─────────────────────────────────────

function ArtifactLineageGraph({ artifact }: { artifact: Artifact }) {
  const graphRef = React.useRef<GraphHandle>(null)
  const router = useRouter()
  const routes = useRoutes()
  const [rendered, setRendered] = React.useState(false)

  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact-lineage', artifact.uri],
    queryFn: () => getArtifactLineage({ artifact_url: artifact.uri, direction: 'both', max_depth: 3 }),
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const { nodes, links, meta, focalNodeId } = React.useMemo(
    () =>
      data
        ? buildArtifactLineageGraph(artifact, data.runs)
        : { nodes: [], links: [], meta: new Map(), focalNodeId: artifact.uuid },
    [artifact, data],
  )

  // The current artifact's own node is always highlighted — not click-driven.
  const currentArtifactNode = React.useMemo(
    () => nodes.find((n) => n.id === focalNodeId),
    [nodes, focalNodeId],
  )

  // Upstream/downstream trimming, matching the build lineage tab. A wide graph
  // (the review's example fans out to dozens of eval results) is unreadable
  // whole, and there was previously no way to narrow it here at all.
  const [focusNodeId, setFocusNodeId] = React.useState<string | null>(null)
  const [upstreamLevels, setUpstreamLevels] = React.useState(Infinity)
  const [downstreamLevels, setDownstreamLevels] = React.useState(Infinity)
  const [partial, setPartial] = React.useState(false)

  const { shownNodes, shownLinks } = React.useMemo(() => {
    if (!focusNodeId || (upstreamLevels === Infinity && downstreamLevels === Infinity)) {
      return { shownNodes: nodes, shownLinks: links }
    }
    const sub = getSubgraph(focusNodeId, downstreamLevels, upstreamLevels, nodes, links)
    return { shownNodes: sub.nodes, shownLinks: sub.links }
  }, [focusNodeId, upstreamLevels, downstreamLevels, nodes, links])

  const stepUpstream = () => {
    if (!focusNodeId) return
    const next = upstreamLevels === Infinity ? 2 : upstreamLevels + 1
    const sub = getSubgraph(focusNodeId, downstreamLevels, next, nodes, links)
    setUpstreamLevels(sub.hasMoreUpstream ? next : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  const stepDownstream = () => {
    if (!focusNodeId) return
    const next = downstreamLevels === Infinity ? 2 : downstreamLevels + 1
    const sub = getSubgraph(focusNodeId, next, upstreamLevels, nodes, links)
    setDownstreamLevels(sub.hasMoreDownstream ? next : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  // Clicking a node did nothing at all here, unlike the build lineage tab —
  // the Graph was simply never given an onClick. Run nodes are not artifacts,
  // so they select for trimming but open no dialog.
  const [detailNode, setDetailNode] = React.useState<ElkNodeEx | null>(null)
  const detail = detailNode ? meta.get(detailNode.id) : undefined

  const handleNodeClick = (node: ElkNodeEx) => {
    setFocusNodeId(node.id)
    if (node.type !== 'Build') setDetailNode(node)
  }

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
          onClick={() => graphRef.current?.centerOnNode(focalNodeId)}
        >
          Focus Node
        </Button>
        <Button size="sm" kind="ghost" renderIcon={ArrowLeft}
          disabled={!focusNodeId} onClick={stepUpstream}
          iconDescription="One level upstream">
          Upstream
        </Button>
        <Button size="sm" kind="ghost" renderIcon={ArrowRight}
          disabled={!focusNodeId} onClick={stepDownstream}
          iconDescription="One level downstream">
          Downstream
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
              nodes={shownNodes}
              links={shownLinks}
              allLinks={links}
              selectedNode={currentArtifactNode}
              onClick={handleNodeClick}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {partial && (
        <div style={{
          padding: '0.5rem 1rem',
          fontSize: '0.75rem',
          color: 'var(--cds-text-secondary)',
          borderTop: '1px solid var(--cds-border-subtle-01)',
        }}>
          Showing part of the graph. Use Upstream or Downstream to widen it.
        </div>
      )}

      {/* Clicking a node used to do nothing here. Two footer buttons, so Carbon
          sizes them without an override — and the URI renders as a HuggingFace
          link inside the body, which is where the third button used to be. */}
      <Modal
        open={detailNode !== null}
        onRequestClose={() => setDetailNode(null)}
        modalHeading="Artifact"
        modalLabel="Lineage"
        primaryButtonText={detail?.uuid ? 'View artifact page' : 'Close'}
        secondaryButtonText={detail?.uuid ? 'Cancel' : undefined}
        onRequestSubmit={() => {
          // Lineage refs carry no gbserver id, so there is nothing to navigate
          // to for those: the primary button just dismisses.
          if (detail?.uuid) router.push(routes.artifactHref(detail.uuid))
          setDetailNode(null)
        }}
        onSecondarySubmit={() => setDetailNode(null)}
        size="sm"
      >
        {detail && (
          <ArtifactSummary
            artifact={{
              id: detail.uuid,
              name: detail.name,
              artifactType: detail.artifactType ?? detailNode?.type,
              uri: detail.uri,
            }}
          />
        )}
      </Modal>
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
