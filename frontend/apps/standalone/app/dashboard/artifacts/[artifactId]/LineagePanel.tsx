'use client'

import * as React from 'react'
import { Button, InlineLoading, Loading, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import { ArrowLeft, ArrowRight, Launch, ZoomIn, ZoomFit, ZoomOut } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import type { Artifact } from '@granite-build/ui-core/types'
import OpenArtifactModal, { type OpenArtifactTarget } from '@granite-build/ui-core/components/LineageGraph/OpenArtifactModal'
import { getBuild, getBuildStatus, getLineageGraph } from '@granite-build/ui-core/api/gbserver'
import BuildLineagePanel from '../../builds/[buildId]/LineagePanel'
import layout from '@granite-build/ui-core/components/LineageGraph/LineagePanelLayout.module.scss'
import own from './LineagePanel.module.scss'
import JobDrawer from '@granite-build/ui-core/components/StepDrawer/JobDrawer'
import Graph, { type GraphHandle } from '@granite-build/ui-core/components/LineageGraph/Graph'
import { depthForNextLevel, groupListingRun, indexGraphToElk, type IndexElkNode, mergeElkGraphs, visibleLevels } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import { useLineageExpansion } from '@granite-build/ui-core/components/LineageGraph/useLineageExpansion'

// ── URI-based artifact lineage panel (lineage index) ──────────────────────────

// Levels loaded each way around the artifact before any Upstream/Downstream click.
const INITIAL_DEPTH = 3

function levelText(level: { depth: number; exhausted: boolean }) {
  const n = `${level.depth} ${level.depth === 1 ? 'level' : 'levels'}`
  if (level.exhausted) return level.depth ? `${n} (all)` : 'none'
  return n
}

function ArtifactLineageGraph({ artifact }: { artifact: Artifact }) {
  const graphRef = React.useRef<GraphHandle>(null)
  const [rendered, setRendered] = React.useState(false)
  const [focusNodeId, setFocusNodeId] = React.useState<string | null>(null)
  // A job (run) node opens its details (see JobDrawer).
  const [jobNodeId, setJobNodeId] = React.useState<string | null>(null)
  const [showBuildInfo, setShowBuildInfo] = React.useState(true)

  // The server normalizes any spelling of the URI; lineage_uri is the id it keys on.
  const uri = artifact.lineage_uri || artifact.uri!
  const { data, isLoading, error } = useQuery({
    queryKey: ['lineage-graph', uri],
    queryFn: () => getLineageGraph({ uri, direction: 'both', depth: INITIAL_DEPTH }),
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const expansion = useLineageExpansion()

  const base = React.useMemo(() => (data ? indexGraphToElk(data) : { nodes: [], links: [] }), [data])
  const merged = React.useMemo(() => mergeElkGraphs(base, expansion.extra), [base, expansion.extra])

  // The index keys artifacts by normalized URI, so the root is root_id, not the UUID.
  const rootId = data?.root_id || uri
  // The artifact the graph is about keeps an outline even once another node is selected.
  const nodes = React.useMemo(
    () => merged.nodes.map((n) => (n.id === rootId ? { ...n, highlight: true } : n)),
    [merged.nodes, rootId],
  )
  const links = merged.links
  // A lone run the merge drops in favour of a group listing it hands its
  // selection, and its open drawer, to that group. Derived in render, so the
  // drawer never sees a frame without its node.
  const regroup = React.useCallback(
    (id: string | null) => (!id || nodes.some((n) => n.id === id) ? id : groupListingRun(nodes, id) ?? id),
    [nodes]
  )
  const focusId = React.useMemo(() => regroup(focusNodeId), [regroup, focusNodeId])
  const jobId = React.useMemo(() => regroup(jobNodeId), [regroup, jobNodeId])
  const activeId = focusId ?? rootId

  // The node the buttons expand from is the one highlighted: the artifact itself
  // until another node is clicked.
  const activeNode = React.useMemo(() => nodes.find((n) => n.id === activeId), [nodes, activeId])
  const activeName = activeNode?.title || activeId

  // Levels are read off the graph on screen, so they stay right whichever node
  // was expanded to get there; a click asks for one level past that.
  const levelOf = (direction: 'upstream' | 'downstream') => ({
    depth: visibleLevels(activeId, links, direction),
    exhausted: expansion.isExhausted(activeId, direction),
  })
  const up = levelOf('upstream')
  const down = levelOf('downstream')
  // After an expansion the selected node is centered, so the new nodes are in view.
  const expand = (direction: 'upstream' | 'downstream') => {
    graphRef.current?.centerOnNodeAfterLayout(activeId)
    const metadata = (activeNode as IndexElkNode | undefined)?.indexNode?.metadata
    const depth = depthForNextLevel(activeId, (direction === 'upstream' ? up : down).depth, metadata)
    return expansion.expand(activeId, direction, depth, { nodes, links }, metadata)
  }

  const unexpanded = expansion.extra.nodes.length
    ? expansion.unexpanded
    : (data?.truncated ? data.unexpanded ?? 0 : 0)
  // POST /lineage/graph never 404s: an empty graph means nothing was recorded.
  const noLineage = !isLoading && !error && nodes.length <= 1
  const busy = expansion.loading !== null

  const jobNode = React.useMemo(
    () => (jobId ? (nodes.find((n) => n.id === jobId) as IndexElkNode | undefined) : undefined),
    [nodes, jobId]
  )
  const drawerRef = React.useRef<HTMLDivElement | null>(null)
  React.useEffect(() => {
    if (!jobNodeId) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && drawerRef.current?.contains(document.activeElement)) setJobNodeId(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [jobNodeId])

  // Index nodes are keyed by normalized URI. The dialog opens at once and opens by
  // URI, which the artifact page resolves when "View artifact page" is clicked.
  const [openError, setOpenError] = React.useState<string | null>(null)
  // The page's own artifact is already open, so there is nothing to open from it.
  const canOpen = Boolean(activeNode && activeNode.type !== 'Build' && activeId !== rootId)
  const [openTarget, setOpenTarget] = React.useState<OpenArtifactTarget | null>(null)
  const openArtifact = () => {
    const metadata = (activeNode as IndexElkNode | undefined)?.indexNode?.metadata
    const uri = typeof metadata?.uri === 'string' ? metadata.uri : activeId
    // The index URI is normalized, as lineage_uri is, so another spelling of
    // this page's own URI is caught without a lookup.
    if (uri === artifact.lineage_uri || uri === artifact.uri) {
      setOpenError(`${activeName} is the artifact already open.`)
      return
    }
    setOpenError(null)
    setOpenTarget({ name: activeName, uri })
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Same toolbar as the build lineage panel. */}
      {/* Any toolbar button closes the open drawer. Capture runs before the
          button's own handler, which still sees this render's selection. */}
      <div
        className={layout.toolbar}
        onClickCapture={(e) => {
          if ((e.target as HTMLElement).closest('button')) setJobNodeId(null)
        }}
      >
        <div className={layout.toolbarLeft}>
          <Button size="sm" kind="ghost" renderIcon={ArrowLeft}
            disabled={busy || noLineage || up.exhausted}
            title={up.exhausted ? `Nothing more upstream of ${activeName}` : `Load level ${up.depth + 1} upstream of ${activeName}`}
            onClick={() => expand('upstream')}>
            Upstream
          </Button>
          <Button size="sm" kind="ghost" renderIcon={ArrowRight}
            disabled={busy || noLineage || down.exhausted}
            title={down.exhausted ? `Nothing more downstream of ${activeName}` : `Load level ${down.depth + 1} downstream of ${activeName}`}
            onClick={() => expand('downstream')}>
            Downstream
          </Button>
          {!noLineage && !isLoading && (
            <span className={own.levelIndicator}>
              From <strong>{activeName}</strong> · ← {levelText(up)} · {levelText(down)} →
            </span>
          )}

          <div className={layout.toolbarDivider} />

          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Zoom In (+10%)" renderIcon={ZoomIn}
            onClick={() => graphRef.current?.zoomIn()} />
          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Reset Zoom" renderIcon={ZoomFit}
            onClick={() => graphRef.current?.resetZoom(rootId)} />
          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Zoom Out (-10%)" renderIcon={ZoomOut}
            onClick={() => graphRef.current?.zoomOut()} />

          <div className={layout.toolbarDivider} />

          <OverflowMenu size="sm" selectorPrimaryFocus=".overflow-item" aria-label="More options">
            <OverflowMenuItem
              className="overflow-item"
              itemText="Reset view"
              onClick={() => { setFocusNodeId(null); setJobNodeId(null); expansion.reset(); graphRef.current?.resetView() }}
            />
            <OverflowMenuItem
              className="overflow-item"
              itemText={showBuildInfo ? 'Hide build IDs' : 'Show build IDs'}
              onClick={() => setShowBuildInfo((v) => !v)}
            />
          </OverflowMenu>

          <div className={layout.toolbarDivider} />

          <Button size="sm" kind="ghost" renderIcon={Launch}
            disabled={!canOpen}
            title={canOpen ? `Open ${activeName}` : 'Select another artifact to open it'}
            onClick={openArtifact}>
            Open artifact
          </Button>
        </div>
      </div>

      {(busy || Boolean(expansion.error) || unexpanded > 0 || Boolean(openError)) && (
        <div className={own.expansionStatus}>
          {busy && <InlineLoading description={`Loading ${expansion.loading} lineage…`} />}
          {Boolean(expansion.error) && (
            <span className={own.expansionError}>Failed to expand lineage: {String(expansion.error)}</span>
          )}
          {openError && <span className={own.expansionError}>{openError}</span>}
          {!busy && unexpanded > 0 && <span>{unexpanded} nodes not expanded.</span>}
        </div>
      )}

      <OpenArtifactModal
        target={openTarget}
        onClose={() => setOpenTarget(null)}
      />

      <div className={layout.graphRow}>
      <div className={layout.graphArea}>
        {isLoading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
            <Loading withOverlay={false} description="Loading lineage…" />
          </div>
        )}
        {!isLoading && Boolean(error) && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-support-error)', fontSize: '0.875rem', padding: '1rem', textAlign: 'center' }}>
            Failed to load lineage: {String(error)}
          </div>
        )}
        {noLineage && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-text-secondary)', fontSize: '0.875rem' }}>
            No lineage recorded for this artifact.
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
              graphKey={artifact.uuid}
              centerNodeId={rootId}
              nodes={nodes}
              links={links}
              selectedNode={activeNode}
              onClick={(node) => {
                setFocusNodeId(node.id)
                setJobNodeId(node.type === 'Build' ? node.id : null)
              }}
              showBuildInfo={showBuildInfo}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {jobNode && (
        <div className={layout.drawerSlot}>
          <JobDrawer node={jobNode} onClose={() => setJobNodeId(null)} drawerRef={drawerRef} />
        </div>
      )}
      </div>
    </div>
  )
}

// ── Build-linked lineage panel (artifacts without a URI) ──────────────────────

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

  // Any artifact with a URI reads the lineage index, which covers build outputs
  // and external artifacts alike, whatever the configured provider.
  if (artifact.uri) {
    // Keyed by the artifact so moving to another one starts from a clean view
    // (expansions, exhausted seeds, the open drawer).
    return <ArtifactLineageGraph key={artifact.uuid} artifact={artifact} />
  }

  if (artifact.build_id) {
    return <BuildLinkedLineage artifact={artifact} artifactLoading={loading} />
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--cds-text-secondary)', fontSize: '0.875rem' }}>
      No lineage data available for this artifact.
    </div>
  )
}
