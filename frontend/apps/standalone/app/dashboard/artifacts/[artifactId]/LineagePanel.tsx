'use client'

import * as React from 'react'
import { Button, InlineLoading, Loading, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import { ArrowLeft, ArrowRight, Launch, ZoomIn, ZoomFit, ZoomOut } from '@carbon/icons-react'
import { useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import type { Artifact } from '@granite-build/ui-core/types'
import { getBuild, getBuildStatus, getLineageGraph, listArtifacts } from '@granite-build/ui-core/api/gbserver'
import BuildLineagePanel from '../../builds/[buildId]/LineagePanel'
import styles from '../../builds/[buildId]/LineagePanel.module.scss'
import JobDrawer from '../../builds/[buildId]/JobDrawer'
import Graph, { type GraphHandle } from '@granite-build/ui-core/components/LineageGraph/Graph'
import { depthForNextLevel, indexGraphToElk, type IndexElkNode, mergeElkGraphs, visibleLevels } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
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
  const [showBuildInfo, setShowBuildInfo] = React.useState(true)

  const { data, isLoading, error } = useQuery({
    queryKey: ['lineage-graph', artifact.uri],
    queryFn: () => getLineageGraph({ uri: artifact.uri, direction: 'both', depth: INITIAL_DEPTH }),
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const expansion = useLineageExpansion()

  const base = React.useMemo(() => (data ? indexGraphToElk(data) : { nodes: [], links: [] }), [data])
  const { nodes, links } = React.useMemo(() => mergeElkGraphs(base, expansion.extra), [base, expansion.extra])

  // The index keys artifacts by normalized URI, so the root is root_id, not the UUID.
  const rootId = data?.root_id || artifact.uri!
  const activeId = focusNodeId ?? rootId

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
    return expansion.expand(activeId, direction, depthForNextLevel(activeId, (direction === 'upstream' ? up : down).depth), { nodes, links })
  }

  const unexpanded = expansion.extra.nodes.length
    ? expansion.unexpanded
    : (data?.truncated ? data.unexpanded ?? 0 : 0)
  // GET /lineage/graph never 404s: an empty graph means nothing was recorded.
  const noLineage = !isLoading && !error && nodes.length <= 1
  const busy = expansion.loading !== null

  // A job (run) node opens its details (see JobDrawer).
  const [jobNodeId, setJobNodeId] = React.useState<string | null>(null)
  const jobNode = React.useMemo(
    () => (jobNodeId ? (nodes.find((n) => n.id === jobNodeId) as IndexElkNode | undefined) : undefined),
    [nodes, jobNodeId]
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

  // Index nodes are keyed by normalized URI, not UUID: the artifact page is
  // found by the raw URI the index keeps in metadata, which the registry filters
  // on exactly. The artifact itself is already open.
  const router = useRouter()
  const [opening, setOpening] = React.useState(false)
  const [openError, setOpenError] = React.useState<string | null>(null)
  const canOpenArtifact = Boolean(activeNode && activeNode.type !== 'Build' && activeId !== rootId)
  const openArtifact = async () => {
    const raw = (activeNode as IndexElkNode | undefined)?.indexNode?.metadata?.uri
    const uri = typeof raw === 'string' ? raw : activeId
    setOpening(true)
    setOpenError(null)
    try {
      const { items } = await listArtifacts({ uri })
      const match = items[0]
      if (match) router.push(`/dashboard/artifacts/_/?id=${match.uuid}`)
      else setOpenError(`${activeName} is not in the artifact registry.`)
    } catch (e) {
      setOpenError(`Failed to open ${activeName}: ${String(e)}`)
    } finally {
      setOpening(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Same toolbar as the build lineage panel. */}
      {/* Any toolbar button closes the open drawer. Capture runs before the
          button's own handler, which still sees this render's selection. */}
      <div
        className={styles.toolbar}
        onClickCapture={(e) => {
          if ((e.target as HTMLElement).closest('button')) setJobNodeId(null)
        }}
      >
        <div className={styles.toolbarLeft}>
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
            <span className={styles.levelIndicator}>
              From <strong>{activeName}</strong> · ← {levelText(up)} · {levelText(down)} →
            </span>
          )}

          <div className={styles.toolbarDivider} />

          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Zoom In (+10%)" renderIcon={ZoomIn}
            onClick={() => graphRef.current?.zoomIn()} />
          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Reset Zoom" renderIcon={ZoomFit}
            onClick={() => { setFocusNodeId(null); graphRef.current?.resetZoom() }} />
          <Button size="sm" kind="ghost" hasIconOnly tooltipPosition="right"
            iconDescription="Zoom Out (-10%)" renderIcon={ZoomOut}
            onClick={() => graphRef.current?.zoomOut()} />

          <div className={styles.toolbarDivider} />

          <OverflowMenu size="sm" selectorPrimaryFocus=".overflow-item" aria-label="More options">
            <OverflowMenuItem
              className="overflow-item"
              itemText="Reset view"
              onClick={() => { setFocusNodeId(null); setJobNodeId(null); expansion.reset(); graphRef.current?.resetZoom() }}
            />
            <OverflowMenuItem
              className="overflow-item"
              itemText={showBuildInfo ? 'Hide build IDs' : 'Show build IDs'}
              onClick={() => setShowBuildInfo((v) => !v)}
            />
          </OverflowMenu>

          <div className={styles.toolbarDivider} />

          <Button size="sm" kind="ghost" renderIcon={Launch}
            disabled={!canOpenArtifact || opening}
            title={canOpenArtifact ? `Open ${activeName}` : 'Select another artifact to open it'}
            onClick={openArtifact}>
            Open artifact
          </Button>
        </div>
      </div>

      {(busy || Boolean(expansion.error) || unexpanded > 0 || Boolean(openError)) && (
        <div className={styles.expansionStatus}>
          {busy && <InlineLoading description={`Loading ${expansion.loading} lineage…`} />}
          {Boolean(expansion.error) && (
            <span className={styles.expansionError}>Failed to expand lineage: {String(expansion.error)}</span>
          )}
          {openError && <span className={styles.expansionError}>{openError}</span>}
          {!busy && unexpanded > 0 && <span>{unexpanded} nodes not expanded.</span>}
        </div>
      )}

      <div className={styles.graphRow}>
      <div className={styles.graphArea}>
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
              nodes={nodes}
              links={links}
              allLinks={links}
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
        <div className={styles.drawerSlot}>
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
    return <ArtifactLineageGraph artifact={artifact} />
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
