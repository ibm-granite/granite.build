'use client'

import * as React from 'react'
import { Button, InlineLoading, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import {
  CenterSquare,
  ZoomFit,
  ZoomIn,
  ZoomOut,
} from '@carbon/icons-react'
import layout from '@granite-build/ui-core/components/LineageGraph/LineagePanelLayout.module.scss'
import styles from './LineagePanel.module.scss'
import type { ElkExtendedEdge } from 'elkjs'
import { parse as parseYaml } from 'yaml'
import { useQuery, useQueries, type UseQueryResult } from '@tanstack/react-query'
import type { Artifact, Build, BuildStatusDetail } from '@granite-build/ui-core/types'
import { getArtifact, inferArtifactTypeFromUri } from '@granite-build/ui-core/api/gbserver'
import { getBuildArchiveFiles } from '@granite-build/ui-core/api/gbserver'
import Graph, { type ElkNodeEx, type GraphHandle, type NodeType } from '@granite-build/ui-core/components/LineageGraph/Graph'
import OpenArtifactModal, { type OpenArtifactTarget } from '@granite-build/ui-core/components/LineageGraph/OpenArtifactModal'
import { artifactTypeToNodeType, renameElkGraph } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import StepDrawer from '@granite-build/ui-core/components/StepDrawer/StepDrawer'
import { isUUID } from '@granite-build/ui-core/lib/uuid'

const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending', 'cancel_requested'])

// Target nodes are keyed `target-${name}`; the details panel needs the name back.
const TARGET_NODE_PREFIX = 'target-'

const TARGET_NODE_HEIGHT = 64
// Taller when a step subtitle is present, so the card doesn't clip it.
const TARGET_NODE_HEIGHT_WITH_STEPS = 84

// How many step names to name explicitly in a node subtitle before eliding.
const SUBTITLE_MAX_STEPS = 3

/** "3 steps: fetch → tune → eval", or "" when the target has no step runs. */
function stepSubtitle(steps: { step_name: string }[]): string {
  if (steps.length === 0) return ''
  const names = steps.slice(0, SUBTITLE_MAX_STEPS).map((s) => s.step_name)
  if (steps.length > SUBTITLE_MAX_STEPS) names.push('…')
  const count = `${steps.length} step${steps.length === 1 ? '' : 's'}`
  return `${count}: ${names.join(' → ')}`
}

interface PlannedTarget {
  target_name: string
  inputs: Record<string, string>
  outputs: Record<string, string>
}

function parseDefinitionTargets(yaml: string): PlannedTarget[] {
  try {
    const def = parseYaml(yaml) as {
      targets?: Record<string, {
        inputs?: Record<string, unknown>
        outputs?: Record<string, unknown>
      } | null>
    }
    if (!def?.targets) return []
    return Object.entries(def.targets).map(([name, config]) => ({
      target_name: name,
      inputs: Object.fromEntries(
        Object.entries(config?.inputs ?? {}).map(([k, v]) => [k, String(v ?? '')])
      ),
      outputs: Object.fromEntries(
        Object.entries(config?.outputs ?? {}).map(([k, v]) => [k, String(v ?? '')])
      ),
    }))
  } catch {
    return []
  }
}

interface LineagePanelProps {
  build: Build | undefined
  buildStatus: BuildStatusDetail | undefined
  describe: Build | undefined
  loading: boolean
  statusError?: Error | null
  showFocusNode?: boolean
  initialFocusNodeId?: string
}

// `previousIds` is the last result's artifactIds: returned as-is when the set is
// unchanged, so the artifact lookups keyed on it only recompute when it changes.
function buildGraphData(
  buildStatus: BuildStatusDetail | undefined,
  plannedTargets: PlannedTarget[],
  isActive: boolean,
  previousIds: string[] = EMPTY_IDS,
): {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  artifactIds: string[]
} {
  if (!buildStatus && !plannedTargets.length) {
    return { nodes: [], links: [], artifactIds: EMPTY_IDS }
  }

  const nodes: ElkNodeEx[] = []
  const links: ElkExtendedEdge[] = []
  const seenArtifacts = new Set<string>()
  const seenEdges = new Set<string>()
  const seenTargets = new Set<string>()

  // ── Actual lineage from runtime status ────────────────────────────────────
  for (const [targetName, targetRun] of Object.entries(buildStatus?.targets ?? {})) {
    const targetId = `${TARGET_NODE_PREFIX}${targetName}`
    seenTargets.add(targetName)

    // The node advertises its steps; a target with no step runs yet gets no
    // subtitle (and so keeps its normal height).
    const subtitle = stepSubtitle(targetRun.steps ?? [])

    nodes.push({
      id: targetId,
      title: targetName,
      type: 'Build',
      width: 192,
      height: subtitle ? TARGET_NODE_HEIGHT_WITH_STEPS : TARGET_NODE_HEIGHT,
      labels: [{ text: targetName }],
      ...(subtitle ? { subtitle } : {}),
    })

    for (const [paramName, artifactId] of Object.entries(targetRun.inputs ?? {})) {
      if (!artifactId) continue
      if (!seenArtifacts.has(artifactId)) {
        seenArtifacts.add(artifactId)
        nodes.push({ id: artifactId, title: paramName, type: 'Fileset', width: 224, height: 64, labels: [{ text: paramName }] })
      }
      const edgeId = `${artifactId}-to-${targetId}`
      if (!seenEdges.has(edgeId)) {
        seenEdges.add(edgeId)
        links.push({ id: edgeId, sources: [`${artifactId}-output`], targets: [`${targetId}-input`] })
      }
    }

    for (const [paramName, artifactId] of Object.entries(targetRun.outputs ?? {})) {
      if (!artifactId) continue
      if (!seenArtifacts.has(artifactId)) {
        seenArtifacts.add(artifactId)
        nodes.push({ id: artifactId, title: paramName, type: 'Fileset', width: 224, height: 64, labels: [{ text: paramName }] })
      }
      const edgeId = `${targetId}-to-${artifactId}`
      if (!seenEdges.has(edgeId)) {
        seenEdges.add(edgeId)
        links.push({ id: edgeId, sources: [`${targetId}-output`], targets: [`${artifactId}-input`] })
      }
    }
  }

  // ── Planned lineage overlay from build definition (active builds only) ────
  if (isActive && plannedTargets.length > 0) {
    for (const plannedTarget of plannedTargets) {
      const targetName = plannedTarget.target_name
      if (seenTargets.has(targetName)) continue  // target already in actual lineage

      const targetId = `${TARGET_NODE_PREFIX}${targetName}`
      seenTargets.add(targetName)

      // Planned targets have no step runs to describe, so they keep the plain
      // height and get no subtitle.
      nodes.push({
        id: targetId,
        title: targetName,
        type: 'Build',
        planned: true,
        width: 192,
        height: TARGET_NODE_HEIGHT,
        labels: [{ text: targetName }],
      })

      for (const [paramName, artifactId] of Object.entries(plannedTarget.inputs ?? {})) {
        if (!artifactId) continue
        // If the input artifact already exists in actual lineage, just add the edge
        if (!seenArtifacts.has(artifactId)) {
          seenArtifacts.add(artifactId)
          nodes.push({ id: artifactId, title: paramName, type: plannedNodeType(artifactId), planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
        }
        const edgeId = `${artifactId}-to-${targetId}`
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId)
          links.push({ id: edgeId, sources: [`${artifactId}-output`], targets: [`${targetId}-input`] })
        }
      }

      for (const [paramName, artifactId] of Object.entries(plannedTarget.outputs ?? {})) {
        if (!artifactId) continue
        if (!seenArtifacts.has(artifactId)) {
          seenArtifacts.add(artifactId)
          nodes.push({ id: artifactId, title: paramName, type: plannedNodeType(artifactId), planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
        }
        const edgeId = `${targetId}-to-${artifactId}`
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId)
          links.push({ id: edgeId, sources: [`${targetId}-output`], targets: [`${artifactId}-input`] })
        }
      }
    }
  }

  const ids = Array.from(seenArtifacts)
  const same = ids.length === previousIds.length && ids.every((id, i) => id === previousIds[i])
  return { nodes, links, artifactIds: same ? previousIds : ids }
}

const EMPTY_IDS: string[] = []

// A planned artifact is named by its build.yaml URI and not registered yet, so its
// type is read off the URI; anything the URI does not say stays a Fileset.
function plannedNodeType(uri: string): NodeType {
  return artifactTypeToNodeType(inferArtifactTypeFromUri(uri))
}

// Only the data, so a status poll or refetch that changes nothing keeps the same
// array: combine's result is structurally shared, unlike the raw query results.
function artifactsData(results: UseQueryResult<Artifact>[]) {
  return results.map((r) => (r.isPending ? PENDING : r.data))
}

// An artifact whose fetch has not settled, told apart from one that failed.
const PENDING = Symbol('pending')

const LineagePanelInner = React.forwardRef<GraphHandle, LineagePanelProps>(function LineagePanelInner(
  { build, buildStatus, loading, statusError, showFocusNode = false, initialFocusNodeId },
  ref
) {
  const graphRef = React.useRef<GraphHandle>(null)
  const isActive = ACTIVE_STATUSES.has(build?.status ?? '')

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => graphRef.current?.zoomIn(),
    zoomOut: () => graphRef.current?.zoomOut(),
    resetZoom: (nodeId?: string) => graphRef.current?.resetZoom(nodeId),
    resetView: () => graphRef.current?.resetView(),
    currentZoom: () => graphRef.current?.currentZoom() ?? 90,
    centerOnNode: (nodeId: string) => graphRef.current?.centerOnNode(nodeId),
    centerOnNodeAfterLayout: (nodeId: string) => graphRef.current?.centerOnNodeAfterLayout(nodeId),
  }))

  // Fetch build archive YAML to derive planned targets for active builds
  const { data: archiveFiles } = useQuery({
    queryKey: ['build-archive', build?.uuid],
    queryFn: () => getBuildArchiveFiles(build!.uuid),
    enabled: Boolean(build?.uuid) && isActive,
    staleTime: 60_000,
  })

  const plannedTargets = React.useMemo<PlannedTarget[]>(() => {
    if (!archiveFiles) return []
    const yaml =
      archiveFiles['build.yaml'] ??
      archiveFiles[Object.keys(archiveFiles).find((k) => k.endsWith('.yaml') || k.endsWith('.yml')) ?? '']
    return yaml ? parseDefinitionTargets(yaml) : []
  }, [archiveFiles])

  // Step metadata (issue #224): target nodes advertise their steps, and clicking
  // one opens the step details. The step data already rides along on
  // getBuildStatus, so this costs no extra request.
  const [stepDetailTarget, setStepDetailTarget] = React.useState<string | null>(null)

  // Where focus was before the drawer opened, so we can hand it back on close —
  // otherwise a keyboard user is dropped at the top of the document.
  const drawerReturnFocusRef = React.useRef<HTMLElement | null>(null)
  const drawerCloseButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const drawerRef = React.useRef<HTMLDivElement | null>(null)
  // Focus fallback when the drawer's trigger node has been detached by a re-render.
  const graphContainerRef = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    if (!stepDetailTarget) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // This is a non-modal drawer — the graph behind it stays interactive — so
      // only swallow Escape when focus is actually inside the drawer. Otherwise
      // a user mid-interaction with the graph would have the drawer yanked shut.
      if (drawerRef.current?.contains(document.activeElement)) {
        setStepDetailTarget(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [stepDetailTarget])

  // Focus management for the drawer (role="dialog"): on open, remember the
  // trigger and move focus to the close button; on close, restore focus. This is
  // a non-modal drawer by design (the graph stays interactive), so no focus trap
  // — just entry and restore, which is what keyboard/SR users expect.
  //
  // Capture the trigger only when opening from a *closed* drawer, and restore
  // only when closing to a closed drawer. Switching directly A→B must neither
  // recapture (B's trigger, not A's return element) nor restore (nothing closed
  // yet) — doing either would leave the eventual restore pointing at the wrong
  // element, a real regression for keyboard/SR users.
  const drawerWasOpenRef = React.useRef(false)
  React.useEffect(() => {
    const wasOpen = drawerWasOpenRef.current
    drawerWasOpenRef.current = Boolean(stepDetailTarget)

    if (stepDetailTarget) {
      // Opening from closed — record where focus was so we can hand it back.
      // A→B switches (wasOpen already true) keep the original return element.
      if (!wasOpen) {
        drawerReturnFocusRef.current = document.activeElement as HTMLElement | null
      }
      // preventScroll on every focus move here: focusing scrolls the nearest
      // overflow container even when it is overflow:hidden, which slid the graph
      // sideways each time the drawer opened, switched or closed.
      drawerCloseButtonRef.current?.focus({ preventScroll: true })
      return
    }

    // stepDetailTarget is null. Only restore if a drawer was actually open.
    if (!wasOpen) return
    // The trigger is often a graph node inside an SVG that re-renders on the
    // status poll; by close it may be detached, and focus() on a detached node
    // is a silent no-op that drops focus to <body>. Restore only when the node
    // is still connected, else fall back to the graph container so keyboard
    // focus lands somewhere sensible rather than the top of the document.
    const returnTo = drawerReturnFocusRef.current
    if (returnTo?.isConnected) {
      returnTo.focus?.({ preventScroll: true })
    } else {
      graphContainerRef.current?.focus?.({ preventScroll: true })
    }
    drawerReturnFocusRef.current = null
  }, [stepDetailTarget])

  // The previous result's ids, kept in state (not a ref read during render) so an
  // unchanged id set keeps its array across polls.
  const [previousIds, setPreviousIds] = React.useState<string[]>(EMPTY_IDS)
  const graphData = React.useMemo(
    () => buildGraphData(buildStatus, plannedTargets, isActive, previousIds),
    [buildStatus, plannedTargets, isActive, previousIds],
  )
  const { nodes: rawBuildNodes, links: rawBuildLinks, artifactIds } = graphData
  if (artifactIds !== previousIds) setPreviousIds(artifactIds)

  // Fetch artifact names for all UUID-shaped artifact IDs
  const uuidArtifactIds = React.useMemo(() => artifactIds.filter(isUUID), [artifactIds])
  const artifacts = useQueries({
    queries: uuidArtifactIds.map((id) => ({
      queryKey: ['artifact', id],
      queryFn: () => getArtifact(id),
      retry: false,
      staleTime: 5 * 60 * 1000,
    })),
    combine: artifactsData,
  })
  const artifactById = React.useMemo(() => {
    const map = new Map<string, Artifact>()
    uuidArtifactIds.forEach((id, i) => { const a = artifacts[i]; if (a && a !== PENDING) map.set(id, a) })
    return map
  }, [artifacts, uuidArtifactIds])

  // Finished targets record their artifacts by registry UUID, but a planned target
  // names them by the URI in build.yaml. Without folding, an artifact one target
  // produced and a planned one consumes draws twice: the typed UUID node, and a
  // 'Fileset' beside it holding the planned edge.
  // The planned URI is matched as the build.yaml spells it, against both the
  // registry's recorded uri and its lineage_uri. One build.yaml usually spells one
  // artifact one way; a miss only leaves a second node, never a wrong merge.
  const plannedUriToUuid = React.useMemo(() => {
    const byUri = new Map<string, string>()
    for (const [id, data] of artifactById) {
      for (const uri of [data.uri, data.lineage_uri]) if (uri && !byUri.has(uri)) byUri.set(uri, id)
    }
    const map = new Map<string, string>()
    for (const id of artifactIds) {
      const uuid = isUUID(id) ? undefined : byUri.get(id)
      if (uuid) map.set(id, uuid)
    }
    return map
  }, [artifactById, artifactIds])
  const { nodes: buildNodes, links: buildLinks } = React.useMemo(
    () => renameElkGraph({ nodes: rawBuildNodes, links: rawBuildLinks }, (id) => plannedUriToUuid.get(id) ?? id),
    [rawBuildNodes, rawBuildLinks, plannedUriToUuid]
  )

  // Enrich nodes with resolved artifact names and types
  const enrichedBuildNodes = React.useMemo<ElkNodeEx[]>(() => {
    return buildNodes.map((node) => {
      const enriched = artifactById.get(node.id)
      if (enriched) {
        return { ...node, title: enriched.name, type: artifactTypeToNodeType(enriched.artifact_type) }
      }
      // This build's own targets are runs of this build.
      if (node.type === 'Build' && !node.planned && build?.uuid) return { ...node, buildId: build.uuid }
      return node
    })
  }, [buildNodes, artifactById, build?.uuid])

  // The graph shows what this build did, whole; lineage beyond it is on the
  // artifact page, which a click on an artifact offers to open.

  // Navigation state
  const [focusNodeId, setFocusNodeId] = React.useState<string | null>(initialFocusNodeId ?? null)
  // The clicked node, not a snapshot of its dialog: the target is derived from
  // the artifact queries, so a click before its fetch lands still fills in.
  const [openNode, setOpenNode] = React.useState<ElkNodeEx | null>(null)
  const [rendered, setRendered] = React.useState(false)
  const [showBuildInfo, setShowBuildInfo] = React.useState(true)

  // The current artifact's node is always highlighted on artifact pages
  // (showFocusNode is only true there) — this is not click-driven.
  const currentArtifactNode = React.useMemo(
    () => (showFocusNode && initialFocusNodeId
      ? enrichedBuildNodes.find((n) => n.id === initialFocusNodeId)
      : undefined),
    [showFocusNode, initialFocusNodeId, enrichedBuildNodes]
  )

  // The node whose step drawer is open. Opening that drawer is what shrinks the
  // graph pane, so this is the node the resize compensation must keep on-screen —
  // and, unlike currentArtifactNode, it is set on build pages (where the drawer
  // exists) rather than only on artifact pages.
  //
  // Memoized on `stepDetailTarget` alone, not on `enrichedBuildNodes`: both consumers
  // of `selectedNode` read only `.id` (GraphNode's isSelected check, and the
  // resize compensation's position lookup), and `enrichedBuildNodes` is a fresh array
  // on every status poll. Depending on it would hand Graph a new object each
  // poll, defeating its React.memo and re-running the nodeElements effect on
  // every tick of a live build with the drawer open.
  const openDrawerNode = React.useMemo(
    () => (stepDetailTarget
      ? ({ id: `${TARGET_NODE_PREFIX}${stepDetailTarget}` } as ElkNodeEx)
      : undefined),
    [stepDetailTarget]
  )


  // If the open target disappears from a later relayout (renamed, or dropped from
  // the build), close the drawer rather than leave it pointing at a target that no
  // longer exists. The graph itself is the authority: a target is exactly as real
  // as the node the user clicked, so validate against the rendered nodes rather
  // than rebuilding the answer from buildStatus.targets plus plannedTargets.
  //
  // Those two sources cannot decide it. A planned target is absent from
  // buildStatus.targets by definition, and plannedTargets is empty whenever the
  // build archive has not loaded — the archive query is gated on isActive and
  // yields [] for a finished build or an unparseable build.yaml. Checking them
  // closed the drawer on every planned target, in the same frame it opened.
  //
  // Guarded on the node set being non-empty so a transient empty layout (first
  // render, or a poll that briefly returns no targets) does not close it either.
  React.useEffect(() => {
    if (!stepDetailTarget || enrichedBuildNodes.length === 0) return
    const nodeId = `${TARGET_NODE_PREFIX}${stepDetailTarget}`
    if (enrichedBuildNodes.some((n) => n.id === nodeId)) return
    setStepDetailTarget(null)
  }, [stepDetailTarget, enrichedBuildNodes])

  // A click selects a node. A target (run) node also opens its step details; a registered artifact asks whether to open
  // its page, where its lineage beyond this build is.
  const handleNodeClick = (node: ElkNodeEx) => {
    // Skeleton stubs are placeholders for hidden branches, not real nodes: their
    // id still starts with TARGET_NODE_PREFIX when they hang off a target, and
    // opening the drawer for one would close again at once.
    if (node.type === 'skeleton-source' || node.type === 'skeleton-target') return
    setFocusNodeId(node.id)
    const isTarget = node.type === 'Build' && node.id.startsWith(TARGET_NODE_PREFIX)
    setStepDetailTarget(isTarget ? node.id.slice(TARGET_NODE_PREFIX.length) : null)
    // The page's own artifact is already open, so there is nothing to open from it.
    if (node.type === 'Build' || node.id === initialFocusNodeId) return
    setOpenNode(node)
  }

  // A registered artifact opens by UUID; a planned one by its build.yaml URI,
  // which the artifact page resolves when the dialog's button is clicked.
  const openTarget = React.useMemo<OpenArtifactTarget | null>(() => {
    if (!openNode) return null
    const registered = isUUID(openNode.id)
    const artifact = registered ? artifactById.get(openNode.id) : undefined
    return {
      name: artifact?.name || openNode.title || openNode.id,
      uri: registered ? artifact?.uri : openNode.id,
      uuid: registered ? openNode.id : undefined,
      artifactType: artifact?.artifact_type,
    }
  }, [openNode, artifactById])

  // Back to the page's own artifact, now that clicks can select other nodes.
  const handleFocusNode = () => {
    const target = initialFocusNodeId ?? focusNodeId
    if (!target) return
    setFocusNodeId(target)
    graphRef.current?.centerOnNode?.(target)
  }

  const focusNode = focusNodeId ? enrichedBuildNodes.find((n) => n.id === focusNodeId) : undefined

  const noLineage = !loading && !statusError && enrichedBuildNodes.length === 0

  return (
    <div className={styles.container}>
      {/* Toolbar */}
      {/* Any toolbar button closes the open drawer. Capture runs before the
          button's own handler, which still sees this render's selection. */}
      <div
        className={layout.toolbar}
        onClickCapture={(e) => {
          if ((e.target as HTMLElement).closest('button')) setStepDetailTarget(null)
        }}
      >
        <div className={layout.toolbarLeft}>
          {showFocusNode && (
            <Button
              size="sm"
              kind="ghost"
              renderIcon={CenterSquare}
              disabled={!focusNodeId}
              onClick={handleFocusNode}
            >
              Focus Node
            </Button>
          )}
          {showFocusNode && <div className={layout.toolbarDivider} />}
          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Zoom In (+10%)"
            renderIcon={ZoomIn}
            onClick={() => graphRef.current?.zoomIn()}
          />
          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Reset Zoom"
            renderIcon={ZoomFit}
            onClick={() => graphRef.current?.resetZoom((currentArtifactNode ?? focusNode ?? openDrawerNode)?.id)}
          />
          <Button
            size="sm"
            kind="ghost"
            hasIconOnly
            tooltipPosition="right"
            iconDescription="Zoom Out (-10%)"
            renderIcon={ZoomOut}
            onClick={() => graphRef.current?.zoomOut()}
          />

          <div className={layout.toolbarDivider} />

          <OverflowMenu size="sm" selectorPrimaryFocus=".overflow-item">
            <OverflowMenuItem
              className="overflow-item"
              itemText="Reset view"
              onClick={() => {
                // Clear the selection, then fit. `resetView` fits against the
                // current layout immediately and clears the user-adjusted flag.
                setFocusNodeId(null);
                graphRef.current?.resetView();
              }}
            />
            <OverflowMenuItem
              className="overflow-item"
              itemText={showBuildInfo ? 'Hide build IDs' : 'Show build IDs'}
              onClick={() => setShowBuildInfo((v) => !v)}
            />
          </OverflowMenu>
        </div>
      </div>

      {/* Graph and drawer are flex siblings in a row so opening the drawer
          physically shrinks the graph's width. ELK lays out RIGHT (downstream
          nodes toward the right edge — the same edge the drawer opens on), so if
          the drawer merely overlaid the graph, a node near that edge could hide
          behind the drawer it just triggered; shrinking the SVG instead fires
          the graph's ResizeObserver, which refits the view. */}
      <div className={layout.graphRow}>
      {/* Graph area. tabIndex=-1 so it can receive programmatic focus as the
          fallback when a closed drawer's trigger node is no longer in the DOM. */}
      <div className={layout.graphArea} ref={graphContainerRef} tabIndex={-1}>
        {loading && (
          <div className={styles.centeredContent}>
            <InlineLoading description="Loading lineage…" />
          </div>
        )}

        {!loading && statusError && (
          <div className={styles.errorContent}>
            Failed to load lineage: {String(statusError)}
          </div>
        )}

        {!loading && noLineage && (
          <div className={styles.emptyContent}>
            No lineage data available for build
            {build?.name ? ` "${build.name}"` : ""}.
          </div>
        )}

        {!loading && !noLineage && (
          <>
            {!rendered && (
              <InlineLoading
                className={styles.renderingIndicator}
                description="Lineage is rendering…"
              />
            )}
            <Graph
              ref={graphRef}
              graphKey={build?.uuid}
              nodes={enrichedBuildNodes}
              links={buildLinks}
              selectedNode={focusNode ?? currentArtifactNode ?? openDrawerNode}
              showBuildInfo={showBuildInfo}
              onClick={handleNodeClick}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {stepDetailTarget && (
      <div className={layout.drawerSlot}>
        <StepDrawer
          targetName={stepDetailTarget}
          // Own-property lookup: a bare-object index would return
          // Object.prototype.toString (a function) for a target named `toString`.
          target={buildStatus?.targets && Object.prototype.hasOwnProperty.call(buildStatus.targets, stepDetailTarget)
            ? buildStatus.targets[stepDetailTarget]
            : undefined}
          build={build}
          onClose={() => setStepDetailTarget(null)}
          drawerRef={drawerRef}
          closeButtonRef={drawerCloseButtonRef}
        />
      </div>
      )}
      </div>

      <OpenArtifactModal
        target={openTarget}
        onClose={() => setOpenNode(null)}
      />
    </div>
  );
})

export default LineagePanelInner

// Re-export GraphHandle for use in parent
export type { GraphHandle }
