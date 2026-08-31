'use client'

import * as React from 'react'
import { Button, ComposedModal, IconButton, InlineLoading, Modal, ModalBody, ModalFooter, ModalHeader, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import {
  ArrowLeft,
  ArrowRight,
  CenterSquare,
  Close,
  Launch,
  ZoomFit,
  ZoomIn,
  ZoomOut,
} from '@carbon/icons-react'
import { useRouter } from 'next/navigation'
import styles from './LineagePanel.module.scss'
import type { ElkExtendedEdge } from 'elkjs'
import { parse as parseYaml } from 'yaml'
import { useQuery, useQueries } from '@tanstack/react-query'
import type { Build, BuildStatusDetail } from '@/types'
import { getArtifact } from '@/api/gbserver'
import { getBuildArchiveFiles } from '@/api/gbserver'
import Graph, { type ElkNodeEx, type GraphHandle, type NodeType } from '@/components/LineageGraph/Graph'
import { getSubgraph } from '@/components/LineageGraph/diagramUtilities'
import StepDetailsPanel, { stepDrawerSummary } from './StepDetailsPanel'
import { BuildStatusBadge } from '@/components/BuildStatusBadge'

const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending'])

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

function artifactTypeToNodeType(artifactType: string): NodeType {
  switch (artifactType.toUpperCase()) {
    case 'MODEL': return 'Model'
    case 'DATASET': return 'Dataset'
    case 'FILESET': return 'Fileset'
    case 'BUCKET': return 'Bucket'
    default: return 'Fileset'
  }
}

function buildGraphData(
  buildStatus: BuildStatusDetail | undefined,
  plannedTargets: PlannedTarget[],
  isActive: boolean,
): {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  artifactIds: string[]
} {
  if (!buildStatus && !plannedTargets.length) return { nodes: [], links: [], artifactIds: [] }

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
          nodes.push({ id: artifactId, title: paramName, type: 'Fileset', planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
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
          nodes.push({ id: artifactId, title: paramName, type: 'Fileset', planned: true, width: 224, height: 64, labels: [{ text: paramName }] })
        }
        const edgeId = `${targetId}-to-${artifactId}`
        if (!seenEdges.has(edgeId)) {
          seenEdges.add(edgeId)
          links.push({ id: edgeId, sources: [`${targetId}-output`], targets: [`${artifactId}-input`] })
        }
      }
    }
  }

  return { nodes, links, artifactIds: Array.from(seenArtifacts) }
}

function isUUID(s: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s)
}

// Mirrors src/gbcommon/utils/hf_utils.py:convert_hf_uri_to_url — model URLs
// never include a "models/" segment; datasets/spaces/buckets keep their
// pluralized type segment.
function getHuggingFaceUrl(uri: string): string | null {
  if (!uri) return null

  if (uri.startsWith('hf://')) {
    const remainder = uri.slice(5)
    let parts: string[]

    if (remainder.startsWith('/')) {
      // hf:///[type/]org/name
      parts = remainder.replace(/^\/+/, '').split('/')
    } else if (remainder.startsWith('huggingface.co/')) {
      // hf://huggingface.co/[type/]org/name
      parts = remainder.slice('huggingface.co/'.length).split('/')
    } else if (remainder.includes('/')) {
      // hf://<domain>/[type/]org/name — the domain segment is discarded;
      // the browsable URL is always on huggingface.co
      parts = remainder.split('/').slice(1)
    } else {
      return null
    }

    if (parts.length === 2) {
      const [org, name] = parts
      return `https://huggingface.co/${org}/${name}`
    }
    if (parts.length === 3) {
      const [type, org, name] = parts
      switch (type) {
        case 'models':   return `https://huggingface.co/${org}/${name}`
        case 'datasets': return `https://huggingface.co/datasets/${org}/${name}`
        case 'spaces':   return `https://huggingface.co/spaces/${org}/${name}`
        case 'buckets':  return `https://huggingface.co/buckets/${org}/${name}`
        default: return null
      }
    }
    return null
  }

  if (/huggingface\.co/.test(uri)) return uri.startsWith('http') ? uri : `https://${uri}`
  return null
}

const LineagePanelInner = React.forwardRef<GraphHandle, LineagePanelProps>(function LineagePanelInner(
  { build, buildStatus, loading, statusError, showFocusNode = false, initialFocusNodeId },
  ref
) {
  const graphRef = React.useRef<GraphHandle>(null)
  const isActive = ACTIVE_STATUSES.has(build?.status ?? '')

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => graphRef.current?.zoomIn(),
    zoomOut: () => graphRef.current?.zoomOut(),
    resetZoom: () => graphRef.current?.resetZoom(),
    resetView: () => graphRef.current?.resetView(),
    currentZoom: () => graphRef.current?.currentZoom() ?? 90,
    centerOnNode: (nodeId: string) => graphRef.current?.centerOnNode(nodeId),
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
      drawerCloseButtonRef.current?.focus()
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
      returnTo.focus?.()
    } else {
      graphContainerRef.current?.focus?.()
    }
    drawerReturnFocusRef.current = null
  }, [stepDetailTarget])

  // If the open target disappears from a later status poll (renamed, or dropped
  // from the build), close the drawer rather than leave it pointing at a target
  // that no longer exists. Guarded on buildStatus being loaded so the drawer
  // isn't closed during a transient empty poll. Planned targets (from the build
  // definition, not yet in runtime status) are legitimately absent from
  // buildStatus.targets, so keep the drawer open for those too — otherwise it
  // opens and immediately self-closes on the next render.
  React.useEffect(() => {
    if (!stepDetailTarget || !buildStatus?.targets) return
    if (stepDetailTarget in buildStatus.targets) return
    if (plannedTargets.some((t) => t.target_name === stepDetailTarget)) return
    setStepDetailTarget(null)
  }, [stepDetailTarget, buildStatus, plannedTargets])

  const { nodes: allNodes, links: allLinks, artifactIds } = React.useMemo(
    () => buildGraphData(buildStatus, plannedTargets, isActive),
    [buildStatus, plannedTargets, isActive]
  )

  // Fetch artifact names for all UUID-shaped artifact IDs
  const uuidArtifactIds = artifactIds.filter(isUUID)
  const artifactQueries = useQueries({
    queries: uuidArtifactIds.map((id) => ({
      queryKey: ['artifact', id],
      queryFn: () => getArtifact(id),
      retry: false,
      staleTime: 5 * 60 * 1000,
    })),
  })

  // Enrich nodes with resolved artifact names and types
  const enrichedNodes = React.useMemo<ElkNodeEx[]>(() => {
    const artifactMap = new Map<string, { name: string; type: NodeType }>()
    uuidArtifactIds.forEach((id, i) => {
      const result = artifactQueries[i]?.data
      if (result) {
        artifactMap.set(id, {
          name: result.name,
          type: artifactTypeToNodeType(result.artifact_type),
        })
      }
    })

    return allNodes.map((node) => {
      const enriched = artifactMap.get(node.id)
      if (enriched) {
        return { ...node, title: enriched.name, type: enriched.type }
      }
      return node
    })
  }, [allNodes, artifactQueries, uuidArtifactIds])

  const artifactUriMap = React.useMemo(() => {
    const map = new Map<string, string>()
    uuidArtifactIds.forEach((id, i) => {
      const uri = artifactQueries[i]?.data?.uri
      if (uri) map.set(id, uri)
    })
    return map
  }, [artifactQueries, uuidArtifactIds])

  const artifactNavModalHeader = (artifactNavNode: { node: ElkNodeEx; hfUrl: string | null } | null) => {
    if (artifactNavNode) {
      return <h4>Would you like to view <code>{artifactNavNode.node?.title || artifactNavNode.node?.id}</code> on HuggingFace or proceed to the artifact page?</h4>
    } else {
      return <h4>Would you like to view this artifact on HuggingFace or proceed to the artifact page?</h4>
    }
  }

  // Navigation state
  const [focusNodeId, setFocusNodeId] = React.useState<string | null>(initialFocusNodeId ?? null)
  const [upstreamLevels, setUpstreamLevels] = React.useState(Infinity)
  const [downstreamLevels, setDownstreamLevels] = React.useState(Infinity)
  const [partial, setPartial] = React.useState(false)
  const [artifactNavNode, setArtifactNavNode] = React.useState<{ node: ElkNodeEx; hfUrl: string | null } | null>(null)
  const router = useRouter()
  const [rendered, setRendered] = React.useState(false)

  // The current artifact's node is always highlighted on artifact pages
  // (showFocusNode is only true there) — this is not click-driven.
  const currentArtifactNode = React.useMemo(
    () => (showFocusNode && initialFocusNodeId
      ? enrichedNodes.find((n) => n.id === initialFocusNodeId)
      : undefined),
    [showFocusNode, initialFocusNodeId, enrichedNodes]
  )

  const { filteredNodes, filteredLinks } = React.useMemo(() => {
    if (!focusNodeId || (upstreamLevels === Infinity && downstreamLevels === Infinity)) {
      return { filteredNodes: enrichedNodes, filteredLinks: allLinks }
    }
    const sub = getSubgraph(focusNodeId, downstreamLevels, upstreamLevels, enrichedNodes, allLinks)
    return { filteredNodes: sub.nodes, filteredLinks: sub.links }
  }, [focusNodeId, upstreamLevels, downstreamLevels, enrichedNodes, allLinks])

  const handleNodeClick = (node: ElkNodeEx) => {
    if (!showFocusNode) {
      setFocusNodeId(node.id)
    }
    if (node.type !== 'Build' && isUUID(node.id)) {
      const uri = artifactUriMap.get(node.id)
      // Close the step drawer first — otherwise the artifact modal stacks on top
      // of it and the drawer is left open underneath once the modal is dismissed.
      setStepDetailTarget(null)
      setArtifactNavNode({ node, hfUrl: uri ? getHuggingFaceUrl(uri) : null })
      return
    }
    // Clicking a target (run) node opens its step details. Artifact
    // click-through is handled above and is unchanged.
    if (node.type === 'Build' && node.id.startsWith(TARGET_NODE_PREFIX)) {
      setStepDetailTarget(node.id.slice(TARGET_NODE_PREFIX.length))
    }
  }

  const handleFocusNode = () => {
    if (!focusNodeId) return
    setUpstreamLevels(Infinity)
    setDownstreamLevels(Infinity)
    setPartial(false)
    graphRef.current?.centerOnNode?.(focusNodeId)
  }

  const handleUpstream = () => {
    if (!focusNodeId) return
    const newUp = upstreamLevels === Infinity ? 2 : upstreamLevels + 1
    const sub = getSubgraph(focusNodeId, downstreamLevels, newUp, enrichedNodes, allLinks)
    setUpstreamLevels(sub.hasMoreUpstream ? newUp : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  const handleDownstream = () => {
    if (!focusNodeId) return
    const newDown = downstreamLevels === Infinity ? 2 : downstreamLevels + 1
    const sub = getSubgraph(focusNodeId, newDown, upstreamLevels, enrichedNodes, allLinks)
    setDownstreamLevels(sub.hasMoreDownstream ? newDown : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  const noLineage = !loading && !statusError && allNodes.length === 0

  return (
    <div className={styles.container}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <Button
            size="sm"
            kind="ghost"
            renderIcon={ArrowLeft}
            disabled={!focusNodeId}
            onClick={handleUpstream}
            iconDescription="One level upstream"
          >
            Upstream
          </Button>
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
          <Button
            size="sm"
            kind="ghost"
            renderIcon={ArrowRight}
            disabled={!focusNodeId}
            onClick={handleDownstream}
            iconDescription="One level downstream"
          >
            Downstream
          </Button>

          <div className={styles.toolbarDivider} />

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
            onClick={() => graphRef.current?.resetZoom()}
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

          <div className={styles.toolbarDivider} />

          <OverflowMenu size="sm" selectorPrimaryFocus=".overflow-item">
            <OverflowMenuItem
              className="overflow-item"
              itemText="Reset view"
              onClick={() => {
                // Was the graph filtered? If so, clearing the filter expands the
                // node set and kicks off an async relayout — clear the
                // user-adjusted flag and let that relayout auto-fit once, rather
                // than fitting now against stale positions and re-snapping. If
                // nothing was filtered, no relayout fires, so fit immediately.
                const wasFiltered =
                  focusNodeId !== null &&
                  (upstreamLevels !== Infinity || downstreamLevels !== Infinity);
                setFocusNodeId(null);
                setUpstreamLevels(Infinity);
                setDownstreamLevels(Infinity);
                setPartial(false);
                if (wasFiltered) {
                  graphRef.current?.resetView();
                } else {
                  graphRef.current?.resetZoom();
                }
              }}
            />
          </OverflowMenu>

          <div className={styles.toolbarDivider} />
        </div>
      </div>

      {/* Status messages */}
      {partial && (
        <div className={styles.partialMessage}>
          The lineage graph is partially displayed. Click Upstream or Downstream
          to show more nodes.
        </div>
      )}

      {/* Graph and drawer are flex siblings in a row so opening the drawer
          physically shrinks the graph's width. ELK lays out RIGHT (downstream
          nodes toward the right edge — the same edge the drawer opens on), so if
          the drawer merely overlaid the graph, a node near that edge could hide
          behind the drawer it just triggered; shrinking the SVG instead fires
          the graph's ResizeObserver, which refits the view. */}
      <div className={styles.graphRow}>
      {/* Graph area. tabIndex=-1 so it can receive programmatic focus as the
          fallback when a closed drawer's trigger node is no longer in the DOM. */}
      <div className={styles.graphArea} ref={graphContainerRef} tabIndex={-1}>
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
              nodes={filteredNodes}
              links={filteredLinks}
              allLinks={allLinks}
              selectedNode={currentArtifactNode}
              onClick={handleNodeClick}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {/* A drawer, not a modal: no overlay, so the graph behind stays visible
          and clickable and picking another target just re-points the drawer. */}
      {stepDetailTarget && (
        <div
          ref={drawerRef}
          className={styles.stepSidePanel}
          role="dialog"
          aria-label={`Step details — ${stepDetailTarget}`}
        >
          {(() => {
            const target = buildStatus?.targets?.[stepDetailTarget]
            const { status, subtitle, summary } = stepDrawerSummary(target)
            return (
              <>
                <div className={styles.stepSidePanelHeader}>
                  <div className={styles.stepSidePanelIdentity}>
                    <h4 className={styles.stepSidePanelHeading}>{stepDetailTarget}</h4>
                    <div className={styles.stepSidePanelSubtitle}>{subtitle}</div>
                    {status && (
                      <div className={styles.stepSidePanelStatus}>
                        <BuildStatusBadge status={status} />
                      </div>
                    )}
                    {summary && (
                      <div className={styles.stepSidePanelSummary}>{summary}</div>
                    )}
                  </div>
                  <IconButton
                    ref={drawerCloseButtonRef}
                    kind="ghost"
                    label="Close"
                    align="bottom"
                    onClick={() => setStepDetailTarget(null)}
                  >
                    <Close />
                  </IconButton>
                </div>
                <div className={styles.stepSidePanelBody}>
                  <StepDetailsPanel
                    targetName={stepDetailTarget}
                    target={target}
                    sourceUri={build?.source_uri}
                    buildId={build?.uuid}
                  />
                </div>
              </>
            )
          })()}
        </div>
      )}
      </div>

      {artifactNavNode?.hfUrl ? (
        <ComposedModal
          open={artifactNavNode !== null}
          onClose={() => setArtifactNavNode(null)}
          size="sm"
        >
          <ModalHeader>{artifactNavModalHeader(artifactNavNode)}</ModalHeader>
          <ModalBody />
          <ModalFooter className={styles.navModalActions}>
            <Button
              kind="secondary"
              onClick={() => {
                setArtifactNavNode(null);
              }}
            >
              Cancel
            </Button>
            <Button
              kind="secondary"
              onClick={() => {
                if (artifactNavNode)
                  router.push(`/dashboard/artifacts/_/?id=${artifactNavNode.node.id}`);
                setArtifactNavNode(null);
              }}
            >
              View artifact page
            </Button>

            <Button
              kind="secondary"
              renderIcon={Launch}
              href={artifactNavNode.hfUrl}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setArtifactNavNode(null)}
            >
              Open on HuggingFace
            </Button>
          </ModalFooter>
        </ComposedModal>
      ) : (
        <Modal
          open={artifactNavNode !== null}
          onRequestClose={() => setArtifactNavNode(null)}
          modalHeading="Navigate to artifact"
          primaryButtonText="Proceed"
          secondaryButtonText="Cancel"
          onRequestSubmit={() => {
            if (artifactNavNode)
              router.push(`/dashboard/artifacts/_/?id=${artifactNavNode.node.id}`);
            setArtifactNavNode(null);
          }}
          onSecondarySubmit={() => setArtifactNavNode(null)}
          size="sm"
        >
          <p>
            Go to the artifact page for{" "}
            <strong>
              {artifactNavNode?.node.title || artifactNavNode?.node.id}
            </strong>
            ?
          </p>
        </Modal>
      )}
    </div>
  );
})

export default LineagePanelInner

// Re-export GraphHandle for use in parent
export type { GraphHandle }
