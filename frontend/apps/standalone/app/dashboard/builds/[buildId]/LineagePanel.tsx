'use client'

import * as React from 'react'
import { Button, ComposedModal, InlineLoading, Modal, ModalBody, ModalFooter, ModalHeader, OverflowMenu, OverflowMenuItem } from '@carbon/react'
import {
  ArrowLeft,
  ArrowRight,
  CenterSquare,
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
import type { Build, BuildStatusDetail } from '@granite-build/ui-core/types'
import { getArtifact } from '@granite-build/ui-core/api/gbserver'
import { getBuildArchiveFiles } from '@granite-build/ui-core/api/gbserver'
import Graph, { type ElkNodeEx, type GraphHandle, type NodeType } from '@granite-build/ui-core/components/LineageGraph/Graph'
import { getSubgraph, getHuggingFaceUrl } from '@granite-build/ui-core/components/LineageGraph/diagramUtilities'
import { artifactTypeToNodeType, depthForNextLevel, type IndexElkNode, mergeElkGraphs, visibleLevels } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import { useLineageExpansion, type ExpandDirection } from '@granite-build/ui-core/components/LineageGraph/useLineageExpansion'
import StepDrawer from './StepDrawer'
import JobDrawer from './JobDrawer'

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
  // A job an expansion brought in that is not one of this build's targets: it has
  // no target in buildStatus, so its drawer is read from the lineage index.
  const [jobNodeId, setJobNodeId] = React.useState<string | null>(null)
  // Whichever drawer is open: Escape and focus handling cover both.
  const openDrawerKey = stepDetailTarget ?? jobNodeId

  // Where focus was before the drawer opened, so we can hand it back on close —
  // otherwise a keyboard user is dropped at the top of the document.
  const drawerReturnFocusRef = React.useRef<HTMLElement | null>(null)
  const drawerCloseButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const drawerRef = React.useRef<HTMLDivElement | null>(null)
  // Focus fallback when the drawer's trigger node has been detached by a re-render.
  const graphContainerRef = React.useRef<HTMLDivElement | null>(null)

  React.useEffect(() => {
    if (!openDrawerKey) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // This is a non-modal drawer — the graph behind it stays interactive — so
      // only swallow Escape when focus is actually inside the drawer. Otherwise
      // a user mid-interaction with the graph would have the drawer yanked shut.
      if (drawerRef.current?.contains(document.activeElement)) {
        setStepDetailTarget(null)
        setJobNodeId(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [openDrawerKey])

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
    drawerWasOpenRef.current = Boolean(openDrawerKey)

    if (openDrawerKey) {
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
  }, [openDrawerKey])

  const { nodes: buildNodes, links: buildLinks, artifactIds } = React.useMemo(
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
  const enrichedBuildNodes = React.useMemo<ElkNodeEx[]>(() => {
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

    return buildNodes.map((node) => {
      const enriched = artifactMap.get(node.id)
      if (enriched) {
        return { ...node, title: enriched.name, type: enriched.type }
      }
      // This build's own targets are runs of this build.
      if (node.type === 'Build' && !node.planned && build?.uuid) return { ...node, buildId: build.uuid }
      return node
    })
  }, [buildNodes, artifactQueries, uuidArtifactIds, build?.uuid])

  const artifactUriMap = React.useMemo(() => {
    const map = new Map<string, string>()
    uuidArtifactIds.forEach((id, i) => {
      const uri = artifactQueries[i]?.data?.uri
      if (uri) map.set(id, uri)
    })
    return map
  }, [artifactQueries, uuidArtifactIds])

  // The build graph keys artifacts by UUID, the lineage index by URI: fold index
  // nodes back onto the build's UUIDs so an expansion joins the graph, not beside it.
  const uriToUuid = React.useMemo(
    () => new Map(Array.from(artifactUriMap, ([id, uri]) => [uri, id])),
    [artifactUriMap]
  )
  // Same for runs: the index names a target run `run:<job_id>`, and its job_id is
  // the target run's uuid, so it folds onto the build's `target-<name>` node.
  const runToTarget = React.useMemo(
    () => new Map(
      Object.entries(buildStatus?.targets ?? {})
        .filter(([, run]) => run.uuid)
        .map(([name, run]) => [`run:${run.uuid}`, `target-${name}`])
    ),
    [buildStatus]
  )
  const renameIndexId = React.useCallback(
    (id: string) => uriToUuid.get(id) ?? runToTarget.get(id) ?? id,
    [uriToUuid, runToTarget]
  )
  const expansion = useLineageExpansion(renameIndexId)

  const { nodes: enrichedNodes, links: allLinks } = React.useMemo(
    () => mergeElkGraphs({ nodes: enrichedBuildNodes, links: buildLinks }, expansion.extra),
    [enrichedBuildNodes, buildLinks, expansion.extra]
  )

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
  const [showBuildInfo, setShowBuildInfo] = React.useState(true)

  // The current artifact's node is always highlighted on artifact pages
  // (showFocusNode is only true there) — this is not click-driven.
  const currentArtifactNode = React.useMemo(
    () => (showFocusNode && initialFocusNodeId
      ? enrichedNodes.find((n) => n.id === initialFocusNodeId)
      : undefined),
    [showFocusNode, initialFocusNodeId, enrichedNodes]
  )

  // The node whose step drawer is open. Opening that drawer is what shrinks the
  // graph pane, so this is the node the resize compensation must keep on-screen —
  // and, unlike currentArtifactNode, it is set on build pages (where the drawer
  // exists) rather than only on artifact pages.
  //
  // Memoized on `stepDetailTarget` alone, not on `enrichedNodes`: both consumers
  // of `selectedNode` read only `.id` (GraphNode's isSelected check, and the
  // resize compensation's position lookup), and `enrichedNodes` is a fresh array
  // on every status poll. Depending on it would hand Graph a new object each
  // poll, defeating its React.memo and re-running the nodeElements effect on
  // every tick of a live build with the drawer open.
  const openDrawerNode = React.useMemo(
    () => (stepDetailTarget
      ? ({ id: `${TARGET_NODE_PREFIX}${stepDetailTarget}` } as ElkNodeEx)
      : undefined),
    [stepDetailTarget]
  )

  const jobNode = React.useMemo(
    () => (jobNodeId ? (enrichedNodes.find((n) => n.id === jobNodeId) as IndexElkNode | undefined) : undefined),
    [enrichedNodes, jobNodeId]
  )

  const { filteredNodes, filteredLinks } = React.useMemo(() => {
    if (!focusNodeId || (upstreamLevels === Infinity && downstreamLevels === Infinity)) {
      return { filteredNodes: enrichedNodes, filteredLinks: allLinks }
    }
    const sub = getSubgraph(focusNodeId, downstreamLevels, upstreamLevels, enrichedNodes, allLinks)
    return { filteredNodes: sub.nodes, filteredLinks: sub.links }
  }, [focusNodeId, upstreamLevels, downstreamLevels, enrichedNodes, allLinks])

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
    if (!stepDetailTarget || enrichedNodes.length === 0) return
    const nodeId = `${TARGET_NODE_PREFIX}${stepDetailTarget}`
    if (enrichedNodes.some((n) => n.id === nodeId)) return
    setStepDetailTarget(null)
  }, [stepDetailTarget, enrichedNodes])

  // A click selects the node Upstream/Downstream expand from, and a target
  // (run) node also opens its step details. Navigating to an artifact is the
  // toolbar's "Open artifact", so the two never compete.
  const handleNodeClick = (node: ElkNodeEx) => {
    // Skeleton stubs are placeholders for hidden branches, not real nodes: their
    // id still starts with TARGET_NODE_PREFIX when they hang off a target, and
    // opening the drawer for one would close again at once.
    if (node.type === 'skeleton-source' || node.type === 'skeleton-target') return
    setFocusNodeId(node.id)
    const isTarget = node.type === 'Build' && node.id.startsWith(TARGET_NODE_PREFIX)
    setStepDetailTarget(isTarget ? node.id.slice(TARGET_NODE_PREFIX.length) : null)
    setJobNodeId(node.type === 'Build' && !isTarget && (node as IndexElkNode).indexNode ? node.id : null)
  }

  const handleOpenArtifact = () => {
    const node = focusNodeId ? enrichedNodes.find((n) => n.id === focusNodeId) : undefined
    if (!node || node.type === 'Build' || !isUUID(node.id)) return
    const uri = artifactUriMap.get(node.id)
    setArtifactNavNode({ node, hfUrl: uri ? getHuggingFaceUrl(uri) : null })
  }

  // Back to the page's own artifact, now that clicks can select other nodes.
  const handleFocusNode = () => {
    const target = initialFocusNodeId ?? focusNodeId
    if (!target) return
    setFocusNodeId(target)
    setUpstreamLevels(Infinity)
    setDownstreamLevels(Infinity)
    setPartial(false)
    graphRef.current?.centerOnNode?.(target)
  }

  // The id to seed GET /lineage/graph from: a build artifact's URI, a target's
  // `run:<uuid>`, or the id of a node an earlier expansion brought in (already an
  // index id). Nodes the index has no id for fall back to walking the loaded graph.
  const indexSeedFor = (nodeId: string): string | null => {
    if (artifactUriMap.has(nodeId)) return artifactUriMap.get(nodeId)!
    for (const [runId, targetId] of runToTarget) if (targetId === nodeId) return runId
    if (expansion.extra.nodes.some((n) => n.id === nodeId)) return nodeId
    return null
  }

  const expandFromIndex = (direction: ExpandDirection): boolean => {
    if (!focusNodeId) return false
    const seed = indexSeedFor(focusNodeId)
    if (!seed) return false
    // Show everything: the remote expansion is the new frontier.
    setUpstreamLevels(Infinity)
    setDownstreamLevels(Infinity)
    setPartial(false)
    // One level past what is on screen for this node, however it got there.
    const depth = depthForNextLevel(seed, visibleLevels(focusNodeId, allLinks, direction))
    void expansion.expand(seed, direction, depth, { nodes: enrichedNodes, links: allLinks })
    return true
  }

  const levelText = (n: number, exhausted: boolean) => {
    const text = `${n} ${n === 1 ? 'level' : 'levels'}`
    if (exhausted) return n ? `${text} (all)` : 'none'
    return text
  }

  // What each button will do for the selected node, so the toolbar can say it: a
  // node the index knows loads more lineage beyond the build; any other node only
  // narrows or widens the build graph already on screen.
  const focusNode = focusNodeId ? enrichedNodes.find((n) => n.id === focusNodeId) : undefined
  const focusName = focusNode?.title || focusNodeId || ''
  const focusSeed = focusNodeId ? indexSeedFor(focusNodeId) : null
  // Only artifacts with a gbserver record have a page to open.
  const canOpenArtifact = Boolean(focusNode && focusNode.type !== 'Build' && isUUID(focusNode.id))
  const directionState = (direction: ExpandDirection, localLevels: number) => {
    if (!focusNodeId) {
      return { disabled: true, text: '', title: 'Select a node to expand its lineage' }
    }
    // Measured on the graph on screen, so it holds whichever node was expanded.
    const shown = visibleLevels(focusNodeId, filteredLinks, direction)
    if (focusSeed) {
      const exhausted = expansion.isExhausted(focusSeed, direction)
      return {
        disabled: expansion.loading !== null || exhausted,
        text: levelText(shown, exhausted),
        title: exhausted
          ? `Nothing more ${direction} of ${focusName}`
          : `Load level ${shown + 1} ${direction} of ${focusName} from the lineage index`,
      }
    }
    // Local-only nodes (targets): with every level shown there is nothing to add.
    const all = localLevels === Infinity
    return {
      disabled: all,
      text: levelText(shown, all),
      title: all
        ? `The whole build graph is already shown; ${focusName} has no lineage beyond the build`
        : `Show one more ${direction} level of the build graph`,
    }
  }
  const upState = directionState('upstream', upstreamLevels)
  const downState = directionState('downstream', downstreamLevels)

  const handleUpstream = () => {
    if (!focusNodeId) return
    graphRef.current?.centerOnNodeAfterLayout(focusNodeId)
    if (expandFromIndex('upstream')) return
    const newUp = upstreamLevels === Infinity ? 2 : upstreamLevels + 1
    const sub = getSubgraph(focusNodeId, downstreamLevels, newUp, enrichedNodes, allLinks)
    setUpstreamLevels(sub.hasMoreUpstream ? newUp : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  const handleDownstream = () => {
    if (!focusNodeId) return
    graphRef.current?.centerOnNodeAfterLayout(focusNodeId)
    if (expandFromIndex('downstream')) return
    const newDown = downstreamLevels === Infinity ? 2 : downstreamLevels + 1
    const sub = getSubgraph(focusNodeId, newDown, upstreamLevels, enrichedNodes, allLinks)
    setDownstreamLevels(sub.hasMoreDownstream ? newDown : Infinity)
    setPartial(sub.hasMoreUpstream || sub.hasMoreDownstream)
  }

  const noLineage = !loading && !statusError && enrichedNodes.length === 0

  return (
    <div className={styles.container}>
      {/* Toolbar */}
      {/* Any toolbar button closes the open drawer. Capture runs before the
          button's own handler, which still sees this render's selection. */}
      <div
        className={styles.toolbar}
        onClickCapture={(e) => {
          if (!(e.target as HTMLElement).closest('button')) return
          setStepDetailTarget(null)
          setJobNodeId(null)
        }}
      >
        <div className={styles.toolbarLeft}>
          <Button
            size="sm"
            kind="ghost"
            renderIcon={ArrowLeft}
            disabled={upState.disabled}
            title={upState.title}
            onClick={handleUpstream}
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
            disabled={downState.disabled}
            title={downState.title}
            onClick={handleDownstream}
          >
            Downstream
          </Button>
          <span className={styles.levelIndicator}>
            {focusNodeId
              ? <>From <strong>{focusName}</strong> · ← {upState.text} · {downState.text} →</>
              : 'Click a node to expand its lineage'}
          </span>

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
                // Clear every filter, then fit. `resetView` fits against the
                // current layout immediately and clears the user-adjusted flag,
                // so a relayout triggered by the expanded node set simply
                // auto-fits again over this one with nothing fighting it —
                // whether or not the graph was actually filtered.
                setFocusNodeId(null);
                setUpstreamLevels(Infinity);
                setDownstreamLevels(Infinity);
                setPartial(false);
                expansion.reset();
                setJobNodeId(null);
                graphRef.current?.resetZoom();
              }}
            />
            <OverflowMenuItem
              className="overflow-item"
              itemText={showBuildInfo ? 'Hide build IDs' : 'Show build IDs'}
              onClick={() => setShowBuildInfo((v) => !v)}
            />
          </OverflowMenu>

          <div className={styles.toolbarDivider} />

          <Button
            size="sm"
            kind="ghost"
            renderIcon={Launch}
            disabled={!canOpenArtifact}
            title={canOpenArtifact ? `Open ${focusName}` : 'Select an artifact to open it'}
            onClick={handleOpenArtifact}
          >
            Open artifact
          </Button>
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
        {/* Overlaid, not in the flow: a banner above the graph would push it down
            on every expansion. */}
        {(expansion.loading || Boolean(expansion.error) || expansion.unexpanded > 0) && (
          <div className={styles.expansionStatus}>
            {expansion.loading && <InlineLoading description={`Loading ${expansion.loading} lineage…`} />}
            {Boolean(expansion.error) && (
              <span className={styles.expansionError}>Failed to expand lineage: {String(expansion.error)}</span>
            )}
            {!expansion.loading && expansion.unexpanded > 0 && <span>{expansion.unexpanded} nodes not expanded.</span>}
          </div>
        )}
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
              nodes={filteredNodes}
              links={filteredLinks}
              allLinks={allLinks}
              selectedNode={focusNode ?? currentArtifactNode ?? openDrawerNode}
              showBuildInfo={showBuildInfo}
              onClick={handleNodeClick}
              onSvgRendered={() => setRendered(true)}
            />
          </>
        )}
      </div>

      {(stepDetailTarget || jobNode) && (
      <div className={styles.drawerSlot}>
      {stepDetailTarget && (
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
      )}
      {!stepDetailTarget && jobNode && (
        <JobDrawer
          node={jobNode}
          onClose={() => setJobNodeId(null)}
          drawerRef={drawerRef}
          closeButtonRef={drawerCloseButtonRef}
        />
      )}
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
