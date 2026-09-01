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

const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending'])

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
    const targetId = `target-${targetName}`
    seenTargets.add(targetName)

    nodes.push({
      id: targetId,
      title: targetName,
      type: 'Build',
      width: 192,
      height: 64,
      labels: [{ text: targetName }],
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

      const targetId = `target-${targetName}`
      seenTargets.add(targetName)

      nodes.push({
        id: targetId,
        title: targetName,
        type: 'Build',
        planned: true,
        width: 192,
        height: 64,
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
      return <h4>Would you like to view <code>{artifactNavNode.node?.title || artifactNavNode.node?.id}</code> on HuggingFace or proceed to the artifact page?`</h4>
    } else {
      return <h4>Would you like to view this artifact on HuggingFace or proceed to the artifact page?`</h4>
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
      setArtifactNavNode({ node, hfUrl: uri ? getHuggingFaceUrl(uri) : null })
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
                setFocusNodeId(null);
                setUpstreamLevels(Infinity);
                setDownstreamLevels(Infinity);
                setPartial(false);
                graphRef.current?.resetZoom();
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

      {/* Graph area */}
      <div className={styles.graphArea}>
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
