'use client'

import { ArrowLeftMarker, ArrowRightMarker, CircleMarker, TeeMarker } from '@carbon/charts-react'
import styles from './Graph.module.scss'
import type { ElkExtendedEdge, ElkNode } from 'elkjs'
import ELK from 'elkjs/lib/elk.bundled.js'
import * as React from 'react'
import * as d3 from 'd3'
import LinkArrow from './LinkArrow'
import GraphNode from './GraphNode'

export type NodeType =
  | 'Build'
  | 'Artifact'
  | 'Model'
  | 'Fileset'
  | 'Dataset'
  | 'Table'
  | 'Bucket'
  | 'skeleton-source'
  | 'skeleton-target'

export interface ElkNodeEx extends ElkNode {
  title: string
  subtitle?: string
  type?: NodeType | string
  highlight?: boolean
  planned?: boolean
  children?: ElkNodeEx[]
}

export interface GraphHandle {
  zoomIn(): void
  zoomOut(): void
  resetZoom(): void
  /** Clear the user-adjusted flag so the next relayout auto-fits (no immediate
   *  fit against stale positions). For callers that also change the node set. */
  resetView(): void
  currentZoom(): number
  centerOnNode(nodeId: string): void
}

interface GraphProps {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  onClick?: (node: ElkNodeEx) => void
  selectedNode?: ElkNodeEx
  allLinks?: ElkExtendedEdge[]
  onSvgRendered?: (svg: SVGSVGElement) => void
}

const elk = new ELK()
const INITIAL_TRANSFORM = d3.zoomIdentity.translate(48, 32)
// Breathing room, in px, between the graph's bounding box and the viewport edge.
const FIT_PADDING = 32
// Lower bound on zoom. A large graph's ideal "fit everything" scale can be very
// small, so the floor has to sit below any normal interactive zoom — a higher
// floor would clamp the fit back up and leave content clipped after auto-fit.
const MIN_FIT_K = 0.02

function GraphComponent(props: GraphProps, ref: React.Ref<GraphHandle>) {
  const { onClick } = props

  const nodeMapRef = React.useRef<Map<string, ElkNodeEx>>(new Map())
  const [positions, setPositions] = React.useState<ElkNode | null>(null)
  const positionsRef = React.useRef<ElkNode | null>(null)
  const [nodeElements, setNodeElements] = React.useState<React.ReactNode>(null)
  const [linkElements, setLinkElements] = React.useState<React.ReactNode>(null)
  const [hoverNode, setHoverNode] = React.useState<ElkNodeEx | null>(null)


  const buildSkeleton = (children: ElkNodeEx[], visibleLinks: ElkExtendedEdge[], allLinks: ElkExtendedEdge[]) => {
    const skeletonNodes: ElkNodeEx[] = []
    const skeletonEdges: ElkExtendedEdge[] = []

    children.forEach((node) => {
      const nodeInputId = `${node.id}-input`
      const nodeOutputId = `${node.id}-output`

      const totalIncoming = allLinks.filter((l) => l.targets.includes(nodeInputId)).length
      const totalOutgoing = allLinks.filter((l) => l.sources.includes(nodeOutputId)).length
      const visibleIncoming = visibleLinks.filter((l) => l.targets.includes(nodeInputId)).length
      const visibleOutgoing = visibleLinks.filter((l) => l.sources.includes(nodeOutputId)).length

      if (visibleIncoming < totalIncoming) {
        const skId = `${node.id}-upstream-skeleton`
        skeletonNodes.push({ id: skId, width: 224, height: 32, labels: [{ text: '' }], title: '', type: 'skeleton-source' })
        skeletonEdges.push({ id: `e-${skId}-to-${node.id}`, sources: [skId], targets: [nodeInputId] })
      }

      if (visibleOutgoing < totalOutgoing) {
        const skId = `${node.id}-downstream-skeleton`
        skeletonNodes.push({ id: skId, width: 224, height: 32, labels: [{ text: '' }], title: '', type: 'skeleton-target' })
        skeletonEdges.push({ id: `e-${node.id}-to-${skId}`, sources: [nodeOutputId], targets: [skId] })
      }
    })

    return { skeletonNodes, skeletonEdges }
  }

  const cleanNodePositions = (graph: ElkNode) => {
    if (!graph) return
    if (graph.children) {
      for (const child of graph.children) {
        delete child.x
        delete child.y
        cleanNodePositions(child)
      }
    }
    if (graph.edges) {
      for (const edge of graph.edges) {
        delete (edge as any).sections
      }
    }
  }

  const withPorts = (nodes: ElkNodeEx[]): ElkNodeEx[] =>
    nodes.map((node) => ({
      ...node,
      layoutOptions: {
        ...node.layoutOptions,
        portConstraints: 'FIXED_SIDE',
      } as Record<string, string>,
      ports: [
        { id: `${node.id}-input`,  layoutOptions: { 'port.side': 'WEST', 'port.alignment': 'CENTER' } },
        { id: `${node.id}-output`, layoutOptions: { 'port.side': 'EAST', 'port.alignment': 'CENTER' } },
      ],
    } as ElkNodeEx))

  const updateGraph = React.useCallback(() => {
    setNodeElements(null)
    setLinkElements(null)

    const allLinks = props.allLinks || props.links
    const { skeletonNodes, skeletonEdges } = buildSkeleton(props.nodes, props.links, allLinks)
    const links = [...props.links, ...skeletonEdges]

    // ELK's WebWorker JSON round-trip strips non-schema fields (title, type, etc).
    // Store display data in a ref so buildNodes can always access it regardless of ELK stripping.
    nodeMapRef.current = new Map([...props.nodes, ...skeletonNodes].map((n) => [n.id, n]))

    const graph: ElkNode = {
      id: 'root',
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
        'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
        'layered.contentAlignment': 'V_CENTER',
        'spacing.nodeNodeBetweenLayers': '250',
        'spacing.edgeNode': '35',
        'elk.partitioning.activate': 'true',
        'elk.layered.wrapping.strategy': 'OFF',
        'elk.direction': 'RIGHT',
        'elk.layered.mergeEdges': 'true',
        'elk.layered.spacing.edgeNodeBetweenLayers': '20',
        'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
        'elk.layered.nodePlacement.bk.fixedAlignment': 'BALANCED',
        'elk.layered.cycleBreaking.strategy': 'DEPTH_FIRST',
      },
      children: withPorts([...props.nodes, ...skeletonNodes]),
      edges: links,
    }

    cleanNodePositions(graph)

    elk.layout(graph)
      .then((g) => { setPositions(g); positionsRef.current = g })
      .catch(console.error)
  }, [props.nodes, props.links, props.allLinks])

  React.useEffect(() => {
    updateGraph()
  }, [updateGraph])

  const buildNodes = (p: ElkNode): React.ReactNode => {
    return (p.children || []).map((n, i) => {
      const src = nodeMapRef.current.get(n.id)
      const elkNode = n as ElkNodeEx
      // Use ELK output as base (preserves x/y/width/height), override display fields from ref.
      const node: ElkNodeEx = {
        ...elkNode,
        title: src?.title ?? elkNode.title ?? '',
        type: src?.type ?? elkNode.type,
        highlight: src?.highlight ?? elkNode.highlight,
        subtitle: src?.subtitle ?? elkNode.subtitle,
      }
      return (
        <GraphNode
          key={`node_${i}`}
          node={node}
          onClick={onClick}
          onMouseHover={(hovered) => setHoverNode(hovered)}
          selectedNode={props.selectedNode}
        />
      )
    })
  }

  const buildLinks = (p: ElkNode, hover: ElkNodeEx | null): React.ReactNode => {
    return (p.edges || [])
      .filter((e) => !!(e as any).sections)
      .map((edge, i) => {
        const isHighlighted =
          hover &&
          (edge.targets.includes(`${hover.id}-input`) || edge.sources.includes(`${hover.id}-output`))
        const isSkeleton = edge.id.includes('-skeleton')

        return (
          <LinkArrow
            key={`link_${i}`}
            link={edge}
            color={isSkeleton ? '#E0E0E0' : isHighlighted ? '#5D5D5D' : '#878787'}
            markerEnd={isSkeleton ? 'arrow' : 'arrow-right'}
            markerStart={isSkeleton ? undefined : undefined}
            className={isSkeleton ? styles.linkSkeleton : isHighlighted ? styles.linkHighlighted : styles.linkDefault}
          />
        )
      })
  }

  // Nodes depend on layout + selection only — NOT on hoverNode, which is used
  // solely for edge highlighting below. Rebuilding nodeElements on every hover
  // produced a fresh array that reran the zoom-setup effect (a synchronous
  // getBoundingClientRect + O(n) bounds scan + zoom listener rebind) on every
  // mouse-enter/leave; keeping this off hoverNode avoids that thrash.
  React.useEffect(() => {
    if (positions) setNodeElements(buildNodes(positions))
  }, [positions, props.selectedNode])

  React.useEffect(() => {
    if (positions) setLinkElements(buildLinks(positions, hoverNode))
  }, [positions, hoverNode])

  const svgRef = React.useRef<SVGSVGElement | null>(null)
  const containerRef = React.useRef<SVGGElement | null>(null)
  const zoomRef = React.useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const transformRef = React.useRef(INITIAL_TRANSFORM)
  // Once the user pans or zooms, stop auto-fitting so we never yank their view.
  const hasUserAdjustedRef = React.useRef(false)

  // A different graph (new artifact/build) earns a fresh fit even if the user had
  // panned the previous one. The graph builder emits nodes in a deterministic
  // order, so joining every id captures identity without a per-render sort, and
  // is stable across the status poll's fresh-array-same-nodes churn. (Count plus
  // endpoints alone would collide when two graphs share their first/last node —
  // e.g. a middle target renamed between fetches — and skip a warranted re-fit.)
  const graphIdentity = React.useMemo(
    () => props.nodes.map((n) => n.id).join('|'),
    [props.nodes]
  )
  React.useEffect(() => {
    hasUserAdjustedRef.current = false
  }, [graphIdentity])

  React.useEffect(() => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (props.onSvgRendered && svgRef.current) {
          props.onSvgRendered(svgRef.current)
        }
      })
    })
  }, [linkElements])

  const BASE_SCALE = 0.85

  // Fit the whole laid-out graph — skeleton stubs included — into the viewport.
  // The container's applied scale is always BASE_SCALE * t.k, so we solve for the
  // k that makes the content bounds fit and let the zoom behaviour own the rest.
  const computeFitTransform = React.useCallback((): d3.ZoomTransform | null => {
    const svg = svgRef.current
    const pos = positionsRef.current
    if (!svg || !pos?.children?.length) return null

    const { width: W, height: H } = svg.getBoundingClientRect()
    if (!W || !H) return null

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const n of pos.children) {
      const x = n.x ?? 0
      const y = n.y ?? 0
      minX = Math.min(minX, x)
      minY = Math.min(minY, y)
      maxX = Math.max(maxX, x + (n.width ?? 0))
      maxY = Math.max(maxY, y + (n.height ?? 0))
    }
    if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null

    const contentW = maxX - minX
    const contentH = maxY - minY
    if (contentW <= 0 || contentH <= 0) return null

    // Never zoom past 1:1 — a two-node graph should not balloon to fill the pane.
    const scale = Math.min(
      (W - FIT_PADDING * 2) / contentW,
      (H - FIT_PADDING * 2) / contentH,
      BASE_SCALE,
    )
    // scaleExtent is expressed in k, and the rendered scale is BASE_SCALE * k.
    const k = Math.max(scale / BASE_SCALE, MIN_FIT_K)
    const applied = BASE_SCALE * k

    // Centre the content bounds in the viewport.
    const tx = (W - contentW * applied) / 2 - minX * applied
    const ty = (H - contentH * applied) / 2 - minY * applied

    return d3.zoomIdentity.translate(tx, ty).scale(k)
  }, [])

  React.useEffect(() => {
    if (!svgRef.current || !containerRef.current) return

    const svg = d3.select(svgRef.current)
    const container = d3.select(containerRef.current)

    if (!zoomRef.current) {
      zoomRef.current = d3
        .zoom<SVGSVGElement, unknown>()
        .filter((event) => event.ctrlKey || event.type !== 'wheel')
        // d3.zoom clamps every applied transform to this extent, so the lower
        // bound has to be the fit floor — a higher one would clamp a large
        // graph's computed fit back up and clip it.
        .scaleExtent([MIN_FIT_K, 10])
    }

    zoomRef.current.on('zoom', (event) => {
      const t = event.transform
      container.attr('transform', `translate(${t.x},${t.y}) scale(${BASE_SCALE * t.k})`)
      transformRef.current = event.transform
    })

    svg.call(zoomRef.current)

    // Fit once per layout, until the user takes over panning/zooming — after that
    // their viewport is preserved across re-renders (hover, selection, polling).
    if (!hasUserAdjustedRef.current) {
      const fit = computeFitTransform()
      if (fit) transformRef.current = fit
    }

    svg.call(zoomRef.current.transform, transformRef.current)

    // Registered after the programmatic transform above so the fit itself does
    // not count as a user adjustment.
    zoomRef.current.on('start.userintent', (event) => {
      if (event.sourceEvent) hasUserAdjustedRef.current = true
    })

    return () => {
      svg.on('.zoom', null)
      zoomRef.current?.on('start.userintent', null)
    }
  }, [nodeElements, computeFitTransform])

  // Re-fit on container resize while the user has not taken over the viewport.
  // Coalesce bursts (a drag-resize fires the observer many times per second)
  // into one refit per animation frame rather than recomputing + applying a
  // transform on every single firing.
  React.useEffect(() => {
    const svg = svgRef.current
    if (!svg || typeof ResizeObserver === 'undefined') return

    let rafId = 0
    const observer = new ResizeObserver(() => {
      if (rafId) return
      rafId = requestAnimationFrame(() => {
        rafId = 0
        if (hasUserAdjustedRef.current || !zoomRef.current) return
        const fit = computeFitTransform()
        if (fit) d3.select(svg).call(zoomRef.current.transform, fit)
      })
    })
    observer.observe(svg)
    return () => {
      if (rafId) cancelAnimationFrame(rafId)
      observer.disconnect()
    }
  }, [computeFitTransform])

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (svgRef.current && zoomRef.current) {
        // Programmatic scaleBy fires no sourceEvent, so the start.userintent
        // handler won't flag it — mark it here or a later relayout snaps the
        // toolbar-set zoom back to auto-fit.
        hasUserAdjustedRef.current = true
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1.1)
      }
    },
    zoomOut: () => {
      if (svgRef.current && zoomRef.current) {
        hasUserAdjustedRef.current = true
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1 / 1.1)
      }
    },
    // "Reset" means fit the whole graph, which is what the user wants when
    // something (a skeleton stub, a far-downstream branch) is off-screen. Use
    // this only when the node set is unchanged (the toolbar's Reset Zoom); it
    // fits against the *current* layout immediately.
    resetZoom: () => {
      if (!svgRef.current || !zoomRef.current) return
      hasUserAdjustedRef.current = false
      const target = computeFitTransform() ?? INITIAL_TRANSFORM
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, target)
    },
    // "Reset view" also expands the node set, which kicks off an async ELK
    // relayout. Fitting now would fit against the stale (pre-expansion) layout
    // and then visibly re-snap once the new layout lands. Instead just clear the
    // user-adjusted flag and let the layout-driven auto-fit do the fitting once —
    // when the fresh positions arrive.
    resetView: () => {
      hasUserAdjustedRef.current = false
    },
    currentZoom: () => {
      if (svgRef.current) {
        return d3.zoomTransform(svgRef.current).k * 100
      }
      return 90
    },
    centerOnNode: (nodeId: string) => {
      if (!svgRef.current || !zoomRef.current) return
      const pos = positionsRef.current
      if (!pos?.children) return
      const node = pos.children.find((n) => n.id === nodeId)
      if (!node || node.x === undefined || node.y === undefined) return
      // A one-off centre is not "I've taken over pan/zoom forever" — leave
      // hasUserAdjustedRef alone so a later container resize can still re-fit.
      // (Genuine user pans/zooms set it via the start.userintent handler.)
      const { width: W, height: H } = svgRef.current.getBoundingClientRect()
      const cx = (node.x ?? 0) + (node.width ?? 0) / 2
      const cy = (node.y ?? 0) + (node.height ?? 0) / 2
      // Keep the current zoom if it is already legible, but never leave the node
      // smaller than 1:1 — "Focus" from a zoomed-way-out view must actually bring
      // the node up to a readable size, not just recentre it at a tiny scale.
      // The rendered scale is BASE_SCALE * k, so the translation accounts for k or
      // the node lands off-centre.
      const k = Math.max(d3.zoomTransform(svgRef.current).k, 1)
      const applied = BASE_SCALE * k
      const tx = W / 2 - applied * cx
      const ty = H / 2 - applied * cy
      d3.select(svgRef.current)
        .transition()
        .duration(400)
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty).scale(k))
    },
  }), [computeFitTransform])

  // Compute dimensions from last layout
  const svgWidth = React.useMemo(() => {
    if (!positions?.children) return 4000
    return Math.max(...positions.children.map((n) => (n.x || 0) + (n.width || 0))) + 300
  }, [positions])

  const svgHeight = React.useMemo(() => {
    if (!positions?.children) return 800
    return Math.max(...positions.children.map((n) => (n.y || 0) + (n.height || 0))) + 200
  }, [positions])

  return (
    <div className={styles.container}>
      {linkElements !== undefined && (
        <svg
          id="svg-graph"
          width={svgWidth}
          height={svgHeight}
          style={{ height: '100%', width: '100%', overflow: 'visible' }}
          ref={svgRef}
        >
          <defs>
            <ArrowLeftMarker id="arrow-left" color="#6F6F6F" markerWidth="8" markerHeight="8" refX={4} refY={4} orient="auto" markerUnits="userSpaceOnUse" />
            <ArrowRightMarker id="arrow-right" color="#6F6F6F" markerWidth="8" markerHeight="8" refX={4} refY={4} orient="auto" markerUnits="userSpaceOnUse" />
            <ArrowRightMarker id="arrow" color="#E0E0E0" markerWidth="8" markerHeight="8" refX={4} refY={4} orient="auto" markerUnits="userSpaceOnUse" />
            <TeeMarker id="tee" />
            <CircleMarker id="circleEnd" color="#6F6F6F" />
            <CircleMarker id="circle" position="start" color="#6F6F6F" />
          </defs>
          <g className="zoom-container" ref={containerRef}>
            {linkElements}
            {nodeElements}
          </g>
        </svg>
      )}
    </div>
  )
}

const Graph = React.memo(React.forwardRef(GraphComponent))
export default Graph
