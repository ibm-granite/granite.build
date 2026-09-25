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
  /**
   * Stable identity of the subject being graphed (the build or artifact id).
   * Changing it earns a fresh auto-fit even if the user had panned the previous
   * graph; it must NOT change as the same graph grows or is re-filtered.
   */
  graphKey?: string
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
  // The resize observer needs the *current* selection, but must not re-subscribe
  // every time it changes (that would re-seed lastWidth and lose the delta), so
  // read it through a ref rather than closing over the prop.
  const selectedNodeRef = React.useRef(props.selectedNode)
  selectedNodeRef.current = props.selectedNode

  // A different graph (new artifact/build) earns a fresh fit even if the user had
  // panned the previous one. Identity must be stable for the *same* graph as it
  // changes shape: a live build gaining a node, a node expansion, or a focus/depth
  // re-filter is still the same graph, and deriving identity from the node set
  // (or from nodes[0], which the subgraph filter can reorder) would reset the
  // flag and yank the user's pan/zoom back to a full fit on the next poll.
  // Callers therefore pass the subject's id explicitly.
  const graphIdentity = props.graphKey ?? ''
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
    // A viewport narrower than the padding makes the available space negative, so
    // there is no scale that fits. Clamping to MIN_FIT_K would "fit" by collapsing
    // the graph to an invisible speck (this happens while the drawer squeezes the
    // graph pane), so skip the fit instead and keep the current transform — the
    // resize observer re-fits once the pane is a usable size again.
    if (scale <= 0) return null
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
      // A `zoom` event carrying a sourceEvent is real user movement: d3 emits it
      // once the pointer actually moves (pan) or the wheel turns. This must not
      // be a `start` handler — d3-zoom fires `start` on every mousedown inside
      // the SVG, and nodes deliberately let that propagate so the graph can be
      // dragged by its nodes. Flagging on `start` therefore treated a plain node
      // click (opening the drawer) as a viewport takeover, permanently disabling
      // the auto-fit for nodes added by later status polls.
      //
      // Deliberately NOT gated on "the transform changed": a pan that gets fully
      // clamped (e.g. against a boundary) could in principle emit a `zoom` event
      // with an unchanged transform. That is still intent, and ignoring it would
      // let the next poll re-fit the graph out from under them.
      if (event.sourceEvent) hasUserAdjustedRef.current = true
    })

    svg.call(zoomRef.current)

    // Fit once per layout, until the user takes over panning/zooming — after that
    // their viewport is preserved across re-renders (hover, selection, polling).
    if (!hasUserAdjustedRef.current) {
      const fit = computeFitTransform()
      if (fit) transformRef.current = fit
    }

    // Programmatic: fires the `zoom` handler with no sourceEvent, so the fit
    // itself is not mistaken for a user adjustment.
    //
    // interrupt() first: `resetView` starts a 300ms transition toward the
    // pre-expansion fit, and the ELK relayout it triggers usually lands well
    // inside that window. A plain `.call(zoom.transform, …)` does NOT cancel a
    // running transition, so without this the in-flight transition keeps
    // interpolating past our fresh fit and snaps the view back to the stale one.
    svg.interrupt()
    svg.call(zoomRef.current.transform, transformRef.current)

    return () => {
      svg.on('.zoom', null)
    }
    // Keyed on `positions` (layout), not `nodeElements`: nodeElements also
    // rebuilds on `props.selectedNode` (for the node-highlight prop), and this
    // effect doing the same rebind + fit on every click would repeat the exact
    // "thrash" the hoverNode split above was written to avoid, just gated on
    // click instead of hover.
  }, [positions, computeFitTransform])

  // Handle container resize. Coalesce bursts (a drag-resize fires the observer
  // many times per second) into one update per animation frame rather than
  // recomputing + applying a transform on every single firing.
  //
  // Two cases, split on whether the user owns the viewport:
  //   - untouched: recompute the fit, as before.
  //   - user-adjusted: keep their scale and shift by half the width delta, so
  //     whatever was centered stays centered, then pull the selected node back
  //     inside the pane if the shift still left it outside. Opening the ~33rem
  //     drawer shrinks the SVG by ~528px; without this the transform is unchanged
  //     and focused content slides out of view with no way back but Reset.
  // Their zoom level is never discarded either way.
  React.useEffect(() => {
    const svg = svgRef.current
    if (!svg || typeof ResizeObserver === 'undefined') return

    // Seeded on first observation below, so the initial firing is a no-op rather
    // than a shift against a phantom width of 0.
    let lastWidth = 0
    let rafId = 0
    const observer = new ResizeObserver(() => {
      if (rafId) return
      rafId = requestAnimationFrame(() => {
        rafId = 0
        if (!zoomRef.current) return
        const width = svg.clientWidth
        const previousWidth = lastWidth
        lastWidth = width

        if (!hasUserAdjustedRef.current) {
          const fit = computeFitTransform()
          if (fit) d3.select(svg).call(zoomRef.current.transform, fit)
          return
        }

        // Recentre on the user's own transform. `transformRef` is kept current by
        // the zoom handler, so this composes with their latest pan/zoom rather
        // than a stale one.
        if (!previousWidth || !width || width === previousWidth) return
        const current = transformRef.current
        let shifted = current.translate((width - previousWidth) / 2 / current.k, 0)

        // Half-the-delta keeps the *centre* fixed, which is the right default but
        // not enough on its own: a selected node already near an edge can still
        // land outside the narrowed pane. When there is a selection — the node
        // whose drawer caused the resize, and the one thing the user is certainly
        // looking at — pull it back inside the visible band instead.
        const selected = selectedNodeRef.current
        const pos = selected
          ? positionsRef.current?.children?.find((n) => n.id === selected.id)
          : undefined
        if (pos && width > FIT_PADDING * 2) {
          const applied = BASE_SCALE * shifted.k
          // Node bounds in screen space under the shifted transform.
          const left = shifted.x + (pos.x ?? 0) * applied
          const right = left + (pos.width ?? 0) * applied
          const overflowRight = right - (width - FIT_PADDING)
          const overflowLeft = FIT_PADDING - left
          // Correct the left edge first. A node wider than the pane overflows both
          // sides at once and cannot be fully shown; pinning its left edge reveals
          // where it starts, and picking one side unconditionally also keeps the
          // choice stable instead of alternating between edges on every resize.
          const correction =
            overflowLeft > 0 ? overflowLeft : overflowRight > 0 ? -overflowRight : 0
          if (correction !== 0) shifted = shifted.translate(correction / shifted.k, 0)
        }

        d3.select(svg).call(zoomRef.current.transform, shifted)
      })
    })
    observer.observe(svg)
    lastWidth = svg.clientWidth
    return () => {
      if (rafId) cancelAnimationFrame(rafId)
      observer.disconnect()
    }
  }, [computeFitTransform])

  // Clear the auto-fit lock and transition to the current layout's fit.
  // `fallback` is used when there is no layout to measure: a transform to go to
  // anyway, or null to leave the viewport untouched.
  const fitNow = React.useCallback(
    (fallback: d3.ZoomTransform | null) => {
      hasUserAdjustedRef.current = false
      if (!svgRef.current || !zoomRef.current) return
      const target = computeFitTransform() ?? fallback
      if (!target) return
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, target)
    },
    [computeFitTransform]
  )

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (svgRef.current && zoomRef.current) {
        // Programmatic scaleBy fires no sourceEvent, so the zoom handler's
        // sourceEvent check won't flag it — mark it here or a later relayout
        // snaps the toolbar-set zoom back to auto-fit.
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
    // Both reset entries fit the whole graph — which is what the user wants
    // when something (a skeleton stub, a far-downstream branch) is off-screen —
    // and clear the user-adjusted flag so a later relayout may auto-fit again.
    //
    // They differ only in the no-layout fallback. `resetZoom` is the toolbar
    // control on an unchanged node set: there is always a layout to fit, and if
    // computeFitTransform somehow yields nothing, INITIAL_TRANSFORM is a sane
    // home position. `resetView` is called when the node set is *expected* to
    // expand and an async ELK relayout is likely; with no layout yet there is
    // nothing meaningful to fit, so it leaves the viewport alone and lets the
    // layout-driven auto-fit handle it rather than lurching to a default.
    //
    // Both fit against the *current* layout immediately, because the expansion
    // is not guaranteed: clearing a filter whose subgraph already covered every
    // node yields a shallow-equal `props.nodes`, so the layout effect never
    // re-runs and no auto-fit ever arrives — "Reset view" would look like a
    // no-op. When a relayout does follow, its auto-fit interrupts this
    // transition (see svg.interrupt() in the layout effect) and fits again.
    resetZoom: () => {
      fitNow(INITIAL_TRANSFORM)
    },
    resetView: () => {
      fitNow(null)
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
      // Focusing a node is a deliberate viewport choice, so it must survive the
      // next relayout — on a live build any status poll re-runs the layout effect,
      // and without this flag the auto-fit would immediately discard the centring.
      // (Programmatic transforms fire no sourceEvent, so the zoom handler's
      // sourceEvent check won't set it for us; "Reset view" clears it again.)
      hasUserAdjustedRef.current = true
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
  }), [computeFitTransform, fitNow])

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
