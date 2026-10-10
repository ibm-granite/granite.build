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
  // The Granite.build build a run node belongs to, shown when showBuildInfo is on.
  buildId?: string
  // How many jobs a grouped run node stands for; above 1 it draws as a stack.
  stackCount?: number
  children?: ElkNodeEx[]
}

export interface GraphHandle {
  zoomIn(): void
  zoomOut(): void
  /** Back to 100%, centered on nodeId when given (else the graph's origin). */
  resetZoom(nodeId?: string): void
  /** Fit the whole graph, also on a relayout the same click causes. For callers
   *  that also change the node set. */
  resetView(): void
  currentZoom(): number
  centerOnNode(nodeId: string): void
  // Centers nodeId, zooming out only as far as needed to fit the graph, on the relayout this click causes if
  // any, else right away. Relayouts keep the selected node put, so a later one
  // (an index fetch landing) leaves it centered.
  centerOnNodeAfterLayout(nodeId: string): void
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
  // Centered at 100% on the graph's first layout (per graphKey), if present.
  centerNodeId?: string
  onClick?: (node: ElkNodeEx) => void
  selectedNode?: ElkNodeEx
  allLinks?: ElkExtendedEdge[]
  onSvgRendered?: (svg: SVGSVGElement) => void
  // Show each run node's build id under its title.
  showBuildInfo?: boolean
}

const elk = new ELK()
// The default scale lives inside the d3 transform, and the transform is drawn as
// is: scaling it again at draw time would make d3's anchor maths wrong, so every
// zoom click would drift the graph.
const BASE_SCALE = 0.85
const INITIAL_TRANSFORM = d3.zoomIdentity.translate(48, 32).scale(BASE_SCALE)
// A layout wider than this many viewports is wrapped into rows.
const WRAP_WIDTH_FACTOR = 2
// Multipliers on the viewport's aspect ratio tried when wrapping, narrowest first.
const WRAP_RATIO_FACTORS = [1, 1.5, 2, 3, 4, 6, 8]
// One zoom button click.
const ZOOM_STEP = 1.1
const MIN_READABLE_SCALE = 0.6
// Breathing room, in px, between a selected node and the pane edge on resize.
const FIT_PADDING = 32

// Whether two layouts show the same real nodes (skeleton placeholders aside).
const sameNodes = (a: ElkNode | null, b: ElkNode): boolean => {
  const ids = (l: ElkNode | null) =>
    new Set((l?.children ?? []).filter((n) => !n.id.endsWith('-skeleton')).map((n) => n.id))
  const [x, y] = [ids(a), ids(b)]
  return x.size === y.size && [...x].every((id) => y.has(id))
}

function GraphComponent(props: GraphProps, ref: React.Ref<GraphHandle>) {
  const { onClick } = props

  const nodeMapRef = React.useRef<Map<string, ElkNodeEx>>(new Map())
  const [positions, setPositions] = React.useState<ElkNode | null>(null)
  const positionsRef = React.useRef<ElkNode | null>(null)
  const [nodeElements, setNodeElements] = React.useState<React.ReactNode>(null)
  const [linkElements, setLinkElements] = React.useState<React.ReactNode>(null)
  const [hoverNode, setHoverNode] = React.useState<ElkNodeEx | null>(null)
  const anchorIdRef = React.useRef<string | undefined>(undefined)
  const layoutRunRef = React.useRef(0)
  // Set by resetView: a relayout the same click triggered (e.g. Reset view
  // dropping expanded nodes) fits the new layout instead of re-anchoring.
  const resetOnLayoutRef = React.useRef(false)
  // Set by centerOnNodeAfterLayout, the same way.
  const centerOnLayoutRef = React.useRef<string | null>(null)
  anchorIdRef.current = props.selectedNode?.id
  const centerNodeIdRef = React.useRef(props.centerNodeId)
  centerNodeIdRef.current = props.centerNodeId
  // Cleared once the first layout containing centerNodeId has been centered.
  const pendingInitialCenterRef = React.useRef(true)
  // The initial center is measured before the pane has settled its size (e.g. a
  // tab just shown), so resizes redo it until the user pans or zooms.
  const initialViewRef = React.useRef<string | null>(null)


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

  // Lays the graph out in one row, as always, unless that row is more than
  // WRAP_WIDTH_FACTOR viewports wide: then ELK wraps it into rows stacked
  // downward, shaped like the viewport, so a long chain fits the screen instead of
  // running off to the side. Small graphs keep their straight layout.
  const wrapViewportRef = React.useRef<{ width: number; height: number } | null>(null)
  const layoutFitting = async (graph: ElkNode): Promise<ElkNode> => {
    const straight = await elk.layout(structuredClone(graph))
    // Measured once per graph and reused: the drawer narrows the pane by ~33rem,
    // and a relayout while it is open (or after it closes) would otherwise wrap
    // the same graph into a different shape.
    if (!wrapViewportRef.current) {
      const rect = svgRef.current?.parentElement?.getBoundingClientRect()
      if (rect?.width && rect.height) wrapViewportRef.current = { width: rect.width, height: rect.height }
    }
    const viewport = wrapViewportRef.current
    if (!viewport) return straight
    const widthOf = (g: ElkNode) => Math.max(0, ...(g.children ?? []).map((n) => (n.x ?? 0) + (n.width ?? 0)))
    if (widthOf(straight) <= WRAP_WIDTH_FACTOR * (viewport.width / BASE_SCALE)) return straight
    // A graph long enough to wrap is shown near the readable minimum, so size
    // rows to fill the viewport at that scale, in layout units.
    const rowWidth = (viewport.width - 2 * INITIAL_TRANSFORM.x) / MIN_READABLE_SCALE

    // ELK sizes rows from `aspectRatio`, but undershoots it badly (asked 2.4, a
    // chain came back near-square, two nodes a row), and elkjs cannot take manual
    // cuts. So widen the ratio step by step and keep the widest wrap whose rows
    // still fit the viewport: fewest rows, nothing off to the side.
    const wrapped = (factor: number) => elk.layout({
      ...structuredClone(graph),
      layoutOptions: {
        ...graph.layoutOptions,
        'elk.layered.wrapping.strategy': 'SINGLE_EDGE',
        'elk.aspectRatio': String((viewport.width / viewport.height) * factor),
        'elk.layered.wrapping.additionalEdgeSpacing': '40',
      },
    })
    let best = await wrapped(WRAP_RATIO_FACTORS[0])
    for (const factor of WRAP_RATIO_FACTORS.slice(1)) {
      const g = await wrapped(factor)
      if (widthOf(g) > rowWidth) break
      best = g
    }
    return best
  }

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

    const run = ++layoutRunRef.current
    layoutFitting(graph)
      .then((g) => {
        // Layouts are async (and a wrapped one takes two passes): only the latest
        // may land, or a stale one would move the anchor back.
        if (run !== layoutRunRef.current) return
        // A relayout (nodes added upstream, or the graph wrapping into rows) moves
        // everything. Pan so the selected node stays where it was on screen,
        // measured on the final layout, keeping the zoom level.
        if (resetOnLayoutRef.current) {
          resetOnLayoutRef.current = false
          // Stop resetView's transition, which is still easing to the old fit.
          if (svgRef.current) d3.select(svgRef.current).interrupt()
          transformRef.current = fitTransform(g)
          setPositions(g)
          positionsRef.current = g
          return
        }
        const anchorId = anchorIdRef.current
        const before = anchorId ? positionsRef.current?.children?.find((n) => n.id === anchorId) : undefined
        const after = anchorId ? g.children?.find((n) => n.id === anchorId) : undefined
        if (before && after) {
          const t = transformRef.current
          const dx = ((after.x ?? 0) - (before.x ?? 0)) * t.k
          const dy = ((after.y ?? 0) - (before.y ?? 0)) * t.k
          transformRef.current = d3.zoomIdentity.translate(t.x - dx, t.y - dy).scale(t.k)
        }
        // An expansion that grew the graph fits it like Reset view does, instead
        // of centering on the expanded node, but never below a scale that still reads.
        if (centerOnLayoutRef.current && !sameNodes(positionsRef.current, g) && (g.children?.length ?? 0) > (positionsRef.current?.children?.length ?? 0)) {
          centerOnLayoutRef.current = null
          const fit = fitTransform(g)
          transformRef.current = fit.k >= MIN_READABLE_SCALE ? fit : fit.scale(MIN_READABLE_SCALE / fit.k)
        }
        // An expansion that brought nothing new keeps the view as it was. Kept
        // pending on an unchanged relayout: the click's own relayout lands before
        // the index fetch that may still add nodes.
        const centerId = centerOnLayoutRef.current
        if (centerId && !sameNodes(positionsRef.current, g)) {
          centerOnLayoutRef.current = null
          const t = centerFitTransform(g, centerId, transformRef.current.k)
          if (t) transformRef.current = t
        }
        const initialId = centerNodeIdRef.current
        if (pendingInitialCenterRef.current && initialId) {
          const t = centerFitTransform(g, initialId, BASE_SCALE)
          if (t) {
            pendingInitialCenterRef.current = false
            initialViewRef.current = initialId
            transformRef.current = t
          }
        }
        setPositions(g)
        positionsRef.current = g
      })
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
        buildId: src?.buildId ?? elkNode.buildId,
      }
      return (
        <GraphNode
          key={`node_${i}`}
          node={node}
          onClick={onClick}
          onMouseHover={(hovered) => setHoverNode(hovered)}
          selectedNode={props.selectedNode}
          showBuildInfo={props.showBuildInfo}
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
  }, [positions, props.selectedNode, props.showBuildInfo])

  React.useEffect(() => {
    if (positions) setLinkElements(buildLinks(positions, hoverNode))
  }, [positions, hoverNode])

  const svgRef = React.useRef<SVGSVGElement | null>(null)
  const containerRef = React.useRef<SVGGElement | null>(null)
  const zoomRef = React.useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const transformRef = React.useRef(INITIAL_TRANSFORM)
  // The resize observer needs the *current* selection, but must not re-subscribe
  // every time it changes (that would re-seed lastWidth and lose the delta), so
  // read it through a ref rather than closing over the prop.
  const selectedNodeRef = React.useRef(props.selectedNode)
  selectedNodeRef.current = props.selectedNode

  // A different graph (new artifact/build) starts from the home transform rather
  // than the previous graph's pan/zoom. Callers pass the subject's id: a live
  // build gaining nodes or an expansion is still the same graph.
  const graphIdentity = props.graphKey ?? ''
  React.useEffect(() => {
    transformRef.current = INITIAL_TRANSFORM
    wrapViewportRef.current = null
    pendingInitialCenterRef.current = true
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

  // The transform that shows the whole layout, never enlarging past the default
  // scale. INITIAL_TRANSFORM when it already fits, so small graphs look as before.
  // The scale at which `layout` fits the viewport, capped at the default scale.
  const fitScale = (layout: ElkNode | null): number => {
    if (!svgRef.current || !layout?.children?.length) return BASE_SCALE
    const { width: W, height: H } = svgRef.current.getBoundingClientRect()
    const w = Math.max(...layout.children.map((n) => (n.x ?? 0) + (n.width ?? 0)))
    const h = Math.max(...layout.children.map((n) => (n.y ?? 0) + (n.height ?? 0)))
    const k = Math.min(BASE_SCALE, (W - 2 * INITIAL_TRANSFORM.x) / w, (H - 2 * INITIAL_TRANSFORM.y) / h)
    return k > 0 ? k : BASE_SCALE
  }

  // Pans `layout` so `nodeId` sits at the viewport's center, at scale k.
  const centerTransform = (layout: ElkNode, nodeId: string, k: number): d3.ZoomTransform | null => {
    const node = layout.children?.find((n) => n.id === nodeId)
    if (!svgRef.current || !node || node.x === undefined || node.y === undefined) return null
    const { width: W, height: H } = svgRef.current.getBoundingClientRect()
    const cx = node.x + (node.width ?? 0) / 2
    const cy = node.y + (node.height ?? 0) / 2
    return d3.zoomIdentity.translate(W / 2 - k * cx, H / 2 - k * cy).scale(k)
  }

  // Like centerTransform, but zoomed out (never in, never past the readable
  // minimum) until the whole layout fits around the centered node.
  const centerFitTransform = (layout: ElkNode, nodeId: string, k: number): d3.ZoomTransform | null => {
    const node = layout.children?.find((n) => n.id === nodeId)
    if (!svgRef.current || !node || node.x === undefined || node.y === undefined || !layout.children) return null
    const { width: W, height: H } = svgRef.current.getBoundingClientRect()
    const cx = node.x + (node.width ?? 0) / 2
    const cy = node.y + (node.height ?? 0) / 2
    const spanX = Math.max(...layout.children.map((n) => Math.max(cx - (n.x ?? 0), (n.x ?? 0) + (n.width ?? 0) - cx)))
    const spanY = Math.max(...layout.children.map((n) => Math.max(cy - (n.y ?? 0), (n.y ?? 0) + (n.height ?? 0) - cy)))
    const fitK = Math.min((W / 2 - INITIAL_TRANSFORM.x) / spanX, (H / 2 - INITIAL_TRANSFORM.y) / spanY)
    const scale = Math.max(Math.min(k, fitK), Math.min(k, MIN_READABLE_SCALE))
    const t = centerTransform(layout, nodeId, scale)
    if (!t) return null
    // Held at the readable minimum the graph may still fit off-center: shift
    // the node from the center just enough to keep every node on screen.
    const clamp = (tr: number, lo: number, hi: number, size: number, m: number) => {
      if (scale * (hi - lo) > size - 2 * m) return tr
      return Math.min(Math.max(tr, m - scale * lo), size - m - scale * hi)
    }
    const xs = layout.children.map((n) => [n.x ?? 0, (n.x ?? 0) + (n.width ?? 0)]).flat()
    const ys = layout.children.map((n) => [n.y ?? 0, (n.y ?? 0) + (n.height ?? 0)]).flat()
    const tx = clamp(t.x, Math.min(...xs), Math.max(...xs), W, INITIAL_TRANSFORM.x)
    const ty = clamp(t.y, Math.min(...ys), Math.max(...ys), H, INITIAL_TRANSFORM.y)
    return d3.zoomIdentity.translate(tx, ty).scale(scale)
  }

  const fitTransform = (layout: ElkNode | null = positionsRef.current): d3.ZoomTransform => {
    const [mx, my] = [INITIAL_TRANSFORM.x, INITIAL_TRANSFORM.y]
    const k = fitScale(layout)
    if (k >= BASE_SCALE) return INITIAL_TRANSFORM
    return d3.zoomIdentity.translate(mx, my).scale(k)
  }

  React.useEffect(() => {
    if (!svgRef.current || !containerRef.current) return

    const svg = d3.select(svgRef.current)
    const container = d3.select(containerRef.current)

    if (!zoomRef.current) {
      zoomRef.current = d3
        .zoom<SVGSVGElement, unknown>()
        .filter((event) => event.ctrlKey || event.type !== 'wheel')
        .scaleExtent([0.1, 10])
        // d3 defaults the extent to the svg's width/height attributes, which are
        // sized to the layout, not the visible viewport: zoom buttons would then
        // anchor on an off-screen point and push the graph sideways.
        .extent(function (this: SVGSVGElement): [[number, number], [number, number]] {
          const { width, height } = this.getBoundingClientRect()
          return [[0, 0], [width, height]]
        })
    }

    zoomRef.current.on('zoom', (event) => {
      const t = event.transform
      if (event.sourceEvent) initialViewRef.current = null
      container.attr('transform', `translate(${t.x},${t.y}) scale(${t.k})`)
      transformRef.current = event.transform
    })

    svg.call(zoomRef.current)

    // Programmatic: fires the `zoom` handler with no sourceEvent, so the fit
    // itself is not mistaken for a user adjustment.
    svg.call(zoomRef.current.transform, transformRef.current)

    return () => {
      svg.on('.zoom', null)
    }
    // Keyed on `positions` (layout), not `nodeElements`: nodeElements also
    // rebuilds on `props.selectedNode` (for the node-highlight prop), and this
    // effect doing the same rebind + fit on every click would repeat the exact
    // "thrash" the hoverNode split above was written to avoid, just gated on
    // click instead of hover.
  }, [positions])

  // Handle container resize. Coalesce bursts (a drag-resize fires the observer
  // many times per second) into one update per animation frame rather than
  // recomputing + applying a transform on every single firing.
  //
  // Leave the graph where it is: the SVG is anchored on its left edge, so a
  // narrower pane just shows less on the right. Opening the ~33rem drawer shrinks
  // the SVG by ~528px, so the one exception is the selected node -- the one whose
  // drawer caused the resize -- which is nudged back only if it would be hidden.
  React.useEffect(() => {
    const svg = svgRef.current
    if (!svg || typeof ResizeObserver === 'undefined') return

    // Seeded on first observation below, so the initial firing is a no-op rather
    // than a shift against a phantom width of 0.
    let lastWidth = 0
    let lastHeight = 0
    let rafId = 0
    const observer = new ResizeObserver(() => {
      if (rafId) return
      rafId = requestAnimationFrame(() => {
        rafId = 0
        if (!zoomRef.current) return
        const width = svg.clientWidth
        const height = svg.clientHeight
        const previousWidth = lastWidth
        const previousHeight = lastHeight
        lastWidth = width
        lastHeight = height

        const initialId = initialViewRef.current
        if (initialId && positionsRef.current && width && height && (width !== previousWidth || height !== previousHeight)) {
          const t = centerFitTransform(positionsRef.current, initialId, BASE_SCALE)
          if (t) d3.select(svg).call(zoomRef.current.transform, t)
          return
        }

        // Start from the user's own transform. `transformRef` is kept current by
        // the zoom handler, so this composes with their latest pan/zoom rather
        // than a stale one.
        if (!previousWidth || !width || width === previousWidth) return
        let shifted = transformRef.current

        const selected = selectedNodeRef.current
        const pos = selected
          ? positionsRef.current?.children?.find((n) => n.id === selected.id)
          : undefined
        if (pos && width > FIT_PADDING * 2) {
          const applied = shifted.k
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
        // Nothing needed moving: leave the transform alone.
        if (shifted === transformRef.current) return

        d3.select(svg).call(zoomRef.current.transform, shifted)
      })
    })
    observer.observe(svg)
    lastWidth = svg.clientWidth
    lastHeight = svg.clientHeight
    return () => {
      if (rafId) cancelAnimationFrame(rafId)
      observer.disconnect()
    }
  }, [])

  const resetZoom = (nodeId?: string) => {
    initialViewRef.current = null
    if (!svgRef.current || !zoomRef.current) return
    const t = (nodeId && positionsRef.current && centerTransform(positionsRef.current, nodeId, BASE_SCALE)) || INITIAL_TRANSFORM
    d3.select(svgRef.current)
      .transition()
      .duration(300)
      .call(zoomRef.current.transform, t)
  }

  const resetView = () => {
    initialViewRef.current = null
    // Effects of the click that called this run before a zero timeout, so a
    // relayout it caused has started by then; otherwise nothing will land.
    const run = layoutRunRef.current
    resetOnLayoutRef.current = true
    setTimeout(() => { if (layoutRunRef.current === run) resetOnLayoutRef.current = false }, 0)
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomRef.current.transform, fitTransform())
    }
  }

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => {
      initialViewRef.current = null
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, ZOOM_STEP)
      }
    },
    zoomOut: () => {
      initialViewRef.current = null
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1 / ZOOM_STEP)
      }
    },
    resetZoom,
    resetView,
    currentZoom: () => {
      if (svgRef.current) {
        // Relative to the default scale, as it was before BASE_SCALE moved into k.
        return (d3.zoomTransform(svgRef.current).k / BASE_SCALE) * 100
      }
      return 90
    },
    centerOnNodeAfterLayout: (nodeId: string) => {
      // Applied by the relayout the expansion causes, and only if it added or
      // dropped nodes: an unchanged graph keeps the user's pan and zoom.
      initialViewRef.current = null
      centerOnLayoutRef.current = nodeId
    },
    centerOnNode: (nodeId: string) => {
      initialViewRef.current = null
      if (!svgRef.current || !zoomRef.current || !positionsRef.current) return
      const t = centerTransform(positionsRef.current, nodeId, BASE_SCALE)
      if (!t) return
      d3.select(svgRef.current)
        .transition()
        .duration(400)
        .call(zoomRef.current.transform, t)
    },
  }), [])

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
