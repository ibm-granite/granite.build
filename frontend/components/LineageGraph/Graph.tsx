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

  React.useEffect(() => {
    if (positions) {
      setNodeElements(buildNodes(positions))
      setLinkElements(buildLinks(positions, hoverNode))
    }
  }, [positions, hoverNode, props.selectedNode])

  const svgRef = React.useRef<SVGSVGElement | null>(null)
  const containerRef = React.useRef<SVGGElement | null>(null)
  const zoomRef = React.useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const transformRef = React.useRef(INITIAL_TRANSFORM)

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

  React.useEffect(() => {
    if (!svgRef.current || !containerRef.current) return

    const svg = d3.select(svgRef.current)
    const container = d3.select(containerRef.current)

    if (!zoomRef.current) {
      zoomRef.current = d3
        .zoom<SVGSVGElement, unknown>()
        .filter((event) => event.ctrlKey || event.type !== 'wheel')
        .scaleExtent([0.1, 10])
    }

    zoomRef.current.on('zoom', (event) => {
      const t = event.transform
      container.attr('transform', `translate(${t.x},${t.y}) scale(${BASE_SCALE * t.k})`)
      transformRef.current = event.transform
    })

    svg.call(zoomRef.current)
    svg.call(zoomRef.current.transform, transformRef.current)

    return () => { svg.on('.zoom', null) }
  }, [nodeElements])

  React.useImperativeHandle(ref, () => ({
    zoomIn: () => {
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1.1)
      }
    },
    zoomOut: () => {
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current).call(zoomRef.current.scaleBy, 1 / 1.1)
      }
    },
    resetZoom: () => {
      if (svgRef.current && zoomRef.current) {
        d3.select(svgRef.current)
          .transition()
          .duration(300)
          .call(zoomRef.current.transform, INITIAL_TRANSFORM)
      }
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
      const { width: W, height: H } = svgRef.current.getBoundingClientRect()
      const cx = (node.x ?? 0) + (node.width ?? 0) / 2
      const cy = (node.y ?? 0) + (node.height ?? 0) / 2
      const tx = W / 2 - BASE_SCALE * cx
      const ty = H / 2 - BASE_SCALE * cy
      d3.select(svgRef.current)
        .transition()
        .duration(400)
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(tx, ty))
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
