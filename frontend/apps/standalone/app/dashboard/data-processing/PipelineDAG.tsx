'use client'

import React, { useLayoutEffect, useRef, useState, useMemo, useCallback } from 'react'
import { Button, SkeletonPlaceholder } from '@carbon/react'
import { Copy } from '@carbon/icons-react'
import {
  CardNode,
  CardNodeColumn,
  CardNodeSubtitle,
  CardNodeTitle,
  ArrowRightMarker,  // used in SVG <defs>
} from '@carbon/charts-react'
import type { DPNode, DPEdge, DPDataset } from '@granite-build/ui-core/api/dataProcessing'
import { ARTIFACT_TYPE_CONFIG } from '@granite-build/ui-core/config/artifactTypes'
import styles from './page.module.scss'

interface Props {
  nodes: DPNode[]
  edges: DPEdge[]
  datasets?: DPDataset[]
  nodeCounts?: Record<string, number>
  pipelineStatuses?: Record<string, Record<string, string>>
  completedOnly: boolean
  search: string
  focusedNodeId: string | null
  onFocusNode: (id: string | null) => void
  isLoading: boolean
  scanned?: number
}

// ── Pipeline Paths side panel ────────────────────────────────────────────────

const PATH_LABELS: { key: keyof DPDataset; label: string }[] = [
  { key: 'parquet_path',     label: 'Parquet' },
  { key: 'arrow_path',       label: 'Arrow' },
  { key: 'megatron_path',    label: 'Megatron' },
  { key: 'merged_text_path', label: 'Merged Text' },
  { key: 'merged_bin_path',  label: 'Merged Bin' },
]

function PipelinePathsPanel({ dataset }: { dataset: DPDataset }) {
  const paths = PATH_LABELS.filter((p) => dataset[p.key])

  function copyAll() {
    const text = paths.map((p) => `${p.label}: ${dataset[p.key]}`).join('\n')
    navigator.clipboard.writeText(text).catch(() => {})
  }

  return (
    <div style={{
      background: 'var(--cds-layer-02)',
      padding: '1rem',
      minWidth: '18rem',
      maxWidth: '22rem',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Pipeline Paths</span>
        <Button kind="ghost" size="sm" hasIconOnly renderIcon={Copy} iconDescription="Copy all paths" tooltipPosition="left" onClick={copyAll} />
      </div>
      {paths.map((p) => (
        <div key={p.key} style={{ marginBottom: '0.625rem' }}>
          <div style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--cds-text-secondary)', marginBottom: '0.125rem' }}>
            {p.label}
          </div>
          <div style={{ fontSize: '0.6875rem', fontFamily: 'monospace', wordBreak: 'break-all', color: 'var(--cds-text-primary)' }}>
            {dataset[p.key] as string}
          </div>
        </div>
      ))}
    </div>
  )
}

interface ComputedArrow {
  id: string
  x1: number; y1: number
  x2: number; y2: number
  highlighted: boolean
}

interface Chain {
  id: string
  nodes: DPNode[]
  hasCompletedMegatron: boolean
}

const NODE_WIDTH  = 220
const NODE_HEIGHT = 64
const COLUMN_GAP  = 56  // gap between columns for arrow routing

const COLUMN_LABELS = ['PARQUET', 'ARROW', 'MEGATRON', 'MERGED']

// Skeleton shown while the lineage query is loading.
// 3 representative chains with animated SVG links between the placeholder nodes.
function SkeletonDAG({ colWidth }: { colWidth: number }) {
  const totalWidth = colWidth * 4 - COLUMN_GAP
  const ROW_GAP = 12 // 0.75rem at 16px base
  const chains: Array<(0 | 1 | 2 | 3)[]> = [
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [1, 2, 3],
  ]

  // Compute mid-Y of each chain row (within the nodes container, no header offset)
  const rowMidY = chains.map((_, i) => i * (NODE_HEIGHT + ROW_GAP) + NODE_HEIGHT / 2)

  // Compute SVG link paths from right edge of source node to left edge of target node.
  // Col N right edge: N * colWidth + NODE_WIDTH
  // Col N+1 left edge: (N+1) * colWidth
  const links: { x1: number; y1: number; x2: number; y2: number; delay: number }[] = []
  let linkIndex = 0
  chains.forEach((cols, ri) => {
    const y = rowMidY[ri]
    for (let c = 0; c < cols.length - 1; c++) {
      const srcCol = cols[c]
      const tgtCol = cols[c + 1]
      const x1 = srcCol * colWidth + NODE_WIDTH
      const x2 = tgtCol * colWidth
      // Stagger delay left-to-right and top-to-bottom so shimmer "travels" across the graph
      links.push({ x1, y1: y, x2, y2: y, delay: linkIndex * 0.12 })
      linkIndex++
    }
  })

  const svgHeight = chains.length * (NODE_HEIGHT + ROW_GAP) - ROW_GAP

  return (
    <div className={styles.dagBackground}>
      {/* Column headers */}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(4, ${colWidth}px)`, marginBottom: '0.75rem', paddingLeft: '0.25rem' }}>
        {COLUMN_LABELS.map((label) => (
          <div key={label} style={{ fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--cds-text-secondary)' }}>
            {label}
          </div>
        ))}
      </div>

      {/* Nodes + SVG overlay in a positioned container */}
      <div style={{ position: 'relative', width: totalWidth }}>
        <svg
          style={{ position: 'absolute', top: 0, left: 0, width: totalWidth, height: svgHeight, overflow: 'visible', pointerEvents: 'none' }}
          aria-hidden
        >
          <defs>
            {/* One marker per link so each arrowhead inherits its line's animation delay */}
            {links.map((l, i) => (
              <marker key={i} id={`skel-arrow-${i}`} markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                <polygon
                  points="0 0, 6 2, 0 4"
                  fill="#878787"
                  className={styles.skeletonLinkFill}
                  style={{ animationDelay: `${l.delay}s` }}
                />
              </marker>
            ))}
          </defs>
          {links.map((l, i) => {
            const dx = (l.x2 - l.x1) * 0.5
            return (
              <path
                key={i}
                d={`M${l.x1},${l.y1} C${l.x1 + dx},${l.y1} ${l.x2 - dx},${l.y2} ${l.x2},${l.y2}`}
                fill="none"
                stroke="#878787"
                strokeWidth={1.5}
                markerEnd={`url(#skel-arrow-${i})`}
                className={styles.skeletonLink}
                style={{ animationDelay: `${l.delay}s` }}
              />
            )
          })}
        </svg>

        {/* Skeleton node rows */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: ROW_GAP }}>
          {chains.map((cols, ri) => (
            <div key={ri} style={{ display: 'grid', gridTemplateColumns: `repeat(4, ${colWidth}px)`, alignItems: 'start' }}>
              {([0, 1, 2, 3] as const).map((col) => (
                <div key={col} style={{ width: NODE_WIDTH }}>
                  {cols.includes(col) && (
                    <SkeletonPlaceholder style={{ width: NODE_WIDTH, height: NODE_HEIGHT }} />
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


function buildChains(nodes: DPNode[], edges: DPEdge[]): Chain[] {
  const adj = new Map<string, Set<string>>()
  for (const n of nodes) adj.set(n.id, new Set())
  for (const e of edges) {
    adj.get(e.source)?.add(e.target)
    adj.get(e.target)?.add(e.source)
  }
  const visited = new Set<string>()
  const chains: Chain[] = []
  const nodeById = new Map(nodes.map((n) => [n.id, n]))
  for (const n of nodes) {
    if (visited.has(n.id)) continue
    const chainNodes: DPNode[] = []
    const queue = [n.id]
    while (queue.length > 0) {
      const id = queue.shift()!
      if (visited.has(id)) continue
      visited.add(id)
      const node = nodeById.get(id)
      if (node) chainNodes.push(node)
      for (const nb of adj.get(id) ?? []) {
        if (!visited.has(nb)) queue.push(nb)
      }
    }
    chains.push({
      id: chainNodes[0]?.id ?? String(chains.length),
      nodes: chainNodes,
      hasCompletedMegatron: chainNodes.some((n) => n.type === 'megatron'),
    })
  }
  return chains
}

function getConnected(nodeId: string, edges: DPEdge[]): Set<string> {
  const connected = new Set([nodeId])
  const adj = new Map<string, Set<string>>()
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, new Set())
    if (!adj.has(e.target)) adj.set(e.target, new Set())
    adj.get(e.source)!.add(e.target)
    adj.get(e.target)!.add(e.source)
  }
  const queue = [nodeId]
  while (queue.length > 0) {
    const id = queue.shift()!
    for (const nb of adj.get(id) ?? []) {
      if (!connected.has(nb)) { connected.add(nb); queue.push(nb) }
    }
  }
  return connected
}

export function PipelineDAG({
  nodes, edges, datasets, nodeCounts, pipelineStatuses,
  completedOnly, search, focusedNodeId, onFocusNode, isLoading, scanned,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const nodeRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const [arrows, setArrows] = useState<ComputedArrow[]>([])

  const chains = useMemo(() => buildChains(nodes, edges), [nodes, edges])
  const searchLower = search.toLowerCase()

  // completedOnly hides whole chains; search only dims non-matching nodes (stays visible)
  const filteredChains = useMemo(
    () => chains.filter((chain) => !(completedOnly && !chain.hasCompletedMegatron)),
    [chains, completedOnly],
  )

  // Nodes that match the search term — null means no active search (all bright)
  const matchingNodeIds = useMemo<Set<string> | null>(() => {
    if (!searchLower) return null
    const ids = new Set<string>()
    for (const n of nodes) {
      if (n.short_name.toLowerCase().includes(searchLower) || n.path.toLowerCase().includes(searchLower))
        ids.add(n.id)
    }
    for (const e of edges) {
      if (e.builds.some((b) => b.uuid.includes(searchLower) || b.name.toLowerCase().includes(searchLower))) {
        ids.add(e.source)
        ids.add(e.target)
      }
    }
    return ids
  }, [nodes, edges, searchLower])

  const focusedConnected = useMemo(
    () => focusedNodeId ? getConnected(focusedNodeId, edges) : null,
    [focusedNodeId, edges],
  )

  const visibleNodeIds = useMemo(() => {
    const ids = new Set<string>()
    for (const c of filteredChains) for (const n of c.nodes) ids.add(n.id)
    return ids
  }, [filteredChains])

  const visibleEdges = useMemo(
    () => edges.filter((e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target)),
    [edges, visibleNodeIds]
  )

  // Map each node id → its dataset by matching node.path to any dataset path
  const nodeToDataset = useMemo(() => {
    const map = new Map<string, DPDataset>()
    if (!datasets?.length) return map
    for (const node of nodes) {
      const norm = (p: string) => p.toLowerCase().replace(/\/$/, '')
      const match = datasets.find((ds) =>
        [ds.parquet_path, ds.arrow_path, ds.megatron_path, ds.merged_text_path, ds.merged_bin_path]
          .some((p) => p && norm(p) === norm(node.path))
      )
      if (match) map.set(node.id, match)
    }
    return map
  }, [nodes, datasets])

  const focusedDataset = focusedNodeId ? (nodeToDataset.get(focusedNodeId) ?? null) : null

  // Representative name for each chain (prefer megatron → arrow → first node)
  function chainName(chain: Chain): string {
    return (
      chain.nodes.find((n) => n.type === 'megatron') ??
      chain.nodes.find((n) => n.type === 'arrow') ??
      chain.nodes[0]
    )?.short_name ?? ''
  }

  const computeArrows = useCallback(() => {
    const container = containerRef.current
    if (!container) return
    const cr = container.getBoundingClientRect()
    const newArrows: ComputedArrow[] = []
    for (const edge of visibleEdges) {
      const srcEl = nodeRefs.current.get(edge.source)
      const tgtEl = nodeRefs.current.get(edge.target)
      if (!srcEl || !tgtEl) continue
      const sr = srcEl.getBoundingClientRect()
      const tr = tgtEl.getBoundingClientRect()
      newArrows.push({
        id: edge.id,
        x1: sr.right - cr.left,
        y1: sr.top + sr.height / 2 - cr.top,
        x2: tr.left - cr.left,
        y2: tr.top + tr.height / 2 - cr.top,
        highlighted: focusedNodeId
          ? (focusedConnected?.has(edge.source) && focusedConnected?.has(edge.target)) ?? false
          : false,
      })
    }
    setArrows(newArrows)
  }, [visibleEdges, focusedNodeId, focusedConnected])

  useLayoutEffect(() => {
    computeArrows()
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(computeArrows)
    ro.observe(container)
    return () => ro.disconnect()
  }, [computeArrows])

  if (isLoading) return <SkeletonDAG colWidth={NODE_WIDTH + COLUMN_GAP} />

  if (filteredChains.length === 0) {
    return (
      <div style={{ padding: '3rem 0', textAlign: 'center', color: 'var(--cds-text-secondary)' }}>
        {nodes.length === 0 ? (
          scanned === 0
            ? "No builds scanned. Set GB_UI_GBSERVER_DB_URL to gbserver's own database to enable build scanning."
            : `No data processing builds found in this time range (${scanned} builds scanned). Try "7 days" or "30 days".`
        ) : 'No datasets match the current filters.'}
      </div>
    )
  }

  const colWidth = NODE_WIDTH + COLUMN_GAP
  const totalWidth = colWidth * 4 - COLUMN_GAP

  // Height of a chain row (varies if MERGED column has 2 nodes)
  function chainRowHeight(chain: Chain): number {
    const merged = chain.nodes.filter((n) => n.column === 3).length
    return merged > 1 ? NODE_HEIGHT * merged + 8 * (merged - 1) : NODE_HEIGHT
  }

  return (
    <div className={styles.dagBackground} style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>

      {/* ── DAG + name labels ── */}
      <div style={{ flex: '1 1 auto', minWidth: 0, overflowX: 'auto' }}>
        {/* Column headers */}
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(4, ${colWidth}px)`, marginBottom: '0.75rem', paddingLeft: '0.25rem' }}>
          {COLUMN_LABELS.map((label) => (
            <div key={label} style={{ fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--cds-text-secondary)' }}>
              {label}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1.5rem' }}>
          {/* Positioned container for nodes + SVG arrow overlay */}
          <div ref={containerRef} style={{ position: 'relative', width: totalWidth, minWidth: totalWidth, flexShrink: 0 }}>
            {/* SVG arrow overlay */}
            <svg
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', overflow: 'visible', pointerEvents: 'none' }}
              aria-hidden
            >
              <defs>
                <ArrowRightMarker id="dp-arrow"    color="#6F6F6F" markerWidth={8} markerHeight={8} />
                <ArrowRightMarker id="dp-arrow-hi" color="#5D5D5D" markerWidth={8} markerHeight={8} />
              </defs>
              {arrows.map((a) => {
                const dx = (a.x2 - a.x1) * 0.5
                return (
                  <path
                    key={a.id}
                    d={`M${a.x1},${a.y1} C${a.x1 + dx},${a.y1} ${a.x2 - dx},${a.y2} ${a.x2},${a.y2}`}
                    fill="none"
                    stroke={a.highlighted ? '#5D5D5D' : '#878787'}
                    strokeWidth={a.highlighted ? 2 : 1.5}
                    markerEnd={`url(#${a.highlighted ? 'dp-arrow-hi' : 'dp-arrow'})`}
                  />
                )
              })}
            </svg>

            {/* Chain rows */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {filteredChains.map((chain) => (
                <div key={chain.id} style={{ display: 'grid', gridTemplateColumns: `repeat(4, ${colWidth}px)`, alignItems: 'start' }}>
                  {([0, 1, 2, 3] as const).map((col) => (
                    <div key={col} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', width: NODE_WIDTH }}>
                      {chain.nodes
                        .filter((n) => n.column === col)
                        .map((node) => {
                          const cfg = ARTIFACT_TYPE_CONFIG[node.type]
                          const isFocused = focusedNodeId === node.id
                          const isDimmed =
                            (focusedConnected !== null && !focusedConnected.has(node.id)) ||
                            (matchingNodeIds !== null && !matchingNodeIds.has(node.id))
                          const count = nodeCounts?.[node.id]
                          const stages = pipelineStatuses?.[node.id]
                          return (
                            <div
                              key={node.id}
                              ref={(el) => { if (el) nodeRefs.current.set(node.id, el); else nodeRefs.current.delete(node.id) }}
                              style={{ width: NODE_WIDTH, height: NODE_HEIGHT, opacity: isDimmed ? 0.35 : 1, transition: 'opacity 150ms' }}
                            >
                              <div
                                className={[styles.nodeWrapper, isFocused ? styles.nodeWrapperFocused : ''].filter(Boolean).join(' ')}
                                style={{ '--dp-node-color': cfg?.color } as React.CSSProperties}
                              >
                                <CardNode
                                  color={cfg?.color}
                                  onClick={() => onFocusNode(isFocused ? null : node.id)}
                                  style={{ cursor: 'pointer' } as React.CSSProperties}
                                >
                                  <CardNodeColumn style={{ paddingTop: '0.25rem', marginRight: '0.25rem', flexShrink: 0 }}>
                                    {cfg?.icon}
                                  </CardNodeColumn>
                                  <CardNodeColumn style={{ minWidth: 0, overflow: 'hidden' }}>
                                    <CardNodeTitle style={{ fontSize: 14, fontWeight: 600, lineHeight: '18px', color: 'var(--cds-text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                      {(node.type === 'arrow' || node.type === 'parquet')
                                        ? (chainName(chain) || node.short_name)
                                        : node.short_name}
                                    </CardNodeTitle>
                                    <CardNodeSubtitle style={{ fontSize: '0.75rem', color: 'var(--cds-text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                      {node.path.split('/').filter(Boolean).slice(-2).join('/')}
                                      {count !== undefined ? ` · ${count.toLocaleString()} obj` : ''}
                                      {stages ? ` · ${Object.keys(stages).join(' ')}` : ''}
                                    </CardNodeSubtitle>
                                  </CardNodeColumn>
                                </CardNode>
                              </div>
                            </div>
                          )
                        })}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Dataset name labels — one per chain row, height-matched for alignment */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', width: '14rem', flexShrink: 0 }}>
            {filteredChains.map((chain) => (
              <div
                key={chain.id}
                style={{ height: chainRowHeight(chain), display: 'flex', alignItems: 'center' }}
              >
                <span style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--cds-text-primary)', wordBreak: 'break-word' }}>
                  {chainName(chain)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Pipeline Paths side panel (shown when a node is clicked) ── */}
      {focusedDataset && (
        <PipelinePathsPanel dataset={focusedDataset} />
      )}
    </div>
  )
}
