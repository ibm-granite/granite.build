import type { ElkExtendedEdge } from 'elkjs'
import type { LineageGraphNode, LineageGraphResult } from '../../api/gbserver'
import type { ElkNodeEx, NodeType } from './Graph'

export interface ElkGraph {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
}

// An ElkNodeEx that came from the lineage index, keeping the wire node so a later
// click can seed another GET /lineage/graph from it.
export interface IndexElkNode extends ElkNodeEx {
  indexNode?: LineageGraphNode
}

export function artifactTypeToNodeType(artifactType: string | null | undefined): NodeType {
  switch ((artifactType ?? '').toUpperCase()) {
    case 'MODEL': return 'Model'
    case 'DATASET': return 'Dataset'
    case 'FILESET': return 'Fileset'
    case 'BUCKET': return 'Bucket'
    default: return 'Fileset'
  }
}

// Seed for expanding a node through GET /lineage/graph. Run nodes are `run:<job_id>`;
// collapsed in-place rewrites are `runs:<uri>` and expand from their artifact.
export function expansionSeed(nodeId: string): { uri?: string; job_id?: string } {
  if (nodeId.startsWith('runs:')) return { uri: nodeId.slice('runs:'.length) }
  if (nodeId.startsWith('run:')) return { job_id: nodeId.slice('run:'.length) }
  return { uri: nodeId }
}

// The `depth` to ask GET /lineage/graph for so it returns exactly one level past
// the `shown` ones. The API counts depth from its seeds, and a job is seeded from
// every endpoint of that execution -- already one hop out -- so depth d from a run
// reaches d + 1 levels, but d from an artifact reaches d. depth 0 is rejected, so a
// run with nothing shown yet still gets two levels.
export function depthForNextLevel(seedId: string, shown: number): number {
  return expansionSeed(seedId).job_id ? Math.max(1, shown) : shown + 1
}

// `rename` maps index ids onto ids already in the caller's graph (e.g. a build
// panel keys artifacts by UUID, the index by URI) so merging does not duplicate them.
export function indexGraphToElk(
  result: LineageGraphResult,
  rename: (id: string) => string = (id) => id,
): ElkGraph {
  const nodes: IndexElkNode[] = result.nodes.map((n) => {
    const isRun = n.node_type === 'run'
    const title = n.name || n.id
    const buildId = n.metadata?.gb_build_id
    return {
      id: rename(n.id),
      title,
      type: isRun ? 'Build' : artifactTypeToNodeType(n.artifact_type),
      width: isRun ? 192 : 224,
      height: 64,
      labels: [{ text: title }],
      indexNode: n,
      buildId: typeof buildId === 'string' ? buildId : undefined,
    }
  })
  const links: ElkExtendedEdge[] = result.edges.map((e) => {
    const source = rename(e.source)
    const target = rename(e.target)
    return { id: `${source}→${target}`, sources: [`${source}-output`], targets: [`${target}-input`] }
  })
  return { nodes, links }
}

// How many lineage levels of `nodeId` are on screen in one direction, measured on
// the graph itself so it holds whichever node was expanded to get there. A level
// is one job hop (artifact -> run -> artifact), i.e. two edges, as GET
// /lineage/graph counts `depth`.
export function visibleLevels(
  nodeId: string,
  links: ElkExtendedEdge[],
  direction: 'upstream' | 'downstream',
): number {
  const next = new Map<string, string[]>()
  for (const l of links) {
    const source = l.sources[0]?.replace(/-output$/, '')
    const target = l.targets[0]?.replace(/-input$/, '')
    if (!source || !target) continue
    const [from, to] = direction === 'downstream' ? [source, target] : [target, source]
    next.set(from, [...(next.get(from) ?? []), to])
  }
  const dist = new Map([[nodeId, 0]])
  const queue = [nodeId]
  let furthest = 0
  while (queue.length) {
    const id = queue.shift()!
    for (const n of next.get(id) ?? []) {
      if (dist.has(n)) continue
      const d = dist.get(id)! + 1
      dist.set(n, d)
      furthest = Math.max(furthest, d)
      queue.push(n)
    }
  }
  return Math.ceil(furthest / 2)
}

export function mergeElkGraphs(a: ElkGraph, b: ElkGraph): ElkGraph {
  const nodeIds = new Set(a.nodes.map((n) => n.id))
  const linkKeys = new Set(a.links.map((l) => `${l.sources[0]}|${l.targets[0]}`))
  return {
    nodes: [...a.nodes, ...b.nodes.filter((n) => !nodeIds.has(n.id))],
    links: [...a.links, ...b.links.filter((l) => !linkKeys.has(`${l.sources[0]}|${l.targets[0]}`))],
  }
}
