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
// grouped jobs are `runs:<inputs> → <outputs>` (several endpoints on a side joined by
// ` + `, an empty side as `∅`; `runs:<uri>` for in-place rewrites). A group that
// lists its jobs (`metadata.jobs_query.job_ids`) is seeded from all of them, so every
// input and output is walked, not just the first; otherwise from its first endpoint.
export interface ExpansionSeed { uri?: string; job_id?: string; job_ids?: string[] }

// The API's cap on `job_ids` (LineageQueryRequest.job_ids max_length).
export const MAX_SEED_JOB_IDS = 1000

export function expansionSeed(nodeId: string, metadata?: Record<string, unknown>): ExpansionSeed {
  if (nodeId.startsWith('runs:')) {
    const jobIds = groupJobIds(metadata)
    // Every job in a group has the same inputs and outputs, so past the API's
    // cap one of them seeds the same endpoints; the walk regroups the rest.
    if (jobIds.length > MAX_SEED_JOB_IDS) return { job_ids: jobIds.slice(0, 1) }
    if (jobIds.length) return { job_ids: jobIds }
    const endpoints = nodeId.slice('runs:'.length).split(' → ').flatMap((side) => side.split(' + '))
    return { uri: endpoints.find((uri) => uri && uri !== '∅') ?? endpoints[0] }
  }
  if (nodeId.startsWith('run:')) return { job_id: nodeId.slice('run:'.length) }
  return { uri: nodeId }
}

function groupJobIds(metadata?: Record<string, unknown>): string[] {
  const query = metadata?.jobs_query
  const ids = query && typeof query === 'object' ? (query as { job_ids?: unknown }).job_ids : undefined
  return Array.isArray(ids) ? ids.filter((id): id is string => typeof id === 'string' && id !== '') : []
}

// The `depth` to ask GET /lineage/graph for so it returns exactly one level past
// the `shown` ones. The API counts depth from its seeds, and a job is seeded from
// every endpoint of that execution -- already one hop out -- so depth d from a run
// (or a group seeded by its jobs) reaches d + 1 levels, but d from an artifact
// reaches d. depth 0 is rejected, so a run with nothing shown yet still gets two levels.
export function depthForNextLevel(seedId: string, shown: number, metadata?: Record<string, unknown>): number {
  const seed = expansionSeed(seedId, metadata)
  return seed.job_id || seed.job_ids ? Math.max(1, shown) : shown + 1
}

// `rename` maps index ids onto ids already in the caller's graph (e.g. a build
// panel keys artifacts by UUID, the index by URI) so merging does not duplicate them.
export function indexGraphToElk(
  result: LineageGraphResult,
  rename: (id: string) => string = (id) => id,
): ElkGraph {
  const nodes: IndexElkNode[] = result.nodes.map((n) => {
    const isRun = n.node_type === 'run'
    const runCount = isRun && typeof n.metadata?.run_count === 'number' ? n.metadata.run_count : undefined
    const stackCount = runCount !== undefined && runCount > 1 ? runCount : undefined
    const title = n.name || n.id
    // Present only on run nodes the caller may read (the server checks the space).
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
      stackCount,
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

// A node already in `a` keeps its place and rendering, but takes what a later fetch
// knows about it -- its index node (run_count, job_ids), stack count and build id --
// so a grouped node does not stay frozen at the first copy seen. A truncated
// expansion sees fewer of a group's runs, so the copy with more runs wins.
function runCountOf(n: IndexElkNode | undefined): number {
  const count = n?.indexNode?.metadata?.run_count
  return typeof count === 'number' ? count : 0
}

export function mergeElkGraphs(a: ElkGraph, b: ElkGraph): ElkGraph {
  const later = new Map(b.nodes.map((n) => [n.id, n as IndexElkNode]))
  const linkKeys = new Set(a.links.map((l) => `${l.sources[0]}|${l.targets[0]}`))
  const nodes = a.nodes.map((n) => {
    const next = later.get(n.id)
    if (!next?.indexNode) return n
    if (runCountOf(next) < runCountOf(n as IndexElkNode)) return n
    return {
      ...n,
      indexNode: next.indexNode,
      stackCount: next.stackCount ?? n.stackCount,
      buildId: n.buildId ?? next.buildId,
    } as IndexElkNode
  })
  const nodeIds = new Set(a.nodes.map((n) => n.id))
  return {
    nodes: [...nodes, ...b.nodes.filter((n) => !nodeIds.has(n.id))],
    links: [...a.links, ...b.links.filter((l) => !linkKeys.has(`${l.sources[0]}|${l.targets[0]}`))],
  }
}

// Whether an expansion's response added nothing to the graph that was on screen
// when it was asked (`known`, its node ids). Both sides go through `rename`, so the
// answer follows a rename that lands after the response: a node that read as new
// under its raw id stops counting once it folds onto one already shown.
export function addsNothing(
  result: LineageGraphResult,
  known: Iterable<string>,
  rename: (id: string) => string = (id) => id,
): boolean {
  if (result.truncated) return false
  const onScreen = new Set(Array.from(known, rename))
  return indexGraphToElk(result, rename).nodes.every((n) => onScreen.has(n.id))
}
