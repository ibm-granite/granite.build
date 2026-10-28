import type { ElkExtendedEdge } from 'elkjs'
import type { LineageGraphNode, LineageGraphResult } from '../../api/gbserver'
import type { ElkNodeEx, NodeType } from './Graph'

export interface ElkGraph {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
}

// An ElkNodeEx that came from the lineage index, keeping the wire node so a later
// click can seed another POST /lineage/graph from it.
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

// Seed for expanding a node through POST /lineage/graph. Run nodes are `run:<job_id>`;
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

// The `depth` to ask POST /lineage/graph for so it returns exactly one level past
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
    // On run nodes and on artifact nodes (their producer's build), only when the
    // caller may read that job (the server checks the space).
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
// is one job hop (artifact -> run -> artifact), i.e. two edges, as POST
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
  return dropGroupedRuns({
    nodes: [...nodes, ...b.nodes.filter((n) => !nodeIds.has(n.id))],
    links: [...a.links, ...b.links.filter((l) => !linkKeys.has(`${l.sources[0]}|${l.targets[0]}`))],
  })
}

// The server groups jobs per response, so one job can arrive as `run:<id>` from a
// walk that reached it alone (cut by a limit, or before a sibling was indexed) and
// inside a `runs:` group from another. The group lists its jobs and has the same
// endpoints, so the lone run node and its links are dropped in its favour.
function dropGroupedRuns(graph: ElkGraph): ElkGraph {
  const grouped = groupedRunIds(graph.nodes)
  if (!grouped.size) return graph
  const endpoint = (port: string | undefined) => port?.replace(/-(input|output)$/, '') ?? ''
  return {
    nodes: graph.nodes.filter((n) => !grouped.has(n.id)),
    links: graph.links.filter((l) => !grouped.has(endpoint(l.sources[0])) && !grouped.has(endpoint(l.targets[0]))),
  }
}

// The `run:<id>` ids every `runs:` group in `nodes` lists.
function groupedRunIds(nodes: ElkNodeEx[]): Set<string> {
  const grouped = new Set<string>()
  for (const n of nodes as IndexElkNode[]) {
    if (n.id.startsWith('runs:')) for (const id of groupJobIds(n.indexNode?.metadata)) grouped.add(`run:${id}`)
  }
  return grouped
}

// `graph` with every node id passed through `rename`, so two ids for one artifact
// (a planned target's YAML URI and the registry UUID a finished target recorded)
// draw as one node. The first node under an id wins -- the caller lists the one it
// renders properly first -- and links that now coincide are kept once.
export function renameElkGraph(graph: ElkGraph, rename: (id: string) => string): ElkGraph {
  const nodes = new Map<string, ElkNodeEx>()
  for (const n of graph.nodes) {
    const id = rename(n.id)
    if (!nodes.has(id)) nodes.set(id, id === n.id ? n : { ...n, id })
  }
  const links = new Map<string, ElkExtendedEdge>()
  for (const l of graph.links) {
    const source = rename(l.sources[0].replace(/-output$/, ''))
    const target = rename(l.targets[0].replace(/-input$/, ''))
    const key = `${source}|${target}`
    if (!links.has(key)) {
      links.set(key, { ...l, id: `${source}-to-${target}`, sources: [`${source}-output`], targets: [`${target}-input`] })
    }
  }
  return { nodes: [...nodes.values()], links: [...links.values()] }
}

// Whether an expansion's response added nothing to the graph that was on screen
// when it was asked (`known`, its nodes or their ids). Both sides go through
// `rename`, so the answer follows a rename that lands after the response: a node
// that read as new under its raw id stops counting once it folds onto one already
// shown. A lone run that a group lists is not new either: the merge drops it (see
// dropGroupedRuns), so it never reaches the screen.
export function addsNothing(
  result: LineageGraphResult,
  known: Iterable<string | ElkNodeEx>,
  rename: (id: string) => string = (id) => id,
): boolean {
  if (result.truncated) return false
  const knownNodes = Array.from(known, (k) => (typeof k === 'string' ? ({ id: k } as ElkNodeEx) : k))
  const onScreen = new Set(knownNodes.map((n) => rename(n.id)))
  const incoming = indexGraphToElk(result, rename).nodes
  const grouped = groupedRunIds([...knownNodes, ...incoming])
  return incoming.every((n) => onScreen.has(n.id) || grouped.has(n.id))
}

// The `runs:` group that lists the lone run `id`, when one in `nodes` does: the
// node a selection on that run moves to once the merge drops it.
export function groupListingRun(nodes: ElkNodeEx[], id: string): string | undefined {
  if (!id.startsWith('run:')) return undefined
  const jobId = id.slice('run:'.length)
  return (nodes as IndexElkNode[]).find(
    (n) => n.id.startsWith('runs:') && groupJobIds(n.indexNode?.metadata).includes(jobId),
  )?.id
}
