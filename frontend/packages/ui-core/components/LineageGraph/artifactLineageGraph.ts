import type { ElkExtendedEdge } from 'elkjs'
import type { ElkNodeEx } from './Graph'
import { artifactNodeType, canonicalArtifactKey } from './diagramUtilities'

/** The subset of an artifact record this builder needs. */
export interface FocalArtifact {
  uuid: string
  name: string
  artifact_type: string
  uri?: string
}

/** One input or output of a lineage run, as `POST /lineage/artifact` returns it. */
export interface LineageRef {
  name: string
  uri?: string
  url?: string
  artifact_type?: string
}

/** One run in the artifact lineage graph. */
export interface LineageRun {
  job_name: string
  run_id: string
  status?: string
  inputs: LineageRef[]
  outputs: LineageRef[]
}

/** What a graph node stands for, so a click can describe or navigate to it. */
export interface ArtifactNodeMeta {
  /** gbserver UUID — known only for the focal artifact; lineage refs carry none. */
  uuid?: string
  name: string
  artifactType?: string
  uri?: string
}

export interface ArtifactLineageGraph {
  nodes: ElkNodeEx[]
  links: ElkExtendedEdge[]
  /** Keyed by node id, for whatever the click handler wants to show. */
  meta: Map<string, ArtifactNodeMeta>
  /** The focal artifact's node id, for highlighting and Focus Node. */
  focalNodeId: string
}

const ARTIFACT_NODE = { width: 224, height: 64 } as const
const RUN_NODE = { width: 192, height: 64 } as const

/**
 * Build the artifact lineage graph.
 *
 * The bug this replaces: the focal node was keyed by gbserver UUID while every
 * run input and output was keyed by its URI, so the artifact you were looking at
 * appeared **twice** — once correctly typed but connected to nothing, and once as
 * a hardcoded "Fileset" carrying all the edges. That is the duplicate, and the
 * mislabel, in the ux0910 review.
 *
 * Both sides are now keyed through `canonicalArtifactKey`, which has to be
 * canonical rather than literal: gbserver spells a model URI
 * `hf:///models/org/name` while the lineage API reconstructs it without the
 * `models/` segment, so the two strings differ for the same artifact.
 *
 * Run nodes keep a `run-` prefix. They are not artifacts and must never collide
 * with one, even if a run id happened to look like a URI.
 */
export function buildArtifactLineageGraph(
  artifact: FocalArtifact,
  runs: LineageRun[],
): ArtifactLineageGraph {
  const nodes: ElkNodeEx[] = []
  const links: ElkExtendedEdge[] = []
  const meta = new Map<string, ArtifactNodeMeta>()
  const seenNodes = new Set<string>()
  const seenEdges = new Set<string>()

  const addNode = (node: ElkNodeEx, nodeMeta?: ArtifactNodeMeta) => {
    if (seenNodes.has(node.id)) {
      // Already present. A later mention may still know more than the first --
      // the focal artifact is added with its uuid and real type, then reappears
      // as a run ref with only a name -- so fill gaps without overwriting.
      if (nodeMeta) {
        const existing = meta.get(node.id)
        if (existing) {
          meta.set(node.id, {
            uuid: existing.uuid ?? nodeMeta.uuid,
            name: existing.name || nodeMeta.name,
            artifactType: existing.artifactType ?? nodeMeta.artifactType,
            uri: existing.uri ?? nodeMeta.uri,
          })
        }
      }
      return
    }
    seenNodes.add(node.id)
    nodes.push(node)
    if (nodeMeta) meta.set(node.id, nodeMeta)
  }

  const addEdge = (id: string, from: string, to: string) => {
    if (seenEdges.has(id)) return
    seenEdges.add(id)
    links.push({ id, sources: [`${from}-output`], targets: [`${to}-input`] })
  }

  // The focal artifact. Keyed canonically when it has a URI so run refs to the
  // same artifact land on this node; by uuid only when it has no URI to key on.
  const focalNodeId = artifact.uri ? canonicalArtifactKey(artifact.uri) : artifact.uuid
  addNode(
    {
      id: focalNodeId,
      title: artifact.name,
      type: artifactNodeType(artifact.artifact_type, artifact.uri),
      ...ARTIFACT_NODE,
      labels: [{ text: artifact.name }],
    },
    {
      uuid: artifact.uuid,
      name: artifact.name,
      artifactType: artifact.artifact_type,
      uri: artifact.uri,
    },
  )

  for (const run of runs) {
    const runId = `run-${run.run_id}`
    const runTitle = run.job_name || run.run_id
    addNode({
      id: runId,
      title: runTitle,
      type: 'Build',
      ...RUN_NODE,
      labels: [{ text: runTitle }],
    })

    const addRef = (ref: LineageRef, direction: 'input' | 'output') => {
      // A ref with neither URI nor name cannot be identified or deduped; keying
      // it on the empty string would merge every such ref into one node.
      const key = ref.uri ? canonicalArtifactKey(ref.uri) : ref.name ? `ref:${ref.name}` : null
      if (!key) return

      addNode(
        {
          id: key,
          title: ref.name,
          type: artifactNodeType(ref.artifact_type, ref.uri),
          ...ARTIFACT_NODE,
          labels: [{ text: ref.name }],
        },
        { name: ref.name, artifactType: ref.artifact_type, uri: ref.uri },
      )

      if (direction === 'input') addEdge(`${key}->${runId}`, key, runId)
      else addEdge(`${runId}->${key}`, runId, key)
    }

    for (const ref of run.inputs) addRef(ref, 'input')
    for (const ref of run.outputs) addRef(ref, 'output')
  }

  return { nodes, links, meta, focalNodeId }
}
