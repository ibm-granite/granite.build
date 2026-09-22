import type { ElkExtendedEdge } from 'elkjs'
import type { ElkNodeEx, NodeType } from './Graph'

export function getDownstream(startId: string, level: number, nodes: ElkNodeEx[], links: ElkExtendedEdge[]) {
  const visitedNodes = new Set<string>()
  const nodeLevels = new Map<string, number>()

  let currentLevelNodes = new Set<string>([startId])
  nodeLevels.set(startId, 0)
  visitedNodes.add(startId)

  let currentLevel = 0

  while (currentLevelNodes.size > 0 && (level === -1 || currentLevel < level)) {
    const nextLevelNodes = new Set<string>()

    for (const link of links) {
      for (const sourceId of link.sources) {
        const fromId = sourceId.replace('-output', '')

        if (currentLevelNodes.has(fromId)) {
          for (const targetId of link.targets) {
            const toId = targetId.replace('-input', '')

            if (!visitedNodes.has(toId)) {
              visitedNodes.add(toId)
              nodeLevels.set(toId, currentLevel + 1)
              nextLevelNodes.add(toId)
            }
          }
          break
        }
      }
    }

    currentLevelNodes = nextLevelNodes
    currentLevel++
  }

  let hasMoreLevels = false
  if (level !== Infinity && currentLevelNodes.size > 0) {
    const nextLevelNodes = new Set<string>()

    for (const link of links) {
      const sourcesMatch = link.sources.some((sourceId) => currentLevelNodes.has(sourceId.replace(/-output$/, '')))

      if (sourcesMatch) {
        for (const targetId of link.targets) {
          const toId = targetId.replace(/-input$/, '')
          if (!visitedNodes.has(toId)) {
            nextLevelNodes.add(toId)
          }
        }
      }
    }

    if (nextLevelNodes.size > 0) hasMoreLevels = true
  }

  const filteredLinks = links.filter((link) => {
    const allFromIn = link.sources.every((s) => visitedNodes.has(s.replace('-output', '')))
    const allToIn = link.targets.every((t) => visitedNodes.has(t.replace('-input', '')))
    return allFromIn && allToIn
  })

  return {
    nodes: nodes.filter((node) => visitedNodes.has(node.id)),
    links: filteredLinks,
    levels: nodeLevels,
    hasMoreLevels,
  }
}

export function getUpstream(startId: string, level: number, nodes: ElkNodeEx[], links: ElkExtendedEdge[]) {
  const visitedNodes = new Set<string>()
  const nodeLevels = new Map<string, number>()

  let currentLevelNodes = new Set<string>([startId])
  nodeLevels.set(startId, 0)
  visitedNodes.add(startId)

  let currentLevel = 0

  while (currentLevelNodes.size > 0 && (level === -1 || currentLevel < level)) {
    const nextLevelNodes = new Set<string>()

    for (const link of links) {
      for (const targetId of link.targets) {
        const toId = targetId.replace('-input', '')

        if (currentLevelNodes.has(toId)) {
          for (const sourceId of link.sources) {
            const fromId = sourceId.replace('-output', '')

            if (!visitedNodes.has(fromId)) {
              visitedNodes.add(fromId)
              nodeLevels.set(fromId, currentLevel + 1)
              nextLevelNodes.add(fromId)
            }
          }
          break
        }
      }
    }

    currentLevelNodes = nextLevelNodes
    currentLevel++
  }

  let hasMoreLevels = false
  if (level !== Infinity && currentLevelNodes.size > 0) {
    const nextLevelNodes = new Set<string>()

    for (const link of links) {
      const targetsMatch = link.targets.some((targetId) => currentLevelNodes.has(targetId.replace(/-input$/, '')))

      if (targetsMatch) {
        for (const sourceId of link.sources) {
          const fromId = sourceId.replace(/-output$/, '')
          if (!visitedNodes.has(fromId)) {
            nextLevelNodes.add(fromId)
          }
        }
      }
    }

    if (nextLevelNodes.size > 0) hasMoreLevels = true
  }

  const filteredLinks = links.filter((link) => {
    const allFromIn = link.sources.every((s) => visitedNodes.has(s.replace('-output', '')))
    const allToIn = link.targets.every((t) => visitedNodes.has(t.replace('-input', '')))
    return allFromIn && allToIn
  })

  return {
    nodes: nodes.filter((node) => visitedNodes.has(node.id)),
    links: filteredLinks,
    levels: nodeLevels,
    hasMoreLevels,
  }
}

export function getSubgraph(
  startId: string,
  downstreamLevels: number,
  upstreamLevels: number,
  nodes: ElkNodeEx[],
  links: ElkExtendedEdge[]
) {
  const downstreamResult = getDownstream(startId, downstreamLevels, nodes, links)
  const upstreamResult = getUpstream(startId, upstreamLevels, nodes, links)

  const mergedNodeIds = new Set<string>()
  const mergedNodes: ElkNodeEx[] = []

  for (const node of [...downstreamResult.nodes, ...upstreamResult.nodes]) {
    if (!mergedNodeIds.has(node.id)) {
      mergedNodeIds.add(node.id)
      mergedNodes.push(node)
    }
  }

  const seenLinks = new Set<string>()
  const mergedLinks: ElkExtendedEdge[] = []

  for (const link of [...downstreamResult.links, ...upstreamResult.links]) {
    if (!seenLinks.has(link.id)) {
      seenLinks.add(link.id)
      const allFromInNodes = link.sources.every((fromId) => mergedNodeIds.has(fromId.replace('-output', '')))
      const allToInNodes = link.targets.every((toId) => mergedNodeIds.has(toId.replace('-input', '')))
      if (allFromInNodes && allToInNodes) mergedLinks.push(link)
    }
  }

  return {
    nodes: mergedNodes,
    links: mergedLinks,
    hasMoreDownstream: downstreamResult.hasMoreLevels,
    hasMoreUpstream: upstreamResult.hasMoreLevels,
  }
}

/** A HuggingFace repo reference, with the type normalised to singular. */
interface HfRef {
  type: 'model' | 'dataset' | 'space' | 'bucket'
  org: string
  name: string
}

// The pluralized path segment as it appears in a URI/URL, mapped to the
// singular type. Models are the odd one out: they have no segment at all.
const HF_TYPE_SEGMENTS: Record<string, HfRef['type']> = {
  models: 'model',
  datasets: 'dataset',
  spaces: 'space',
  buckets: 'bucket',
}

/**
 * Parse the many spellings of a HuggingFace reference into one shape.
 *
 * Both `hf://` URIs and browsable `huggingface.co` URLs reach the frontend, and
 * the same artifact can arrive spelled several ways — gbserver's own
 * `get_hf_artifact_uri` emits `hf:///models/org/name`, while the lineage API's
 * `_uri_from_url` reconstructs a URI *without* the `models/` segment. String
 * comparison therefore cannot tell that two references are the same artifact;
 * this is what {@link canonicalArtifactKey} exists to fix.
 *
 * Returns null for anything that is not recognisably a HuggingFace reference.
 */
function parseHfRef(uri: string): HfRef | null {
  if (!uri) return null

  let parts: string[] | null = null

  if (uri.startsWith('hf://')) {
    const remainder = uri.slice(5)
    if (remainder.startsWith('/')) {
      // hf:///[type/]org/name
      parts = remainder.replace(/^\/+/, '').split('/')
    } else if (remainder.startsWith('huggingface.co/')) {
      // hf://huggingface.co/[type/]org/name
      parts = remainder.slice('huggingface.co/'.length).split('/')
    } else if (remainder.includes('/')) {
      // hf://<domain>/[type/]org/name — the domain segment is discarded; the
      // browsable URL is always on huggingface.co
      parts = remainder.split('/').slice(1)
    } else {
      return null
    }
  } else if (/huggingface\.co/.test(uri)) {
    // https://huggingface.co/[type/]org/name, or the same without a scheme
    const afterHost = uri.split('huggingface.co/')[1]
    if (afterHost === undefined) return null
    parts = afterHost.split('/')
  }

  if (!parts) return null

  if (parts.length === 2) {
    const [org, name] = parts
    return org && name ? { type: 'model', org, name } : null
  }
  if (parts.length === 3) {
    const [segment, org, name] = parts
    const type = HF_TYPE_SEGMENTS[segment]
    return type && org && name ? { type, org, name } : null
  }
  return null
}

// Mirrors src/gbcommon/utils/hf_utils.py:convert_hf_uri_to_url — model URLs
// never include a "models/" segment; datasets/spaces/buckets keep their
// pluralized type segment.
export function getHuggingFaceUrl(uri: string): string | null {
  if (!uri) return null

  // A non-hf:// huggingface.co URL is already browsable — hand it back rather
  // than round-tripping it through the parser, which would drop anything after
  // the org/name (a /tree/main path, a query string).
  if (!uri.startsWith('hf://') && /huggingface\.co/.test(uri)) {
    return uri.startsWith('http') ? uri : `https://${uri}`
  }

  const ref = parseHfRef(uri)
  if (!ref) return null

  const segment = ref.type === 'model' ? '' : `${ref.type}s/`
  return `https://huggingface.co/${segment}${ref.org}/${ref.name}`
}

/**
 * A spelling-independent identity for an artifact reference, for use as a graph
 * node key.
 *
 * The artifact lineage graph keyed its focal node by gbserver UUID but every
 * run input/output by URI, so one artifact became two nodes — the UUID one
 * carrying no edges at all, and the URI one carrying all of them. Keying both
 * through this collapses them, and it has to be canonical rather than literal
 * because the two sides genuinely spell model URIs differently (see
 * {@link parseHfRef}).
 *
 * Non-HuggingFace URIs (`space://`, `s3://`, …) are returned unchanged: exact
 * string identity is already correct for those, and inventing a canonical form
 * for schemes we do not model would risk collapsing distinct artifacts.
 */
export function canonicalArtifactKey(uri: string): string {
  const ref = parseHfRef(uri)
  return ref ? `hf:${ref.type}:${ref.org}/${ref.name}` : uri
}

/**
 * Decide how to draw an artifact node.
 *
 * Both lineage panels previously hardcoded `'Fileset'` for every run input and
 * output, which is why a model showed up labelled "Fileset" next to itself. The
 * explicit type is preferred where the API supplies one; where it does not, a
 * HuggingFace URI already implies the type, which recovers the common cases
 * without waiting on a backend change.
 *
 * Ordering matters: `artifactType` wins over the URI because the record is
 * authoritative and a URI is only a hint. Falls back to `'Fileset'` — the
 * previous unconditional answer — when neither says anything, so this can only
 * improve on the old behaviour, never regress it.
 */
export function artifactNodeType(artifactType?: string, uri?: string): NodeType {
  switch (artifactType?.toUpperCase()) {
    case 'MODEL':    return 'Model'
    case 'DATASET':  return 'Dataset'
    case 'FILESET':  return 'Fileset'
    case 'BUCKET':   return 'Bucket'
    case 'TABLE':    return 'Table'
    case 'ARTIFACT': return 'Artifact'
  }

  // No explicit type — infer from the URI when it is a HuggingFace reference.
  // `space` has no node type of its own; it renders as a generic fileset.
  const ref = uri ? parseHfRef(uri) : null
  switch (ref?.type) {
    case 'model':   return 'Model'
    case 'dataset': return 'Dataset'
    case 'bucket':  return 'Bucket'
    case 'space':   return 'Fileset'
  }

  return 'Fileset'
}
