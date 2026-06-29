import type { ElkExtendedEdge } from 'elkjs'
import type { ElkNodeEx } from './Graph'

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
