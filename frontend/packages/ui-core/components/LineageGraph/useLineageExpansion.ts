import * as React from 'react'
import { getLineageGraph } from '../../api/gbserver'
import { expansionSeed, indexGraphToElk, mergeElkGraphs, type ElkGraph } from './indexGraph'

export type ExpandDirection = 'upstream' | 'downstream'

const EMPTY: ElkGraph = { nodes: [], links: [] }

// Grows a lineage graph through GET /lineage/graph, seeded from the selected node.
// The caller says how deep to ask -- one level past what is already on screen for
// that node (see visibleLevels) -- so the count follows the graph, not the clicks.
export function useLineageExpansion(rename?: (id: string) => string) {
  const [extra, setExtra] = React.useState<ElkGraph>(EMPTY)
  const [loading, setLoading] = React.useState<ExpandDirection | null>(null)
  const [error, setError] = React.useState<unknown>(null)
  const [unexpanded, setUnexpanded] = React.useState(0)
  // Seeds whose last request in a direction found nothing new: nothing more to load.
  const [exhausted, setExhausted] = React.useState<Set<string>>(new Set())

  const isExhausted = React.useCallback(
    (seedId: string, direction: ExpandDirection) => exhausted.has(`${direction}|${seedId}`),
    [exhausted]
  )

  // `seedId` is the node's id in the index (a URI or `run:<job_id>`). `known` is the
  // graph on screen, used to tell whether this level added anything.
  const expand = React.useCallback(async (
    seedId: string, direction: ExpandDirection, depth: number, known: ElkGraph,
  ) => {
    setLoading(direction)
    setError(null)
    try {
      const result = await getLineageGraph({ ...expansionSeed(seedId), direction, depth })
      const incoming = indexGraphToElk(result, rename)
      const onScreen = new Set(known.nodes.map((n) => n.id))
      if (incoming.nodes.every((n) => onScreen.has(n.id)) && !result.truncated) {
        setExhausted((prev) => new Set(prev).add(`${direction}|${seedId}`))
      }
      setUnexpanded(result.truncated ? result.unexpanded ?? 0 : 0)
      setExtra((prev) => mergeElkGraphs(prev, incoming))
    } catch (e) {
      setError(e)
    } finally {
      setLoading(null)
    }
  }, [rename])

  const reset = React.useCallback(() => {
    setExhausted(new Set())
    setExtra(EMPTY)
    setUnexpanded(0)
    setError(null)
  }, [])

  return { extra, expand, reset, isExhausted, loading, error, unexpanded }
}
