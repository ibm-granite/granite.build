import * as React from 'react'
import { getLineageGraph, type LineageGraphResult } from '../../api/gbserver'
import { expansionSeed, indexGraphToElk, mergeElkGraphs, type ElkGraph } from './indexGraph'

export type ExpandDirection = 'upstream' | 'downstream'

const EMPTY: ElkGraph = { nodes: [], links: [] }

// Grows a lineage graph through GET /lineage/graph, seeded from the selected node.
// The caller says how deep to ask -- one level past what is already on screen for
// that node (see visibleLevels) -- so the count follows the graph, not the clicks.
// `groupRuns: false` asks for one node per job, matching an ungrouped base graph.
// Responses are kept as the index returned them and renamed on render, so a node
// fetched before `rename` knew its id (a build artifact whose registry lookup was
// still pending) folds onto the build's node once it does.
export function useLineageExpansion(rename?: (id: string) => string, groupRuns = true) {
  const [results, setResults] = React.useState<LineageGraphResult[]>([])
  const extra = React.useMemo(
    () => results.reduce<ElkGraph>((acc, r) => mergeElkGraphs(acc, indexGraphToElk(r, rename)), EMPTY),
    [results, rename]
  )
  const [loading, setLoading] = React.useState<ExpandDirection | null>(null)
  const [error, setError] = React.useState<unknown>(null)
  const [unexpanded, setUnexpanded] = React.useState(0)
  // Seeds whose last request in a direction found nothing new: nothing more to load.
  const [exhausted, setExhausted] = React.useState<Set<string>>(new Set())
  // Bumped by reset: a response from an older generation (an expansion still in
  // flight when the view was reset) is dropped instead of bringing back what was
  // cleared.
  const generation = React.useRef(0)
  // The latest request per `direction|seed`: a response superseded by a later click
  // on the same seed is dropped, while one for another seed still lands.
  const latest = React.useRef(new Map<string, number>())
  const nextRequest = React.useRef(0)
  const inFlight = React.useRef(0)

  const isExhausted = React.useCallback(
    (seedId: string, direction: ExpandDirection) => exhausted.has(`${direction}|${seedId}`),
    [exhausted]
  )

  // `seedId` is the node's id in the index (a URI or `run:<job_id>`), `metadata` its
  // index metadata (a grouped node's jobs_query). `known` is the graph on screen,
  // used to tell whether this level added anything.
  const expand = React.useCallback(async (
    seedId: string, direction: ExpandDirection, depth: number, known: ElkGraph,
    metadata?: Record<string, unknown>,
  ) => {
    const gen = generation.current
    const key = `${direction}|${seedId}`
    const request = ++nextRequest.current
    latest.current.set(key, request)
    const current = () => gen === generation.current && latest.current.get(key) === request
    inFlight.current += 1
    setLoading(direction)
    setError(null)
    try {
      const result = await getLineageGraph({ ...expansionSeed(seedId, metadata), direction, depth, group_runs: groupRuns })
      if (!current()) return
      const incoming = indexGraphToElk(result, rename)
      const onScreen = new Set(known.nodes.map((n) => n.id))
      if (incoming.nodes.every((n) => onScreen.has(n.id)) && !result.truncated) {
        setExhausted((prev) => new Set(prev).add(key))
      }
      setUnexpanded(result.truncated ? result.unexpanded ?? 0 : 0)
      setResults((prev) => [...prev, result])
    } catch (e) {
      if (current()) setError(e)
    } finally {
      if (gen === generation.current) {
        inFlight.current -= 1
        if (inFlight.current === 0) setLoading(null)
      }
    }
  }, [rename, groupRuns])

  const reset = React.useCallback(() => {
    generation.current += 1
    latest.current = new Map()
    inFlight.current = 0
    setLoading(null)
    setExhausted(new Set())
    setResults([])
    setUnexpanded(0)
    setError(null)
  }, [])

  return { extra, expand, reset, isExhausted, loading, error, unexpanded }
}
