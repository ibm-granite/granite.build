import * as React from 'react'
import { getLineageGraph, type LineageGraphResult } from '../../api/gbserver'
import { addsNothing, expansionSeed, indexGraphToElk, mergeElkGraphs, type ElkGraph } from './indexGraph'
import type { ElkNodeEx } from './Graph'

export type ExpandDirection = 'upstream' | 'downstream'

const EMPTY: ElkGraph = { nodes: [], links: [] }

// The last response for one `direction|seed`, with the nodes on screen when it was
// asked for, so whether it added anything can be re-judged under a later rename.
// Only ids are kept, except for `runs:` groups, whose job list says which lone
// runs they already show.
interface Expansion {
  result: LineageGraphResult
  known: Array<string | ElkNodeEx>
}

// Grows a lineage graph through POST /lineage/graph, seeded from the selected node.
// The caller says how deep to ask -- one level past what is already on screen for
// that node (see visibleLevels) -- so the count follows the graph, not the clicks.
// `groupRuns: false` asks for one node per job, matching an ungrouped base graph.
// Responses are kept as the index returned them and renamed on render, so a node
// fetched before `rename` knew its id (a build artifact whose registry lookup was
// still pending) folds onto the build's node once it does.
//
// One complete response is kept per `direction|seed`, the latest: a repeated
// expansion of the same seed asks for at least as much and replaces the earlier
// one, so the kept set grows with the seeds expanded, not with the clicks. A
// truncated response is kept beside it instead.
export function useLineageExpansion(rename?: (id: string) => string, groupRuns = true) {
  const [expansions, setExpansions] = React.useState<Map<string, Expansion>>(new Map())
  const extra = React.useMemo(
    () => Array.from(expansions.values()).reduce<ElkGraph>(
      (acc, e) => mergeElkGraphs(acc, indexGraphToElk(e.result, rename)), EMPTY
    ),
    [expansions, rename]
  )
  const [loading, setLoading] = React.useState<ExpandDirection | null>(null)
  const [error, setError] = React.useState<unknown>(null)
  const [unexpanded, setUnexpanded] = React.useState(0)
  // Seeds whose last request in a direction found nothing new: nothing more to load.
  // Derived on render, so it is judged under the current `rename`.
  const exhausted = React.useMemo(() => {
    const keys = new Set<string>()
    for (const [key, e] of expansions) if (addsNothing(e.result, e.known, rename)) keys.add(key)
    return keys
  }, [expansions, rename])
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
      setUnexpanded(result.truncated ? result.unexpanded ?? 0 : 0)
      const entry: Expansion = { result, known: known.nodes.map((n) => (n.id.startsWith('runs:') ? n : n.id)) }
      setExpansions((prev) => {
        const next = new Map(prev)
        if (result.truncated) {
          // A cut response may lack nodes the earlier one drew, so it is kept
          // beside it rather than in its place.
          next.set(`${key}#${request}`, entry)
        } else {
          // Re-inserted at the end: merge order is response order, later wins.
          next.delete(key)
          next.set(key, entry)
        }
        return next
      })
    } catch (e) {
      if (current()) setError(e)
    } finally {
      if (gen === generation.current) {
        inFlight.current -= 1
        if (inFlight.current === 0) setLoading(null)
      }
    }
  }, [groupRuns])

  const reset = React.useCallback(() => {
    generation.current += 1
    latest.current = new Map()
    inFlight.current = 0
    setLoading(null)
    setExpansions(new Map())
    setUnexpanded(0)
    setError(null)
  }, [])

  return { extra, expand, reset, isExhausted, loading, error, unexpanded }
}
