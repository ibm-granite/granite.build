/**
 * Unit tests for the lineage index graph helpers: how a node seeds an expansion
 * through POST /lineage/graph, how deep to ask, and how an expansion merges into
 * the graph on screen.
 *
 * Usage: node --test tests/lineage-index-graph.test.js
 */

const { describe, it, before } = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const { registerHooks } = require('node:module')

// The module under test is TypeScript with extensionless imports (bundler
// resolution). Node strips the types; this hook supplies the extension.
registerHooks({
  resolve(specifier, context, next) {
    try {
      return next(specifier, context)
    } catch (err) {
      if (err.code !== 'ERR_MODULE_NOT_FOUND' && err.code !== 'MODULE_NOT_FOUND') throw err
      return next(`${specifier}.ts`, context)
    }
  },
})

const ROOT = path.join(__dirname, '..')
let expansionSeed
let depthForNextLevel
let mergeElkGraphs
let indexGraphToElk
let addsNothing

before(async () => {
  ;({ expansionSeed, depthForNextLevel, mergeElkGraphs, indexGraphToElk, addsNothing } = await import(
    path.join(ROOT, '../../packages/ui-core/components/LineageGraph/indexGraph.ts')
  ))
})

const link = (source, target) => ({ id: `${source}→${target}`, sources: [`${source}-output`], targets: [`${target}-input`] })

describe('expansionSeed', () => {
  it('seeds an artifact from its URI and a run from its job id', () => {
    assert.deepEqual(expansionSeed('s3://b/k'), { uri: 's3://b/k' })
    assert.deepEqual(expansionSeed('run:J1'), { job_id: 'J1' })
  })

  it('seeds a grouped node from every job it groups, not its first input', () => {
    const metadata = { jobs_query: { job_ids: ['J1', 'J2'] } }
    assert.deepEqual(expansionSeed('runs:s3://b/a + s3://b/b → s3://b/c', metadata), { job_ids: ['J1', 'J2'] })
  })

  it('falls back to the first recorded endpoint when the group lists no jobs', () => {
    assert.deepEqual(expansionSeed('runs:∅ → s3://b/c'), { uri: 's3://b/c' })
    assert.deepEqual(expansionSeed('runs:s3://b/a', { jobs_query: { uri: 's3://b/a', self_loop: true } }), { uri: 's3://b/a' })
  })

  it('seeds a group past the API cap from one of its jobs, since all share endpoints', () => {
    const ids = Array.from({ length: 1001 }, (_, i) => `J${i}`)
    assert.deepEqual(expansionSeed('runs:s3://b/a → s3://b/c', { jobs_query: { job_ids: ids } }), { job_ids: ['J0'] })
    const atCap = ids.slice(0, 1000)
    assert.deepEqual(expansionSeed('runs:s3://b/a → s3://b/c', { jobs_query: { job_ids: atCap } }), { job_ids: atCap })
  })

  it('ignores job ids that are not non-empty strings', () => {
    assert.deepEqual(expansionSeed('runs:s3://b/a → s3://b/c', { jobs_query: { job_ids: ['', 3, 'J1'] } }), { job_ids: ['J1'] })
  })
})

describe('depthForNextLevel', () => {
  it('asks one level past what is shown for an artifact', () => {
    assert.equal(depthForNextLevel('s3://b/k', 0), 1)
    assert.equal(depthForNextLevel('s3://b/k', 2), 3)
  })

  it('counts a job seed as already one hop out, for runs and groups alike', () => {
    const metadata = { jobs_query: { job_ids: ['J1'] } }
    assert.equal(depthForNextLevel('run:J1', 0), 1)
    assert.equal(depthForNextLevel('run:J1', 2), 2)
    assert.equal(depthForNextLevel('runs:a → b', 2, metadata), 2)
  })
})

describe('indexGraphToElk', () => {
  it('shows the build id only when the server sent it', () => {
    const result = {
      nodes: [
        { id: 'run:J1', node_type: 'run', name: 'train', metadata: { gb_build_id: 'B1' } },
        { id: 'run:J2', node_type: 'run', name: 'eval', metadata: {} },
      ],
      edges: [],
    }
    const { nodes } = indexGraphToElk(result)
    assert.equal(nodes[0].buildId, 'B1')
    assert.equal(nodes[1].buildId, undefined)
  })
})

describe('mergeElkGraphs', () => {
  it('adds new nodes and links without duplicating known ones', () => {
    const a = { nodes: [{ id: 'x' }, { id: 'y' }], links: [link('x', 'y')] }
    const b = { nodes: [{ id: 'y' }, { id: 'z' }], links: [link('x', 'y'), link('y', 'z')] }
    const merged = mergeElkGraphs(a, b)
    assert.deepEqual(merged.nodes.map((n) => n.id), ['x', 'y', 'z'])
    assert.equal(merged.links.length, 2)
  })

  it('refreshes a known node with what a later fetch knows about it', () => {
    const stale = { id: 'g', title: 'kept', indexNode: { id: 'g', metadata: { run_count: 2 } }, stackCount: 2 }
    const fresh = { id: 'g', title: 'other', indexNode: { id: 'g', metadata: { run_count: 5 } }, stackCount: 5, buildId: 'B1' }
    const [node] = mergeElkGraphs({ nodes: [stale], links: [] }, { nodes: [fresh], links: [] }).nodes
    assert.equal(node.title, 'kept')
    assert.equal(node.indexNode.metadata.run_count, 5)
    assert.equal(node.stackCount, 5)
    assert.equal(node.buildId, 'B1')
  })

  it('keeps the larger group when a truncated expansion sees fewer of its runs', () => {
    const full = { id: 'g', indexNode: { id: 'g', metadata: { run_count: 50 } }, stackCount: 50 }
    const partial = { id: 'g', indexNode: { id: 'g', metadata: { run_count: 3 } }, stackCount: 3 }
    const [node] = mergeElkGraphs({ nodes: [full], links: [] }, { nodes: [partial], links: [] }).nodes
    assert.equal(node.indexNode.metadata.run_count, 50)
    assert.equal(node.stackCount, 50)
  })

  it('keeps a base node untouched when the later copy has no index data', () => {
    const base = { id: 'u', title: 'artifact', buildId: 'B0' }
    const [node] = mergeElkGraphs({ nodes: [base], links: [] }, { nodes: [{ id: 'u' }], links: [] }).nodes
    assert.equal(node, base)
  })
})

describe('addsNothing', () => {
  const artifact = (id) => ({ id, node_type: 'artifact', name: id, metadata: {} })
  const result = (ids, truncated = false) => ({ root_id: ids[0], nodes: ids.map(artifact), edges: [], truncated })

  it('is true only when every returned node is already on screen', () => {
    assert.equal(addsNothing(result(['s3://b/a']), ['s3://b/a']), true)
    assert.equal(addsNothing(result(['s3://b/a', 's3://b/new']), ['s3://b/a']), false)
  })

  it('is never true for a truncated response', () => {
    assert.equal(addsNothing(result(['s3://b/a'], true), ['s3://b/a']), false)
  })

  it('follows a rename that lands after the response', () => {
    // Asked before the registry lookup resolved: the URI looked new next to the UUID.
    const res = result(['hf://o/m'])
    const known = ['uuid-1']
    assert.equal(addsNothing(res, known), false)
    const rename = (id) => (id === 'hf://o/m' ? 'uuid-1' : id)
    assert.equal(addsNothing(res, known, rename), true)
  })

  it('renames the on-screen ids too, so a raw id shown earlier still matches', () => {
    const rename = (id) => (id === 'hf://o/m' ? 'uuid-1' : id)
    assert.equal(addsNothing(result(['hf://o/m']), ['hf://o/m'], rename), true)
  })

  it('does not count a lone run that a group on screen already lists', () => {
    const group = { id: 'runs:g', type: 'Build', indexNode: { metadata: { jobs_query: { job_ids: ['J1', 'J2'] } } } }
    const res = { root_id: 'run:J1', nodes: [{ id: 'run:J1', node_type: 'run', name: 'J1', metadata: {} }], edges: [], truncated: false }
    assert.equal(addsNothing(res, [group]), true)
    assert.equal(addsNothing(res, ['runs:g']), false)
  })
})

describe('groupListingRun', () => {
  let groupListingRun
  before(async () => {
    ;({ groupListingRun } = await import(path.join(ROOT, '../../packages/ui-core/components/LineageGraph/indexGraph.ts')))
  })
  const group = { id: 'runs:g', indexNode: { metadata: { jobs_query: { job_ids: ['J1'] } } } }

  it('finds the group a dropped run moves its selection to', () => {
    assert.equal(groupListingRun([group], 'run:J1'), 'runs:g')
    assert.equal(groupListingRun([group], 'run:J9'), undefined)
    assert.equal(groupListingRun([group], 's3://b/k'), undefined)
  })
})

describe('renameElkGraph', () => {
  let renameElkGraph
  before(async () => {
    ;({ renameElkGraph } = await import(path.join(ROOT, '../../packages/ui-core/components/LineageGraph/indexGraph.ts')))
  })
  const node = (id, type, extra = {}) => ({ id, type, title: id, ...extra })

  it('folds a planned target URI onto the UUID a finished target recorded', () => {
    // target-a produced the model (UUID); planned target-b consumes it by URI.
    const graph = {
      nodes: [node('target-a', 'Build'), node('uuid-1', 'Model'), node('target-b', 'Build', { planned: true }),
        node('hf://o/m', 'Fileset', { planned: true })],
      links: [link('target-a', 'uuid-1'), link('hf://o/m', 'target-b')],
    }
    const out = renameElkGraph(graph, (id) => (id === 'hf://o/m' ? 'uuid-1' : id))
    assert.deepEqual(out.nodes.map((n) => [n.id, n.type]), [['target-a', 'Build'], ['uuid-1', 'Model'], ['target-b', 'Build']])
    assert.deepEqual(out.links.map((l) => [l.sources[0], l.targets[0]]), [
      ['target-a-output', 'uuid-1-input'],
      ['uuid-1-output', 'target-b-input'],
    ])
  })

  it('keeps a link once when two spellings of it coincide', () => {
    const graph = { nodes: [node('a', 'Model'), node('b', 'Model'), node('t', 'Build')], links: [link('a', 't'), link('b', 't')] }
    const out = renameElkGraph(graph, (id) => (id === 'b' ? 'a' : id))
    assert.equal(out.nodes.length, 2)
    assert.equal(out.links.length, 1)
  })

  it('leaves a graph with nothing to rename as it was', () => {
    const graph = { nodes: [node('a', 'Model'), node('t', 'Build')], links: [link('a', 't')] }
    const out = renameElkGraph(graph, (id) => id)
    assert.deepEqual(out.nodes, graph.nodes)
    assert.deepEqual(out.links.map((l) => [l.sources[0], l.targets[0]]), [['a-output', 't-input']])
  })
})

describe('mergeElkGraphs across grouped and ungrouped responses', () => {
  const art = (id) => ({ id, node_type: 'artifact', name: id })
  const lone = {
    nodes: [art('a'), { id: 'run:J1', node_type: 'run', name: 'J1' }, art('b')],
    edges: [{ source: 'a', target: 'run:J1' }, { source: 'run:J1', target: 'b' }],
  }
  const group = {
    nodes: [art('a'), { id: 'runs:a → b', node_type: 'run', name: 'g', metadata: { run_count: 2, jobs_query: { job_ids: ['J1', 'J2'] } } }, art('b')],
    edges: [{ source: 'a', target: 'runs:a → b' }, { source: 'runs:a → b', target: 'b' }],
  }
  const runIds = (g) => g.nodes.filter((n) => n.id.startsWith('run')).map((n) => n.id)

  it('drops a lone run that a later group lists', () => {
    const out = mergeElkGraphs(indexGraphToElk(lone), indexGraphToElk(group))
    assert.deepEqual(runIds(out), ['runs:a → b'])
    assert.ok(out.links.every((l) => !l.sources[0].startsWith('run:J1') && !l.targets[0].startsWith('run:J1')))
    assert.equal(out.links.length, 2)
  })

  it('drops it the other way round too', () => {
    const out = mergeElkGraphs(indexGraphToElk(group), indexGraphToElk(lone))
    assert.deepEqual(runIds(out), ['runs:a → b'])
    assert.equal(out.links.length, 2)
  })

  it('keeps a run no group lists', () => {
    const other = { nodes: [art('a'), { id: 'run:J9', node_type: 'run', name: 'J9' }, art('c')], edges: [{ source: 'a', target: 'run:J9' }, { source: 'run:J9', target: 'c' }] }
    const out = mergeElkGraphs(indexGraphToElk(group), indexGraphToElk(other))
    assert.deepEqual(runIds(out).sort(), ['run:J9', 'runs:a → b'])
  })
})
