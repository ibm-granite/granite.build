/**
 * Unit tests for the lineage index graph helpers: how a node seeds an expansion
 * through GET /lineage/graph, how deep to ask, and how an expansion merges into
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
})
