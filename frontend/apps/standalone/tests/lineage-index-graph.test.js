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

before(async () => {
  ;({ expansionSeed, depthForNextLevel, mergeElkGraphs, indexGraphToElk } = await import(
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

  it('keeps a base node untouched when the later copy has no index data', () => {
    const base = { id: 'u', title: 'artifact', buildId: 'B0' }
    const [node] = mergeElkGraphs({ nodes: [base], links: [] }, { nodes: [{ id: 'u' }], links: [] }).nodes
    assert.equal(node, base)
  })
})
