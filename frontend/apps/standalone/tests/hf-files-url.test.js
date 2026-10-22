/**
 * Tests for the Results tab's "View in HuggingFace" link.
 *
 * The link must point at the repo the Results file list is read from -- the
 * tuning task's artifact_uri (AutoTuneX services/assets.py `_resolve_source`) --
 * and must not exist for an output that is not on the Hub. The panel turns the
 * URI into a page with the existing getHuggingFaceUrl; its cases below pin that
 * conversion for the URI shapes a tuning output arrives in.
 *
 * Usage: node --test tests/hf-files-url.test.js
 */
const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { tuningHfArtifactUri } = require('../../../packages/ui-core/lib/autotunex/hfFilesUrl.ts')
const { getHuggingFaceUrl } = require('../../../packages/ui-core/components/LineageGraph/diagramUtilities.ts')

const task = (overrides) => ({
  task_id: 't',
  build_id: 'b',
  task_status: 'COMPLETED',
  task_type: 'TUNING',
  github_pr_url: '',
  artifact_id: 'a',
  artifact_uri: '',
  ...overrides,
})

describe('tuningHfArtifactUri', () => {
  it("returns the tuning task's hf:// uri", () => {
    const uri = 'hf://huggingface.co/models/ibm-research/autotunex_c9edaccd'
    assert.equal(tuningHfArtifactUri([task({ artifact_uri: uri })]), uri)
  })

  it('reads the TUNING task, not another one', () => {
    const tasks = [
      task({ task_type: 'DOWNLOAD', artifact_uri: 'hf://huggingface.co/models/other/download' }),
      task({ artifact_uri: 'hf://huggingface.co/models/ibm-research/tuned' }),
    ]
    assert.equal(tuningHfArtifactUri(tasks), 'hf://huggingface.co/models/ibm-research/tuned')
  })

  it('is null for an output that is not on the Hub', () => {
    assert.equal(tuningHfArtifactUri([task({ artifact_uri: 'file:///data/outputs/job/results' })]), null)
  })

  it('is null with no artifact uri, no tuning task, or no tasks', () => {
    assert.equal(tuningHfArtifactUri([task({ artifact_uri: '' })]), null)
    assert.equal(tuningHfArtifactUri([task({ task_type: 'DOWNLOAD', artifact_uri: 'hf:///o/r' })]), null)
    assert.equal(tuningHfArtifactUri([]), null)
    assert.equal(tuningHfArtifactUri(undefined), null)
  })
})

describe('getHuggingFaceUrl for tuning outputs', () => {
  it('drops the models/ segment of a model repo', () => {
    assert.equal(
      getHuggingFaceUrl('hf://huggingface.co/models/ibm-research/autotunex_c9edaccd'),
      'https://huggingface.co/ibm-research/autotunex_c9edaccd',
    )
  })

  it('accepts the hostless hf:/// form', () => {
    assert.equal(getHuggingFaceUrl('hf:///ibm-research/autotunex_c9edaccd'), 'https://huggingface.co/ibm-research/autotunex_c9edaccd')
  })

  it('is null for an hf:// uri with no repo in it', () => {
    assert.equal(getHuggingFaceUrl('hf://huggingface.co'), null)
  })
})
