'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type {
  ColumnMapping,
  Dataset,
  HfDatasetSplits,
  HfImportPreview,
} from '@granite-build/ui-core/types'
import {
  getDataset,
  getHfSplits,
  importHfDataset,
  previewHfDataset,
  searchHfDatasets,
} from '@granite-build/ui-core/api/autotunex'
import {
  HF_IMPORT_POLL_MS,
  HF_IMPORT_READY_TIMEOUT_MS,
} from '@granite-build/ui-core/lib/autotunex/datasetReady'
import {
  canImport as canImportGate,
  defaultConfig,
  defaultTrainSplit,
  deriveDatasetName,
  hfErrorStatus,
  isDatasetNameValid,
  isMappingComplete,
  mappedPreviewKey,
  pollStep,
  probeMapping,
  problemDetail,
  pruneMapping,
  suffixWithRevision,
  survivalSummary,
  type SurvivalSummary,
} from './hfImport'

const SEARCH_DEBOUNCE_MS = 300
const SEARCH_LIMIT = 20
// Carbon's Select needs a real option value; null is not one.
export const NO_VALIDATION = '__none__'

export interface UseHfImportOptions {
  /** The HuggingFace source is the active tab. Replaces the modal's `open`. */
  active: boolean
  requiredColumns: string[]
  onImported: (datasetId: string) => void
}

export interface UseHfImportResult {
  suggestions: string[]
  repoId: string | null
  setRepoId: (id: string | null) => void
  handleSearchInput: (term: string) => void

  splits: HfDatasetSplits | undefined
  splitsFetching: boolean
  splitsError: unknown
  splitsErrorStatus: number | undefined
  refetchSplits: () => void
  configNames: string[]
  splitNames: string[]
  config: string
  handleConfigChange: (next: string) => void
  trainSplit: string
  handleTrainSplitChange: (next: string) => void
  validationSplit: string
  setValidationSplit: (next: string) => void

  preview: HfImportPreview | null
  freshMappedPreview: HfImportPreview | null
  probeLoading: boolean
  mappedLoading: boolean
  previewError: string
  mappedPreviewError: string

  mapping: ColumnMapping
  setMapping: (next: ColumnMapping) => void
  survival: SurvivalSummary

  name: string
  setName: (next: string) => void
  nameValid: boolean
  validationPercentage: number
  setValidationPercentage: (next: number) => void

  canSubmit: boolean
  importing: boolean
  importStatus: string
  error: string
  handleImport: () => Promise<void>
  resetState: () => void
}

export function useHfImport({ active, requiredColumns, onImported }: UseHfImportOptions): UseHfImportResult {
  const queryClient = useQueryClient()
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [repoId, setRepoId] = useState<string | null>(null)
  const [config, setConfig] = useState('')
  const [trainSplit, setTrainSplit] = useState('')
  const [validationSplit, setValidationSplit] = useState(NO_VALIDATION)

  const [preview, setPreview] = useState<HfImportPreview | null>(null)
  // Two independent booleans, each set and cleared by exactly one effect below --
  // sharing one meant whichever request resolved first cleared the spinner while
  // the other was still in flight. Both feed the same loading indicator in the
  // JSX (`probeLoading || mappedLoading`); there is still only one render site.
  const [probeLoading, setProbeLoading] = useState(false)
  const [mappedLoading, setMappedLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')
  const [mapping, setMapping] = useState<ColumnMapping>({})
  // Tagged with the key it was fetched for (repo/config/split/mapping), so a
  // stale result from a since-changed selection is never mistaken for a fresh
  // one -- see `freshMappedPreview` below.
  const [mappedPreview, setMappedPreview] = useState<{ key: string; preview: HfImportPreview } | null>(null)
  const [mappedPreviewError, setMappedPreviewError] = useState('')
  const [validationPercentage, setValidationPercentage] = useState(10)
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [importing, setImporting] = useState(false)
  const [importStatus, setImportStatus] = useState('')

  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const suggestTokenRef = useRef(0)
  // Exclusive to the probe. The mapped-preview effect has always fired in the
  // same commit as the probe on a repo/config/split change (it just went from
  // incomplete to complete, or vice versa), so a shared counter let the mapped
  // effect's token bump win and made the probe's own response fail its own
  // staleness check -- see mappedTokenRef.
  const previewTokenRef = useRef(0)
  // Exclusive to the mapped-preview effect, for the same reason in reverse.
  const mappedTokenRef = useRef(0)
  // Every import run captures the id it started with and compares before writing
  // state; bumping the counter abandons all prior runs for good. Same guard as
  // SettingsDatasetCreate's, and it is what makes closing mid-import safe.
  const runIdRef = useRef(0)

  const mappingComplete = isMappingComplete(mapping, requiredColumns)
  const mappingKey = JSON.stringify(mapping)

  // `requiredColumns` is a fresh array on every parent render, so it cannot be an
  // effect dependency directly -- the probe effect would refire forever. This
  // string stands in for it, and changes exactly when its contents do.
  const requiredKey = requiredColumns.join(',')

  // The probe's request body needs the current required columns, but its RESPONSE
  // does not depend on them: the server returns source-side `columns` and
  // `raw_rows`, and the payload carries no target_format for it to validate keys
  // against. Reading them through a ref keeps `requiredKey` out of the probe's
  // dependency array below -- otherwise an AI-suggested algorithm change refires
  // the probe, whose body calls setMapping({}), wiping the mapping the AI just
  // wrote.
  const requiredColumnsRef = useRef(requiredColumns)
  requiredColumnsRef.current = requiredColumns

  const {
    data: splits,
    isFetching: splitsFetching,
    error: splitsError,
    refetch: refetchSplits,
  } = useQuery({
    queryKey: ['autotunex', 'hfSplits', repoId],
    queryFn: () => getHfSplits(repoId as string),
    enabled: active && !!repoId,
    // Overrides the app default of retry: 1. A 422 ("no tabular data") is
    // permanent, and a 503 gets an explicit Retry button, so an automatic retry
    // only doubles the wait before either message appears.
    retry: false,
  })

  const configNames = useMemo(() => Object.keys(splits?.configs ?? {}), [splits])
  const splitNames = useMemo(() => splits?.configs?.[config] ?? [], [splits, config])

  // Preselect a config and train split once the repo resolves. Both stay visible
  // and changeable: a silently auto-picked wrong split is only discovered after a
  // multi-hour tuning run.
  useEffect(() => {
    if (!splits) return
    const nextConfig = defaultConfig(Object.keys(splits.configs))
    setConfig(nextConfig)
    setTrainSplit(defaultTrainSplit(splits.configs[nextConfig] ?? []))
    setMapping({})
    setValidationSplit(NO_VALIDATION)
    setName(deriveDatasetName(splits.repo_id))
  }, [splits])

  // The probe. Its only job is to fetch `columns` and `raw_rows`; `sampled` and
  // `survived` are meaningless for a blank-source mapping and are not read here.
  useEffect(() => {
    if (!active || !repoId || !splits || !config || !trainSplit) return
    // The chosen pair must exist in the splits currently loaded. On switching
    // datasets, `splits` briefly holds the new repo's data while `config`/`trainSplit`
    // still hold the previous repo's (the preselect effect's setState only lands on the
    // next render), which would fire a probe for a config the new repo does not have
    // and flash its error before the real config arrives.
    if (!splits.configs[config]?.includes(trainSplit)) return
    const token = ++previewTokenRef.current
    setProbeLoading(true)
    setPreviewError('')
    setPreview(null)
    setMapping({})
    previewHfDataset({
      repo_id: repoId,
      config,
      train_split: trainSplit,
      // Always null, and validationSplit is deliberately not a dependency below:
      // the server samples rows from the train split only, so the validation split
      // cannot change `columns` or `raw_rows`. Including it would re-probe (and, in
      // the next commit, discard the user's mapping) every time they change it.
      validation_split: null,
      column_mapping: probeMapping(requiredColumnsRef.current),
    })
      .then((result) => {
        if (previewTokenRef.current !== token) return
        setPreview(result)
      })
      .catch((err) => {
        if (previewTokenRef.current !== token) return
        setPreviewError(problemDetail(err, 'Could not preview this dataset.'))
      })
      .finally(() => {
        if (previewTokenRef.current === token) setProbeLoading(false)
      })
  }, [active, repoId, splits, config, trainSplit])

  // The required columns changed -- an AI-suggested algorithm, or the user
  // changing it. The mapping still holds the previous algorithm's targets, which
  // would be sent in the import body. Prune rather than clear: a target that is
  // still required keeps the source the user (or the AI) chose for it.
  //
  // `requiredKey` stands in for `requiredColumns`, which is a fresh array on every
  // parent render. pruneMapping returns its argument when nothing is dropped, so
  // this cannot loop.
  useEffect(() => {
    setMapping((current) => pruneMapping(current, requiredColumnsRef.current))
  }, [requiredKey])

  // The second preview: the real one. Fires only once every required column has a
  // source, because the server counts survivors over the mapping's own keys -- a
  // partial mapping would return a high number describing only the columns chosen
  // so far. validation_split is null here for the same reason as in the probe.
  useEffect(() => {
    if (!active || !repoId || !config || !trainSplit || !mappingComplete) {
      setMappedPreview(null)
      setMappedPreviewError('')
      return
    }
    // Same guard as the probe, and for the same reason: on switching datasets
    // `splits` briefly holds the new repo's data while `config`/`trainSplit` still
    // hold the previous repo's, which would fire a mapped request for a config the
    // new repo does not have.
    if (!splits || !splits.configs[config]?.includes(trainSplit)) {
      setMappedPreviewError('')
      return
    }
    const key = mappedPreviewKey({ repoId, config, trainSplit, mappingKey })
    const token = ++mappedTokenRef.current
    setMappedLoading(true)
    setMappedPreviewError('')
    previewHfDataset({
      repo_id: repoId,
      config,
      train_split: trainSplit,
      validation_split: null,
      column_mapping: mapping,
    })
      .then((result) => {
        if (mappedTokenRef.current !== token) return
        setMappedPreview({ key, preview: result })
      })
      .catch((err) => {
        if (mappedTokenRef.current !== token) return
        setMappedPreviewError(problemDetail(err, 'Could not preview this mapping.'))
      })
      .finally(() => {
        if (mappedTokenRef.current === token) setMappedLoading(false)
      })
  }, [active, repoId, splits, config, trainSplit, mappingComplete, mappingKey])

  function resetState() {
    runIdRef.current += 1
    previewTokenRef.current += 1
    mappedTokenRef.current += 1
    suggestTokenRef.current += 1
    if (debounceRef.current) clearTimeout(debounceRef.current)
    setSuggestions([])
    setRepoId(null)
    setConfig('')
    setTrainSplit('')
    setValidationSplit(NO_VALIDATION)
    setPreview(null)
    setProbeLoading(false)
    setMappedLoading(false)
    setPreviewError('')
    setMapping({})
    setMappedPreview(null)
    setMappedPreviewError('')
    setValidationPercentage(10)
    setName('')
    setError('')
    setImporting(false)
    setImportStatus('')
  }

  // Tokenized exactly like Step0GetStarted's model search: suggestion requests are
  // not ordered, so a slow earlier query must not overwrite a newer term's results.
  async function fetchSuggestions(term: string) {
    const token = ++suggestTokenRef.current
    const trimmed = term.trim()
    if (!trimmed) {
      if (suggestTokenRef.current === token) setSuggestions([])
      return
    }
    try {
      const results = await searchHfDatasets(trimmed, SEARCH_LIMIT)
      if (suggestTokenRef.current !== token) return
      setSuggestions(results)
    } catch {
      if (suggestTokenRef.current === token) setSuggestions([])
    }
  }

  function handleSearchInput(term: string) {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => fetchSuggestions(term), SEARCH_DEBOUNCE_MS)
  }

  function handleConfigChange(nextConfig: string) {
    setConfig(nextConfig)
    setTrainSplit(defaultTrainSplit(splits?.configs?.[nextConfig] ?? []))
    setMapping({})
    setValidationSplit(NO_VALIDATION)
  }

  function handleTrainSplitChange(nextSplit: string) {
    setTrainSplit(nextSplit)
    setMapping({})
  }

  const splitsErrorStatus = hfErrorStatus(splitsError)

  // `mappedPreview` is tagged with the key it was fetched for; a mismatch means
  // the selection has since moved on (repo, config, split, or mapping) and the
  // stored result describes a mapping that is no longer the current one. Reading
  // through this rather than the raw state makes a stale survival count or a
  // stale mapped-rows table structurally impossible, not merely cleared on one
  // particular transition.
  const currentMappedKey = mappedPreviewKey({ repoId: repoId ?? '', config, trainSplit, mappingKey })
  const freshMappedPreview = mappedPreview?.key === currentMappedKey ? mappedPreview.preview : null

  const survival = survivalSummary({
    sampled: freshMappedPreview?.sampled ?? 0,
    survived: freshMappedPreview?.survived ?? 0,
    // Not just mappingComplete: until the mapped preview has come back there is no
    // count to describe, and the probe's own numbers must never be shown.
    mappingComplete: mappingComplete && freshMappedPreview !== null,
  })

  const nameValid = isDatasetNameValid(name)
  const splitFromTrain = validationSplit === NO_VALIDATION
  const canSubmit = canImportGate({
    hasRepo: !!repoId,
    hasConfig: !!config,
    hasTrainSplit: !!trainSplit,
    mappingComplete,
    nameValid,
    survivalKind: survival.kind,
    importing,
    splitFromTrain,
    validationPercentage,
  })

  async function pollUntilReady(datasetId: string, runId: number): Promise<Dataset> {
    const deadline = Date.now() + HF_IMPORT_READY_TIMEOUT_MS
    while (runId === runIdRef.current) {
      let dataset: Dataset | null = null
      try {
        dataset = await getDataset(datasetId)
      } catch {
        // A failed poll is just a poll to retry: the import is already running
        // server-side, and one transient GET blip must not abandon it. Whatever
        // this rejection was, `pollStep` below decides purely from the deadline,
        // not from this error -- so a post-deadline network blip surfaces the
        // authored timeout copy rather than an unrelated transport error.
      }
      if (runId !== runIdRef.current) throw new Error('superseded')
      if (dataset) setImportStatus(dataset.status)
      const decision = pollStep({ status: dataset?.status, expired: Date.now() > deadline })
      if (decision === 'ready') {
        // Only reachable when `dataset.status === 'ready'`, which requires `dataset`.
        return dataset as Dataset
      }
      if (decision === 'error') {
        throw new Error(dataset?.status_detail || 'Import failed.')
      }
      if (decision === 'timeout') {
        throw new Error(
          'Import timed out while processing. It may still finish -- check Settings > Datasets.'
        )
      }
      await new Promise((resolve) => setTimeout(resolve, HF_IMPORT_POLL_MS))
    }
    throw new Error('superseded')
  }

  async function handleImport() {
    if (!repoId || !canSubmit) return
    const runId = ++runIdRef.current
    // Captured now, not read from `splits` inside the catch below: changing the
    // repo mid-import re-fires the splits query, so by the time a 409 lands
    // `splits` may briefly be undefined and `suffixWithRevision` would silently
    // leave the name unchanged while the error text claims a suffix was added.
    const revision = splits?.revision ?? ''
    setImporting(true)
    setError('')
    setImportStatus('importing')
    try {
      const created = await importHfDataset({
        name: name.trim(),
        // Not required by the server. Set because the dataset otherwise reaches the
        // table and the wizard's own name/description form with an empty
        // description; strike this line if that is not wanted.
        description: `Imported from HuggingFace ${repoId}`,
        repo_id: repoId,
        config,
        train_split: trainSplit,
        validation_split: validationSplit === NO_VALIDATION ? null : validationSplit,
        validation_percentage: validationSplit === NO_VALIDATION ? validationPercentage : null,
        column_mapping: mapping,
      })
      // Unconditional and ahead of the runId check: the row exists server-side now,
      // whether or not this run gets abandoned next (e.g. the user switches away
      // from the HuggingFace tab while polling starts), so the list must learn
      // about it either way.
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'datasets'] })
      const ready = await pollUntilReady(created.id, runId)
      // Also unconditional, and in addition to the invalidate above rather than
      // instead of it: that one only makes an abandoned run's row visible while it
      // is still `importing`. Because the datasets list is an active query while
      // this form is mounted, it refetches immediately and caches that snapshot;
      // with no second invalidate here, nothing ever told it the row reached
      // `ready`, so it stayed missing from Step 1's existing-dataset dropdown for
      // the rest of the wizard session.
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'datasets'] })
      if (runId !== runIdRef.current) return
      onImported(ready.id)
      resetState()
    } catch (err) {
      // Unconditional and ahead of the runId check, mirroring the invalidate above:
      // the record may already exist server-side whatever went wrong after the
      // POST -- including a client-side timeout or dropped connection after the
      // server had already committed the row -- so make it visible either way
      // rather than leaving an orphan the user cannot see and whose name then
      // collides on retry, even if this run has since been abandoned.
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'datasets'] })
      if (runId !== runIdRef.current) return
      const status = hfErrorStatus(err)
      if (status === 409) {
        // The same repo at a pinned revision is a legitimate second dataset, so the
        // revision is what distinguishes it. Suffix and stop -- the second POST is
        // the user's to make, not ours to fire silently. The server's own detail is
        // rendered too: problemDetail alone would show only "already exists" and
        // never tell the user the name changed underneath them.
        setName((current) => suffixWithRevision(current, revision))
        setError(
          `${problemDetail(err, 'A dataset with that name already exists.')} A revision suffix has been added -- review the name and import again.`
        )
      } else {
        setError(problemDetail(err, (err as Error)?.message || 'Import failed.'))
      }
    } finally {
      if (runId === runIdRef.current) {
        setImporting(false)
        setImportStatus('')
      }
    }
  }

  return {
    suggestions,
    repoId,
    setRepoId,
    handleSearchInput,

    splits,
    splitsFetching,
    splitsError,
    splitsErrorStatus,
    refetchSplits,
    configNames,
    splitNames,
    config,
    handleConfigChange,
    trainSplit,
    handleTrainSplitChange,
    validationSplit,
    setValidationSplit,

    preview,
    freshMappedPreview,
    probeLoading,
    mappedLoading,
    previewError,
    mappedPreviewError,

    mapping,
    setMapping,
    survival,

    name,
    setName,
    nameValid,
    validationPercentage,
    setValidationPercentage,

    canSubmit,
    importing,
    importStatus,
    error,
    handleImport,
    resetState,
  }
}
