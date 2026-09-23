'use client'

import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type {
  ColumnMapping,
  Dataset,
  HfDatasetSplits,
  HfImportPreview,
} from '@granite-build/ui-core/types'
import {
  getAutotuneDatasetTypes,
  getDataset,
  getHfSplits,
  importHfDataset,
  previewHfDataset,
  searchHfDatasets,
  suggestColumnMappingAI,
} from '@granite-build/ui-core/api/autotunex'
import { aiMappingToColumnMapping } from '@granite-build/ui-core/lib/autotunex/aiColumnMapping'
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
  HF_VALIDATION_PERCENTAGE,
  isDatasetNameValid,
  isMappingComplete,
  mappedPreviewKey,
  NO_VALIDATION,
  pollStep,
  probeMapping,
  problemDetail,
  pruneMapping,
  reconcileValidationSplit,
  suffixWithRevision,
  survivalSummary,
  type SurvivalSummary,
} from './hfImport'
import {
  extractColumnMetadata,
  suggestColumnMapping as suggestColumnMappingHeuristic,
} from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { ALGORITHM_TO_DATASET_TYPE } from '@granite-build/ui-core/config/autotunexAlgorithms'

const SEARCH_DEBOUNCE_MS = 300
const SEARCH_LIMIT = 20
export { NO_VALIDATION } from './hfImport'

export interface UseHfImportOptions {
  /** The HuggingFace source is the active tab. Replaces the modal's `open`. */
  active: boolean
  /** Targets that must have a source before the import may be submitted. */
  requiredColumns: string[]
  /**
   * Every target the form renders a select for -- the required columns plus the
   * optional ones the dataset type declares. This, not `requiredColumns`, is the
   * vocabulary the AI suggestion is filtered against and the mapping is pruned to:
   * an optional target the user can see and clear is a legitimate mapping key.
   */
  mappableColumns: string[]
  onImported: (datasetId: string) => void
  selectedAlgorithm: string
  datasetTypes: Record<string, any>

  // The chosen selection is WIZARD state, passed in rather than owned here: Step 1
  // unmounts on wizard navigation, so a user who picked a repo, waited for the
  // probe and then went Back to check something in Step 0 returned to an empty
  // HuggingFace tab. Only the selection is lifted -- every transient value
  // (previews, loading flags, errors, the import status, the AI fields) stays local.
  //
  // `mapping` is deliberately NOT lifted: the probe effect calls `setMapping({})`
  // unconditionally, so a restored mapping would be wiped on remount anyway. The
  // probe refires on return and the AI re-derives the mapping, which is intended.
  //
  // The setters are the raw `useState` dispatchers because `handleImport`'s 409
  // branch calls `setName` with an updater function.
  repoId: string | null
  setRepoId: Dispatch<SetStateAction<string | null>>
  config: string
  setConfig: Dispatch<SetStateAction<string>>
  trainSplit: string
  setTrainSplit: Dispatch<SetStateAction<string>>
  validationSplit: string
  setValidationSplit: Dispatch<SetStateAction<string>>
  name: string
  setName: Dispatch<SetStateAction<string>>
  validationPercentage: number
  setValidationPercentage: Dispatch<SetStateAction<number>>
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
  /** True when no separate validation split is chosen. Drives the split toggle. */
  splitFromTrain: boolean

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

  canSubmit: boolean
  importing: boolean
  importStatus: string
  error: string
  handleImport: () => Promise<void>
  resetState: () => void

  aiSuggestion: { confidence: number; reasoning: string } | null
  isAiSuggesting: boolean
  showAiReasoning: boolean
  setShowAiReasoning: (next: boolean) => void
}

export function useHfImport({
  active,
  requiredColumns,
  mappableColumns,
  onImported,
  selectedAlgorithm,
  datasetTypes,
  // Local names match the lifted wizard state exactly, so the effects,
  // `resetState` and the handlers below read no differently than when these were
  // `useState` pairs owned here.
  repoId,
  setRepoId,
  config,
  setConfig,
  trainSplit,
  setTrainSplit,
  validationSplit,
  setValidationSplit,
  name,
  setName,
  validationPercentage,
  setValidationPercentage,
}: UseHfImportOptions): UseHfImportResult {
  const queryClient = useQueryClient()
  const [suggestions, setSuggestions] = useState<string[]>([])

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
  const [error, setError] = useState('')
  const [importing, setImporting] = useState(false)
  const [importStatus, setImportStatus] = useState('')

  const [aiSuggestion, setAiSuggestion] = useState<
    { confidence: number; reasoning: string } | null
  >(null)
  const [isAiSuggesting, setIsAiSuggesting] = useState(false)
  const [showAiReasoning, setShowAiReasoning] = useState(false)

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

  // `mappableColumns` is a fresh array on every parent render, so it cannot be an
  // effect dependency directly -- the prune effect would refire forever. This
  // string stands in for it, and changes exactly when its contents do.
  const mappableKey = mappableColumns.join(',')

  // The probe's request body needs the current required columns, but its RESPONSE
  // does not depend on them: the server returns source-side `columns` and
  // `raw_rows`, and the payload carries no target_format for it to validate keys
  // against. Reading them through a ref keeps the column lists out of the probe's
  // dependency array below -- otherwise an AI-suggested algorithm change refires
  // the probe, whose body calls setMapping({}), wiping the mapping the AI just
  // wrote.
  const requiredColumnsRef = useRef(requiredColumns)
  requiredColumnsRef.current = requiredColumns
  // Read through a ref for the same reason: the prune effect and the AI suggestion
  // both need the current list without taking it as a dependency.
  const mappableColumnsRef = useRef(mappableColumns)
  mappableColumnsRef.current = mappableColumns

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
  //
  // Keyed on the resolved dataset rather than the response object: `getHfSplits`
  // returns a fresh object per fetch, so a window-focus refetch or a tab
  // round-trip would otherwise reset the user's config, split and edited name.
  //
  // The validity guard covers the other direction: Step 1 unmounts on wizard
  // navigation, and the selection is restored from wizard state on remount, so a
  // freshly-mounted effect must not overwrite a restored selection with defaults.
  useEffect(() => {
    if (!splits) return
    const configNames = Object.keys(splits.configs)
    if (config && configNames.includes(config) && splits.configs[config]?.includes(trainSplit)) {
      return
    }
    const nextConfig = defaultConfig(configNames)
    setConfig(nextConfig)
    setTrainSplit(defaultTrainSplit(splits.configs[nextConfig] ?? []))
    setMapping({})
    setValidationSplit(NO_VALIDATION)
    setName(deriveDatasetName(splits.repo_id))
  }, [splits?.repo_id, splits?.revision, config, trainSplit])

  /**
   * Ask the AI for a column mapping for the probed dataset.
   *
   * Tokenized against `previewTokenRef`, the probe's own counter: the response can
   * outlive the selection it was asked about, and committing a superseded mapping
   * would pass the import gate against the wrong dataset's columns.
   */
  async function suggestMappingWithAI(probe: HfImportPreview, token: number) {
    if (probe.raw_rows.length === 0 || probe.columns.length === 0) return
    const isCurrent = () => previewTokenRef.current === token
    const mappable = mappableColumnsRef.current

    setIsAiSuggesting(true)
    setAiSuggestion(null)
    setShowAiReasoning(false)

    /**
     * The heuristic guess, applied only when the AI cannot supply a mapping.
     *
     * Not applied eagerly, unlike the Upload path's heuristic effect, for two
     * reasons: the form hides its mapping rows while `isAiSuggesting`, so an eager
     * guess would never be seen; and here a complete mapping immediately triggers a
     * mapped-preview request to the backend, so an eager guess would cost a second
     * round trip that the AI's own answer then invalidates. Observable behaviour
     * matches Upload either way -- the mapping is never left empty when the
     * heuristic could fill it.
     */
    const applyHeuristic = () => {
      if (isCurrent()) setMapping(suggestColumnMappingHeuristic(probe.columns, mappable))
    }

    try {
      const metadata = extractColumnMetadata(probe.raw_rows)
      const colSamples: Record<string, string[]> = {}
      for (const col of metadata) colSamples[col.name] = col.sampleValues.slice(0, 3)

      const types =
        Object.keys(datasetTypes).length > 0 ? datasetTypes : await getAutotuneDatasetTypes()
      if (!isCurrent()) return

      const result = await suggestColumnMappingAI({
        sample_data: probe.raw_rows.slice(0, 8),
        column_names: probe.columns,
        column_samples: colSamples,
        target_format: ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm],
      })
      if (!isCurrent()) return

      setAiSuggestion({ confidence: result.confidence, reasoning: result.reasoning ?? '' })

      // `result.tuning_type` is deliberately NOT read. The endpoint returns a
      // dataset-type key there ("dataset_type_a"), not an algorithm id, so it can
      // never name an algorithm to adopt -- confirmed against the live backend. The
      // selected algorithm stays the user's.

      if (!result.column_mapping) {
        applyHeuristic()
        return
      }

      // The vocabulary is exactly the set of rows the form renders -- which is now
      // every column the dataset type declares, optional ones included, matching the
      // Upload path. That equality is load-bearing, not incidental: while the form
      // rendered required columns only, an accepted suggestion for an optional target
      // (dataset_type_a carries `documents_col` and `tools_col`) landed in `mapping`
      // with no select rendered for it, and a sparse source column there dropped
      // `survived` to zero, so `survivalSummary` returned `blocked` and Import was
      // disabled with nothing on screen the user could change. If the form ever goes
      // back to rendering a subset, narrow this to match it. Out-of-range targets are
      // dropped by `aiMappingToColumnMapping`'s own `targetColumns.includes(...)`
      // check, and if that empties the mapping the `applyHeuristic()` fallback below
      // takes over.
      const targetColumns = mappable
      const typeKey = ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm]

      const { mapping: next } = aiMappingToColumnMapping(
        result.column_mapping,
        probe.columns,
        { targetColumns, columnsDict: types[typeKey]?.columns || {} }
      )

      if (Object.keys(next).length > 0) {
        setMapping(next)
      } else {
        // Every entry filtered out -- a key we cannot match, or a source column
        // absent from this split. Leaving `{}` would strand the form with no
        // mapping at all.
        applyHeuristic()
      }
    } catch {
      // Swallowed, matching the Upload path -- an explicit human ruling, not an
      // oversight. NOTE: a 502 from an unconfigured LLM provider is
      // indistinguishable from a no-op here. Surfacing it is user-facing copy and
      // is tracked separately.
      applyHeuristic()
    } finally {
      // A superseded suggestion must not clear the flag its replacement now owns.
      if (isCurrent()) setIsAiSuggesting(false)
    }
  }

  // The probe. Its job is to fetch `columns` and `raw_rows`; `survived` is
  // meaningless for a blank-source mapping and is not read from this response.
  // `sampled` is read: it counts the rows the server looked at, is independent of
  // the mapping, and drives both the "N of M sampled rows" line in HfImportPreview
  // and the zero-rows notice in HfImportForm.
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
      revision: splits.revision,
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
        // Fire-and-forget: it owns its own staleness check against the same token,
        // and awaiting it here would hold the probe's loading flag for the LLM's
        // full round trip.
        void suggestMappingWithAI(result, token)
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
  // `mappableKey` stands in for `mappableColumns`, which is a fresh array on every
  // parent render. pruneMapping returns its argument when nothing is dropped, so
  // this cannot loop.
  //
  // Pruned against every mappable target, not just the required ones: an optional
  // target the form renders is a key the user chose, and pruning to
  // `requiredColumns` would delete it on the next render.
  useEffect(() => {
    setMapping((current) => pruneMapping(current, mappableColumnsRef.current))
  }, [mappableKey])

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
      revision: splits.revision,
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
    setValidationPercentage(HF_VALIDATION_PERCENTAGE)
    setName('')
    setError('')
    setImporting(false)
    setImportStatus('')
    setAiSuggestion(null)
    setIsAiSuggesting(false)
    setShowAiReasoning(false)
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
    // The candidate list is keyed on the *incoming* split, so a selection can go
    // stale: `handleConfigChange` resets it outright, and without this the import
    // could post one split as both train and validation.
    setValidationSplit((current) =>
      reconcileValidationSplit(
        current,
        splitNames.filter((split) => split !== nextSplit),
      ),
    )
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
    hasValidationSplit: validationSplit !== '',
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
    // The server requires that sha, so an unresolved one is a guaranteed 422. Bail
    // before touching the importing/status flags rather than after the round trip.
    if (!revision) return
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
        revision,
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
    splitFromTrain,

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

    canSubmit,
    importing,
    importStatus,
    error,
    handleImport,
    resetState,

    aiSuggestion,
    isAiSuggesting,
    showAiReasoning,
    setShowAiReasoning,
  }
}
