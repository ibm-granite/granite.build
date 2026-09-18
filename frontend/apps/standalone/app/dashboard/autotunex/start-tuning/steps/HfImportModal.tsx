'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Callout,
  ComboBox,
  InlineLoading,
  InlineNotification,
  Modal,
  Select,
  SelectItem,
} from '@carbon/react'
import type { HfImportConfig, HfImportPreview } from '@granite-build/ui-core/types'
import { getHfSplits, previewHfDataset, searchHfDatasets } from '@granite-build/ui-core/api/autotunex'
import { PreviewTable } from '@granite-build/ui-core/components/autotunex/shared/PreviewTable'
import { formatBytes } from '@granite-build/ui-core/lib/autotunex/formatBytes'
import { defaultConfig, defaultTrainSplit, hfErrorStatus, probeMapping, problemDetail } from './hfImport'
import styles from './HfImportModal.module.scss'

const SEARCH_DEBOUNCE_MS = 300
const SEARCH_LIMIT = 20
const PREVIEW_ROWS = 10
const CELL_MAX = 120
// Carbon's Select needs a real option value; null is not one.
const NO_VALIDATION = '__none__'

interface HfImportModalProps {
  open: boolean
  onClose: () => void
  requiredColumns: string[]
  hfConfig: HfImportConfig
}

export function HfImportModal({ open, onClose, requiredColumns, hfConfig }: HfImportModalProps) {
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [repoId, setRepoId] = useState<string | null>(null)
  const [config, setConfig] = useState('')
  const [trainSplit, setTrainSplit] = useState('')
  const [validationSplit, setValidationSplit] = useState(NO_VALIDATION)

  const [preview, setPreview] = useState<HfImportPreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState('')

  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const suggestTokenRef = useRef(0)
  const previewTokenRef = useRef(0)

  // `requiredColumns` is a fresh array on every parent render, so it cannot be an
  // effect dependency directly -- the probe effect would refire forever. This
  // string stands in for it, and changes exactly when its contents do.
  const requiredKey = requiredColumns.join(',')

  const {
    data: splits,
    isFetching: splitsFetching,
    error: splitsError,
    refetch: refetchSplits,
  } = useQuery({
    queryKey: ['autotunex', 'hfSplits', repoId],
    queryFn: () => getHfSplits(repoId as string),
    enabled: open && !!repoId,
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
    setValidationSplit(NO_VALIDATION)
  }, [splits])

  // The probe. Its only job is to fetch `columns` and `raw_rows`; `sampled` and
  // `survived` are meaningless for a blank-source mapping and are not read here.
  useEffect(() => {
    if (!open || !repoId || !splits || !config || !trainSplit) return
    // The chosen pair must exist in the splits currently loaded. On switching
    // datasets, `splits` briefly holds the new repo's data while `config`/`trainSplit`
    // still hold the previous repo's (the preselect effect's setState only lands on the
    // next render), which would fire a probe for a config the new repo does not have
    // and flash its error before the real config arrives.
    if (!splits.configs[config]?.includes(trainSplit)) return
    const token = ++previewTokenRef.current
    setPreviewLoading(true)
    setPreviewError('')
    setPreview(null)
    previewHfDataset({
      repo_id: repoId,
      config,
      train_split: trainSplit,
      // Always null, and validationSplit is deliberately not a dependency below:
      // the server samples rows from the train split only, so the validation split
      // cannot change `columns` or `raw_rows`. Including it would re-probe (and, in
      // the next commit, discard the user's mapping) every time they change it.
      validation_split: null,
      column_mapping: probeMapping(requiredColumns),
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
        if (previewTokenRef.current === token) setPreviewLoading(false)
      })
  }, [open, repoId, splits, config, trainSplit, requiredKey])

  function resetState() {
    previewTokenRef.current += 1
    suggestTokenRef.current += 1
    if (debounceRef.current) clearTimeout(debounceRef.current)
    setSuggestions([])
    setRepoId(null)
    setConfig('')
    setTrainSplit('')
    setValidationSplit(NO_VALIDATION)
    setPreview(null)
    setPreviewLoading(false)
    setPreviewError('')
  }

  function handleClose() {
    resetState()
    onClose()
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
    setValidationSplit(NO_VALIDATION)
  }

  const splitsErrorStatus = hfErrorStatus(splitsError)

  return (
    <Modal
      open={open}
      passiveModal
      modalHeading="Import a HuggingFace dataset"
      onRequestClose={handleClose}
      size="lg"
    >
      <ComboBox
        id="hf-dataset-search"
        titleText="Dataset"
        placeholder="Search the Hub, e.g. vicgalle/alpaca-gpt4"
        items={suggestions}
        itemToString={(item) => item ?? ''}
        selectedItem={repoId}
        // Server-side search, so the local filter must be disabled or it would
        // filter the results a second time against the same term.
        shouldFilterItem={() => true}
        onInputChange={handleSearchInput}
        onChange={({ selectedItem }) => setRepoId(selectedItem ?? null)}
      />
      <p className={styles.limits}>
        Up to {hfConfig.max_rows.toLocaleString('en-US')} rows and {formatBytes(hfConfig.max_bytes)}{' '}
        per import.
      </p>

      {splitsFetching && <InlineLoading description="Resolving configs and splits..." />}

      {/* Two components rather than one with a conditional prop, because
          `InlineNotification` in @carbon/react 1.108 has NO `actions` prop: its
          implementation destructures title/subtitle/kind/lowContrast/hideCloseButton
          and spreads the rest onto a div, so an `actions` prop would silently become
          a DOM attribute instead of rendering a button.
          `Callout` — not `ActionableNotification` — is the component that carries an
          action button without alertdialog semantics. `ActionableNotification`
          defaults to `role="alertdialog"` with `hasFocus`, which focuses its action
          button on mount AND wraps Tab inside the notification while it is open
          (Notification.js:315,334,352), so on a 503 the user could not reach this
          modal's own Close button by keyboard — while the 503's message is precisely
          "Try again later, or upload a file", i.e. it recommends leaving. Carbon's own
          `@deprecated` note on `hasFocus` points at `Callout` for this case.
          Only a 503 is worth retrying: a 422 means the dataset has no tabular data at
          all, which waiting cannot change, so a Retry button there would promise
          something it cannot deliver. */}
      {!!splitsError &&
        (splitsErrorStatus === 503 ? (
          <Callout
            kind="error"
            title="Could not read this dataset"
            subtitle={problemDetail(splitsError, 'Could not read this dataset.')}
            lowContrast
            className={styles.section}
            actionButtonLabel="Retry"
            onActionButtonClick={() => refetchSplits()}
          />
        ) : (
          <InlineNotification
            kind="error"
            title="Could not read this dataset"
            subtitle={problemDetail(splitsError, 'Could not read this dataset.')}
            lowContrast
            hideCloseButton
            className={styles.section}
          />
        ))}

      {splits && (
        <>
          <div className={styles.row}>
            <div className={styles.rowItem}>
              {/* A ComboBox, not a Select: allenai/c4 has 112 configs and
                  HuggingFaceFW/finewiki has 325. The local filter is left ON here
                  (unlike the search box above) because every config arrives in
                  this one response and filtering is client-side. */}
              <ComboBox
                id="hf-config"
                titleText="Config"
                placeholder="Select a config"
                items={configNames}
                itemToString={(item) => item ?? ''}
                selectedItem={config || null}
                onChange={({ selectedItem }) => handleConfigChange(selectedItem ?? '')}
              />
            </div>
            <div className={styles.rowItem}>
              <Select
                id="hf-train-split"
                labelText="Train split"
                value={trainSplit}
                onChange={(event) => setTrainSplit(event.target.value)}
              >
                {splitNames.map((split) => (
                  <SelectItem key={split} value={split} text={split} />
                ))}
              </Select>
            </div>
            <div className={styles.rowItem}>
              <Select
                id="hf-validation-split"
                labelText="Validation split"
                value={validationSplit}
                onChange={(event) => setValidationSplit(event.target.value)}
              >
                {/* Defaults to none rather than guessing at a split named
                    "validation" or "test" -- see the comment on the preselect
                    effect. */}
                <SelectItem value={NO_VALIDATION} text="None (split from train)" />
                {splitNames
                  .filter((split) => split !== trainSplit)
                  .map((split) => (
                    <SelectItem key={split} value={split} text={split} />
                  ))}
              </Select>
            </div>
          </div>
          <p className={styles.revision} title={splits.revision}>
            Revision {splits.revision.slice(0, 7)}
          </p>
        </>
      )}

      {previewLoading && <InlineLoading description="Loading sample rows..." />}

      {!!previewError && (
        <InlineNotification
          kind="error"
          title="Could not preview this dataset"
          subtitle={previewError}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}

      {preview && (
        <>
          <p className={styles.subheading}>Sample rows</p>
          <PreviewTable rows={preview.raw_rows} maxRows={PREVIEW_ROWS} maxCellChars={CELL_MAX} />
        </>
      )}
    </Modal>
  )
}
