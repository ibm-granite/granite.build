'use client'

import {
  Callout,
  ComboBox,
  InlineLoading,
  InlineNotification,
  Modal,
  Select,
  SelectItem,
  TextInput,
} from '@carbon/react'
import type { HfImportConfig } from '@granite-build/ui-core/types'
import { PreviewTable } from '@granite-build/ui-core/components/autotunex/shared/PreviewTable'
import { formatBytes } from '@granite-build/ui-core/lib/autotunex/formatBytes'
import { problemDetail } from './hfImport'
import { NO_VALIDATION, useHfImport } from './useHfImport'
import styles from './HfImportModal.module.scss'

const PREVIEW_ROWS = 10
const CELL_MAX = 120

interface HfImportModalProps {
  open: boolean
  onClose: () => void
  onImported: (datasetId: string) => void
  requiredColumns: string[]
  hfConfig: HfImportConfig
}

export function HfImportModal({
  open,
  onClose,
  onImported,
  requiredColumns,
  hfConfig,
}: HfImportModalProps) {
  const hf = useHfImport({ active: open, requiredColumns, onImported })
  const {
    suggestions, repoId, setRepoId, handleSearchInput,
    splits, splitsFetching, splitsError, splitsErrorStatus, refetchSplits,
    configNames, splitNames, config, handleConfigChange,
    trainSplit, handleTrainSplitChange, validationSplit, setValidationSplit,
    preview, freshMappedPreview, probeLoading, mappedLoading,
    previewError, mappedPreviewError,
    mapping, setMapping, survival,
    name, setName, nameValid, validationPercentage, setValidationPercentage,
    canSubmit, importing, importStatus, error, handleImport, resetState,
  } = hf

  function handleClose() {
    // The dataset row, once the POST resolves, is made visible by the
    // unconditional invalidate in handleImport -- right after importHfDataset
    // resolves, regardless of whether this run is later abandoned. Gating an
    // invalidate here on `importing` fired before that POST had resolved, which
    // was too early for the row to exist yet. Closing only abandons this run's own
    // polling loop (the runIdRef bump below); the server-side import, once
    // started, keeps running either way.
    resetState()
    onClose()
  }

  return (
    <Modal
      open={open}
      modalHeading="Import a HuggingFace dataset"
      primaryButtonText={importing ? 'Importing...' : 'Import'}
      secondaryButtonText="Cancel"
      primaryButtonDisabled={!canSubmit}
      onRequestSubmit={handleImport}
      onRequestClose={handleClose}
      onSecondarySubmit={handleClose}
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
        disabled={importing}
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
                disabled={importing}
              />
            </div>
            <div className={styles.rowItem}>
              <Select
                id="hf-train-split"
                labelText="Train split"
                value={trainSplit}
                onChange={(event) => handleTrainSplitChange(event.target.value)}
                disabled={importing}
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
                disabled={importing}
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

      {(probeLoading || mappedLoading) && <InlineLoading description="Loading sample rows..." />}

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

      {/* Own state and own title: this error is about the mapped re-preview, not
          the probe above, and sharing `previewError` made "Could not preview this
          mapping." render under the title "Could not preview this dataset". */}
      {!!mappedPreviewError && (
        <InlineNotification
          kind="error"
          title="Could not preview this mapping"
          subtitle={mappedPreviewError}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}

      {preview && (
        <>
          <p className={styles.subheading}>Sample rows</p>
          <PreviewTable
            rows={preview.raw_rows}
            maxRows={PREVIEW_ROWS}
            maxCellChars={CELL_MAX}
            emptyMessage="This split returned no rows."
          />

          {/* Independent of mappingComplete and survivalSummary: with zero raw
              rows there are no columns, so the mapping below can never complete
              and survivalSummary's own zero-sampled branch is unreachable. This
              is the only thing that tells the user why, before they hit a
              disabled Import button with no explanation. */}
          {preview.sampled === 0 && (
            <InlineNotification
              kind="info"
              title="No rows in this split"
              subtitle="This split returned no rows. Choose a different split."
              lowContrast
              hideCloseButton
              className={styles.section}
            />
          )}

          <p className={styles.subheading}>Column mapping</p>
          <div className={styles.row}>
            {requiredColumns.map((required) => (
              <div className={styles.rowItem} key={required}>
                <Select
                  id={`hf-mapping-${required}`}
                  labelText={required}
                  value={mapping[required] ?? ''}
                  onChange={(event) =>
                    setMapping({ ...mapping, [required]: event.target.value })
                  }
                  disabled={importing}
                >
                  <SelectItem value="" text="Choose a column..." />
                  {preview.columns.map((column) => (
                    <SelectItem key={column} value={column} text={column} />
                  ))}
                </Select>
              </div>
            ))}
          </div>

          {survival.kind !== 'hidden' && (
            <InlineNotification
              kind={
                survival.kind === 'blocked'
                  ? 'error'
                  : survival.kind === 'warning'
                    ? 'warning'
                    : 'success'
              }
              title={survival.kind === 'blocked' ? 'This mapping keeps no rows' : 'Mapped rows'}
              subtitle={survival.text}
              lowContrast
              hideCloseButton
              className={styles.section}
            />
          )}

          {freshMappedPreview && (
            <>
              <p className={styles.subheading}>Mapped rows</p>
              <PreviewTable rows={freshMappedPreview.mapped_rows} maxRows={PREVIEW_ROWS} maxCellChars={CELL_MAX} />
            </>
          )}

          <div className={styles.row}>
            <div className={styles.rowItem}>
              <TextInput
                id="hf-dataset-name"
                labelText="Dataset name"
                value={name}
                invalid={name.length > 0 && !nameValid}
                invalidText={"Use up to 255 characters, without '/', '\\' or '..'."}
                onChange={(event) => setName(event.target.value)}
                disabled={importing}
              />
            </div>
            {validationSplit === NO_VALIDATION && (
              <div className={styles.rowItem}>
                <TextInput
                  id="hf-validation-percentage"
                  labelText="Validation split (%)"
                  type="number"
                  min={1}
                  max={50}
                  value={String(validationPercentage)}
                  onChange={(event) => setValidationPercentage(Number(event.target.value))}
                  invalid={
                    !Number.isInteger(validationPercentage) ||
                    validationPercentage < 1 ||
                    validationPercentage > 50
                  }
                  invalidText="Choose between 1 and 50."
                  disabled={importing}
                />
              </div>
            )}
          </div>
        </>
      )}

      {importing && (
        <InlineLoading
          description={`Importing from HuggingFace (${importStatus || 'importing'})... this can take several minutes.`}
          className={styles.section}
        />
      )}

      {!!error && (
        <InlineNotification
          kind="error"
          title="Import failed"
          subtitle={error}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}
    </Modal>
  )
}
