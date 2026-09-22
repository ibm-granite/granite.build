'use client'

import {
  Button,
  Callout,
  ComboBox,
  FileUploaderItem,
  InlineLoading,
  InlineNotification,
  Select,
  SelectItem,
  Tag,
  TextInput,
} from '@carbon/react'
import type { HfImportConfig } from '@granite-build/ui-core/types'
import { formatBytes } from '@granite-build/ui-core/lib/autotunex/formatBytes'
import { toUpperCase } from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { InfoTooltip } from './InfoTooltip'
import { problemDetail } from './hfImport'
import { NO_VALIDATION, type UseHfImportResult } from './useHfImport'
import styles from './HfImportForm.module.scss'

interface HfImportFormProps {
  hf: UseHfImportResult
  /**
   * Every target to render a mapping select for, required first, as
   * Step1DatasetUpload's own mapping block does. Optional targets get a "None"
   * option; required ones gate the Import button via the hook.
   */
  mappableColumns: { name: string; desc: string; required: boolean }[]
  hfConfig: HfImportConfig
}

export function HfImportForm({ hf, mappableColumns, hfConfig }: HfImportFormProps) {
  // Only a split other than the one being trained on can serve as validation. With
  // none left over there is nothing to choose, and with a single split there is
  // nothing to choose for training either -- so both selects are hidden rather than
  // rendered with one inevitable option, matching how the Upload path shows no
  // validation affordance until there is a file to attach.
  const validationCandidates = hf.splitNames.filter((split) => split !== hf.trainSplit)

  return (
    <div>
      <div className={styles.field}>
        {hf.repoId ? (
          /* Labelled on the left like the Upload path's Train file row: on its own,
             a bare repo id under the tab strip did not say what it was. */
          <div className={styles.fileRow}>
            <span className={styles.fileLabel}>
              Dataset
              <InfoTooltip label="The HuggingFace repository this dataset is imported from." />
            </span>
            <FileUploaderItem name={hf.repoId} status="edit" onDelete={() => hf.resetState()} />
          </div>
        ) : (
          <>
            <ComboBox
              id="hf-dataset-search"
              titleText="HuggingFace dataset"
              placeholder="Search the Hub, e.g. vicgalle/alpaca-gpt4"
              items={hf.suggestions}
              itemToString={(item) => item ?? ''}
              // Server-side search, so the local filter must be disabled or it would
              // filter the results a second time against the same term.
              shouldFilterItem={() => true}
              onInputChange={hf.handleSearchInput}
              // Deliberately uncontrolled: this branch renders only while
              // `hf.repoId` is falsy (a chosen repo is shown as the
              // FileUploaderItem above), so there is no selection to feed back in
              // and a cleared selection is not reachable here.
              onChange={({ selectedItem }) => {
                if (selectedItem) hf.setRepoId(selectedItem)
              }}
              disabled={hf.importing}
            />
            <p className={styles.limits}>
              Up to {hfConfig.max_rows.toLocaleString('en-US')} rows and{' '}
              {formatBytes(hfConfig.max_bytes)} per import.
            </p>
          </>
        )}
      </div>

      {/* Directly under the selected repo: the name is the field the user is most
          likely to edit, and it sat below the whole mapping before. Gated on
          `splits` rather than on `repoId` so it appears already filled by the
          preselect effect, instead of flashing empty while the repo resolves. */}
      {hf.splits && (
        <div className={styles.field}>
          <TextInput
            id="hf-dataset-name"
            labelText="Dataset Name"
            value={hf.name}
            invalid={hf.name.length > 0 && !hf.nameValid}
            invalidText={"Use up to 255 characters, without '/', '\\' or '..'."}
            onChange={(event) => hf.setName(event.target.value)}
            disabled={hf.importing}
          />
        </div>
      )}

      {hf.splitsFetching && <InlineLoading description="Resolving configs and splits..." />}

      {/* Only a 503 is worth retrying: a 422 means the dataset has no tabular data
          at all, which waiting cannot change. `Callout` rather than
          `ActionableNotification` because the latter defaults to role="alertdialog"
          with `hasFocus`, which traps Tab inside the notification -- and
          `InlineNotification` in @carbon/react 1.108 has no `actions` prop at all
          (it would become a DOM attribute). */}
      {!!hf.splitsError &&
        (hf.splitsErrorStatus === 503 ? (
          <Callout
            kind="error"
            title="Could not read this dataset"
            subtitle={problemDetail(hf.splitsError, 'Could not read this dataset.')}
            lowContrast
            className={styles.section}
            actionButtonLabel="Retry"
            onActionButtonClick={() => hf.refetchSplits()}
          />
        ) : (
          <InlineNotification
            kind="error"
            title="Could not read this dataset"
            subtitle={problemDetail(hf.splitsError, 'Could not read this dataset.')}
            lowContrast
            hideCloseButton
            className={styles.section}
          />
        ))}

      {hf.splits && (
        <>
          {/* A ComboBox, not a Select: allenai/c4 has 112 configs and
              HuggingFaceFW/finewiki has 325. The local filter is left ON here
              (unlike the search box above) because every config arrives in this one
              response and filtering is client-side. */}
          <div className={styles.field}>
            <ComboBox
              id="hf-config"
              titleText="Config"
              placeholder="Select a config"
              items={hf.configNames}
              itemToString={(item) => item ?? ''}
              selectedItem={hf.config || null}
              onChange={({ selectedItem }) => hf.handleConfigChange(selectedItem ?? '')}
              disabled={hf.importing}
            />
          </div>
          {/* One split means one possible answer, so the control asks a question the
              user cannot answer differently. `trainSplit` is still set to it by the
              preselect effect -- this hides the select, not the choice. */}
          {hf.splitNames.length > 1 && (
            <div className={styles.field}>
              <Select
                id="hf-train-split"
                labelText="Train split"
                value={hf.trainSplit}
                onChange={(event) => hf.handleTrainSplitChange(event.target.value)}
                disabled={hf.importing}
              >
                {hf.splitNames.map((split) => (
                  <SelectItem key={split} value={split} text={split} />
                ))}
              </Select>
            </div>
          )}
          {validationCandidates.length > 0 && (
            <div className={styles.field}>
              <Select
                id="hf-validation-split"
                labelText="Validation split"
                value={hf.validationSplit}
                onChange={(event) => hf.setValidationSplit(event.target.value)}
                disabled={hf.importing}
              >
                {/* Defaults to none rather than guessing at a split named
                    "validation" or "test": a silently auto-picked wrong split is only
                    discovered after a multi-hour tuning run. */}
                <SelectItem value={NO_VALIDATION} text="None (split from train)" />
                {validationCandidates.map((split) => (
                  <SelectItem key={split} value={split} text={split} />
                ))}
              </Select>
            </div>
          )}
          {/* Outside both blocks above: a single-split dataset hides them, and the
              revision is what identifies the snapshot being imported either way. */}
          <p className={styles.revision} title={hf.splits.revision}>
            Revision {hf.splits.revision.slice(0, 7)}
          </p>
        </>
      )}

      {(hf.probeLoading || hf.mappedLoading) && (
        <InlineLoading description="Loading sample rows..." />
      )}

      {!!hf.previewError && (
        <InlineNotification
          kind="error"
          title="Could not preview this dataset"
          subtitle={hf.previewError}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}

      {/* Own state and own title: this error is about the mapped re-preview, not the
          probe above. */}
      {!!hf.mappedPreviewError && (
        <InlineNotification
          kind="error"
          title="Could not preview this mapping"
          subtitle={hf.mappedPreviewError}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}

      {hf.preview && (
        <>
          {/* Independent of the mapping and of survivalSummary: with zero raw rows
              there are no columns, so the mapping below can never complete and
              survivalSummary's own zero-sampled branch is unreachable. This is the
              only thing that tells the user why, before they hit a disabled Import
              button with no explanation. */}
          {hf.preview.sampled === 0 && (
            <InlineNotification
              kind="info"
              title="No rows in this split"
              subtitle="This split returned no rows. Choose a different split."
              lowContrast
              hideCloseButton
              className={styles.section}
            />
          )}

          <hr className={styles.sectionDivider} />
          <div className={styles.mappingHeaderRow}>
            <p className={styles.subheading} style={{ margin: 0 }}>
              Column mapping
            </p>
            {hf.isAiSuggesting ? (
              <InlineLoading description="AI analyzing..." />
            ) : hf.aiSuggestion ? (
              <button
                type="button"
                className={styles.aiTagButton}
                onClick={() => hf.setShowAiReasoning(!hf.showAiReasoning)}
              >
                <Tag type="green" size="sm">
                  {/* Guarded: the response adapter casts `confidence` without
                      validating it, so a missing field would render "(NaN%)". */}
                  AI Suggested
                  {Number.isFinite(hf.aiSuggestion.confidence)
                    ? ` (${Math.round(hf.aiSuggestion.confidence * 100)}%)`
                    : ''}{' '}
                  {hf.showAiReasoning ? '▴' : '▾'}
                </Tag>
              </button>
            ) : null}
          </div>

          {hf.aiSuggestion?.reasoning && hf.showAiReasoning && (
            <InlineNotification
              kind="info"
              title="AI Insight"
              subtitle={hf.aiSuggestion.reasoning}
              hideCloseButton
              lowContrast
              className={styles.section}
            />
          )}

          {!hf.isAiSuggesting && (
            <>
              <div className={styles.mappingHeader}>
                <span>Field</span>
                <span>Source Column</span>
              </div>
              {mappableColumns.map((colInfo) => (
                <div className={styles.mappingRow} key={colInfo.name}>
                  <div className={styles.mappingLabel}>
                    {toUpperCase(colInfo.name) ?? colInfo.name}
                    {colInfo.desc && <InfoTooltip label={colInfo.desc} />}
                  </div>
                  <Select
                    id={`hf-mapping-${colInfo.name}`}
                    // The visual label is the div above and `labelText` stays empty
                    // for the layout, but Carbon renders that as an empty <label for>,
                    // which leaves the select with no accessible name.
                    aria-label={colInfo.name}
                    labelText=""
                    size="sm"
                    value={hf.mapping[colInfo.name] ?? ''}
                    onChange={(event) =>
                      hf.setMapping({ ...hf.mapping, [colInfo.name]: event.target.value })
                    }
                    disabled={hf.importing}
                  >
                    {/* An optional target is legitimately unmapped, so its empty
                        option reads as a choice rather than as something missing.
                        Selecting it writes a blank source, which the server's
                        apply_mapping skips. */}
                    <SelectItem
                      value=""
                      text={colInfo.required ? 'Choose a column...' : 'None'}
                    />
                    {hf.preview!.columns.map((column) => (
                      <SelectItem key={column} value={column} text={column} />
                    ))}
                  </Select>
                </div>
              ))}
            </>
          )}

          {hf.survival.kind !== 'hidden' && (
            <InlineNotification
              kind={
                hf.survival.kind === 'blocked'
                  ? 'error'
                  : hf.survival.kind === 'warning'
                    ? 'warning'
                    : 'success'
              }
              title={hf.survival.kind === 'blocked' ? 'This mapping keeps no rows' : 'Mapped rows'}
              subtitle={hf.survival.text}
              lowContrast
              hideCloseButton
              className={styles.section}
            />
          )}

          {/* No validation-percentage field: the split is a constant
              (HF_VALIDATION_PERCENTAGE), because the Upload path offers no ratio
              control either. */}
          <hr className={styles.sectionDivider} />

          <div className={styles.importRow}>
            <Button
              kind="primary"
              size="sm"
              disabled={!hf.canSubmit}
              onClick={() => hf.handleImport()}
            >
              {hf.importing ? 'Importing...' : 'Import'}
            </Button>
          </div>
        </>
      )}

      {hf.importing && (
        <InlineLoading
          description={`Importing from HuggingFace (${hf.importStatus || 'importing'})... this can take several minutes.`}
          className={styles.section}
        />
      )}

      {!!hf.error && (
        <InlineNotification
          kind="error"
          title="Import failed"
          subtitle={hf.error}
          lowContrast
          hideCloseButton
          className={styles.section}
        />
      )}
    </div>
  )
}
