'use client'

import {
  Button,
  Callout,
  ComboBox,
  InlineLoading,
  InlineNotification,
  Select,
  SelectItem,
  TextInput,
} from '@carbon/react'
import type { HfImportConfig } from '@granite-build/ui-core/types'
import { formatBytes } from '@granite-build/ui-core/lib/autotunex/formatBytes'
import { problemDetail } from './hfImport'
import { NO_VALIDATION, type UseHfImportResult } from './useHfImport'
import styles from './HfImportForm.module.scss'

interface HfImportFormProps {
  hf: UseHfImportResult
  requiredColumns: string[]
  hfConfig: HfImportConfig
}

export function HfImportForm({ hf, requiredColumns, hfConfig }: HfImportFormProps) {
  return (
    <div>
      <div className={styles.field}>
        <ComboBox
          id="hf-dataset-search"
          titleText="HuggingFace dataset"
          placeholder="Search the Hub, e.g. vicgalle/alpaca-gpt4"
          items={hf.suggestions}
          itemToString={(item) => item ?? ''}
          selectedItem={hf.repoId}
          // Server-side search, so the local filter must be disabled or it would
          // filter the results a second time against the same term.
          shouldFilterItem={() => true}
          onInputChange={hf.handleSearchInput}
          onChange={({ selectedItem }) => hf.setRepoId(selectedItem ?? null)}
          disabled={hf.importing}
        />
        <p className={styles.limits}>
          Up to {hfConfig.max_rows.toLocaleString('en-US')} rows and{' '}
          {formatBytes(hfConfig.max_bytes)} per import.
        </p>
      </div>

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
              {hf.splitNames
                .filter((split) => split !== hf.trainSplit)
                .map((split) => (
                  <SelectItem key={split} value={split} text={split} />
                ))}
            </Select>
            <p className={styles.revision} title={hf.splits.revision}>
              Revision {hf.splits.revision.slice(0, 7)}
            </p>
          </div>
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
          <p className={styles.subheading}>Column mapping</p>

          {requiredColumns.map((required) => (
            <div className={styles.mappingRow} key={required}>
              <div className={styles.mappingLabel}>{required}</div>
              <Select
                id={`hf-mapping-${required}`}
                labelText=""
                size="sm"
                value={hf.mapping[required] ?? ''}
                onChange={(event) =>
                  hf.setMapping({ ...hf.mapping, [required]: event.target.value })
                }
                disabled={hf.importing}
              >
                <SelectItem value="" text="Choose a column..." />
                {hf.preview!.columns.map((column) => (
                  <SelectItem key={column} value={column} text={column} />
                ))}
              </Select>
            </div>
          ))}

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

          <hr className={styles.sectionDivider} />

          <div className={styles.field}>
            <TextInput
              id="hf-dataset-name"
              labelText="Dataset name"
              value={hf.name}
              invalid={hf.name.length > 0 && !hf.nameValid}
              invalidText={"Use up to 255 characters, without '/', '\\' or '..'."}
              onChange={(event) => hf.setName(event.target.value)}
              disabled={hf.importing}
            />
          </div>

          {hf.validationSplit === NO_VALIDATION && (
            <div className={styles.field}>
              <TextInput
                id="hf-validation-percentage"
                labelText="Validation split (%)"
                type="number"
                min={1}
                max={50}
                value={String(hf.validationPercentage)}
                onChange={(event) => hf.setValidationPercentage(Number(event.target.value))}
                invalid={
                  !Number.isInteger(hf.validationPercentage) ||
                  hf.validationPercentage < 1 ||
                  hf.validationPercentage > 50
                }
                invalidText="Choose between 1 and 50."
                disabled={hf.importing}
              />
            </div>
          )}

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
