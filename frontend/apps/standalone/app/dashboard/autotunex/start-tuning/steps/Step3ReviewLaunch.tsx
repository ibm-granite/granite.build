'use client'

import { Tile, TextInput, Tag, Button, ProgressBar } from '@carbon/react'
import { DataBase, Settings, ModelTuned, Checkmark, Edit } from '@carbon/icons-react'
import type { ColumnMetadata, Configuration, Dataset, DatasetForm, LaunchPhase, ModelSource, Resources } from '@granite-build/ui-core/types'
import { getConfigSummary } from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { MODEL_SOURCE_LABELS } from '../../modelSources'
import styles from './Step3ReviewLaunch.module.scss'
import layoutStyles from '@granite-build/ui-core/components/layout.module.scss'

function formatFileSize(bytes: number): string {
  if (!bytes || bytes <= 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatTimeBudget(seconds: number): string {
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  const h = Math.floor(seconds / 3600)
  const m = Math.round((seconds % 3600) / 60)
  return m > 0 ? `${h}h ${m}m` : `${h}h`
}

function formatPercent(val: number): string {
  return `${(val * 100).toFixed(0)}%`
}

interface Step3ReviewLaunchProps {
  uploadedFile: File | null
  datasetForm: DatasetForm
  selectedExistingDataset: Dataset | null
  selectedConfig: Configuration | null
  selectedModel: string
  modelSource: ModelSource
  experimentName: string
  setExperimentName: (v: string) => void
  isPendingDataset: boolean
  isPendingConfig: boolean
  launchPhase: LaunchPhase
  uploadProgress: number
  resourceEstimation: Resources | null
  estimationUnavailable: boolean
  totalRecords: number
  splitRatio: number
  isSplitEnabled: boolean
  validationFile: File | null
  autotuneEnabled: boolean
  columnMetadata: ColumnMetadata[]
  onEditStep: (step: number) => void
}

export function Step3ReviewLaunch({
  uploadedFile,
  datasetForm,
  selectedExistingDataset,
  selectedConfig,
  selectedModel,
  modelSource,
  experimentName,
  setExperimentName,
  isPendingDataset,
  isPendingConfig,
  launchPhase,
  uploadProgress,
  resourceEstimation,
  estimationUnavailable,
  totalRecords,
  splitRatio,
  isSplitEnabled,
  validationFile,
  autotuneEnabled,
  columnMetadata,
  onEditStep,
}: Step3ReviewLaunchProps) {
  const isExisting = !!selectedExistingDataset

  const trainFileName = isExisting ? selectedExistingDataset!.train_file : uploadedFile?.name ?? null
  const valFileName = isExisting
    ? selectedExistingDataset!.validation_file
    : validationFile?.name ?? (isSplitEnabled ? 'Auto-split from train' : null)
  const trainRecords = isExisting
    ? selectedExistingDataset!.train_records
    : isSplitEnabled
      ? Math.round((totalRecords * splitRatio) / 100)
      : totalRecords
  const valRecords = isExisting
    ? selectedExistingDataset!.validation_records || 0
    : isSplitEnabled
      ? Math.round((totalRecords * (100 - splitRatio)) / 100)
      : null
  const trainFileSize = isExisting ? selectedExistingDataset!.train_file_size : uploadedFile?.size ?? 0
  const valFileSize = isExisting ? selectedExistingDataset!.validation_file_size : validationFile?.size ?? 0

  const tuneConfig = selectedConfig?.config_data?.tune_config ?? null
  const trainingConfig = selectedConfig?.config_data?.training_config ?? null

  const numGpusPerTrial = (trainingConfig as any)?.num_gpus_per_trial?.default as number | undefined
  const maxLength = (trainingConfig as any)?.max_length?.default as number | undefined
  const trainImpl = (trainingConfig as any)?.train_implementation?.default as string | undefined
  const dsStrategy = (trainingConfig as any)?.ds_strategy?.default as string | undefined
  const fsdpStrategy = (trainingConfig as any)?.fsdp_strategy?.default as string | undefined

  const laterPhases = (...phases: LaunchPhase[]) => phases.includes(launchPhase)

  return (
    <div>
      <div className={styles.experimentNameField}>
          <TextInput
            id="experiment-name"
            labelText="Experiment Name"
            placeholder="Enter a unique name for this tuning job"
            value={experimentName}
            onChange={(e) => setExperimentName(e.target.value)}
            onBlur={() => setExperimentName(experimentName.trim().replace(/\s+/g, '_'))}
          />
      </div>

      <div className={layoutStyles.rowWrap} style={{ marginTop: '1.5rem' }}>
        <div className={styles.cardColumn}>
          <Tile className={styles.reviewCard}>
            <div className={styles.cardHeader}>
              <ModelTuned size={20} className={styles.cardIcon} />
              <h6 className={styles.cardHeading}>Model</h6>
              <div style={{ marginLeft: 'auto' }}>
                <Button kind="ghost" size="sm" renderIcon={Edit} iconDescription="Edit model" hasIconOnly onClick={() => onEditStep(0)} />
              </div>
            </div>
            <div className={styles.cardBody}>
              <div className={styles.cardRow}>
                <span className={styles.cardLabel}>Model</span>
                <span className={styles.cardValue}>{selectedModel}</span>
              </div>
              <div className={styles.cardRow}>
                <span className={styles.cardLabel}>Source</span>
                <span className={styles.cardValue}>
                  {MODEL_SOURCE_LABELS[modelSource]}
                </span>
              </div>
              <div className={styles.cardRow}>
                <span className={styles.cardLabel}>AutoTune</span>
                <span className={styles.cardValue}>
                  <Tag size="sm" type={autotuneEnabled ? 'green' : 'cool-gray'}>{autotuneEnabled ? 'Enabled' : 'Disabled'}</Tag>
                </span>
              </div>
            </div>
          </Tile>
        </div>

        <div className={styles.cardColumn}>
          <Tile className={styles.reviewCard}>
            <div className={styles.cardHeader}>
              <DataBase size={20} className={styles.cardIcon} />
              <h6 className={styles.cardHeading}>Dataset</h6>
              {isPendingDataset && <Tag type="cyan" size="sm">New</Tag>}
              <div style={{ marginLeft: 'auto' }}>
                <Button kind="ghost" size="sm" renderIcon={Edit} iconDescription="Edit dataset" hasIconOnly onClick={() => onEditStep(1)} />
              </div>
            </div>
            <div className={styles.cardBody}>
              <div className={styles.cardRow}>
                <span className={styles.cardLabel}>Name</span>
                <span className={styles.cardValue}>{datasetForm.name}</span>
              </div>
              {datasetForm.description && (
                <div className={styles.cardRow}>
                  <span className={styles.cardLabel}>Description</span>
                  <span className={`${styles.cardValue} ${styles.cardValueTruncate}`} title={datasetForm.description}>
                    {datasetForm.description}
                  </span>
                </div>
              )}

              {trainFileName && (
                <>
                  <div className={styles.sectionDivider} />
                  <span className={styles.sectionLabel}>Training</span>
                  <div className={styles.cardRow}>
                    <span className={styles.cardLabel}>File</span>
                    <span className={`${styles.cardValue} ${styles.cardValueTruncate}`} title={trainFileName}>{trainFileName}</span>
                  </div>
                  {trainRecords > 0 && (
                    <div className={styles.cardRow}>
                      <span className={styles.cardLabel}>Records</span>
                      <span className={styles.cardValue}>{trainRecords.toLocaleString()}</span>
                    </div>
                  )}
                  {trainFileSize > 0 && (
                    <div className={styles.cardRow}>
                      <span className={styles.cardLabel}>Size</span>
                      <span className={styles.cardValue}>{formatFileSize(trainFileSize)}</span>
                    </div>
                  )}
                </>
              )}

              {valFileName && (
                <>
                  <div className={styles.sectionDivider} />
                  <span className={styles.sectionLabel}>Validation</span>
                  <div className={styles.cardRow}>
                    <span className={styles.cardLabel}>File</span>
                    <span className={`${styles.cardValue} ${styles.cardValueTruncate}`} title={valFileName}>{valFileName}</span>
                  </div>
                  <div className={styles.cardRow}>
                    <span className={styles.cardLabel}>Records</span>
                    <span className={styles.cardValue}>{valRecords != null ? valRecords.toLocaleString() : '—'}</span>
                  </div>
                  {valFileSize > 0 && (
                    <div className={styles.cardRow}>
                      <span className={styles.cardLabel}>Size</span>
                      <span className={styles.cardValue}>{formatFileSize(valFileSize)}</span>
                    </div>
                  )}
                </>
              )}

              {columnMetadata.length > 0 && (
                <>
                  <div className={styles.sectionDivider} />
                  <div className={`${styles.cardRow} ${styles.cardRowTop}`}>
                    <span className={styles.cardLabel}>Columns</span>
                    <span className={styles.cardValue}>
                      <div className={styles.columnTags}>
                        {columnMetadata.map((col) => (
                          <Tag key={col.name} size="sm" type="cool-gray">{col.name}</Tag>
                        ))}
                      </div>
                    </span>
                  </div>
                </>
              )}
            </div>
          </Tile>
        </div>

        <div className={styles.cardColumnWide}>
          <Tile className={styles.reviewCard}>
            <div className={styles.cardHeader}>
              <Settings size={20} className={styles.cardIcon} />
              <h6 className={styles.cardHeading}>Configuration</h6>
              {isPendingConfig && <Tag type="cyan" size="sm">New</Tag>}
              <div style={{ marginLeft: 'auto' }}>
                <Button kind="ghost" size="sm" renderIcon={Edit} iconDescription="Edit configuration" hasIconOnly onClick={() => onEditStep(2)} />
              </div>
            </div>
            {selectedConfig ? (
              <div className={styles.cardBody}>
                <div className={styles.cardRow}>
                  <span className={styles.cardLabel}>Name</span>
                  <span className={styles.cardValue}>{selectedConfig.name}</span>
                </div>
                <div className={styles.cardRow}>
                  <span className={styles.cardLabel}>Algorithm</span>
                  <span className={styles.cardValue}>{getConfigSummary(selectedConfig)}</span>
                </div>

                {(tuneConfig || trainingConfig) && (
                  <>
                    <div className={styles.sectionDivider} />
                    <span className={styles.sectionLabel}>HPO Settings</span>
                    <div className={styles.configGrid}>
                      {tuneConfig?.num_samples && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>Trials</span>
                          <span className={styles.configVal}>{tuneConfig.num_samples.default}</span>
                        </div>
                      )}
                      {tuneConfig?.max_concurrent_trials && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>Concurrent</span>
                          <span className={styles.configVal}>{tuneConfig.max_concurrent_trials.default}</span>
                        </div>
                      )}
                      {numGpusPerTrial != null && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>GPUs / Trial</span>
                          <span className={styles.configVal}>{numGpusPerTrial}</span>
                        </div>
                      )}
                      {tuneConfig?.time_budget_s?.default != null && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>Time Budget</span>
                          <span className={styles.configVal}>{formatTimeBudget(tuneConfig.time_budget_s.default)}</span>
                        </div>
                      )}
                      {trainingConfig?.hpo_dataset_percentage && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>HPO Data %</span>
                          <span className={styles.configVal}>{formatPercent(Number(trainingConfig.hpo_dataset_percentage.default))}</span>
                        </div>
                      )}
                      {maxLength != null && (
                        <div className={styles.configField}>
                          <span className={styles.configKey}>Max Length</span>
                          <span className={styles.configVal}>{maxLength.toLocaleString()}</span>
                        </div>
                      )}
                    </div>

                    {trainImpl && (
                      <>
                        <div className={styles.sectionDivider} />
                        <span className={styles.sectionLabel}>Distribution</span>
                        <div className={styles.configGrid}>
                          <div className={styles.configField}>
                            <span className={styles.configKey}>Train Impl</span>
                            <span className={styles.configVal}>{trainImpl}</span>
                          </div>
                          {trainImpl.toLowerCase() === 'deepspeed' && dsStrategy && (
                            <div className={styles.configField}>
                              <span className={styles.configKey}>DS Strategy</span>
                              <span className={styles.configVal}>{dsStrategy}</span>
                            </div>
                          )}
                          {trainImpl.toLowerCase() === 'fsdp' && fsdpStrategy && (
                            <div className={styles.configField}>
                              <span className={styles.configKey}>FSDP Strategy</span>
                              <span className={styles.configVal}>{fsdpStrategy}</span>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            ) : (
              <p className={styles.emptyHint}>No configuration selected</p>
            )}
          </Tile>
        </div>
      </div>

      {resourceEstimation ? (
        <div style={{ marginTop: '1rem' }}>
            <Tile className={styles.reviewCard}>
              <h6 className={styles.cardHeading} style={{ marginBottom: '0.75rem' }}>Estimated Resources</h6>
              <div className={styles.resourceRow}>
                <div className={styles.resourceItem}>
                  <span className={styles.configKey}>Model Size</span>
                  <span className={styles.configVal}>{resourceEstimation.model_size_billion_params.toFixed(1)}B params</span>
                </div>
                <div className={styles.resourceItem}>
                  <span className={styles.configKey}>GPU Memory</span>
                  <span className={styles.configVal}>{resourceEstimation.gpu_memory_gb.toFixed(1)} GB</span>
                </div>
                <div className={styles.resourceItem}>
                  <span className={styles.configKey}>GPUs Required</span>
                  <span className={styles.configVal}>{resourceEstimation.num_gpus}</span>
                </div>
                <div className={styles.resourceItem}>
                  <span className={styles.configKey}>CPU Memory</span>
                  <span className={styles.configVal}>{resourceEstimation.cpu_memory_gb.toFixed(1)} GB</span>
                </div>
              </div>
            </Tile>
        </div>
      ) : estimationUnavailable ? (
        <div style={{ marginTop: '1rem' }}>
            <Tile className={styles.reviewCard}>
              <h6 className={styles.cardHeading} style={{ marginBottom: '0.75rem' }}>Estimated Resources</h6>
              <p className={styles.emptyHint}>Resource estimation is temporarily unavailable. You can still launch your tuning job.</p>
            </Tile>
        </div>
      ) : null}

      {launchPhase && (
        <div style={{ marginTop: '1.5rem' }}>
            <Tile className={styles.reviewCard}>
              <h6 className={styles.cardHeading} style={{ marginBottom: '0.75rem' }}>Launching...</h6>
              <div className={styles.launchSteps}>
                {isPendingDataset && (
                  <>
                    <div
                      className={`${styles.launchStep} ${launchPhase === 'creating_dataset' ? styles.launchStepActive : ''} ${
                        laterPhases('uploading_files', 'creating_config', 'launching_job') ? styles.launchStepDone : ''
                      }`}
                    >
                      {laterPhases('uploading_files', 'creating_config', 'launching_job') && <Checkmark size={16} />}
                      <span>Create dataset</span>
                    </div>
                    <div
                      className={`${styles.launchStep} ${launchPhase === 'uploading_files' ? styles.launchStepActive : ''} ${
                        laterPhases('creating_config', 'launching_job') ? styles.launchStepDone : ''
                      }`}
                    >
                      {laterPhases('creating_config', 'launching_job') && <Checkmark size={16} />}
                      <span>Upload files</span>
                      {launchPhase === 'uploading_files' && uploadProgress > 0 && (
                        <>
                          <div style={{ flex: 1, maxWidth: 200 }}>
                            <ProgressBar value={uploadProgress} max={100} size="small" label="Upload progress" hideLabel />
                          </div>
                          <span className={styles.progressLabel}>{uploadProgress}%</span>
                        </>
                      )}
                    </div>
                  </>
                )}
                {isPendingConfig && (
                  <div className={`${styles.launchStep} ${launchPhase === 'creating_config' ? styles.launchStepActive : ''} ${launchPhase === 'launching_job' ? styles.launchStepDone : ''}`}>
                    {launchPhase === 'launching_job' && <Checkmark size={16} />}
                    <span>Create configuration</span>
                  </div>
                )}
                <div className={`${styles.launchStep} ${launchPhase === 'launching_job' ? styles.launchStepActive : ''}`}>
                  <span>Launch job</span>
                </div>
              </div>
            </Tile>
        </div>
      )}
    </div>
  )
}
