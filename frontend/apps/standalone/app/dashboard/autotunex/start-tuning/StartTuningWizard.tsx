'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useQuery } from '@tanstack/react-query'
import { ProgressIndicator, ProgressStep, Button, InlineLoading, InlineNotification, Breadcrumb, BreadcrumbItem } from '@carbon/react'
import { ArrowLeft, ArrowRight, Rocket, Close } from '@carbon/icons-react'
import type {
  ColumnMapping,
  ColumnMetadata,
  Configuration,
  Dataset,
  DatasetForm,
  DatasetFormatType,
  LaunchPhase,
  ModelSource,
  ParsedDataRow,
  PendingConfigData,
  PendingConfigUpdate,
  Resources,
  TuningForm,
  TuningGoal,
  WizardDraft,
} from '@granite-build/ui-core/types'
import {
  AUTOTUNEX_FEATURES,
  createDataset,
  estimateUsage,
  getAutotuneDatasetTypes,
  getConfigurations,
  getDataset,
  getDatasets,
  getHFModels,
  startJob,
  updateConfiguration as apiUpdateConfiguration,
  createConfiguration as apiCreateConfiguration,
  uploadDataset,
} from '@granite-build/ui-core/api/autotunex'
import { getRequiredColumns, isModelSelectionValid, normalizeTokenizerListFields, overlayColumnMapping } from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { normalizeVerlRows } from '@granite-build/ui-core/lib/autotunex/verlNormalize'
import { ALGORITHM_DETAILS, ALGORITHM_OPTIONS } from '@granite-build/ui-core/config/autotunexAlgorithms'
import { clearDraft, saveDraft } from './wizardDraft'
import { Step0GetStarted } from './steps/Step0GetStarted'
import { Step1DatasetUpload } from './steps/Step1DatasetUpload'
import { Step2Configure } from './steps/Step2Configure'
import { StepRewardFunction } from './steps/StepRewardFunction'
import { Step3ReviewLaunch } from './steps/Step3ReviewLaunch'
import styles from './StartTuningWizard.module.scss'

const DRAFT_DEBOUNCE_MS = 500
const DATASET_READY_POLL_MS = 1500
const DATASET_READY_TIMEOUT_MS = 5 * 60 * 1000

/**
 * The v0.3.5 multipart upload returns 202 with status "uploading" — the
 * server processes the file off-request. Poll the dataset row until the
 * server marks it ready (or error) before referencing it in the job-create
 * call.
 */
async function waitForDatasetReady(id: string): Promise<void> {
  const deadline = Date.now() + DATASET_READY_TIMEOUT_MS
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const ds = await getDataset(id)
    if (ds.status === 'ready') return
    if (ds.status === 'error') throw new Error(ds.status_detail || 'Dataset processing failed.')
    if (Date.now() > deadline) throw new Error('Dataset upload timed out while processing. Please try again.')
    await new Promise((resolve) => setTimeout(resolve, DATASET_READY_POLL_MS))
  }
}

export function StartTuningWizard() {
  const router = useRouter()

  // Step tracking
  const [currentStep, setCurrentStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState<boolean[]>([false, false, false, false, false])

  // Step 0: Get Started
  // Pre-select the first goal so a tile is always selected (the tiles are radio,
  // not deselectable). `prevGoalRef` below is seeded with the same value, so this
  // does not trigger the goal-change reset on mount.
  const [selectedGoal, setSelectedGoal] = useState<TuningGoal | null>('sft')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('lora')
  const [selectedModel, setSelectedModel] = useState('ibm-granite/granite-4.0-h-micro')
  const [modelSource, setModelSource] = useState<ModelSource>('huggingface')
  const [autotuneEnabled, setAutotuneEnabled] = useState(true)

  // Step 1: Dataset
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [parsedData, setParsedData] = useState<ParsedDataRow[]>([])
  const [columnMetadata, setColumnMetadata] = useState<ColumnMetadata[]>([])
  const [detectedFormat, setDetectedFormat] = useState<DatasetFormatType>('unknown')
  const [datasetForm, setDatasetForm] = useState<DatasetForm>({ name: '', description: '', train_file: null, validation_file: null })
  const [totalRecords, setTotalRecords] = useState(0)
  const [datasetId, setDatasetId] = useState<string | null>(null)
  const [existingDatasetId, setExistingDatasetId] = useState<string | null>(null)
  const [selectedExistingDataset, setSelectedExistingDataset] = useState<Dataset | null>(null)
  const splitRatio = 80 // fixed 80/20 auto-split — no ratio control exists in the source UI (matched as-is)
  const [validationFile, setValidationFile] = useState<File | null>(null)
  const [isSplitEnabled, setIsSplitEnabled] = useState(true)
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({})
  const [isDatasetCompatible, setIsDatasetCompatible] = useState(true)

  // Step 2: Config
  const [selectedConfigId, setSelectedConfigId] = useState<string | null>(null)
  const [selectedConfig, setSelectedConfig] = useState<Configuration | null>(null)
  const [pendingNewConfig, setPendingNewConfig] = useState<PendingConfigData | null>(null)
  const [pendingConfigUpdate, setPendingConfigUpdate] = useState<PendingConfigUpdate | null>(null)
  const [isEditingConfig, setIsEditingConfig] = useState(false)
  const [isCreatingConfig, setIsCreatingConfig] = useState(false)

  // Step 2.5: Reward function (Online RL only)
  const [rewardFunctionCode, setRewardFunctionCode] = useState('')
  const [rewardFunctionName, setRewardFunctionName] = useState('compute_score')
  const [allTestsPassed, setAllTestsPassed] = useState(false)

  // Step 3: Launch
  const [experimentName, setExperimentName] = useState('')
  const [isLaunching, setIsLaunching] = useState(false)
  const [transitionError, setTransitionError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [launchPhase, setLaunchPhase] = useState<LaunchPhase>(null)

  // Idempotent retry: resources already created on a failed launch attempt
  const createdDatasetIdRef = useRef<string | null>(null)
  const createdConfigIdRef = useRef<string | null>(null)

  const [resourceEstimation, setResourceEstimation] = useState<Resources | null>(null)
  const [estimationUnavailable, setEstimationUnavailable] = useState(false)

  // Pre-fetch parallel API calls on wizard open — same cache keys the step
  // components consume via useQuery, so this just primes the cache.
  useQuery({ queryKey: ['autotunex', 'datasets'], queryFn: () => getDatasets({ page: 1, pageSize: 100 }) })
  useQuery({ queryKey: ['autotunex', 'configurations'], queryFn: () => getConfigurations({ page: 1, pageSize: 100 }) })
  useQuery({ queryKey: ['autotunex', 'datasetTypes'], queryFn: getAutotuneDatasetTypes })
  const { data: prefetchedModels } = useQuery({
    queryKey: ['autotunex', 'hfModels', 'ibm-granite/granite-4.0-h-micro', 20],
    queryFn: () => getHFModels('ibm-granite/granite-4.0-h-micro', 20),
  })

  const hasRewardStep = selectedGoal === 'online_rl'
  const totalSteps = hasRewardStep ? 5 : 4
  const lastStepIndex = totalSteps - 1

  // Reset Step 1/2/3 state when the tuning goal changes (user changed their mind on Step 0)
  const prevGoalRef = useRef<TuningGoal | null>(selectedGoal)
  useEffect(() => {
    if (selectedGoal === null || prevGoalRef.current === null || selectedGoal === prevGoalRef.current || currentStep !== 0) {
      prevGoalRef.current = selectedGoal
      return
    }

    setUploadedFile(null)
    setParsedData([])
    setColumnMetadata([])
    setDetectedFormat('unknown')
    setDatasetForm({ name: '', description: '', train_file: null, validation_file: null })
    setTotalRecords(0)
    setDatasetId(null)
    setExistingDatasetId(null)
    setValidationFile(null)
    setIsSplitEnabled(true)
    setColumnMapping({})

    setSelectedConfigId(null)
    setSelectedConfig(null)
    setPendingNewConfig(null)
    setPendingConfigUpdate(null)

    setExperimentName('')
    setResourceEstimation(null)
    setEstimationUnavailable(false)

    setRewardFunctionCode('')
    setRewardFunctionName('compute_score')
    setAllTestsPassed(false)

    setCompletedSteps([false, false, false, false, false])

    prevGoalRef.current = selectedGoal
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedGoal, currentStep])

  // Sync the goal when the algorithm changes — only while on Step 0 (user picking a goal directly).
  useEffect(() => {
    if (!selectedAlgorithm || currentStep !== 0) return
    const algo = ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm) || ALGORITHM_OPTIONS.find((a) => a.id === selectedAlgorithm)
    if (algo && algo.category !== selectedGoal) setSelectedGoal(algo.category)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAlgorithm, currentStep])

  const canProceed = useMemo(() => {
    switch (currentStep) {
      case 0:
        return selectedGoal !== null && selectedAlgorithm !== '' && isModelSelectionValid(modelSource, selectedModel)
      case 1: {
        const hasDataset = existingDatasetId !== null || parsedData.length > 0
        const hasName = datasetForm.name.trim() !== ''
        const requiredCols = getRequiredColumns(selectedAlgorithm)
        const allMapped = requiredCols.every((c) => columnMapping[c])
        const hasValidation = existingDatasetId !== null || isSplitEnabled || validationFile !== null
        return hasDataset && hasName && allMapped && hasValidation && isDatasetCompatible
      }
      case 2:
        return selectedConfigId !== null && !isEditingConfig && !isCreatingConfig
      case 3:
        if (hasRewardStep) {
          // Reward-function validation is gated off in this environment (see
          // AUTOTUNEX_FEATURES.rewardValidation) — don't block the wizard on
          // a test-pass signal that can never arrive.
          return (
            rewardFunctionCode.trim().length > 0 &&
            rewardFunctionName.trim().length > 0 &&
            (allTestsPassed || !AUTOTUNEX_FEATURES.rewardValidation)
          )
        }
        return experimentName.trim() !== '' && !isLaunching
      case 4:
        return experimentName.trim() !== '' && !isLaunching
      default:
        return false
    }
  }, [
    currentStep,
    selectedGoal,
    selectedAlgorithm,
    selectedModel,
    modelSource,
    existingDatasetId,
    parsedData.length,
    datasetForm.name,
    columnMapping,
    isSplitEnabled,
    validationFile,
    isDatasetCompatible,
    selectedConfigId,
    isEditingConfig,
    isCreatingConfig,
    hasRewardStep,
    rewardFunctionCode,
    rewardFunctionName,
    allTestsPassed,
    experimentName,
    isLaunching,
  ])

  const breadcrumbItems = useMemo(() => {
    const items: { label: string; step: number }[] = []
    if (!completedSteps[0] || currentStep === 0) return items

    if (selectedGoal) {
      const goalLabels: Record<TuningGoal, string> = { sft: 'SFT', offline_rl: 'Offline RL', online_rl: 'Online RL' }
      items.push({ label: goalLabels[selectedGoal] || selectedGoal, step: 0 })
    }
    if (selectedModel) items.push({ label: selectedModel.split('/').pop() || selectedModel, step: 0 })

    if (completedSteps[1] && currentStep > 1) items.push({ label: datasetForm.name || 'Dataset', step: 1 })
    if (completedSteps[2] && currentStep > 2) items.push({ label: selectedConfig?.name || pendingNewConfig?.name || 'Config', step: 2 })

    return items
  }, [completedSteps, currentStep, selectedGoal, selectedModel, datasetForm.name, selectedConfig, pendingNewConfig])

  // Debounced draft autosave to localStorage
  const saveDraftTimeout = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    if (!selectedGoal) return
    clearTimeout(saveDraftTimeout.current)
    saveDraftTimeout.current = setTimeout(() => {
      const draft: WizardDraft = {
        savedAt: new Date().toISOString(),
        currentStep,
        completedSteps,
        selectedGoal,
        selectedAlgorithm,
        selectedModel,
        modelSource,
        datasetForm: { name: datasetForm.name, description: datasetForm.description },
        existingDatasetId,
        splitRatio,
        selectedConfigId: selectedConfigId === '__pending__' ? null : selectedConfigId,
        experimentName,
        autotuneEnabled,
      }
      saveDraft(draft)
    }, DRAFT_DEBOUNCE_MS)
    return () => clearTimeout(saveDraftTimeout.current)
  }, [
    currentStep,
    completedSteps,
    selectedGoal,
    selectedAlgorithm,
    selectedModel,
    modelSource,
    datasetForm.name,
    datasetForm.description,
    existingDatasetId,
    selectedConfigId,
    experimentName,
    autotuneEnabled,
  ])

  function goToStep(step: number) {
    if (step < 0 || step > lastStepIndex) return
    if (step <= currentStep || completedSteps[step - 1]) setCurrentStep(step)
  }

  async function handleNext() {
    setTransitionError('')

    if (currentStep === 0) {
      setCompletedSteps((prev) => prev.map((v, i) => (i === 0 ? true : v)))
      setCurrentStep(1)
    } else if (currentStep === 1) {
      if (existingDatasetId) setDatasetId(existingDatasetId)
      setCompletedSteps((prev) => prev.map((v, i) => (i === 1 ? true : v)))
      setCurrentStep(2)
    } else if (currentStep === 2) {
      setCompletedSteps((prev) => prev.map((v, i) => (i === 2 ? true : v)))
      if (hasRewardStep) {
        setCurrentStep(3)
      } else {
        setCurrentStep(3)
        prepareReviewStep()
      }
    } else if (currentStep === 3 && hasRewardStep) {
      setCompletedSteps((prev) => prev.map((v, i) => (i === 3 ? true : v)))
      setCurrentStep(4)
      prepareReviewStep()
    }
  }

  function prepareReviewStep() {
    setExperimentName((prev) => {
      if (prev) return prev
      const modelShort = selectedModel.split('/').pop() || selectedModel
      const configName = selectedConfig?.name || pendingNewConfig?.name || 'config'
      return `${modelShort}_${configName}`.substring(0, 50)
    })

    if (!AUTOTUNEX_FEATURES.estimation) {
      setResourceEstimation(null)
      setEstimationUnavailable(true)
      return
    }

    if (selectedModel && selectedConfigId) {
      // estimate-usages requires exactly one of config_id / config_data: send
      // config_data for a pending (not-yet-saved) config, config_id otherwise.
      const estimation =
        selectedConfigId === '__pending__'
          ? pendingNewConfig
            ? { model_name: selectedModel, config_data: pendingNewConfig.config_data, gpu_memory: 80 }
            : null
          : { model_name: selectedModel, config_id: selectedConfigId, gpu_memory: 80 }
      if (estimation) {
        estimateUsage(estimation)
          .then((result) => {
            if (result && 'unavailable' in result) {
              setResourceEstimation(null)
              setEstimationUnavailable(true)
            } else {
              setResourceEstimation(result)
              setEstimationUnavailable(false)
            }
          })
          .catch(() => setResourceEstimation(null))
      } else {
        setResourceEstimation(null)
      }
    }
  }

  function handlePendingConfig(data: PendingConfigData) {
    setPendingNewConfig(data)
    createdConfigIdRef.current = null
  }

  function handlePendingConfigUpdate(data: PendingConfigUpdate) {
    setPendingConfigUpdate(data)
  }

  function handleClearPendingConfig() {
    setPendingNewConfig(null)
    setPendingConfigUpdate(null)
    createdConfigIdRef.current = null
  }

  function handleDatasetChanged() {
    setSelectedConfigId(null)
    setSelectedConfig(null)
    setPendingNewConfig(null)
    setPendingConfigUpdate(null)
    setExperimentName('')
    setResourceEstimation(null)
    setEstimationUnavailable(false)
    setCompletedSteps((prev) => prev.map((v, i) => (i === 2 || i === 3 ? false : v)))
  }

  async function handleLaunch() {
    setIsLaunching(true)
    setTransitionError('')
    setUploadProgress(0)

    try {
      let finalDatasetId = datasetId || existingDatasetId

      if (!finalDatasetId && uploadedFile) {
        setLaunchPhase('creating_dataset')

        if (!createdDatasetIdRef.current) {
          const resp = await createDataset({ name: datasetForm.name.trim(), description: datasetForm.description })
          if (!resp?.id) throw new Error('Failed to create dataset metadata.')
          createdDatasetIdRef.current = resp.id
        }
        finalDatasetId = createdDatasetIdRef.current

        setLaunchPhase('uploading_files')
        await uploadDataset(
          finalDatasetId!,
          {
            trainFile: uploadedFile,
            validationFile: validationFile ?? null,
            validationPercentage: validationFile ? null : 100 - splitRatio,
            columnMapping,
          },
          setUploadProgress
        )

        // Multipart upload responds 202 ("uploading") — the server finishes
        // processing off-request, so wait for it before referencing this
        // dataset in the job-create call below.
        await waitForDatasetReady(finalDatasetId!)

        setDatasetId(finalDatasetId)
      }

      if (pendingConfigUpdate && selectedConfigId !== '__pending__') {
        setLaunchPhase('updating_config')
        normalizeTokenizerListFields(pendingConfigUpdate.config_data)
        await apiUpdateConfiguration(pendingConfigUpdate.configId, pendingConfigUpdate)
      }

      let finalConfigId = selectedConfigId

      if (pendingNewConfig && selectedConfigId === '__pending__') {
        setLaunchPhase('creating_config')

        if (!createdConfigIdRef.current) {
          normalizeTokenizerListFields(pendingNewConfig.config_data)
          const createdConfig = await apiCreateConfiguration(pendingNewConfig)
          createdConfigIdRef.current = createdConfig.id
        }
        finalConfigId = createdConfigIdRef.current
      }

      setLaunchPhase('launching_job')

      const tuningForm: TuningForm = {
        config_id: finalConfigId!,
        dataset_id: (finalDatasetId ?? datasetId ?? existingDatasetId)!,
        model: selectedModel.trim(),
        model_source: modelSource,
        experiment_name: experimentName.trim().replace(/\s+/g, '_'),
        autotune: autotuneEnabled,
        // No seed control exists in this wizard's UI — use the API default.
        seed: 42,
        ...(hasRewardStep && rewardFunctionCode.trim()
          ? { reward_function_code: rewardFunctionCode, reward_function_name: rewardFunctionName || 'compute_score' }
          : {}),
      }

      const { id: jobId } = await startJob(tuningForm)
      setCompletedSteps((prev) => prev.map((v, i) => (i === lastStepIndex ? true : v)))
      clearDraft()
      router.push(`/dashboard/autotunex/_/?id=${jobId}`)
    } catch (err: any) {
      setTransitionError(err.message || 'Launch failed. Please try again.')
    } finally {
      setIsLaunching(false)
      setLaunchPhase(null)
      setUploadProgress(0)
    }
  }

  function handleBack() {
    if (currentStep > 0) setCurrentStep((prev) => prev - 1)
  }

  const goalHeading =
    selectedGoal === 'sft' ? 'Supervised Fine-Tuning' : selectedGoal === 'offline_rl' ? 'Preference Learning' : selectedGoal === 'online_rl' ? 'Reinforcement Learning' : 'Tuning'

  return (
    <div className={styles.wizardContainer}>
      <div className={styles.wizardHeader}>
        <div className={styles.wizardHeaderText}>
          <h3>Configure {goalHeading}</h3>
          <p className={styles.wizardSubtitle}>Follow the steps to configure and launch your fine-tuning job</p>
        </div>
        <Button kind="ghost" size="sm" renderIcon={Close} iconDescription="Close wizard" hasIconOnly onClick={() => router.push('/dashboard/autotunex')} />
      </div>

      {breadcrumbItems.length > 0 && (
        <div className={styles.wizardBreadcrumb}>
          <Breadcrumb noTrailingSlash>
            {breadcrumbItems.map((item, i) => {
              const isCurrent = i === breadcrumbItems.length - 1
              return (
                <BreadcrumbItem key={i} isCurrentPage={isCurrent}>
                  {isCurrent ? (
                    item.label
                  ) : (
                    <button type="button" className={styles.breadcrumbButton} onClick={() => goToStep(item.step)}>
                      {item.label}
                    </button>
                  )}
                </BreadcrumbItem>
              )
            })}
          </Breadcrumb>
        </div>
      )}

      <ProgressIndicator key={hasRewardStep ? 'with-reward' : 'no-reward'} currentIndex={currentStep} spaceEqually onChange={goToStep}>
        <ProgressStep complete={completedSteps[0]} label="Get Started" description="Choose your approach" />
        <ProgressStep disabled={!completedSteps[0]} complete={completedSteps[1]} label="Upload Dataset" description="Upload and preview your data" />
        <ProgressStep disabled={!completedSteps[1]} complete={completedSteps[2]} label="Configure" description="Select or create a configuration" />
        {hasRewardStep && (
          <ProgressStep disabled={!completedSteps[2]} complete={completedSteps[3]} label="Reward Function" description="Define your reward function" />
        )}
        <ProgressStep disabled={!completedSteps[hasRewardStep ? 3 : 2]} complete={completedSteps[hasRewardStep ? 4 : 3]} label="Review & Launch" description="Review and start tuning" />
      </ProgressIndicator>

      <div className={styles.stepContent}>
        {currentStep === 0 && (
          <Step0GetStarted
            selectedAlgorithm={selectedAlgorithm}
            setSelectedAlgorithm={setSelectedAlgorithm}
            selectedGoal={selectedGoal}
            setSelectedGoal={setSelectedGoal}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            modelSource={modelSource}
            setModelSource={setModelSource}
            prefetchedModels={prefetchedModels ?? null}
          />
        )}
        {currentStep === 1 && (
          <Step1DatasetUpload
            uploadedFile={uploadedFile}
            setUploadedFile={setUploadedFile}
            parsedData={parsedData}
            setParsedData={setParsedData}
            columnMetadata={columnMetadata}
            setColumnMetadata={setColumnMetadata}
            detectedFormat={detectedFormat}
            setDetectedFormat={setDetectedFormat}
            datasetForm={datasetForm}
            setDatasetForm={setDatasetForm}
            totalRecords={totalRecords}
            setTotalRecords={setTotalRecords}
            existingDatasetId={existingDatasetId}
            setExistingDatasetId={setExistingDatasetId}
            splitRatio={splitRatio}
            validationFile={validationFile}
            setValidationFile={setValidationFile}
            isSplitEnabled={isSplitEnabled}
            setIsSplitEnabled={setIsSplitEnabled}
            selectedAlgorithm={selectedAlgorithm}
            setSelectedAlgorithm={setSelectedAlgorithm}
            selectedGoal={selectedGoal}
            columnMapping={columnMapping}
            setColumnMapping={setColumnMapping}
            setIsDatasetCompatible={setIsDatasetCompatible}
            selectedExistingDataset={selectedExistingDataset}
            setSelectedExistingDataset={setSelectedExistingDataset}
            onDatasetChanged={handleDatasetChanged}
          />
        )}
        {currentStep === 2 && (
          <Step2Configure
            selectedAlgorithm={selectedAlgorithm}
            selectedGoal={selectedGoal}
            selectedConfigId={selectedConfigId}
            setSelectedConfigId={setSelectedConfigId}
            selectedConfig={selectedConfig}
            setSelectedConfig={setSelectedConfig}
            isEditingConfig={isEditingConfig}
            setIsEditingConfig={setIsEditingConfig}
            isCreatingConfig={isCreatingConfig}
            setIsCreatingConfig={setIsCreatingConfig}
            onPendingConfig={handlePendingConfig}
            onPendingConfigUpdate={handlePendingConfigUpdate}
            onClearPendingConfig={handleClearPendingConfig}
            autotuneEnabled={autotuneEnabled}
            setAutotuneEnabled={setAutotuneEnabled}
          />
        )}
        {currentStep === 3 && hasRewardStep && (
          <StepRewardFunction
            rewardFunctionCode={rewardFunctionCode}
            setRewardFunctionCode={setRewardFunctionCode}
            rewardFunctionName={rewardFunctionName}
            setRewardFunctionName={setRewardFunctionName}
            allTestsPassed={allTestsPassed}
            setAllTestsPassed={setAllTestsPassed}
            datasetId={datasetId || existingDatasetId}
            parsedData={parsedData.length > 0 && !existingDatasetId ? normalizeVerlRows(overlayColumnMapping(parsedData, columnMapping)) : []}
          />
        )}
        {currentStep === lastStepIndex && (
          <Step3ReviewLaunch
            uploadedFile={uploadedFile}
            datasetForm={datasetForm}
            selectedExistingDataset={selectedExistingDataset}
            selectedConfig={selectedConfig}
            selectedModel={selectedModel}
            modelSource={modelSource}
            resourceEstimation={resourceEstimation}
            estimationUnavailable={estimationUnavailable}
            totalRecords={totalRecords}
            splitRatio={splitRatio}
            isSplitEnabled={isSplitEnabled}
            validationFile={validationFile}
            autotuneEnabled={autotuneEnabled}
            columnMetadata={columnMetadata}
            experimentName={experimentName}
            setExperimentName={setExperimentName}
            isPendingDataset={!existingDatasetId && !datasetId && !!uploadedFile}
            isPendingConfig={selectedConfigId === '__pending__'}
            launchPhase={launchPhase}
            uploadProgress={uploadProgress}
            onEditStep={goToStep}
          />
        )}
      </div>

      {transitionError && (
        <InlineNotification kind="error" title="Error" subtitle={transitionError} onClose={() => setTransitionError('')} style={{ marginBottom: '1rem' }} />
      )}

      <div className={styles.wizardFooter}>
        <Button kind="tertiary" onClick={() => router.push('/dashboard/autotunex')}>Cancel</Button>
        {currentStep > 0 && (
          <Button kind="secondary" renderIcon={ArrowLeft} onClick={handleBack} disabled={isLaunching}>Back</Button>
        )}
        {currentStep < lastStepIndex ? (
          <Button kind="primary" renderIcon={ArrowRight} onClick={handleNext} disabled={!canProceed}>Next</Button>
        ) : (
          <Button kind="primary" renderIcon={Rocket} onClick={handleLaunch} disabled={!canProceed || isLaunching}>
            {isLaunching ? (
              <InlineLoading
                description={
                  launchPhase === 'creating_dataset'
                    ? 'Creating dataset...'
                    : launchPhase === 'uploading_files'
                      ? `Uploading files (${uploadProgress}%)...`
                      : launchPhase === 'updating_config'
                        ? 'Updating configuration...'
                        : launchPhase === 'creating_config'
                          ? 'Creating configuration...'
                          : launchPhase === 'launching_job'
                            ? 'Launching job...'
                            : 'Launching...'
                }
              />
            ) : (
              'Launch Tuning'
            )}
          </Button>
        )}
      </div>
    </div>
  )
}
