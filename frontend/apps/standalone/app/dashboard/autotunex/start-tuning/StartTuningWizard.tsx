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
  getConfiguration,
  getConfigurations,
  getDataset,
  getDatasets,
  getHFModels,
  startJob,
  updateConfiguration as apiUpdateConfiguration,
  createConfiguration as apiCreateConfiguration,
  uploadDataset,
} from '@granite-build/ui-core/api/autotunex'
import { getRequiredColumnsFromTypes, isModelSelectionValid, normalizeTokenizerListFields, overlayColumnMapping } from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { normalizeVerlRows } from '@granite-build/ui-core/lib/autotunex/verlNormalize'
import { DATASET_READY_TIMEOUT_MS } from '@granite-build/ui-core/lib/autotunex/datasetReady'
import { ALGORITHM_DETAILS, ALGORITHM_OPTIONS } from '@granite-build/ui-core/config/autotunexAlgorithms'
import { clearDraft, loadDraft, resolveDraft, saveDraft } from './wizardDraft'
import { Step0GetStarted } from './steps/Step0GetStarted'
import { Step1DatasetUpload } from './steps/Step1DatasetUpload'
import { Step2Configure } from './steps/Step2Configure'
import { StepRewardFunction } from './steps/StepRewardFunction'
import { Step3ReviewLaunch } from './steps/Step3ReviewLaunch'
import styles from './StartTuningWizard.module.scss'

const DRAFT_DEBOUNCE_MS = 500

// Draft autosave and the "Resume your previous setup?" notification are deferred
// for this release: nothing is written to localStorage and no offer is made. The
// draft code (wizardDraft.ts, resumeDraft, resolveDraft and their tests) is kept
// intact -- set this back to true to switch the feature on.
const DRAFT_ENABLED = false
const DATASET_READY_POLL_MS = 1500

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
    // A failed poll is just a poll to retry. Letting the rejection propagate meant
    // one transient GET blip aborted a launch whose upload had already succeeded,
    // and the caller never reached `setDatasetId`, so a retry re-uploaded the same
    // file into the populated record. Only the deadline ends the wait unhappily.
    let ds: Dataset | null = null
    try {
      ds = await getDataset(id)
    } catch (err) {
      if (Date.now() > deadline) throw err
    }
    if (ds?.status === 'ready') return
    if (ds?.status === 'error') throw new Error(ds.status_detail || 'Dataset processing failed.')
    if (Date.now() > deadline) throw new Error('Dataset upload timed out while processing. Please try again.')
    await new Promise((resolve) => setTimeout(resolve, DATASET_READY_POLL_MS))
  }
}

export function StartTuningWizard() {
  const router = useRouter()

  // A draft saved by a previous visit, offered rather than applied: silently
  // restoring would be a surprise, and some of it may no longer be restorable.
  // Read synchronously at first render because the debounced autosave below would
  // otherwise overwrite the stored draft with this session's empty state within
  // DRAFT_DEBOUNCE_MS, before the user could answer.
  const [draftOffer, setDraftOffer] = useState<WizardDraft | null>(() =>
    DRAFT_ENABLED ? loadDraft() : null
  )
  const [draftNotes, setDraftNotes] = useState<string[]>([])
  const [isResumingDraft, setIsResumingDraft] = useState(false)
  // A restore jumps straight to the saved step, so it skips `handleNext` and with it
  // `prepareReviewStep` -- the review step came up with no resource estimate. This
  // cannot just be called at the end of `resumeDraft`, because prepareReviewStep
  // reads the very state those setState calls are still queuing; the effect below
  // runs it once React has applied them.
  const [prepareReviewAfterRestore, setPrepareReviewAfterRestore] = useState(false)

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
  // The dataset the file has already been uploaded into. `createdDatasetIdRef`
  // only covers the metadata POST, so a retry after the upload succeeded but the
  // readiness wait failed re-POSTed the same file into a populated record.
  const uploadedDatasetIdRef = useRef<string | null>(null)
  const createdConfigIdRef = useRef<string | null>(null)

  const [resourceEstimation, setResourceEstimation] = useState<Resources | null>(null)
  const [estimationUnavailable, setEstimationUnavailable] = useState(false)
  const estimationTokenRef = useRef(0)

  // Pre-fetch parallel API calls on wizard open — same cache keys the step
  // components consume via useQuery, so this just primes the cache.
  useQuery({ queryKey: ['autotunex', 'datasets'], queryFn: () => getDatasets({ page: 1, pageSize: 100 }) })
  useQuery({ queryKey: ['autotunex', 'configurations'], queryFn: () => getConfigurations({ page: 1, pageSize: 100 }) })
  const { data: datasetTypes } = useQuery({ queryKey: ['autotunex', 'datasetTypes'], queryFn: getAutotuneDatasetTypes })
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
        const requiredCols = getRequiredColumnsFromTypes(selectedAlgorithm, datasetTypes ?? {})
        // Column mapping applies only to a fresh upload. An existing dataset was
        // already uploaded with its mapping applied, `columnMapping` is unused
        // downstream for it (neither the uploadDataset call nor the preview
        // overlay runs), and Step 1 renders no mapping controls for it — so
        // requiring it here left Next permanently disabled with nothing the user
        // could do about it. Same bypass the adjacent hasValidation check uses.
        const allMapped = existingDatasetId !== null || requiredCols.every((c) => columnMapping[c])
        const hasValidation = existingDatasetId !== null || isSplitEnabled || validationFile !== null
        return hasDataset && hasName && allMapped && hasValidation && isDatasetCompatible
      }
      case 2: {
        // A pending config's name now goes straight into the payload
        // apiCreateConfiguration POSTs, so an empty name has to be caught here.
        // Otherwise the launch fails in the creating_config phase — after the
        // dataset has already been created and uploaded.
        const pendingIsNamed =
          selectedConfigId !== '__pending__' || (pendingNewConfig?.name ?? '').trim() !== ''
        return (
          selectedConfigId !== null && !isEditingConfig && !isCreatingConfig && pendingIsNamed
        )
      }
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
    datasetTypes,
    isSplitEnabled,
    validationFile,
    isDatasetCompatible,
    selectedConfigId,
    pendingNewConfig,
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
  // Set once the job is away. handleLaunch marks the last step complete before
  // clearing the draft, and that state change re-runs this effect -- so the debounce
  // fired *after* clearDraft() and wrote the draft back. The component stays mounted
  // across router.push, which is longer than the debounce, so unmount did not cancel
  // it: the next visit offered to resume a run that had already launched.
  const launchedRef = useRef(false)
  useEffect(() => {
    if (!DRAFT_ENABLED || !selectedGoal || launchedRef.current) return
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

  async function resumeDraft() {
    if (!draftOffer) return
    setIsResumingDraft(true)
    try {
      // A draft lives up to 24 hours, so confirm its references still resolve before
      // restoring them; a failed lookup counts as gone. `resolveDraft` then rewinds to
      // the step that owns anything missing, rather than letting the user walk to
      // Review and launch against a dead id.
      const [dataset, config] = await Promise.all([
        draftOffer.existingDatasetId
          ? getDataset(draftOffer.existingDatasetId).catch(() => null)
          : Promise.resolve(null),
        draftOffer.selectedConfigId
          ? getConfiguration(draftOffer.selectedConfigId).catch(() => null)
          : Promise.resolve(null),
      ])
      const { draft, notes } = resolveDraft(draftOffer, {
        dataset: Boolean(dataset),
        config: Boolean(config),
      })

      setSelectedGoal(draft.selectedGoal)
      // Suppress the goal-change reset, which fires on Step 0 and would wipe
      // everything else this function is about to restore.
      prevGoalRef.current = draft.selectedGoal
      setSelectedAlgorithm(draft.selectedAlgorithm)
      setSelectedModel(draft.selectedModel)
      setModelSource(draft.modelSource)
      setAutotuneEnabled(draft.autotuneEnabled ?? true)
      setDatasetForm((prev) => ({
        ...prev,
        name: draft.datasetForm.name,
        description: draft.datasetForm.description,
      }))
      // `splitRatio` is a fixed const in this wizard, not state, so the draft's copy
      // has nothing to restore into.
      setExistingDatasetId(draft.existingDatasetId)
      setSelectedExistingDataset(draft.existingDatasetId ? dataset : null)
      setSelectedConfigId(draft.selectedConfigId)
      setSelectedConfig(draft.selectedConfigId ? config : null)
      setExperimentName(draft.experimentName)
      setCompletedSteps(draft.completedSteps)
      setCurrentStep(draft.currentStep)

      setDraftNotes(notes)
      setDraftOffer(null)
      setPrepareReviewAfterRestore(true)
    } finally {
      setIsResumingDraft(false)
    }
  }

  useEffect(() => {
    if (!prepareReviewAfterRestore) return
    setPrepareReviewAfterRestore(false)
    if (currentStep === lastStepIndex) prepareReviewStep()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prepareReviewAfterRestore, currentStep, lastStepIndex])

  function discardDraft() {
    clearDraft()
    setDraftOffer(null)
  }

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
    // Review can be re-entered after editing the config, so an earlier estimate can
    // still be in flight and would otherwise land under the newly chosen config.
    const estimationToken = ++estimationTokenRef.current
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
            if (estimationTokenRef.current !== estimationToken) return
            if (result && 'unavailable' in result) {
              setResourceEstimation(null)
              setEstimationUnavailable(true)
            } else {
              setResourceEstimation(result)
              setEstimationUnavailable(false)
            }
          })
          .catch(() => {
            if (estimationTokenRef.current !== estimationToken) return
            // Without the flag, Step 3 has neither an estimate to show nor a reason
            // to explain, so the whole Estimated Resources section silently vanished.
            setResourceEstimation(null)
            setEstimationUnavailable(true)
          })
      } else {
        setResourceEstimation(null)
      }
    }
  }

  function handlePendingConfig(data: PendingConfigData) {
    setPendingNewConfig(data)
    createdConfigIdRef.current = null
  }

  // The pending-config card renames `selectedConfig` (its display state), but
  // `pendingNewConfig` is the object POSTed by apiCreateConfiguration at launch.
  // Without this the configuration was created under its previous name while the
  // review card and the generated experiment name showed the new one.
  function handlePendingConfigRename(name: string) {
    setPendingNewConfig((prev) => (prev ? { ...prev, name } : prev))
    // Matches handlePendingConfig: the name is part of what gets created, so a
    // config created by an earlier failed launch must not be reused.
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
    // The dataset being launched against changed, so drop every id derived from
    // the previous one. Leaving `datasetId` set made `datasetId ||
    // existingDatasetId` in handleLaunch resolve to the OLD dataset and skip the
    // upload branch entirely, so the job trained on the previous dataset while
    // the review card showed the newly-picked file.
    setDatasetId(null)
    // Only set during a launch attempt; a stale value would upload the new file
    // into the dataset record created by a previous, failed attempt.
    createdDatasetIdRef.current = null
    uploadedDatasetIdRef.current = null
    setSelectedConfigId(null)
    setSelectedConfig(null)
    setPendingNewConfig(null)
    setPendingConfigUpdate(null)
    setExperimentName('')
    setResourceEstimation(null)
    setEstimationUnavailable(false)
    // Step 1 too, not just what follows it: `clearTrainFile` leaves no dataset at
    // all, and leaving step 1 marked complete kept Review reachable (goToStep only
    // checks the preceding step) with nothing to train on.
    setCompletedSteps((prev) => prev.map((v, i) => (i >= 1 && i <= 3 ? false : v)))
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
        if (uploadedDatasetIdRef.current !== finalDatasetId) {
          await uploadDataset(
            finalDatasetId!,
            {
              // Keyed off the toggle, the way SettingsDatasetCreate does it. Reading
              // intent off `validationFile` instead let a stale file silently win
              // over an enabled split. Step 1's Next gate requires isSplitEnabled or
              // a file, so "neither" cannot reach here.
              trainFile: uploadedFile,
              validationFile: isSplitEnabled ? null : validationFile,
              validationPercentage: isSplitEnabled ? 100 - splitRatio : null,
              columnMapping,
            },
            setUploadProgress
          )
          uploadedDatasetIdRef.current = finalDatasetId!
        }

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

      // Deleting the train file after reaching Review used to arrive here with
      // nothing selected, and the non-null assertion sent `dataset_id: null` past
      // the type checker to fail server-side mid-launch.
      const launchDatasetId = finalDatasetId ?? datasetId ?? existingDatasetId
      if (!launchDatasetId) {
        throw new Error('No dataset selected. Go back to Upload Dataset and choose or upload a file.')
      }

      const tuningForm: TuningForm = {
        config_id: finalConfigId!,
        dataset_id: launchDatasetId,
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
      launchedRef.current = true
      setCompletedSteps((prev) => prev.map((v, i) => (i === lastStepIndex ? true : v)))
      clearTimeout(saveDraftTimeout.current)
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

      {draftOffer && (
        <div style={{ marginBottom: '1rem' }}>
          <InlineNotification
            kind="info"
            lowContrast
            hideCloseButton
            title="Resume your previous setup?"
            subtitle={`You left a draft on ${new Date(draftOffer.savedAt).toLocaleString()}.`}
            style={{ marginBottom: '0.5rem' }}
          />
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <Button size="sm" kind="tertiary" onClick={resumeDraft} disabled={isResumingDraft}>
              {isResumingDraft ? 'Restoring…' : 'Resume'}
            </Button>
            <Button size="sm" kind="ghost" onClick={discardDraft} disabled={isResumingDraft}>
              Start fresh
            </Button>
          </div>
        </div>
      )}

      {draftNotes.length > 0 && (
        <InlineNotification
          kind="warning"
          lowContrast
          title="Some of your draft could not be restored"
          subtitle={draftNotes.join(' ')}
          onCloseButtonClick={() => setDraftNotes([])}
          style={{ marginBottom: '1rem' }}
        />
      )}

      {/* The steps are built as an array rather than written inline with a
          `{hasRewardStep && ...}` hole. React.Children.map invokes its callback for
          a `false` child and still advances the index, and Carbon's
          ProgressIndicator numbers its steps by that index -- so with the reward
          step absent, "Review & Launch" sat at index 4 while `currentIndex` was 3.
          The highlighted step was the invisible one, the last step never read as
          current, and clicking it called onChange(4), which `goToStep` rejects as
          past `lastStepIndex`. Keep this list free of falsy entries. */}
      <ProgressIndicator key={hasRewardStep ? 'with-reward' : 'no-reward'} currentIndex={currentStep} spaceEqually onChange={goToStep}>
        {[
          <ProgressStep key="get-started" complete={completedSteps[0]} label="Get Started" description="Choose your approach" />,
          <ProgressStep key="dataset" disabled={!completedSteps[0]} complete={completedSteps[1]} label="Upload Dataset" description="Upload and preview your data" />,
          <ProgressStep key="configure" disabled={!completedSteps[1]} complete={completedSteps[2]} label="Configure" description="Select or create a configuration" />,
          ...(hasRewardStep
            ? [
                <ProgressStep key="reward" disabled={!completedSteps[2]} complete={completedSteps[3]} label="Reward Function" description="Define your reward function" />,
              ]
            : []),
          <ProgressStep key="review" disabled={!completedSteps[hasRewardStep ? 3 : 2]} complete={completedSteps[hasRewardStep ? 4 : 3]} label="Review & Launch" description="Review and start tuning" />,
        ]}
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
            onPendingConfigRename={handlePendingConfigRename}
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
