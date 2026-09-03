'use client'

import { useMemo, useRef, useState } from 'react'
import {
  Modal,
  TextInput,
  Dropdown,
  Toggle,
  NumberInput,
  Select,
  SelectItem,
  Button,
  FileUploaderDropContainer,
  FileUploaderItem,
  FormLabel,
  ProgressBar,
  InlineNotification,
  InlineLoading,
} from '@carbon/react'
import { MagicWand } from '@carbon/icons-react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import type { ColumnMapping, ColumnMetadata, DatasetStatus } from '../types'
import { createDataset, uploadDataset, getDataset, getAutotuneDatasetTypes, suggestColumnMappingAI } from '../api/autotunex'
import { processUploadedFileAsync } from '../lib/autotunex/processUploadedFile'
import { extractColumnMetadata, getColumnsFromTypes, getRequiredColumnsFromTypes } from '../lib/autotunex/wizardUtils'
import { ALGORITHM_TO_DATASET_TYPE } from '../config/autotunexAlgorithms'
import styles from './SettingsDatasetCreate.module.scss'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

// Matches the wizard's `waitForDatasetReady` timeout (StartTuningWizard.tsx)
// so both upload paths give up on a stuck "uploading" dataset after the same
// window instead of polling forever.
const POLL_TIMEOUT_MS = 5 * 60 * 1000

// Dataset-type dropdown items → algorithm id (matches AutoTuneX).
const TYPE_ITEMS = [
  { id: 'lora', label: 'SFT' },
  { id: 'dpo', label: 'DPO' },
  { id: 'kto', label: 'KTO' },
  { id: 'grpo', label: 'Online RL' },
]

export function SettingsDatasetCreate({ open, onClose, onCreated }: Props) {
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [algorithm, setAlgorithm] = useState('lora')
  const [trainFile, setTrainFile] = useState<File | null>(null)
  const [validationFile, setValidationFile] = useState<File | null>(null)
  const [split, setSplit] = useState(true)
  const [trainPercentage, setTrainPercentage] = useState(80)
  const [detectedColumns, setDetectedColumns] = useState<ColumnMetadata[]>([])
  const [sampleRows, setSampleRows] = useState<Record<string, any>[]>([])
  const [columnMapping, setColumnMapping] = useState<ColumnMapping>({})
  const [aiBusy, setAiBusy] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [polling, setPolling] = useState(false)
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null)
  const [error, setError] = useState('')
  const [createdId, setCreatedId] = useState<string | null>(null)
  // Guards against touching state after the modal is closed mid upload/poll
  // (the component stays mounted — only its `open` prop toggles).
  const cancelledRef = useRef(false)

  const { data: datasetTypes = {} } = useQuery({
    queryKey: ['autotunex', 'datasetTypes'],
    queryFn: getAutotuneDatasetTypes,
    enabled: open,
  })

  const requiredColumns = useMemo(
    () => getRequiredColumnsFromTypes(algorithm, datasetTypes),
    [algorithm, datasetTypes]
  )

  const userColumnNames = detectedColumns.map((c) => c.name)

  async function onTrainFileSelect(file: File) {
    setTrainFile(file)
    setError('')
    try {
      const rows = await processUploadedFileAsync(file, 100)
      setSampleRows(rows)
      setDetectedColumns(extractColumnMetadata(rows))
    } catch {
      setSampleRows([])
      setDetectedColumns([])
      setError('Could not parse the uploaded file. Supported: .jsonl, .json, .csv, .parquet.')
    }
  }

  function updateMapping(requiredCol: string, userCol: string) {
    setColumnMapping((prev) => ({ ...prev, [requiredCol]: userCol }))
  }

  async function handleAiSuggest() {
    if (sampleRows.length === 0 || detectedColumns.length === 0) return
    setAiBusy(true)
    try {
      const colNames = detectedColumns.map((c) => c.name)
      const colSamples: Record<string, string[]> = {}
      for (const c of detectedColumns) colSamples[c.name] = c.sampleValues.slice(0, 3)
      const targetType = ALGORITHM_TO_DATASET_TYPE[algorithm]
      const result = await suggestColumnMappingAI({
        sample_data: sampleRows.slice(0, 8),
        column_names: colNames,
        column_samples: colSamples,
        target_format: targetType,
      })

      // Map the AI's dataset-type-keyed suggestions (flat requiredCol -> sourceCol
      // strings) onto our required column names.
      const typeCols = getColumnsFromTypes(algorithm, datasetTypes).map((c) => c.name)
      const newMapping: ColumnMapping = {}
      for (const [aiKey, sourceColumn] of Object.entries(result.column_mapping ?? {})) {
        if (!sourceColumn || !colNames.includes(sourceColumn)) continue
        const normalized = aiKey.replace(/_col$/, '')
        const matched = typeCols.find((rc) => rc === aiKey || rc === normalized) ?? (requiredColumns.includes(normalized) ? normalized : '')
        if (matched) newMapping[matched] = sourceColumn
      }
      if (Object.keys(newMapping).length > 0) setColumnMapping((prev) => ({ ...prev, ...newMapping }))
    } catch {
      // AI suggest is best-effort; leave existing mappings untouched.
    } finally {
      setAiBusy(false)
    }
  }

  const canSubmit = name.trim().length > 0 && !!trainFile && (split || !!validationFile) && progress == null && !polling

  // Polls GET /datasets/{id} every ~3s until the server-side processing
  // triggered by the multipart upload settles into 'ready' or 'error'.
  async function pollUntilReady(datasetId: string): Promise<void> {
    setPolling(true)
    const deadline = Date.now() + POLL_TIMEOUT_MS
    while (!cancelledRef.current) {
      const ds = await getDataset(datasetId)
      if (cancelledRef.current) return
      setDatasetStatus(ds.status)
      if (ds.status === 'ready') return
      if (ds.status === 'error') {
        throw new Error(ds.status_detail || 'Dataset processing failed. Please check the file and try again.')
      }
      if (Date.now() > deadline) {
        throw new Error('Dataset processing timed out. Please try again.')
      }
      await new Promise((resolve) => setTimeout(resolve, 3000))
    }
  }

  async function handleSubmit() {
    if (!trainFile) return
    setError('')
    setProgress(0)
    setDatasetStatus(null)
    cancelledRef.current = false
    try {
      let datasetId = createdId
      if (!datasetId) {
        const info = await createDataset({ name: name.trim(), description: description.trim() })
        datasetId = info.id
        setCreatedId(datasetId)
      }
      await uploadDataset(
        datasetId,
        {
          trainFile,
          validationFile: split ? null : validationFile,
          validationPercentage: split ? 100 - trainPercentage : null,
          columnMapping: Object.keys(columnMapping).length > 0 ? columnMapping : null,
        },
        (p) => { if (!cancelledRef.current) setProgress(p) }
      )
      if (cancelledRef.current) return
      setProgress(100)
      await pollUntilReady(datasetId)
      if (cancelledRef.current) return
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'datasets'] })
      resetAndClose(true)
    } catch (err) {
      if (cancelledRef.current) return
      setProgress(null)
      setError(err instanceof Error && err.message ? err.message : 'Upload failed. You can retry — the dataset was created and will be reused.')
    } finally {
      if (!cancelledRef.current) setPolling(false)
    }
  }

  function resetAndClose(created: boolean) {
    cancelledRef.current = true
    setName(''); setDescription(''); setAlgorithm('lora')
    setTrainFile(null); setValidationFile(null); setSplit(true); setTrainPercentage(80)
    setDetectedColumns([]); setSampleRows([]); setColumnMapping({})
    setProgress(null); setPolling(false); setDatasetStatus(null); setError(''); setCreatedId(null)
    if (created) onCreated()
    onClose()
  }

  return (
    <Modal
      open={open}
      size="lg"
      modalHeading="Create New Dataset"
      primaryButtonText={polling ? 'Processing…' : progress != null ? 'Uploading…' : 'Save'}
      secondaryButtonText="Cancel"
      primaryButtonDisabled={!canSubmit}
      onRequestClose={() => resetAndClose(false)}
      onRequestSubmit={handleSubmit}
    >
      <div className={styles.field}>
        <TextInput id="ds-name" labelText="Dataset name" value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      <div className={styles.field}>
        <TextInput id="ds-desc" labelText="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />
      </div>
      <div className={styles.field}>
        <Dropdown
          id="ds-type"
          titleText="Dataset type"
          label="Select a type"
          items={TYPE_ITEMS}
          selectedItem={TYPE_ITEMS.find((i) => i.id === algorithm) ?? null}
          itemToString={(i) => (i ? i.label : '')}
          onChange={({ selectedItem }) => { if (selectedItem) { setAlgorithm(selectedItem.id); setColumnMapping({}) } }}
        />
      </div>

      <div className={styles.field}>
        <FormLabel>Training file</FormLabel>
        {trainFile ? (
          <FileUploaderItem name={trainFile.name} status="edit" onDelete={() => { setTrainFile(null); setDetectedColumns([]); setSampleRows([]); setColumnMapping({}) }} />
        ) : (
          <FileUploaderDropContainer
            accept={['.jsonl', '.json', '.csv', '.parquet']}
            labelText="Drag and drop a file here or click to upload"
            onAddFiles={(_e, { addedFiles }) => { if (addedFiles[0]) onTrainFileSelect(addedFiles[0]) }}
          />
        )}
      </div>

      <div className={styles.field}>
        <Toggle id="ds-split" labelText="Split into training + validation" toggled={split} onToggle={setSplit} size="sm" />
      </div>

      {split ? (
        <div className={styles.field}>
          <NumberInput
            id="ds-train-pct"
            label="Training set percentage"
            min={1}
            max={99}
            value={trainPercentage}
            onChange={(_e, { value }) => setTrainPercentage(Number(value) || 80)}
          />
        </div>
      ) : (
        <div className={styles.field}>
          <FormLabel>Validation file</FormLabel>
          {validationFile ? (
            <FileUploaderItem name={validationFile.name} status="edit" onDelete={() => setValidationFile(null)} />
          ) : (
            <FileUploaderDropContainer
              accept={['.jsonl', '.json', '.csv', '.parquet']}
              labelText="Drag and drop a validation file here or click to upload"
              onAddFiles={(_e, { addedFiles }) => { if (addedFiles[0]) setValidationFile(addedFiles[0]) }}
            />
          )}
        </div>
      )}

      {trainFile && requiredColumns.length > 0 && userColumnNames.length > 0 && (
        <div className={styles.field}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <FormLabel>Column mapping</FormLabel>
            <Button kind="ghost" size="sm" renderIcon={MagicWand} disabled={aiBusy} onClick={handleAiSuggest}>
              {aiBusy ? 'Suggesting…' : 'AI suggest'}
            </Button>
          </div>
          {requiredColumns.map((reqCol) => (
            <div key={reqCol} className={styles.mappingRow}>
              <Select
                id={`map-${reqCol}`}
                className={styles.mappingSelect}
                labelText={reqCol}
                value={columnMapping[reqCol] ?? ''}
                onChange={(e) => updateMapping(reqCol, e.target.value)}
              >
                <SelectItem value="" text="— select a column —" />
                {userColumnNames.map((col) => (
                  <SelectItem key={col} value={col} text={col} />
                ))}
              </Select>
            </div>
          ))}
        </div>
      )}

      {progress != null && !polling && (
        <div className={styles.field}>
          <ProgressBar label="Uploading dataset" value={progress} max={100} />
        </div>
      )}

      {polling && (
        <div className={styles.field}>
          <InlineLoading
            description={
              datasetStatus === 'uploading' || datasetStatus == null
                ? 'Processing dataset…'
                : `Processing dataset (${datasetStatus})…`
            }
          />
        </div>
      )}

      {error && (
        <InlineNotification kind="error" title="Error" subtitle={error} lowContrast hideCloseButton />
      )}
    </Modal>
  )
}
