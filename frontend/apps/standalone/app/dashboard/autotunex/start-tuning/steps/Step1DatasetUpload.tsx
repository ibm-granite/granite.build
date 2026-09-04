'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  FileUploaderButton,
  FileUploaderDropContainer,
  FileUploaderItem,
  TextInput,
  Tile,
  Select,
  SelectItem,
  Toggle,
  Button,
  InlineLoading,
  InlineNotification,
  Tag,
  ContentSwitcher,
  Switch,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Tooltip,
  Table,
  TableHead,
  TableRow,
  TableHeader,
  TableBody,
  TableCell,
} from '@carbon/react'
import { Reset, Information } from '@carbon/icons-react'
import type { ColumnMapping, ColumnMetadata, Dataset, DatasetForm, DatasetFormatType, ParsedDataRow, TuningGoal } from '@granite-build/ui-core/types'
import { getAutotuneDatasetTypes, getDataset, getDatasets, suggestColumnMappingAI } from '@granite-build/ui-core/api/autotunex'
import { countLinesInFileAsync, processUploadedFileAsync } from '@granite-build/ui-core/lib/autotunex/processUploadedFile'
import {
  applyColumnMapping,
  detectDatasetFormat,
  extractColumnMetadata,
  generateFormatExamples,
  getColumnsFromTypes,
  getDatasetExamples,
  getDatasetExamplesFromTypes,
  getDefaultAlgorithmForGoal,
  getRequiredColumns,
  getRequiredColumnsFromTypes,
  suggestAlgorithm,
  suggestColumnMapping as suggestColumnMappingHeuristic,
  toUpperCase,
  validateDatasetForGoal,
} from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { ALGORITHM_DETAILS, ALGORITHM_TO_DATASET_TYPE } from '@granite-build/ui-core/config/autotunexAlgorithms'
import styles from './Step1DatasetUpload.module.scss'
import layoutStyles from '@granite-build/ui-core/components/layout.module.scss'

const ACCEPTED_TYPES = ['.jsonl', '.json', '.csv', '.parquet']

type PreviewHeader = { key: string; header: string }

function buildPreviewData(data: ParsedDataRow[]): { headers: PreviewHeader[]; rows: Record<string, any>[] } {
  const cols = Object.keys(data[0] || {})
  const headers = cols.map((col) => ({ key: col, header: toUpperCase(col) || col }))
  const rows = data.slice(0, 15).map((row, i) => {
    const processedRow: Record<string, any> = { id: String(i) }
    for (const col of cols) {
      const val = row[col]
      if (val === null || val === undefined) {
        processedRow[col] = ''
      } else if (typeof val === 'string') {
        processedRow[col] = val.length > 120 ? val.substring(0, 120) + '...' : val
      } else {
        const str = JSON.stringify(val)
        processedRow[col] = str.length > 120 ? str.substring(0, 120) + '...' : str
      }
    }
    return processedRow
  })
  return { headers, rows }
}

function PreviewTable({ headers, rows }: { headers: PreviewHeader[]; rows: Record<string, any>[] }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <Table size="sm">
        <TableHead>
          <TableRow>
            {headers.map((h) => (
              <TableHeader key={h.key}>{h.header}</TableHeader>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.id}>
              {headers.map((h) => (
                <TableCell key={h.key}>{row[h.key]}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function InfoTooltip({ label }: { label: string }) {
  return (
    <Tooltip label={label}>
      <button type="button" className={styles.tooltipTrigger} aria-label={label}>
        <Information size={16} />
      </button>
    </Tooltip>
  )
}

interface Step1DatasetUploadProps {
  uploadedFile: File | null
  setUploadedFile: (f: File | null) => void
  parsedData: ParsedDataRow[]
  setParsedData: (d: ParsedDataRow[]) => void
  columnMetadata: ColumnMetadata[]
  setColumnMetadata: (d: ColumnMetadata[]) => void
  detectedFormat: DatasetFormatType
  setDetectedFormat: (d: DatasetFormatType) => void
  datasetForm: DatasetForm
  setDatasetForm: (updater: (prev: DatasetForm) => DatasetForm) => void
  totalRecords: number
  setTotalRecords: (n: number) => void
  existingDatasetId: string | null
  setExistingDatasetId: (id: string | null) => void
  splitRatio: number
  validationFile: File | null
  setValidationFile: (f: File | null) => void
  isSplitEnabled: boolean
  setIsSplitEnabled: (b: boolean) => void
  selectedAlgorithm: string
  setSelectedAlgorithm: (a: string) => void
  selectedGoal: TuningGoal | null
  columnMapping: ColumnMapping
  setColumnMapping: (m: ColumnMapping) => void
  setIsDatasetCompatible: (b: boolean) => void
  selectedExistingDataset: Dataset | null
  setSelectedExistingDataset: (d: Dataset | null) => void
  onDatasetChanged: () => void
}

export function Step1DatasetUpload({
  uploadedFile,
  setUploadedFile,
  parsedData,
  setParsedData,
  columnMetadata,
  setColumnMetadata,
  detectedFormat,
  setDetectedFormat,
  datasetForm,
  setDatasetForm,
  totalRecords,
  setTotalRecords,
  existingDatasetId,
  setExistingDatasetId,
  splitRatio,
  validationFile,
  setValidationFile,
  isSplitEnabled,
  setIsSplitEnabled,
  selectedAlgorithm,
  setSelectedAlgorithm,
  selectedGoal,
  columnMapping,
  setColumnMapping,
  setIsDatasetCompatible,
  selectedExistingDataset,
  setSelectedExistingDataset,
  onDatasetChanged,
}: Step1DatasetUploadProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingProgress, setProcessingProgress] = useState('')
  const [error, setError] = useState('')
  const [previewRows, setPreviewRows] = useState<Record<string, any>[]>([])
  const [previewHeaders, setPreviewHeaders] = useState<PreviewHeader[]>([])
  const [valPreviewRows, setValPreviewRows] = useState<Record<string, any>[]>([])
  const [valPreviewHeaders, setValPreviewHeaders] = useState<PreviewHeader[]>([])
  const [validationRecordCount, setValidationRecordCount] = useState(0)
  const [activePreviewTab, setActivePreviewTab] = useState(0)
  const [userColumns, setUserColumns] = useState<string[]>([])
  // Increments per upload so a late record count can tell whether it is still
  // describing the file currently selected.
  const countTokenRef = useRef(0)
  const [dataSourceIndex, setDataSourceIndex] = useState(0)

  const [isAiSuggesting, setIsAiSuggesting] = useState(false)
  const [aiSuggestion, setAiSuggestion] = useState<{ confidence: number; reasoning: string; algorithm: string } | null>(null)
  const [aiSuggestedFields, setAiSuggestedFields] = useState<Set<string>>(new Set())
  const [showAiReasoning, setShowAiReasoning] = useState(false)

  const { data: existingDatasetsResult, isLoading: isLoadingDatasets } = useQuery({
    queryKey: ['autotunex', 'datasets'],
    queryFn: () => getDatasets({ page: 1, pageSize: 100 }),
  })
  const existingDatasets = existingDatasetsResult?.items ?? []
  const { data: datasetTypes = {} } = useQuery({
    queryKey: ['autotunex', 'datasetTypes'],
    queryFn: getAutotuneDatasetTypes,
  })

  const hasDatasetTypes = Object.keys(datasetTypes).length > 0

  const allColumns = useMemo(
    () => (hasDatasetTypes ? getColumnsFromTypes(selectedAlgorithm, datasetTypes) : []),
    [hasDatasetTypes, selectedAlgorithm, datasetTypes]
  )
  const requiredColumns = useMemo(
    () => (hasDatasetTypes ? getRequiredColumnsFromTypes(selectedAlgorithm, datasetTypes) : getRequiredColumns(selectedAlgorithm)),
    [hasDatasetTypes, selectedAlgorithm, datasetTypes]
  )
  const allColumnNames = useMemo(() => allColumns.map((c) => c.name), [allColumns])
  const datasetGoalWarning = useMemo(
    () => (selectedGoal && detectedFormat !== 'unknown' ? validateDatasetForGoal(detectedFormat, selectedGoal) : { valid: true, message: '' }),
    [selectedGoal, detectedFormat]
  )

  useEffect(() => {
    setIsDatasetCompatible(datasetGoalWarning.valid)
  }, [datasetGoalWarning.valid, setIsDatasetCompatible])

  // Heuristic column-mapping suggestion when the algorithm changes (skipped once AI has suggested)
  useEffect(() => {
    if (!selectedAlgorithm || userColumns.length === 0 || aiSuggestion || isAiSuggesting) return
    const alreadyMapped = requiredColumns.length > 0 && requiredColumns.every((c) => columnMapping[c])
    if (alreadyMapped) return
    const columnsToMap = allColumnNames.length > 0 ? allColumnNames : requiredColumns
    setColumnMapping(suggestColumnMappingHeuristic(userColumns, columnsToMap))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAlgorithm, userColumns, aiSuggestion, isAiSuggesting])

  // Build train preview whenever parsed data changes
  useEffect(() => {
    if (parsedData.length === 0) return
    setUserColumns(columnMetadata.map((col) => col.name))
    const result = buildPreviewData(parsedData)
    setPreviewHeaders(result.headers)
    setPreviewRows(result.rows)
  }, [parsedData, columnMetadata])

  // Build validation preview for split mode
  useEffect(() => {
    if (!isSplitEnabled || parsedData.length === 0 || !uploadedFile) return
    const splitIndex = Math.floor((parsedData.length * splitRatio) / 100)
    const valSlice = parsedData.slice(splitIndex)
    if (valSlice.length > 0) {
      const result = buildPreviewData(valSlice)
      setValPreviewHeaders(result.headers)
      setValPreviewRows(result.rows)
      setValidationRecordCount(totalRecords - Math.floor((totalRecords * splitRatio) / 100))
    }
  }, [isSplitEnabled, parsedData, uploadedFile, splitRatio, totalRecords])

  // Build validation preview for a manually-uploaded validation file
  useEffect(() => {
    if (isSplitEnabled || !validationFile) return
    let alive = true
    processUploadedFileAsync(validationFile, 50)
      .then((rawData) => {
        if (!alive) return
        const result = buildPreviewData(rawData)
        setValPreviewHeaders(result.headers)
        setValPreviewRows(result.rows)
        setValidationRecordCount(rawData.length)
        // Returned so the chained catch below covers this rejection too.
        return countLinesInFileAsync(validationFile).then((count) => {
          if (alive) setValidationRecordCount(count)
        })
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [isSplitEnabled, validationFile])

  // Clear validation preview when split is off and no manual validation file
  useEffect(() => {
    if (isSplitEnabled || validationFile) return
    setValPreviewRows([])
    setValPreviewHeaders([])
    setValidationRecordCount(0)
  }, [isSplitEnabled, validationFile])

  // Restore an existing dataset's details on remount (e.g. navigating back to this step)
  useEffect(() => {
    if (!existingDatasetId || selectedExistingDataset) return
    getDataset(existingDatasetId).then(setSelectedExistingDataset).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [existingDatasetId])

  const trainRecordCount = existingDatasetId
    ? selectedExistingDataset?.train_records || totalRecords
    : isSplitEnabled && uploadedFile
      ? Math.floor((totalRecords * splitRatio) / 100)
      : totalRecords

  async function suggestMappingWithAI(data: ParsedDataRow[], metadata: ColumnMetadata[]) {
    if (data.length === 0 || metadata.length === 0) return

    setIsAiSuggesting(true)
    setAiSuggestion(null)
    setAiSuggestedFields(new Set())
    setShowAiReasoning(false)

    try {
      const colNames = metadata.map((c) => c.name)
      const colSamples: Record<string, string[]> = {}
      for (const col of metadata) colSamples[col.name] = col.sampleValues.slice(0, 3)

      const targetType = ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm]
      const result = await suggestColumnMappingAI({
        sample_data: data.slice(0, 8),
        column_names: colNames,
        column_samples: colSamples,
        target_format: targetType,
      })

      setAiSuggestion({ confidence: result.confidence, reasoning: result.reasoning ?? '', algorithm: result.tuning_type })

      // Tracks the algorithm this mapping should be filtered against. `setSelectedAlgorithm`
      // does not update the `selectedAlgorithm` captured by this closure, so reading
      // that below applied the AI's column mapping against the PREVIOUS algorithm
      // whenever the AI changed it.
      let effectiveAlgorithm = selectedAlgorithm

      if (result.tuning_type) {
        const aiAlgoDetail = ALGORITHM_DETAILS.find((a) => a.id === result.tuning_type)
        if (!selectedGoal || (aiAlgoDetail && aiAlgoDetail.category === selectedGoal)) {
          setSelectedAlgorithm(result.tuning_type)
          // Only when the suggestion is actually adopted.
          effectiveAlgorithm = result.tuning_type
        }
      }

      if (result.column_mapping) {
        const newMapping: ColumnMapping = {}
        const newSuggested = new Set<string>()

        const types = hasDatasetTypes ? datasetTypes : await getAutotuneDatasetTypes()
        const algo = effectiveAlgorithm
        const aiAllCols = hasDatasetTypes || Object.keys(types).length > 0 ? getColumnsFromTypes(algo, types).map((c) => c.name) : getRequiredColumns(algo)

        const typeKey = ALGORITHM_TO_DATASET_TYPE[algo]
        const columnsDict = types[typeKey]?.columns || {}
        const dictKeyToName: Record<string, string> = {}
        for (const [key, col] of Object.entries(columnsDict)) dictKeyToName[key] = (col as any).name

        for (const [aiKey, sourceColumn] of Object.entries(result.column_mapping)) {
          if (!sourceColumn || !colNames.includes(sourceColumn)) continue

          let matchedCol = dictKeyToName[aiKey]
          if (!matchedCol) {
            const normalized = aiKey.replace(/_col$/, '')
            matchedCol = aiAllCols.find((rc) => rc === aiKey || rc === normalized || rc === sourceColumn) || ''
          }

          if (matchedCol && aiAllCols.includes(matchedCol)) {
            newMapping[matchedCol] = sourceColumn
            newSuggested.add(matchedCol)
          }
        }

        setColumnMapping(newMapping)
        setAiSuggestedFields(newSuggested)
      }
    } catch {
      // Heuristic mapping already applied by the effect above; nothing else to do.
    } finally {
      setIsAiSuggesting(false)
    }
  }

  async function handleFileUpload(file: File) {
    setIsProcessing(true)
    setProcessingProgress('')
    setError('')
    setExistingDatasetId(null)
    setSelectedExistingDataset(null)

    try {
      setUploadedFile(file)
      setDatasetForm((prev) =>
        prev.name ? prev : { ...prev, name: file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9-_]/g, '-') }
      )

      if (file.size > 5 * 1024 * 1024) setProcessingProgress('Processing large file...')

      const rawData = await processUploadedFileAsync(file, 50)
      setParsedData(rawData)

      const columns = Object.keys(rawData[0] || {})
      const metadata = extractColumnMetadata(rawData)
      const format = detectDatasetFormat(columns)
      setColumnMetadata(metadata)
      setDetectedFormat(format)

      const suggestedAlgo = suggestAlgorithm(columns)
      const suggestedDetail = ALGORITHM_DETAILS.find((a) => a.id === suggestedAlgo)
      if (!selectedGoal || (suggestedDetail && suggestedDetail.category === selectedGoal)) {
        setSelectedAlgorithm(suggestedAlgo)
      }

      setTotalRecords(rawData.length)
      // The exact count streams in a worker and can outlive this upload. Without a
      // token, a slow count from a file the user has since replaced overwrites the
      // newer file's total; without a catch, a failure (a malformed file now
      // reports an unterminated quoted field) is an unhandled rejection. The
      // sample-derived estimate just set above stands if the count cannot finish.
      const countToken = ++countTokenRef.current
      countLinesInFileAsync(file)
        .then((count) => {
          if (countTokenRef.current === countToken) setTotalRecords(count)
        })
        .catch(() => {})

      onDatasetChanged()
      suggestMappingWithAI(rawData, metadata)
    } catch (err: any) {
      setError(err.message || 'Failed to process file')
    } finally {
      setIsProcessing(false)
      setProcessingProgress('')
    }
  }

  async function handleExistingDatasetSelect(datasetId: string) {
    if (!datasetId) {
      setSelectedExistingDataset(null)
      setExistingDatasetId(null)
      setParsedData([])
      setColumnMetadata([])
      setDetectedFormat('unknown')
      setTotalRecords(0)
      setUserColumns([])
      return
    }

    setIsProcessing(true)
    setError('')
    onDatasetChanged()

    try {
      const dataset = await getDataset(datasetId, { preview: true, previewRows: 50 })
      setSelectedExistingDataset(dataset)
      setExistingDatasetId(dataset.id)
      setTotalRecords((dataset.train_records || 0) + (dataset.validation_records || 0))
      setDatasetForm(() => ({ name: dataset.name, description: dataset.description, train_file: null, validation_file: null }))

      const trainPreview = dataset.preview?.train ?? []
      if (trainPreview.length > 0) {
        setParsedData(trainPreview)
        const columns = Object.keys(trainPreview[0] || {})
        setColumnMetadata(extractColumnMetadata(trainPreview))
        setDetectedFormat(detectDatasetFormat(columns))
        const suggestedAlgo = suggestAlgorithm(columns)
        const suggestedDetail = ALGORITHM_DETAILS.find((a) => a.id === suggestedAlgo)
        if (!selectedGoal || (suggestedDetail && suggestedDetail.category === selectedGoal)) {
          setSelectedAlgorithm(suggestedAlgo)
        }
      } else {
        setParsedData([])
        setColumnMetadata([])
        setDetectedFormat('unknown')
      }

      const validationPreview = dataset.preview?.validation ?? []
      if (validationPreview.length > 0) {
        const result = buildPreviewData(validationPreview)
        setValPreviewHeaders(result.headers)
        setValPreviewRows(result.rows)
        setValidationRecordCount(dataset.validation_records || validationPreview.length)
      } else {
        setValPreviewRows([])
        setValPreviewHeaders([])
        setValidationRecordCount(0)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load dataset')
    } finally {
      setIsProcessing(false)
    }
  }

  function clearTrainFile() {
    setUploadedFile(null)
    setParsedData([])
    setColumnMetadata([])
    setDetectedFormat('unknown')
    setDatasetForm((prev) => ({ ...prev, name: '' }))
    setTotalRecords(0)
    setUserColumns([])
    setPreviewRows([])
    setPreviewHeaders([])
    setValPreviewRows([])
    setValPreviewHeaders([])
    setValidationRecordCount(0)
    setValidationFile(null)
    setColumnMapping({})
    setSelectedAlgorithm(selectedGoal ? getDefaultAlgorithmForGoal(selectedGoal) : 'lora')
    setActivePreviewTab(0)
    setAiSuggestion(null)
    setAiSuggestedFields(new Set())
  }

  function updateColumnMapping(requiredCol: string, userCol: string) {
    setColumnMapping({ ...columnMapping, [requiredCol]: userCol })
    setAiSuggestedFields((prev) => {
      const next = new Set(prev)
      next.delete(requiredCol)
      return next
    })
  }

  function resetForm() {
    setUploadedFile(null)
    setParsedData([])
    setColumnMetadata([])
    setDetectedFormat('unknown')
    setDatasetForm(() => ({ name: '', description: '', train_file: null, validation_file: null }))
    setTotalRecords(0)
    setExistingDatasetId(null)
    setSelectedExistingDataset(null)
    setValidationFile(null)
    setSelectedAlgorithm(selectedGoal ? getDefaultAlgorithmForGoal(selectedGoal) : 'lora')
    setColumnMapping({})
    setUserColumns([])
    setPreviewRows([])
    setPreviewHeaders([])
    setValPreviewRows([])
    setValPreviewHeaders([])
    setValidationRecordCount(0)
    setActivePreviewTab(0)
    setError('')
    setIsSplitEnabled(true)
    setAiSuggestion(null)
    setAiSuggestedFields(new Set())
    onDatasetChanged()
  }

  const showColumnMapping = uploadedFile && parsedData.length > 0 && userColumns.length > 0 && !existingDatasetId
  const sortedColumns = [...allColumns].sort((a, b) => Number(b.required) - Number(a.required))

  return (
    <div className={layoutStyles.rowWrap}>
      <div className={styles.settingsColumn}>
          <Tile style={{ padding: '1.25rem' }}>
            <h6 className={styles.tileHeading}>Dataset Settings</h6>

            {uploadedFile && !existingDatasetId && (
              <>
                <div className={styles.nameField}>
                  <TextInput
                    id="dataset-name"
                    labelText="Dataset Name"
                    placeholder="my-dataset"
                    value={datasetForm.name}
                    onChange={(e) => setDatasetForm((prev) => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div className={styles.splitToggleRow}>
                  <div className={styles.toggleLabelRow}>
                    <span>Split dataset</span>
                    <InfoTooltip label="Automatically splits your uploaded file into training and validation sets. Disable this to upload separate files for each." />
                  </div>
                  <Toggle id="split-toggle" labelText="" hideLabel toggled={isSplitEnabled} onToggle={setIsSplitEnabled} size="sm" />
                </div>
              </>
            )}

            {!uploadedFile && !existingDatasetId && (
              <>
                {existingDatasets.length > 0 && (
                  <ContentSwitcher selectedIndex={dataSourceIndex} onChange={({ index }) => setDataSourceIndex(index ?? 0)} style={{ marginBottom: '0.75rem' }}>
                    <Switch name="upload" text="Upload" />
                    <Switch name="existing" text="Select Existing" />
                  </ContentSwitcher>
                )}

                {dataSourceIndex === 0 || existingDatasets.length === 0 ? (
                  <div className={styles.dropZone}>
                    <FileUploaderDropContainer
                      labelText="Drag and drop a file here or click to upload"
                      accept={ACCEPTED_TYPES}
                      onAddFiles={(_e, { addedFiles }) => {
                        if (addedFiles.length > 0) handleFileUpload(addedFiles[0])
                      }}
                    />
                    <p className={styles.dropZoneHint}>Accepted formats: .jsonl, .json, .csv, .parquet</p>
                  </div>
                ) : (
                  <Select
                    id="existing-dataset-select"
                    labelText=""
                    defaultValue=""
                    onChange={(e) => handleExistingDatasetSelect(e.target.value)}
                  >
                    <SelectItem value="" text="Choose a dataset..." />
                    {existingDatasets.map((ds) => (
                      <SelectItem key={ds.id} value={ds.id} text={`${ds.name} (${(ds.train_records || 0) + (ds.validation_records || 0)} records)`} />
                    ))}
                  </Select>
                )}
              </>
            )}

            {existingDatasetId ? (
              <FileUploaderItem name={`${selectedExistingDataset?.name || ''} (${totalRecords.toLocaleString()} records)`} status="edit" onDelete={resetForm} />
            ) : uploadedFile && !isSplitEnabled ? (
              <div className={styles.fileRow}>
                <span className={styles.fileLabel}>
                  Train file
                  <InfoTooltip label="The main dataset used to train the model. This is where the model learns patterns from your data." />
                </span>
                <FileUploaderItem name={uploadedFile.name} status="edit" onDelete={clearTrainFile} />
              </div>
            ) : null}

            {isLoadingDatasets && !uploadedFile && !existingDatasetId && (
              <InlineLoading description="Loading datasets..." style={{ marginTop: '0.5rem' }} />
            )}

            {isProcessing && <InlineLoading description={processingProgress || 'Processing...'} style={{ marginTop: '0.5rem' }} />}

            {error && <InlineNotification kind="error" title="Error" subtitle={error} style={{ marginTop: '0.5rem' }} />}

            {uploadedFile && !existingDatasetId && !isSplitEnabled && (
              <div style={{ marginTop: '0.75rem' }}>
                {!validationFile ? (
                  <FileUploaderButton
                    labelText="Upload Validation File"
                    buttonKind="secondary"
                    size="sm"
                    accept={ACCEPTED_TYPES}
                    onChange={(e) => {
                      const file = e.target.files?.[0]
                      if (file) setValidationFile(file)
                    }}
                  />
                ) : (
                  <div className={styles.fileRow}>
                    <span className={styles.fileLabel}>
                      Validation file
                      <InfoTooltip label="A separate dataset used to evaluate model performance during training. Helps detect overfitting." />
                    </span>
                    <FileUploaderItem name={validationFile.name} status="edit" onDelete={() => setValidationFile(null)} />
                  </div>
                )}
              </div>
            )}

            {showColumnMapping && (
              <>
                <hr className={styles.sectionDivider} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <p className={styles.sectionTitle} style={{ margin: 0 }}>Column Mapping</p>
                  {isAiSuggesting ? (
                    <InlineLoading description="AI analyzing..." />
                  ) : aiSuggestion ? (
                    <button type="button" className={styles.aiTagButton} onClick={() => setShowAiReasoning((v) => !v)}>
                      <Tag type="green" size="sm">
                        AI Suggested ({Math.round(aiSuggestion.confidence * 100)}%) {showAiReasoning ? '▴' : '▾'}
                      </Tag>
                    </button>
                  ) : null}
                </div>

                {aiSuggestion?.reasoning && showAiReasoning && (
                  <InlineNotification kind="info" title="AI Insight" subtitle={aiSuggestion.reasoning} hideCloseButton lowContrast style={{ marginBottom: '0.75rem' }} />
                )}

                {!isAiSuggesting && (
                  <>
                    <div className={styles.mappingHeader}>
                      <span>Field</span>
                      <span>Source Column</span>
                    </div>
                    {sortedColumns.length > 0
                      ? sortedColumns.map((colInfo) => (
                          <div className={styles.mappingRow} key={colInfo.name}>
                            <div className={styles.mappingLabel}>
                              {toUpperCase(colInfo.name)}
                              {colInfo.desc && <InfoTooltip label={colInfo.desc} />}
                            </div>
                            <Select
                              id={`column-map-${colInfo.name}`}
                              labelText=""
                              size="sm"
                              value={columnMapping[colInfo.name] || ''}
                              onChange={(e) => updateColumnMapping(colInfo.name, e.target.value)}
                            >
                              <SelectItem value="" text={colInfo.required ? 'Select column...' : 'None'} />
                              {userColumns.map((col) => (
                                <SelectItem key={col} value={col} text={col} />
                              ))}
                            </Select>
                          </div>
                        ))
                      : requiredColumns.map((reqCol) => (
                          <div className={styles.mappingRow} key={reqCol}>
                            <div className={styles.mappingLabel}>
                              {toUpperCase(reqCol)}
                              <span className={styles.requiredMarker}>*</span>
                            </div>
                            <Select
                              id={`column-map-${reqCol}`}
                              labelText=""
                              size="sm"
                              value={columnMapping[reqCol] || ''}
                              onChange={(e) => updateColumnMapping(reqCol, e.target.value)}
                            >
                              <SelectItem value="" text="Select column..." />
                              {userColumns.map((col) => (
                                <SelectItem key={col} value={col} text={col} />
                              ))}
                            </Select>
                          </div>
                        ))}
                  </>
                )}
              </>
            )}

            {(uploadedFile || existingDatasetId) && (
              <Button kind="tertiary" size="sm" style={{ marginTop: '0.5rem' }} renderIcon={Reset} onClick={resetForm}>
                Reset All
              </Button>
            )}
          </Tile>
      </div>

        <div className={styles.previewColumn}>
          {previewRows.length > 0 && previewHeaders.length > 0 ? (
            <Tile className={styles.previewTile}>
              {valPreviewRows.length > 0 ? (
                <Tabs selectedIndex={activePreviewTab} onChange={({ selectedIndex }) => setActivePreviewTab(selectedIndex)}>
                  <TabList aria-label="Dataset preview tabs">
                    <Tab>{`Train (${trainRecordCount.toLocaleString()})`}</Tab>
                    <Tab>{`Validation (${validationRecordCount.toLocaleString()})`}</Tab>
                  </TabList>
                  <TabPanels>
                    <TabPanel style={{ padding: '0.5rem 0' }}>
                      <PreviewTable headers={previewHeaders} rows={previewRows} />
                    </TabPanel>
                    <TabPanel style={{ padding: '0.5rem 0' }}>
                      <PreviewTable headers={valPreviewHeaders} rows={valPreviewRows} />
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              ) : (
                <>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <h6 className={styles.tileHeading} style={{ margin: 0 }}>Data Preview</h6>
                    <span className={styles.helperTextInline}>
                      {previewRows.length} of {totalRecords.toLocaleString()} records
                    </span>
                  </div>
                  <PreviewTable headers={previewHeaders} rows={previewRows} />
                </>
              )}
            </Tile>
          ) : !uploadedFile && !selectedExistingDataset ? (
            <ExpectedFormatPanel selectedAlgorithm={selectedAlgorithm} datasetTypes={datasetTypes} />
          ) : null}
        </div>
    </div>
  )
}

function ExpectedFormatPanel({ selectedAlgorithm, datasetTypes }: { selectedAlgorithm: string; datasetTypes: Record<string, any> }) {
  const hasDatasetTypes = Object.keys(datasetTypes).length > 0
  const examples = hasDatasetTypes ? getDatasetExamplesFromTypes(selectedAlgorithm, datasetTypes) : getDatasetExamples(selectedAlgorithm)
  const typeKey = ALGORITHM_TO_DATASET_TYPE[selectedAlgorithm]
  const typeDesc = datasetTypes[typeKey]?.desc
  const typeColumns = datasetTypes[typeKey]?.columns
  const formats = typeColumns ? generateFormatExamples(Object.values(typeColumns)) : null

  return (
    <Tile style={{ padding: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
        <h6 className={styles.tileHeading} style={{ margin: 0 }}>Expected Dataset Format</h6>
      </div>
      {typeDesc && <p className={styles.helperTextInline} style={{ marginBottom: '0.75rem' }}>{typeDesc}</p>}

      <Tabs>
        <TabList aria-label="Dataset format examples">
          <Tab>JSON</Tab>
          <Tab>JSONL</Tab>
          <Tab>CSV</Tab>
          <Tab>Parquet</Tab>
        </TabList>
        <TabPanels>
          <TabPanel style={{ padding: '0.5rem 0' }}>
            <pre className={styles.exampleCode}>{formats?.json || JSON.stringify(examples, null, 2)}</pre>
          </TabPanel>
          <TabPanel style={{ padding: '0.5rem 0' }}>
            <pre className={styles.exampleCode}>{formats?.jsonl || examples.map((ex) => JSON.stringify(ex)).join('\n')}</pre>
          </TabPanel>
          <TabPanel style={{ padding: '0.5rem 0' }}>
            <pre className={styles.exampleCode}>{formats?.csv || examples.map((ex) => Object.values(ex).join(', ')).join('\n')}</pre>
          </TabPanel>
          <TabPanel style={{ padding: '0.5rem 0' }}>
            <pre className={styles.exampleCode}>
              {'Apache Parquet is a columnar storage format.\nUse the same column structure as CSV/JSON.\nGenerate with: df.to_parquet("data.parquet")'}
            </pre>
          </TabPanel>
        </TabPanels>
      </Tabs>

      <p className={styles.emptyStateHint}>Upload a dataset or select an existing one to get started.</p>
    </Tile>
  )
}
