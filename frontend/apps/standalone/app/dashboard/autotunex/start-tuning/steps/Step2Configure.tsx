'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Tile, Select, SelectItem, Button, TextInput, Tag, InlineLoading, InlineNotification, Checkbox } from '@carbon/react'
import { Add, Settings, Edit } from '@carbon/icons-react'
import type { Configuration, ConfigData, ConfigForm, ListResult, PendingConfigData, PendingConfigUpdate, TuningGoal } from '@granite-build/ui-core/types'
import { getConfiguration, getConfigurations, getConfigurationTemplate } from '@granite-build/ui-core/api/autotunex'
import { ALGORITHM_DETAILS } from '@granite-build/ui-core/config/autotunexAlgorithms'
import { ConfigDisplay } from '@granite-build/ui-core/components/ConfigDisplay'
import { CreateConfigForm } from '@granite-build/ui-core/components/CreateConfigForm'
import styles from './Step2Configure.module.scss'
import layoutStyles from '@granite-build/ui-core/components/layout.module.scss'

const SFT_ALGORITHMS = ['lora', 'sft', 'alora', 'lokr', 'loha', 'vera']
const RL_ALGORITHMS = ['dpo', 'kto', 'ppo', 'grpo', 'dapo']
const ONLINE_RL_ALGORITHMS = ['ppo', 'grpo', 'dapo']
const SYSTEM_USER_ID = '00000000-0000-0000-0000-000000000001'

function suggestConfigs(configs: Configuration[], goal: TuningGoal | null, algorithm: string): Configuration[] {
  const filtered = configs.filter((config) => {
    const rlTuner = config.rl_tuner_type?.toLowerCase()
    const tunerType = config.tuner_type?.toLowerCase()

    if (goal === 'sft') {
      return SFT_ALGORITHMS.includes(tunerType || '') && (!rlTuner || rlTuner === 'none')
    } else if (goal === 'online_rl') {
      return ONLINE_RL_ALGORITHMS.includes(rlTuner || '')
    } else if (goal === 'offline_rl') {
      return rlTuner === algorithm
    }

    if (SFT_ALGORITHMS.includes(algorithm)) {
      return SFT_ALGORITHMS.includes(tunerType || '') && (!rlTuner || rlTuner === 'none')
    } else if (RL_ALGORITHMS.includes(algorithm)) {
      return rlTuner === algorithm
    }
    return true
  })

  return filtered.sort((a, b) => {
    const aIsSystem = a.user_id === SYSTEM_USER_ID
    const bIsSystem = b.user_id === SYSTEM_USER_ID
    if (aIsSystem !== bIsSystem) return aIsSystem ? -1 : 1
    return (a.name || '').localeCompare(b.name || '')
  })
}

interface Step2ConfigureProps {
  selectedAlgorithm: string
  selectedGoal: TuningGoal | null
  selectedConfigId: string | null
  setSelectedConfigId: (id: string | null) => void
  selectedConfig: Configuration | null
  setSelectedConfig: (c: Configuration | null) => void
  isEditingConfig: boolean
  setIsEditingConfig: (b: boolean) => void
  isCreatingConfig: boolean
  setIsCreatingConfig: (b: boolean) => void
  onPendingConfig: (data: PendingConfigData) => void
  /** Rename of an already-pending config; must reach the payload POSTed at launch. */
  onPendingConfigRename: (name: string) => void
  onPendingConfigUpdate: (data: PendingConfigUpdate) => void
  onClearPendingConfig: () => void
  autotuneEnabled: boolean
  setAutotuneEnabled: (v: boolean) => void
}

export function Step2Configure({
  selectedAlgorithm,
  selectedGoal,
  selectedConfigId,
  setSelectedConfigId,
  selectedConfig,
  setSelectedConfig,
  isEditingConfig,
  setIsEditingConfig,
  isCreatingConfig,
  setIsCreatingConfig,
  onPendingConfig,
  onPendingConfigRename,
  onPendingConfigUpdate,
  onClearPendingConfig,
  autotuneEnabled,
  setAutotuneEnabled,
}: Step2ConfigureProps) {
  const queryClient = useQueryClient()
  const { data: configsResult, isLoading } = useQuery({
    queryKey: ['autotunex', 'configurations'],
    queryFn: () => getConfigurations({ page: 1, pageSize: 100 }),
  })
  const allConfigs = configsResult?.items ?? []

  const [suggestedConfigs, setSuggestedConfigs] = useState<Configuration[]>([])
  const prevAlgorithm = useRef(selectedAlgorithm)
  const prevGoal = useRef(selectedGoal)
  const didAutoSelect = useRef(false)

  const presetGoal = useMemo(
    () => (ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm)?.category ?? null) as TuningGoal | null,
    [selectedAlgorithm]
  )

  // Config creation state
  const [newConfigForm, setNewConfigForm] = useState<ConfigForm | null>(null)
  const [newConfigName, setNewConfigName] = useState('')
  const [saveError, setSaveError] = useState('')
  const [configTemplateLoaded, setConfigTemplateLoaded] = useState(false)
  const [isLoadingCreateConfig, setIsLoadingCreateConfig] = useState(false)

  // Config editing state
  const [editableConfig, setEditableConfig] = useState<ConfigForm | null>(null)
  const [editSaveError, setEditSaveError] = useState('')
  const [needsSaveAs, setNeedsSaveAs] = useState(false)
  const [editConfigName, setEditConfigName] = useState('')
  const [isLoadingEditConfig, setIsLoadingEditConfig] = useState(false)

  useEffect(() => {
    if (!isEditingConfig || !selectedConfig) {
      setNeedsSaveAs(false)
      return
    }
    const isSystemConfig = selectedConfig.user_id === SYSTEM_USER_ID
    const hasAssociatedJobs = (selectedConfig.associated_jobs?.length ?? 0) > 0
    const requiresSaveAs = isSystemConfig || hasAssociatedJobs
    setNeedsSaveAs(requiresSaveAs)
    if (requiresSaveAs) setEditConfigName((prev) => prev || `${selectedConfig.name}_modified`)
  }, [isEditingConfig, selectedConfig])

  async function selectConfig(config: Configuration) {
    if (isEditingConfig) cancelEditMode()
    if (isCreatingConfig) cancelCreateMode()
    setSelectedConfigId(config.id)
    setSelectedConfig(config)
    onClearPendingConfig()

    if (!config.config_data && config.id) {
      try {
        const fullConfig = await getConfiguration(config.id)
        setSelectedConfig(fullConfig)
      } catch {
        // Keep the summary config; preview will show what's available.
      }
    }
  }

  // Initial load: populate suggestions, auto-select the first one
  useEffect(() => {
    if (allConfigs.length === 0 || didAutoSelect.current) return
    const suggestions = suggestConfigs(allConfigs, selectedGoal, selectedAlgorithm)
    setSuggestedConfigs(suggestions)
    if (!selectedConfigId && suggestions.length > 0) {
      selectConfig(suggestions[0])
    }
    didAutoSelect.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allConfigs])

  // Reactively update suggestions and reset selection when algorithm or goal changes
  useEffect(() => {
    if (allConfigs.length === 0 || !selectedAlgorithm) return
    if (selectedAlgorithm === prevAlgorithm.current && selectedGoal === prevGoal.current) return

    if (isEditingConfig) cancelEditMode()
    if (isCreatingConfig) cancelCreateMode()

    const suggestions = suggestConfigs(allConfigs, selectedGoal, selectedAlgorithm)
    setSuggestedConfigs(suggestions)
    if (suggestions.length > 0) {
      selectConfig(suggestions[0])
    } else {
      setSelectedConfigId(null)
      setSelectedConfig(null)
    }
    prevAlgorithm.current = selectedAlgorithm
    prevGoal.current = selectedGoal
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allConfigs, selectedAlgorithm, selectedGoal])

  async function enterEditMode() {
    if (!selectedConfig) return
    if (isCreatingConfig) cancelCreateMode()
    setIsEditingConfig(true)
    setEditSaveError('')
    setEditConfigName(selectedConfig.name)
    setIsLoadingEditConfig(true)

    try {
      const fullConfig = await getConfiguration(selectedConfig.id)
      setSelectedConfig(fullConfig)
      setEditableConfig({
        name: fullConfig.name,
        tuner_type: fullConfig.tuner_type,
        rl_tuner_type: fullConfig.rl_tuner_type || '',
        // getConfiguration(id) always returns populated config_data in
        // practice (only the list endpoint, GET /configs, nulls it) — this
        // fallback exists purely to satisfy ConfigForm's required ConfigData
        // fields at the type level.
        ...(fullConfig.config_data ?? ({} as ConfigData)),
      })
    } catch {
      setEditSaveError('Failed to load configuration details for editing.')
    } finally {
      setIsLoadingEditConfig(false)
    }
  }

  function cancelEditMode() {
    setIsEditingConfig(false)
    setEditableConfig(null)
    setEditSaveError('')
    setEditConfigName('')
    setNeedsSaveAs(false)
  }

  function confirmConfigEdit() {
    if (!selectedConfig || !editableConfig) return

    const tuner_type = editableConfig.tuner_type || selectedConfig.tuner_type
    const rl_tuner_type = editableConfig.rl_tuner_type || selectedConfig.rl_tuner_type || null
    const configData: ConfigData = {
      tune_config: editableConfig.tune_config,
      tuners_config: editableConfig.tuners_config,
      training_config: editableConfig.training_config,
      ...(editableConfig.training_rl_config ? { training_rl_config: editableConfig.training_rl_config } : {}),
      ...(editableConfig.tuners_rl_config ? { tuners_rl_config: editableConfig.tuners_rl_config } : {}),
    }

    if (needsSaveAs) {
      if (!editConfigName.trim()) {
        setEditSaveError('Please provide a name for the new configuration.')
        return
      }
      if (allConfigs.some((c) => c.name === editConfigName.trim())) {
        setEditSaveError(`A configuration named "${editConfigName.trim()}" already exists.`)
        return
      }

      const pendingData: PendingConfigData = {
        name: editConfigName.trim(),
        tuner_type: tuner_type || null,
        rl_tuner_type,
        config_data: configData,
      }

      const virtualConfig: Configuration = {
        id: '__pending__',
        user_id: '',
        name: pendingData.name,
        tuner_type: pendingData.tuner_type || '',
        rl_tuner_type: pendingData.rl_tuner_type,
        config_data: pendingData.config_data,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      }

      setSelectedConfigId('__pending__')
      setSelectedConfig(virtualConfig)
      onPendingConfig(pendingData)
    } else {
      if (!editConfigName.trim()) {
        setEditSaveError('Please provide a configuration name.')
        return
      }
      if (editConfigName.trim() !== selectedConfig.name && allConfigs.some((c) => c.name === editConfigName.trim())) {
        setEditSaveError(`A configuration named "${editConfigName.trim()}" already exists.`)
        return
      }

      const updatedName = editConfigName.trim()
      const pendingUpdate: PendingConfigUpdate = {
        configId: selectedConfig.id,
        name: updatedName,
        tuner_type: tuner_type || null,
        rl_tuner_type,
        config_data: configData,
      }

      setSelectedConfig({ ...selectedConfig, name: updatedName, tuner_type: tuner_type || '', rl_tuner_type, config_data: configData })
      onPendingConfigUpdate(pendingUpdate)
      setSuggestedConfigs((prev) => prev.map((c) => (c.id === selectedConfig.id ? { ...c, name: updatedName } : c)))
      queryClient.setQueryData<ListResult<Configuration>>(['autotunex', 'configurations'], (prev) =>
        prev ? { ...prev, items: prev.items.map((c) => (c.id === selectedConfig.id ? { ...c, name: updatedName } : c)) } : prev
      )
    }

    cancelEditMode()
  }

  async function openCreateForm() {
    if (isEditingConfig) cancelEditMode()
    setIsCreatingConfig(true)
    setSaveError('')
    setNewConfigName(selectedConfigId === '__pending__' && selectedConfig ? selectedConfig.name : '')

    if (!configTemplateLoaded) {
      setIsLoadingCreateConfig(true)
      try {
        const template = await getConfigurationTemplate()
        setNewConfigForm(template)
        setConfigTemplateLoaded(true)
      } catch {
        setSaveError('Failed to load configuration template.')
      } finally {
        setIsLoadingCreateConfig(false)
      }
    }
  }

  function cancelCreateMode() {
    setIsCreatingConfig(false)
    setSaveError('')
    setNewConfigName('')
  }

  function confirmNewConfig() {
    if (!newConfigName.trim()) {
      setSaveError('Please enter a configuration name.')
      return
    }
    if (allConfigs.some((c) => c.name === newConfigName.trim())) {
      setSaveError(`A configuration named "${newConfigName.trim()}" already exists.`)
      return
    }
    if (!newConfigForm) return

    const { name: _name, tuner_type, rl_tuner_type, ...configSections } = newConfigForm

    const pendingData: PendingConfigData = {
      name: newConfigName.trim(),
      tuner_type: presetGoal === 'online_rl' ? null : tuner_type || 'lora',
      rl_tuner_type: rl_tuner_type || null,
      config_data: configSections as ConfigData,
    }

    const virtualConfig: Configuration = {
      id: '__pending__',
      user_id: '',
      name: pendingData.name,
      tuner_type: pendingData.tuner_type || '',
      rl_tuner_type: pendingData.rl_tuner_type,
      config_data: pendingData.config_data,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    setSelectedConfigId('__pending__')
    setSelectedConfig(virtualConfig)
    setIsCreatingConfig(false)
    onPendingConfig(pendingData)
  }

  if (isLoading) {
    return <InlineLoading description="Loading configurations..." />
  }

  const hpoHeader = (
    <div className={styles.hpoToggleRow}>
      <Checkbox
        id="autotune-enabled"
        checked={autotuneEnabled}
        onChange={(_, { checked }) => setAutotuneEnabled(checked)}
        labelText="Use hyperparameter optimization"
      />
      <p className={styles.hpoToggleHelper}>
        {autotuneEnabled
          ? 'Search across each hyperparameter’s value space.'
          : 'Off — a single run using each hyperparameter’s Default value.'}
      </p>
    </div>
  )

  if (isEditingConfig || isCreatingConfig) {
    return (
      <div>
        {hpoHeader}
        <div className={layoutStyles.rowWrap} key={selectedAlgorithm}>
          <div className={styles.leftColumn}>
            <Tile style={{ padding: '1.25rem' }}>
              {isEditingConfig ? (
                <>
                  <h5 className={styles.panelTitle}>Edit Configuration</h5>
                  {needsSaveAs && (
                    <InlineNotification
                      kind="info"
                      lowContrast
                      hideCloseButton
                      title="Save as new"
                      subtitle={
                        selectedConfig?.user_id === SYSTEM_USER_ID
                          ? 'System config — cannot modify directly.'
                          : `Has ${selectedConfig?.associated_jobs?.length || 0} job(s) — cannot modify directly.`
                      }
                      style={{ marginBottom: '0.75rem' }}
                    />
                  )}
                  <TextInput
                    id="edit-config-name"
                    labelText={needsSaveAs ? 'New Configuration Name' : 'Configuration Name'}
                    placeholder={needsSaveAs ? 'Enter new configuration name' : undefined}
                    value={editConfigName}
                    onChange={(e) => setEditConfigName(e.target.value)}
                    invalid={
                      editConfigName.trim() !== '' &&
                      editConfigName.trim() !== (needsSaveAs ? undefined : selectedConfig?.name) &&
                      allConfigs.some((c) => c.name === editConfigName.trim())
                    }
                    invalidText={`"${editConfigName.trim()}" already exists`}
                  />
                  {editSaveError && (
                    <InlineNotification kind="error" title="Error:" subtitle={editSaveError} onClose={() => setEditSaveError('')} style={{ marginTop: '0.75rem' }} />
                  )}
                  <div className={styles.leftPanelActions}>
                    <Button size="sm" kind="ghost" onClick={cancelEditMode}>Cancel</Button>
                    <Button size="sm" kind="primary" disabled={isLoadingEditConfig || (needsSaveAs && !editConfigName.trim())} onClick={confirmConfigEdit}>
                      {needsSaveAs ? 'Confirm as New' : 'Confirm'}
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <h5 className={styles.panelTitle}>New Configuration</h5>
                  <TextInput
                    id="new-config-name"
                    labelText="Configuration Name"
                    placeholder="my-config"
                    value={newConfigName}
                    onChange={(e) => setNewConfigName(e.target.value)}
                    invalid={saveError !== '' && !newConfigName.trim()}
                    invalidText="Name is required"
                  />
                  {saveError && <InlineNotification kind="error" title="Error" subtitle={saveError} onClose={() => setSaveError('')} style={{ marginTop: '0.75rem' }} />}
                  <div className={styles.leftPanelActions}>
                    <Button size="sm" kind="ghost" onClick={cancelCreateMode}>Cancel</Button>
                    <Button size="sm" kind="primary" disabled={!newConfigName.trim()} onClick={confirmNewConfig}>Confirm</Button>
                  </div>
                </>
              )}
            </Tile>
          </div>

          <div className={styles.rightColumn}>
            <Tile className={styles.configPreviewTile}>
              {isEditingConfig ? (
                isLoadingEditConfig ? (
                  <InlineLoading description="Loading configuration..." />
                ) : (
                  editableConfig && (
                    <CreateConfigForm
                      config={editableConfig}
                      setConfig={setEditableConfig as any}
                      configurations={allConfigs}
                      editMode
                      existingConfig={selectedConfig}
                      hideNameField
                      presetGoal={presetGoal}
                      presetAlgorithm={selectedAlgorithm}
                      hpoEnabled={autotuneEnabled}
                    />
                  )
                )
              ) : isLoadingCreateConfig ? (
                <InlineLoading description="Loading configuration template..." />
              ) : (
                configTemplateLoaded &&
                newConfigForm && (
                  <CreateConfigForm
                    key={`${presetGoal}-${selectedAlgorithm}`}
                    config={newConfigForm}
                    setConfig={setNewConfigForm as any}
                    configurations={allConfigs}
                    hideNameField
                    presetGoal={presetGoal}
                    presetAlgorithm={selectedAlgorithm}
                    hpoEnabled={autotuneEnabled}
                  />
                )
              )}
            </Tile>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      {hpoHeader}
      <div className={layoutStyles.rowWrap} key={selectedAlgorithm}>
        <div className={styles.leftColumn}>
          <Tile style={{ padding: '1.25rem' }}>
            {selectedConfigId === '__pending__' && selectedConfig ? (
              <div className={styles.createdConfigCard}>
                <TextInput
                  id="pending-config-name"
                  value={selectedConfig.name}
                  labelText="Configuration Name"
                  onChange={(e) => {
                    setSelectedConfig({ ...selectedConfig, name: e.target.value })
                    onPendingConfigRename(e.target.value)
                  }}
                />
                <div className={styles.createdConfigActions}>
                  <Button kind="tertiary" renderIcon={Edit} size="sm" onClick={openCreateForm}>Edit</Button>
                  {suggestedConfigs.length > 0 && (
                    <Button
                      kind="ghost"
                      size="sm"
                      onClick={() => {
                        setSelectedConfigId(null)
                        setSelectedConfig(null)
                        onClearPendingConfig()
                      }}
                    >
                      Choose Existing
                    </Button>
                  )}
                </div>
              </div>
            ) : suggestedConfigs.length === 0 ? (
              <>
                <p className={styles.suggestionText}>
                  {allConfigs.length === 0
                    ? 'No configurations yet. Create one to get started.'
                    : `No matching configurations for ${selectedGoal === 'sft' ? 'SFT' : selectedGoal === 'online_rl' ? 'Online RL' : selectedAlgorithm.toUpperCase()}.`}
                </p>
                <Button kind="tertiary" renderIcon={Add} size="sm" onClick={openCreateForm}>Create New Configuration</Button>
              </>
            ) : (
              <>
                <Select
                  id="config-select"
                  labelText="Configurations"
                  value={selectedConfigId || ''}
                  onChange={(e) => {
                    const found = suggestedConfigs.find((c) => c.id === e.target.value)
                    if (found) selectConfig(found)
                  }}
                >
                  <SelectItem value="" text="Choose a configuration..." />
                  {suggestedConfigs.map((config) => (
                    <SelectItem key={config.id} value={config.id} text={config.name} />
                  ))}
                </Select>
                <div style={{ marginTop: '0.75rem' }}>
                  <Button kind="tertiary" renderIcon={Add} size="sm" onClick={openCreateForm}>Create New Configuration</Button>
                </div>
              </>
            )}
          </Tile>
        </div>

        <div className={styles.rightColumn}>
          {selectedConfig ? (
            <Tile className={styles.configPreviewTile}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <h6 className={styles.tileHeading}>Configuration Preview</h6>
                {selectedConfig.id === '__pending__' && <Tag type="cyan" size="sm">New</Tag>}
                {selectedConfig.id !== '__pending__' && (
                  <div style={{ marginLeft: 'auto' }}>
                    <Button size="sm" kind="primary" renderIcon={Edit} onClick={enterEditMode}>Edit</Button>
                  </div>
                )}
              </div>
              <ConfigDisplay configuration={selectedConfig} />
            </Tile>
          ) : (
            <Tile className={styles.emptyPreview}>
              <div>
                <Settings size={32} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
                <p>Select a configuration to preview</p>
              </div>
            </Tile>
          )}
        </div>
      </div>
    </div>
  )
}
