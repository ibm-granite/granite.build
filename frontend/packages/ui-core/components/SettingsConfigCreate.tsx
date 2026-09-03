'use client'

import { useEffect, useMemo, useState } from 'react'
import { Modal, TextInput, Dropdown, InlineLoading, InlineNotification } from '@carbon/react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import type { ConfigForm, ConfigData, Configuration, PendingConfigData, TuningGoal } from '../types'
import { getConfigurationTemplate, createConfiguration, getConfigurations } from '../api/autotunex'
import { ALGORITHM_DETAILS } from '../config/autotunexAlgorithms'
import { CreateConfigForm } from './CreateConfigForm'
import { normalizeTokenizerListFields } from '../lib/autotunex/wizardUtils'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: () => void
}

const ALGORITHM_ITEMS = ALGORITHM_DETAILS.map((a) => ({ id: a.id, label: a.name }))

export function SettingsConfigCreate({ open, onClose, onCreated }: Props) {
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('lora')
  const [configForm, setConfigForm] = useState<ConfigForm | null>(null)
  const [nameError, setNameError] = useState('')

  const presetGoal = useMemo<TuningGoal | null>(
    () => (ALGORITHM_DETAILS.find((a) => a.id === selectedAlgorithm)?.category ?? null) as TuningGoal | null,
    [selectedAlgorithm]
  )

  // Fetched only to check for duplicate names before submit; a generous
  // pageSize keeps this a single request for the (own-scope) common case.
  const { data: configsResult } = useQuery({
    queryKey: ['autotunex', 'configurations', 'for-validation'],
    queryFn: () => getConfigurations({ page: 1, pageSize: 100, scope: 'own' }),
  })
  const configurations = configsResult?.items ?? []

  // Seed the ConfigForm from the backend template when the modal opens.
  const { data: template, isLoading: templateLoading } = useQuery({
    queryKey: ['autotunex', 'configTemplate'],
    queryFn: getConfigurationTemplate,
    enabled: open,
  })

  useEffect(() => {
    if (open && template && configForm == null) {
      setConfigForm({ name: '', tuner_type: '', rl_tuner_type: '', ...(template as ConfigData) } as ConfigForm)
    }
    if (!open) {
      // Reset for next open.
      setConfigForm(null)
      setName('')
      setNameError('')
      setSelectedAlgorithm('lora')
      createMutation.reset()
    }
  }, [open, template, configForm])

  const createMutation = useMutation({
    mutationFn: (payload: PendingConfigData) => createConfiguration(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'configurations'] })
      onCreated()
      onClose()
    },
  })

  function handleSubmit() {
    const trimmed = name.trim()
    if (!trimmed) { setNameError('Please enter a configuration name.'); return }
    if (configurations.some((c: Configuration) => c.name === trimmed)) {
      setNameError(`A configuration named "${trimmed}" already exists.`); return
    }
    if (!configForm) return
    setNameError('')

    const { name: _name, tuner_type, rl_tuner_type, ...configSections } = configForm
    const pendingData: PendingConfigData = {
      name: trimmed,
      tuner_type: presetGoal === 'online_rl' ? null : (tuner_type || 'lora'),
      rl_tuner_type: rl_tuner_type || null,
      config_data: configSections as ConfigData,
    }
    normalizeTokenizerListFields(pendingData.config_data)
    createMutation.mutate(pendingData)
  }

  return (
    <Modal
      open={open}
      size="lg"
      modalHeading="Create New Configuration"
      primaryButtonText={createMutation.isPending ? 'Creating…' : 'Create'}
      secondaryButtonText="Cancel"
      primaryButtonDisabled={createMutation.isPending || templateLoading || !configForm}
      onRequestClose={onClose}
      onRequestSubmit={handleSubmit}
    >
      {templateLoading || !configForm ? (
        <InlineLoading description="Loading configuration template…" />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <TextInput
            id="settings-config-name"
            labelText="Configuration name"
            value={name}
            invalid={!!nameError}
            invalidText={nameError}
            onChange={(e) => setName(e.target.value)}
          />
          <Dropdown
            id="settings-config-algorithm"
            titleText="Algorithm"
            label="Select an algorithm"
            items={ALGORITHM_ITEMS}
            selectedItem={ALGORITHM_ITEMS.find((i) => i.id === selectedAlgorithm) ?? null}
            itemToString={(i) => (i ? i.label : '')}
            onChange={({ selectedItem }) => {
              if (selectedItem) setSelectedAlgorithm(selectedItem.id)
            }}
          />
          <CreateConfigForm
            key={`${presetGoal}-${selectedAlgorithm}`}
            config={configForm}
            setConfig={setConfigForm as any}
            configurations={configurations}
            presetGoal={presetGoal}
            presetAlgorithm={selectedAlgorithm}
            hideNameField
          />
          {createMutation.isError && (
            <InlineNotification
              kind="error"
              title="Failed to create configuration"
              subtitle="Something went wrong. Please try again."
              lowContrast
              hideCloseButton
            />
          )}
        </div>
      )}
    </Modal>
  )
}
