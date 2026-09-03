'use client'

import { useEffect, useRef, useState } from 'react'
import { NumberInput, Select, SelectItem } from '@carbon/react'
import type { NumberInputColumn } from '../types'
import { toUpperCase } from '../lib/autotunex/wizardUtils'

type TimeUnit = 'seconds' | 'minutes' | 'hours' | 'days'

const CONVERSIONS: Record<TimeUnit, number> = { seconds: 1, minutes: 60, hours: 3600, days: 86400 }

const UNIT_CONSTRAINTS: Record<TimeUnit, { min: number; max: number }> = {
  seconds: { min: 60, max: 1209600 },
  minutes: { min: 1, max: 20160 },
  hours: { min: 1, max: 336 },
  days: { min: 1, max: 14 },
}

function convertToSeconds(value: number, unit: TimeUnit): number {
  return value * CONVERSIONS[unit]
}

function convertFromSeconds(seconds: number, unit: TimeUnit): number {
  return seconds / CONVERSIONS[unit]
}

/** Pick the unit that keeps the displayed value within that unit's typical range. */
function bestUnit(seconds: number): TimeUnit {
  const days = seconds / 86400
  if (days >= 1 && days <= 14) return 'days'
  const hours = seconds / 3600
  if (hours >= 1 && hours <= 336) return 'hours'
  const minutes = seconds / 60
  if (minutes >= 1 && minutes <= 20160) return 'minutes'
  return 'seconds'
}

interface TimeInputProps {
  label?: string
  value: NumberInputColumn
  onChange: (next: NumberInputColumn) => void
}

export function TimeInput({ label = 'Time Budget', value, onChange }: TimeInputProps) {
  const [selectedUnit, setSelectedUnit] = useState<TimeUnit>('hours')
  const [displayedTimeBudget, setDisplayedTimeBudget] = useState<number | null>(null)
  const initialized = useRef(false)
  const previousUnit = useRef<TimeUnit>('hours')

  // Initialize from the incoming value once, picking whichever unit keeps it readable.
  useEffect(() => {
    if (initialized.current) return
    if (value?.default !== null && value?.default !== undefined) {
      const unit = bestUnit(value.default)
      setSelectedUnit(unit)
      setDisplayedTimeBudget(convertFromSeconds(value.default, unit))
      previousUnit.current = unit
    }
    initialized.current = true
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-derive the displayed value when the unit changes (after init).
  useEffect(() => {
    if (!initialized.current || selectedUnit === previousUnit.current) return
    if (value?.default !== null && value?.default !== undefined) {
      setDisplayedTimeBudget(convertFromSeconds(value.default, selectedUnit))
    }
    previousUnit.current = selectedUnit
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUnit])

  const currentMin = UNIT_CONSTRAINTS[selectedUnit].min
  const currentMax = UNIT_CONSTRAINTS[selectedUnit].max
  const isInvalid = displayedTimeBudget !== null && (displayedTimeBudget < currentMin || displayedTimeBudget > currentMax)

  function handleChange(newValue: number | undefined) {
    if (newValue !== null && newValue !== undefined && !Number.isNaN(newValue)) {
      setDisplayedTimeBudget(newValue)
      onChange({ ...value, default: convertToSeconds(newValue, selectedUnit) })
    } else {
      setDisplayedTimeBudget(null)
      onChange({ ...value, default: null })
    }
  }

  return (
    <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'flex-start' }}>
      <div style={{ flex: 1 }}>
        <NumberInput
          id={label}
          hideSteppers
          // An unset time budget (`default: null`) renders as "" and is a valid,
          // expected state — it means "no limit". Carbon React's NumberInput
          // derives its own validity (`invalid: !getInputValidity(...)`) and treats
          // `value === ""` as invalid unless `allowEmpty` is set, which would show a
          // phantom range error on a blank field that our own `isInvalid` never
          // flags. Carbon Svelte honours the `invalid` prop instead, which is why
          // the source form shows no error here.
          allowEmpty
          label={toUpperCase(label) ?? label}
          helperText={value.description?.replace('seconds', selectedUnit)}
          invalid={isInvalid}
          invalidText={
            selectedUnit !== 'days' ? `Value must be between ${currentMin} and ${currentMax} ${selectedUnit}` : `Value must be ${currentMax} ${selectedUnit}`
          }
          min={displayedTimeBudget !== null ? currentMin : undefined}
          max={displayedTimeBudget !== null ? currentMax : undefined}
          step={selectedUnit === 'hours' || selectedUnit === 'days' ? 0.01 : 1}
          value={displayedTimeBudget ?? ''}
          onChange={(_e, { value: newValue }) => handleChange(typeof newValue === 'number' ? newValue : undefined)}
        />
      </div>
      <div style={{ width: 130, paddingTop: '1.475rem' }}>
        <Select id={`${label}-unit`} hideLabel labelText="Unit" value={selectedUnit} onChange={(e) => setSelectedUnit(e.target.value as TimeUnit)}>
          <SelectItem value="seconds" text="seconds" />
          <SelectItem value="minutes" text="minutes" />
          <SelectItem value="hours" text="hours" />
          <SelectItem value="days" text="days" />
        </Select>
      </div>
    </div>
  )
}
