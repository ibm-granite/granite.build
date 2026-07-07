'use client'

import { useState } from 'react'
import { TextInput, Checkbox, Button, InlineNotification, InlineLoading } from '@carbon/react'
import { loadDPReport, type DPReportResult } from '@/api/dataProcessing'
import styles from './page.module.scss'

export function LoadReportForm() {
  const [megatronPath, setMegatronPath] = useState('')
  const [arrowPath, setArrowPath] = useState('')
  const [parquetPath, setParquetPath] = useState('')
  const [includeP1, setIncludeP1] = useState(true)
  const [includeTokens, setIncludeTokens] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DPReportResult | null>(null)

  async function handleLoad() {
    if (!megatronPath.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const report = await loadDPReport({
        megatron_path: megatronPath.trim(),
        arrow_path: arrowPath.trim() || undefined,
        parquet_path: parquetPath.trim() || undefined,
        include_p1: includeP1,
        include_tokens: includeTokens,
      })
      setResult(report)
    } catch {
      setResult({ error: 'Request failed' })
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setMegatronPath('')
    setArrowPath('')
    setParquetPath('')
    setIncludeP1(true)
    setIncludeTokens(false)
    setResult(null)
  }

  const summaryEntries = result?.summary ? Object.entries(result.summary) : []
  const isNotConfigured = result?.error === 'not_configured'
  const isCLIMissing = result?.error === 'megatron_cli_not_available'

  return (
    <div className={styles.reportForm}>
      <div className={styles.reportFormTitle}>Load Report</div>

      <div className={styles.reportFormFields}>
        <div className={styles.reportFormField}>
          <TextInput
            id="megatron-path"
            labelText="Megatron COS Path *"
            placeholder="cos-optimal-llm-pile/path/to/megatron"
            value={megatronPath}
            onChange={(e) => setMegatronPath(e.target.value)}
            size="sm"
          />
        </div>
        <div className={styles.reportFormField}>
          <TextInput
            id="arrow-path"
            labelText="Arrow Path (auto-detected)"
            placeholder="auto-detected if empty"
            value={arrowPath}
            onChange={(e) => setArrowPath(e.target.value)}
            size="sm"
          />
        </div>
        <div className={styles.reportFormField}>
          <TextInput
            id="parquet-path"
            labelText="Parquet Path (auto-detected)"
            placeholder="auto-detected if empty"
            value={parquetPath}
            onChange={(e) => setParquetPath(e.target.value)}
            size="sm"
          />
        </div>
      </div>

      <div className={styles.reportFormOptions}>
        <Checkbox
          id="include-p1"
          labelText="Include P1"
          checked={includeP1}
          onChange={(_, { checked }) => setIncludeP1(checked)}
        />
        <Checkbox
          id="include-tokens"
          labelText="Token Distribution"
          checked={includeTokens}
          onChange={(_, { checked }) => setIncludeTokens(checked)}
        />
      </div>

      <div className={styles.reportFormActions}>
        {loading ? (
          <InlineLoading description="Loading report…" />
        ) : (
          <>
            <Button
              size="sm"
              disabled={!megatronPath.trim()}
              onClick={handleLoad}
            >
              Load Report
            </Button>
            <Button size="sm" kind="ghost" onClick={handleClear}>
              Clear
            </Button>
          </>
        )}
      </div>

      {result && !loading && (
        <div style={{ marginTop: '0.75rem' }}>
          {isNotConfigured && (
            <InlineNotification
              kind="warning"
              title="COS not configured"
              subtitle="Pipeline reports require COS credentials (GB_UI_COS_ENDPOINT not set)."
              lowContrast
            />
          )}
          {isCLIMissing && (
            <InlineNotification
              kind="warning"
              title="Megatron CLI not available"
              subtitle="The vendored megatron data layer is not installed in the server environment."
              lowContrast
            />
          )}
          {result.error && !isNotConfigured && !isCLIMissing && (
            <InlineNotification
              kind="error"
              title="Report failed"
              subtitle={result.error}
              lowContrast
            />
          )}
          {summaryEntries.length > 0 && (
            <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
              {summaryEntries.map(([stage, status]) => (
                <span key={stage} style={{ fontSize: '0.8125rem' }}>
                  <strong>{stage}</strong>{' '}
                  <span style={{ color: status === 'ok' || status === 'complete' ? '#24a148' : '#da1e28' }}>
                    {status === 'ok' || status === 'complete' ? '✓' : '✗'} {status}
                  </span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
