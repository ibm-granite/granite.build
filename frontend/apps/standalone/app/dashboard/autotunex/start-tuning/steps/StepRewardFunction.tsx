'use client'

import { useEffect, useRef, useState } from 'react'
import Editor, { loader } from '@monaco-editor/react'
import {
  Button,
  InlineLoading,
  InlineNotification,
  ContentSwitcher,
  Switch,
  TextArea,
  Tag,
} from '@carbon/react'
import { Play, Add, TrashCan, Checkmark, ListBoxes } from '@carbon/icons-react'

// @monaco-editor/react's default loader pulls Monaco from jsdelivr at runtime, so
// the editor never appears in an air-gapped deployment. Setting MONACO_VS_PATH
// points it at a self-hosted copy of Monaco's `vs` directory instead. Configured
// at module scope so it is applied before the first <Editor> mounts; left alone
// when unset, which keeps the CDN default for ordinary deployments.
if (process.env.MONACO_VS_PATH) {
  loader.config({ paths: { vs: process.env.MONACO_VS_PATH } })
}
import type { ParsedDataRow, RewardFunctionValidationResult } from '@granite-build/ui-core/types'
import { AUTOTUNEX_FEATURES, getDataset, generateTestSolutions, validateRewardFunction } from '@granite-build/ui-core/api/autotunex'
import styles from './StepRewardFunction.module.scss'

const DEFAULT_REWARD_TEMPLATE = `# gsm8k_reward.py
#
# Custom reward function for GSM8K math problems.
# Compatible with verl 0.7.0's NaiveRewardManager which calls:
#   compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
# and expects a scalar float (or dict with "score" key) in return.

import re
from typing import Optional, Union

# -------- parsing helpers --------

_NUMBER_RE = re.compile(r"[-+]?\\d+(?:\\.\\d+)?")


def _normalize_number_str(s) -> Optional[str]:
    """Normalize numeric string: remove commas, strip, and extract a number."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    if _NUMBER_RE.fullmatch(s):
        return s
    m = _NUMBER_RE.search(s)
    return m.group(0) if m else None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from a model response.
    Prefer GSM8K style: '#### <answer>'
    Fallback: last number in the completion.
    """
    if not text:
        return None
    # Prefer #### convention
    m = re.search(r"####\\s*([-+$]?\\d[\\d,]*(?:\\.\\d+)?)", text)
    if m:
        return _normalize_number_str(m.group(1))
    # Fallback: last number
    nums = _NUMBER_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


# -------- core scoring --------

# Hyperparameters
CORRECT_REWARD = 1.0
WRONG_REWARD = -1.0
FORMAT_BONUS = 0.05
LENGTH_PENALTY_COEF = 1.0 / 4000.0
MAX_LENGTH_PENALTY = 0.2


def _score_one(
    response: str,
    gt: Union[str, int, float, None],
) -> float:
    """Score a single response against a ground truth answer."""
    gt_str = _normalize_number_str(gt)
    pred = extract_final_answer(response)

    used_hash = "####" in (response or "")
    bonus = FORMAT_BONUS if used_hash else 0.0

    # Length penalty (small, PPO-stabilizing)
    lp = 0.0
    if response:
        lp = -min(len(response) * LENGTH_PENALTY_COEF, MAX_LENGTH_PENALTY)

    if pred is None or gt_str is None:
        return WRONG_REWARD + bonus + lp

    return (CORRECT_REWARD if pred == gt_str else WRONG_REWARD) + bonus + lp


# -------- entry point for verl 0.7.0 --------

def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Union[str, int, float, None] = None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """
    verl 0.7.0 custom reward function entrypoint.

    Called per-sample by NaiveRewardManager with:
        compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)

    Returns:
        float: scalar reward score for this single sample.
    """
    return _score_one(solution_str, ground_truth)
`

// ─── Test case data model ───
// Each test case stores both structured fields (for Table View) and a JSON
// string (for JSON View). The JSON string is the source of truth for the API.
// When switching modes we sync between the two representations.

interface TestCase {
  id: number
  json: string
  // Table View fields (kept in sync with json)
  data_source: string
  solution_str: string
  ground_truth: string
  // Result fields (populated after validation)
  reward: number | null
  rewardError: string | null
}

function makeTestCase(
  id: number,
  data_source: string,
  solution_str: string,
  ground_truth: string,
  extra: Record<string, any> = {}
): TestCase {
  return {
    id,
    data_source,
    solution_str,
    ground_truth,
    json: JSON.stringify({ data_source, solution_str, ground_truth, extra_info: {}, ...extra }, null, 2),
    reward: null,
    rewardError: null,
  }
}

/** Sync simple fields → JSON string (called on field edits / mode switch). */
function syncFieldsToJson(tc: TestCase): TestCase {
  let parsed: Record<string, any>
  try {
    parsed = JSON.parse(tc.json)
  } catch {
    parsed = { extra_info: {} }
  }
  parsed.data_source = tc.data_source
  parsed.solution_str = tc.solution_str
  parsed.ground_truth = tc.ground_truth
  if (tc.reward !== null) {
    parsed._reward = Number(tc.reward.toFixed(3))
  } else {
    delete parsed._reward
  }
  return { ...tc, json: JSON.stringify(parsed, null, 2) }
}

/** Sync JSON string → simple fields (called when switching to Table View). */
function syncJsonToFields(tc: TestCase): TestCase {
  try {
    const parsed = JSON.parse(tc.json)
    return {
      ...tc,
      data_source: typeof parsed.data_source === 'string' ? parsed.data_source : '',
      solution_str: typeof parsed.solution_str === 'string' ? parsed.solution_str : '',
      ground_truth: parsed.ground_truth != null ? String(parsed.ground_truth) : '',
    }
  } catch {
    // Leave fields as-is if JSON is invalid
    return tc
  }
}

const STANDARD_KEYS = new Set(['data_source', 'solution_str', 'ground_truth', 'extra_info', '_reward'])

function hasExtraKeys(tc: TestCase): string[] {
  try {
    const parsed = JSON.parse(tc.json)
    return Object.keys(parsed).filter((k) => !STANDARD_KEYS.has(k))
  } catch {
    return []
  }
}

function makeWrongAnswer(groundTruth: string): string {
  const num = parseFloat(groundTruth)
  if (!isNaN(num)) {
    return `The answer is #### ${num + 10}`
  }
  return 'I am not sure about the answer.'
}

/** Build up to 5 test cases (3 positive + 2 deliberately-wrong) from dataset/parsed rows. */
async function buildTestCasesFromRows(allRows: ParsedDataRow[]): Promise<TestCase[]> {
  if (allRows.length === 0) {
    return [makeTestCase(1, '', '', '')]
  }

  const rows = allRows.slice(0, 5)
  const positiveCount = Math.min(3, rows.length)
  const negativeCount = Math.min(2, rows.length - positiveCount)
  const positiveRows = rows.slice(0, positiveCount)
  const negativeRows = rows.slice(positiveCount, positiveCount + negativeCount)

  let llmSolutions: string[] = []
  try {
    const prompts = positiveRows
      .map((row) => row.prompt)
      .filter((p): p is any[] => Array.isArray(p) && p.length > 0)
    if (prompts.length > 0) {
      const result = await generateTestSolutions(prompts)
      llmSolutions = result && 'solutions' in result ? result.solutions : []
    }
  } catch {
    // LLM failed — will fall back to placeholders below
  }

  const cases: TestCase[] = []

  positiveRows.forEach((row, i) => {
    const dataSource = row.data_source || ''
    const groundTruth = row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : ''
    const solutionStr = llmSolutions[i] || (groundTruth ? `The answer is #### ${groundTruth}` : '')
    const extraInfo = row.extra_info || {}
    cases.push(makeTestCase(i + 1, dataSource, solutionStr, groundTruth, { extra_info: extraInfo }))
  })

  negativeRows.forEach((row, i) => {
    const dataSource = row.data_source || ''
    const groundTruth = row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : ''
    const solutionStr = makeWrongAnswer(groundTruth)
    const extraInfo = row.extra_info || {}
    cases.push(makeTestCase(positiveCount + i + 1, dataSource, solutionStr, groundTruth, { extra_info: extraInfo }))
  })

  return cases
}

interface StepRewardFunctionProps {
  rewardFunctionCode: string
  setRewardFunctionCode: (v: string) => void
  rewardFunctionName: string
  setRewardFunctionName: (v: string) => void
  allTestsPassed: boolean
  setAllTestsPassed: (v: boolean) => void
  datasetId: string | null
  parsedData: ParsedDataRow[]
}

export function StepRewardFunction({
  rewardFunctionCode,
  setRewardFunctionCode,
  rewardFunctionName,
  setRewardFunctionName,
  allTestsPassed,
  setAllTestsPassed,
  datasetId,
  parsedData,
}: StepRewardFunctionProps) {
  const [isValidating, setIsValidating] = useState(false)
  const [validationResult, setValidationResult] = useState<RewardFunctionValidationResult | null>(null)
  const [showTestPanel, setShowTestPanel] = useState(false)

  // Live reward-function validation has no v0.3.5 backend equivalent yet
  // (see AUTOTUNEX_FEATURES.rewardValidation) — this is a fixed build-time
  // flag, so a plain const (not state) is enough to gate the UI.
  const validationUnavailable = !AUTOTUNEX_FEATURES.rewardValidation

  // View mode: 0 = Table View, 1 = JSON View
  const [viewModeIndex, setViewModeIndex] = useState(0)
  const advancedTestMode = viewModeIndex === 1
  const [modeSwitchWarning, setModeSwitchWarning] = useState<string | null>(null)

  const [testCases, setTestCases] = useState<TestCase[]>([])
  const nextIdRef = useRef(1)
  const [isLoadingDataset, setIsLoadingDataset] = useState(false)

  const testCasesScrollRef = useRef<HTMLDivElement>(null)

  // Initialize with defaults if the parent hasn't set them yet (mirrors the
  // Svelte prop defaults `rewardFunctionCode = DEFAULT_REWARD_TEMPLATE` /
  // `rewardFunctionName = 'compute_score'`).
  useEffect(() => {
    if (!rewardFunctionCode || rewardFunctionCode.trim() === '') {
      setRewardFunctionCode(DEFAULT_REWARD_TEMPLATE)
    }
    if (!rewardFunctionName || rewardFunctionName.trim() === '') {
      setRewardFunctionName('compute_score')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Load test cases from dataset, parsed data, or fall back to a single blank
  // case — once, on arrival to this step (mirrors the Svelte `datasetLoaded`
  // guard, which likewise never re-runs once one of these has fired).
  const hasLoadedTestCasesRef = useRef(false)
  useEffect(() => {
    if (hasLoadedTestCasesRef.current) return
    hasLoadedTestCasesRef.current = true

    if (datasetId) {
      setIsLoadingDataset(true)
      getDataset(datasetId, { preview: true, previewRows: 50 })
        .then((dataset) => buildTestCasesFromRows(dataset?.preview?.train || []))
        .then((cases) => {
          setTestCases(cases)
          nextIdRef.current = cases.length + 1
        })
        .catch(() => {
          setTestCases([makeTestCase(1, '', '', '')])
          nextIdRef.current = 2
        })
        .finally(() => setIsLoadingDataset(false))
    } else if (parsedData.length > 0) {
      setIsLoadingDataset(true)
      buildTestCasesFromRows(parsedData)
        .then((cases) => {
          setTestCases(cases)
          nextIdRef.current = cases.length + 1
        })
        .catch(() => {
          setTestCases([makeTestCase(1, '', '', '')])
          nextIdRef.current = 2
        })
        .finally(() => setIsLoadingDataset(false))
    } else {
      setTestCases([makeTestCase(1, '', '', '')])
      nextIdRef.current = 2
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function addTestCase() {
    setTestCases((prev) => [...prev, makeTestCase(nextIdRef.current++, '', '', '')])
    requestAnimationFrame(() => {
      const el = testCasesScrollRef.current
      if (el) el.scrollTop = el.scrollHeight
    })
  }

  function removeTestCase(id: number) {
    setTestCases((prev) => (prev.length > 1 ? prev.filter((tc) => tc.id !== id) : prev))
  }

  function updateTestCaseField(id: number, field: 'data_source' | 'solution_str' | 'ground_truth' | 'json', value: string) {
    setTestCases((prev) => prev.map((tc) => (tc.id === id ? { ...tc, [field]: value } : tc)))
  }

  function parseTestCases(): Record<string, any>[] | null {
    // Sync simple fields into JSON before parsing (no-op in JSON View mode)
    const source = advancedTestMode ? testCases : testCases.map(syncFieldsToJson)
    if (!advancedTestMode) setTestCases(source)
    try {
      return source.map((tc) => {
        const parsed = JSON.parse(tc.json)
        delete parsed._reward // Strip computed field before sending to API
        return parsed
      })
    } catch {
      return null
    }
  }

  // ─── Simple/Advanced mode helpers ───

  function canSwitchToSimple(): { ok: boolean; error?: string } {
    for (let i = 0; i < testCases.length; i++) {
      try {
        JSON.parse(testCases[i].json)
      } catch {
        return { ok: false, error: `Case ${i + 1} has invalid JSON. Fix it in JSON mode first.` }
      }
    }
    return { ok: true }
  }

  function handleViewModeChange(newIndex: number) {
    setModeSwitchWarning(null)

    if (newIndex === 0) {
      // Switching to Table View
      const check = canSwitchToSimple()
      if (!check.ok) {
        setModeSwitchWarning(check.error ?? null)
        setViewModeIndex(1) // Stay on JSON View
        return
      }
      const synced = testCases.map(syncJsonToFields)
      setTestCases(synced)
      setViewModeIndex(0)

      const allExtras = synced.flatMap((tc, i) => {
        const extras = hasExtraKeys(tc)
        return extras.length > 0 ? [`Case ${i + 1}: ${extras.join(', ')}`] : []
      })
      if (allExtras.length > 0) {
        setModeSwitchWarning(`Custom fields preserved but hidden: ${allExtras.join('; ')}. Use JSON View to edit them.`)
      }
    } else {
      // Switching to JSON View — sync simple fields → JSON first
      setTestCases(testCases.map(syncFieldsToJson))
      setViewModeIndex(1)
    }
  }

  // ─── Validation ───

  async function validateCode(runTest: boolean = false) {
    // Live validation has no v0.3.5 backend equivalent yet (see
    // AUTOTUNEX_FEATURES.rewardValidation) — no-op rather than a failed
    // network call, so this can be safely wired to the editor's blur handler.
    if (!AUTOTUNEX_FEATURES.rewardValidation) return

    setIsValidating(true)
    setValidationResult(null)

    let testInputs: Record<string, any>[] | undefined = undefined
    if (runTest) {
      const parsed = parseTestCases()
      if (!parsed) {
        setValidationResult({
          success: false,
          syntax_errors: [],
          security_issues: [],
          validation: { syntax_valid: true, security_valid: true, function_found: true, function_signature_valid: true },
          test_result: {
            executed: false,
            error: 'Invalid JSON in test cases. Please check your test input syntax.',
            results: [],
          },
        })
        setIsValidating(false)
        return
      }
      testInputs = parsed
    }

    try {
      const result = await validateRewardFunction(rewardFunctionCode, rewardFunctionName, runTest, testInputs)
      setValidationResult('unavailable' in result ? null : result)
    } catch {
      setValidationResult({
        success: false,
        syntax_errors: ['Failed to reach validation server'],
        security_issues: [],
        validation: { syntax_valid: false, security_valid: false, function_found: false, function_signature_valid: false },
        test_result: null,
      })
    }
    setIsValidating(false)
  }

  // Kept fresh every render so the Monaco `onDidBlurEditorText` listener
  // (attached once, on mount) always calls the latest validateCode/state.
  const validateCodeRef = useRef(validateCode)
  const rewardFunctionCodeRef = useRef(rewardFunctionCode)
  const isValidatingRef = useRef(isValidating)
  useEffect(() => {
    validateCodeRef.current = validateCode
    rewardFunctionCodeRef.current = rewardFunctionCode
    isValidatingRef.current = isValidating
  })

  // Track whether all test cases passed (drives the parent's Next-button gate)
  useEffect(() => {
    const passed =
      validationResult?.success === true &&
      validationResult?.test_result?.executed === true &&
      (validationResult?.test_result?.results?.length ?? 0) > 0 &&
      (validationResult?.test_result?.results ?? []).every((r) => !r.error)
    setAllTestsPassed(passed)
  }, [validationResult, setAllTestsPassed])

  // Per-case runtime errors (e.g. a NameError raised inside the reward function)
  const testCaseErrors =
    validationResult?.success === true &&
    validationResult?.test_result?.executed === true &&
    Array.isArray(validationResult?.test_result?.results)
      ? validationResult.test_result.results
          .map((r, i) => (r?.error ? { index: i + 1, error: String(r.error) } : null))
          .filter((x): x is { index: number; error: string } => x !== null)
      : []
  const hasTestCaseErrors = testCaseErrors.length > 0
  const uniqueTestCaseErrors = Array.from(new Set(testCaseErrors.map((e) => e.error)))

  // Reset validation when code or function name changes
  const prevCodeRef = useRef(rewardFunctionCode)
  const prevNameRef = useRef(rewardFunctionName)
  useEffect(() => {
    if (rewardFunctionCode !== prevCodeRef.current || rewardFunctionName !== prevNameRef.current) {
      setValidationResult(null)
      prevCodeRef.current = rewardFunctionCode
      prevNameRef.current = rewardFunctionName
    }
  }, [rewardFunctionCode, rewardFunctionName])

  // Distribute validation results into individual test cases (and clear
  // rewards once validation is reset).
  useEffect(() => {
    const results = validationResult?.test_result?.results
    if (results) {
      setTestCases((prev) => {
        const next = prev.map((tc, i) => {
          const result = results[i]
          const updated: TestCase = result
            ? { ...tc, reward: result.error ? null : result.return_value ?? null, rewardError: result.error || null }
            : { ...tc, reward: null, rewardError: null }
          return updated
        })
        return advancedTestMode ? next.map(syncFieldsToJson) : next
      })
    } else {
      setTestCases((prev) =>
        prev.every((tc) => tc.reward === null && tc.rewardError === null)
          ? prev
          : prev.map((tc) => ({ ...tc, reward: null, rewardError: null }))
      )
    }
  }, [validationResult, advancedTestMode])

  const hasValidationError = validationResult != null && !validationResult.success

  return (
    <div>
          <div className={styles.stepHeader}>
            <div>
              <h4 className={styles.stepTitle}>Define Reward Function</h4>
              <p className={styles.stepSubtitle}>
                Write a Python function that scores model responses during online RL training.
              </p>
            </div>
          </div>

      {/* Code editor (full width) */}
      <div style={{ marginTop: '1rem' }}>
          <div className={styles.editorWrapper}>
            <div className={styles.editorTabBar}>
              <span className={styles.editorTab}>reward_function.py</span>
            </div>
            <div className={styles.codeEditorContainer}>
              <Editor
                height="560px"
                language="python"
                theme="vs-dark"
                value={rewardFunctionCode}
                onChange={(value) => setRewardFunctionCode(value ?? '')}
                onMount={(editorInstance) => {
                  editorInstance.onDidBlurEditorText(() => {
                    if (rewardFunctionCodeRef.current.trim().length > 0 && !isValidatingRef.current) {
                      validateCodeRef.current(false)
                    }
                  })
                }}
                options={{
                  minimap: { enabled: false },
                  fontSize: 13,
                  fontFamily: "'IBM Plex Mono', monospace",
                  lineHeight: 21,
                  padding: { top: 16, bottom: 16 },
                  scrollBeyondLastLine: false,
                  automaticLayout: true,
                  tabSize: 4,
                  wordWrap: 'off',
                  renderLineHighlight: 'line',
                  cursorBlinking: 'smooth',
                  smoothScrolling: true,
                  fixedOverflowWidgets: true,
                }}
              />
            </div>
          </div>

          {/* Action bar below editor */}
          <div className={styles.actionBar}>
            <div className={styles.actionBarLeft}>
              {!hasValidationError && !hasTestCaseErrors && (
                <Button
                  kind={showTestPanel ? 'ghost' : 'secondary'}
                  size="sm"
                  renderIcon={ListBoxes}
                  disabled={isValidating || isLoadingDataset || rewardFunctionCode.trim().length === 0}
                  onClick={() => setShowTestPanel((v) => !v)}
                >
                  {showTestPanel ? 'Hide Test Cases' : 'Show Test Cases'}
                </Button>
              )}
              {!hasValidationError && (
                <Button
                  kind="primary"
                  size="sm"
                  renderIcon={Play}
                  disabled={isValidating || isLoadingDataset || rewardFunctionCode.trim().length === 0 || validationUnavailable}
                  onClick={() => validateCode(true)}
                >
                  Run
                </Button>
              )}
            </div>
            <div className={styles.actionBarRight}>
              {isLoadingDataset ? (
                <InlineLoading description="Generating test cases..." />
              ) : isValidating ? (
                <InlineLoading description="Validating..." />
              ) : null}
              {allTestsPassed && (
                <Tag type="green" size="sm" renderIcon={Checkmark}>
                  Tests passed
                </Tag>
              )}
            </div>
          </div>

          {validationUnavailable && (
            <div style={{ marginTop: '0.5rem' }}>
              <InlineNotification
                kind="info"
                title="Reward function validation unavailable"
                subtitle="Live validation and test execution are temporarily unavailable. You can still write your reward function and continue — it will run when your tuning job launches."
                hideCloseButton
                lowContrast
              />
            </div>
          )}
      </div>

      {/* Test cases panel (below editor, hidden on validation error or per-case runtime error) */}
      {showTestPanel && !hasValidationError && !hasTestCaseErrors && (
        <div>
            <div
              className={styles.testPanel}
              role="region"
              aria-label="Test cases"
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                  e.preventDefault()
                  validateCode(true)
                }
              }}
            >
              <div className={styles.testPanelHeader}>
                <div className={styles.testPanelHeaderRow}>
                  <p className={styles.testPanelTitle}>Test Cases</p>
                  <div className={styles.viewModeSwitcher}>
                    <ContentSwitcher
                      size="sm"
                      selectedIndex={viewModeIndex}
                      onChange={({ index }) => handleViewModeChange(index ?? 0)}
                    >
                      <Switch name="table" text="Table View" />
                      <Switch name="json" text="JSON View" />
                    </ContentSwitcher>
                  </div>
                </div>
                <p className={styles.testPanelDesc}>
                  {advancedTestMode ? 'JSON kwargs passed to your function.' : "Fill in the fields your function expects."}
                </p>
              </div>

              <div className={styles.testCasesScroll} ref={testCasesScrollRef}>
                {modeSwitchWarning && (
                  <InlineNotification
                    kind={modeSwitchWarning.includes('invalid JSON') ? 'error' : 'warning'}
                    title=""
                    subtitle={modeSwitchWarning}
                    hideCloseButton={false}
                    lowContrast
                    onClose={() => setModeSwitchWarning(null)}
                  />
                )}

                {testCases.map((testCase, i) => (
                  <div className={styles.testCaseBlock} key={testCase.id}>
                    <div className={styles.testCaseTopRow}>
                      <span className={styles.testCaseLabel}>Case {i + 1}</span>
                      {testCases.length > 1 && (
                        <button
                          type="button"
                          className={styles.testCaseRemove}
                          onClick={() => removeTestCase(testCase.id)}
                          title="Remove test case"
                        >
                          <TrashCan size={16} />
                        </button>
                      )}
                    </div>

                    {advancedTestMode ? (
                      <textarea
                        className={styles.testCaseInput}
                        value={testCase.json}
                        spellCheck={false}
                        rows={6}
                        placeholder={'{\n  "data_source": "...",\n  "solution_str": "..."\n}'}
                        onChange={(e) => updateTestCaseField(testCase.id, 'json', e.target.value)}
                      />
                    ) : (
                      <div className={styles.simpleFields}>
                        <TextArea
                          id={`test-case-${testCase.id}-ground-truth`}
                          labelText="Expected answer (ground_truth)"
                          rows={2}
                          value={testCase.ground_truth}
                          placeholder="Ground truth value"
                          onChange={(e) => updateTestCaseField(testCase.id, 'ground_truth', e.target.value)}
                        />
                        <TextArea
                          id={`test-case-${testCase.id}-solution`}
                          labelText="Model response (solution_str)"
                          rows={2}
                          value={testCase.solution_str}
                          placeholder="The model's generated response..."
                          onChange={(e) => updateTestCaseField(testCase.id, 'solution_str', e.target.value)}
                        />
                        {/* Inline reward result (Table View) */}
                        {testCase.reward !== null ? (
                          <div className={styles.inlineReward}>
                            <span className={styles.inlineRewardLabel}>Reward</span>
                            <code className={styles.inlineRewardValue}>{testCase.reward.toFixed(3)}</code>
                          </div>
                        ) : testCase.rewardError ? (
                          <div className={`${styles.inlineReward} ${styles.inlineRewardError}`}>
                            <span className={styles.inlineRewardLabel}>Error</span>
                            <span className={styles.inlineRewardErrorText}>{testCase.rewardError}</span>
                          </div>
                        ) : null}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className={styles.testPanelFooter}>
                <Button kind="ghost" size="sm" renderIcon={Add} onClick={addTestCase}>
                  Add Case
                </Button>
              </div>
            </div>
      </div>
      )}

      {/* Per-case runtime errors (code validated OK, but raised at exec time) */}
      {hasTestCaseErrors && validationResult?.test_result && (
        <div style={{ marginTop: '0.5rem' }}>
            {uniqueTestCaseErrors.length === 1 ? (
              <InlineNotification
                kind="error"
                title="Test Execution Error"
                subtitle={`${uniqueTestCaseErrors[0]} (affected ${testCaseErrors.length} of ${validationResult.test_result.results.length} cases)`}
                hideCloseButton
                lowContrast
              />
            ) : (
              <InlineNotification
                kind="error"
                title="Test Execution Errors"
                subtitle={testCaseErrors.map((e) => `Case ${e.index}: ${e.error}`).join(' • ')}
                hideCloseButton
                lowContrast
              />
            )}
      </div>
      )}

      {/* Validation errors only */}
      {hasValidationError && validationResult && (
        <div style={{ marginTop: '0.5rem' }}>
            {validationResult.syntax_errors?.length > 0 && (
              <InlineNotification
                kind="error"
                title="Syntax Error"
                subtitle={validationResult.syntax_errors.join('; ')}
                hideCloseButton
                lowContrast
              />
            )}
            {validationResult.security_issues?.length > 0 && (
              <InlineNotification
                kind="error"
                title="Security Issues"
                subtitle={validationResult.security_issues.join('; ')}
                hideCloseButton
                lowContrast
                style={{ marginTop: '0.5rem' }}
              />
            )}
            {validationResult.test_result?.error && (
              <InlineNotification
                kind="error"
                title="Test Execution Error"
                subtitle={validationResult.test_result.error}
                hideCloseButton
                lowContrast
                style={{ marginTop: '0.5rem' }}
              />
            )}
            {!validationResult.validation?.function_found ? (
              <InlineNotification
                kind="warning"
                title="Function not found"
                subtitle={`Function '${rewardFunctionName}' was not found in the code.`}
                hideCloseButton
                lowContrast
                style={{ marginTop: '0.5rem' }}
              />
            ) : !validationResult.validation?.function_signature_valid ? (
              <InlineNotification
                kind="warning"
                title="Invalid signature"
                subtitle="The function should accept at least 2 parameters: (data_source, solution_str)"
                hideCloseButton
                lowContrast
                style={{ marginTop: '0.5rem' }}
              />
            ) : null}
      </div>
      )}

      {/* stdout output from test (only when validation succeeded) */}
      {validationResult?.success && validationResult?.test_result?.stdout && (
        <div style={{ marginTop: '0.5rem' }}>
            <div className={styles.stdoutWrapper}>
              <span className={styles.stdoutLabel}>stdout</span>
              <pre className={styles.stdoutOutput}>{validationResult.test_result.stdout}</pre>
            </div>
        </div>
      )}
    </div>
  )
}
