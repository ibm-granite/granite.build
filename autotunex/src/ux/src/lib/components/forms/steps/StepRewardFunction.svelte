<!-- Copyright IBM Corp. 2024-2026 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<script lang="ts">
	import {
		InlineNotification,
		Grid,
		Row,
		Column,
		Button,
		InlineLoading,
		ContentSwitcher,
		Switch,
		TextInput,
		Tag,
		TextArea
	} from 'carbon-components-svelte';
	import { Play, Add, TrashCan, Checkmark, ListBoxes } from 'carbon-icons-svelte';
	import { onMount, onDestroy } from 'svelte';
	import { slide } from 'svelte/transition';
	import loader from '@monaco-editor/loader';
	import type * as Monaco from 'monaco-editor';
	import { API } from '$lib/api';

	const api = new API();

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
`;

	export let rewardFunctionCode: string = '';
	export let rewardFunctionName: string = 'compute_score';
	export let allTestsPassed: boolean = false;
	export let datasetId: string | null = null;
	export let parsedData: Record<string, any>[] = [];

	// Initialize with template if empty
	if (!rewardFunctionCode || rewardFunctionCode.trim() === '') {
		rewardFunctionCode = DEFAULT_REWARD_TEMPLATE;
	}

	$: isValid = rewardFunctionCode.trim().length > 0 && rewardFunctionName.trim().length > 0;

	// Monaco editor state
	let editorContainer: HTMLDivElement = null!;
	let editor: Monaco.editor.IStandaloneCodeEditor | null = null;
	let isUpdatingFromEditor = false;
	let isUpdatingFromProp = false;
	let resizeObserver: ResizeObserver | null = null;

	onMount(() => {
		loader.init().then((monaco) => {
			if (!editorContainer) return;

			editor = monaco.editor.create(editorContainer, {
				value: rewardFunctionCode,
				language: 'python',
				theme: 'vs-dark',
				minimap: { enabled: false },
				fontSize: 13,
				fontFamily: "'IBM Plex Mono', monospace",
				lineHeight: 21,
				padding: { top: 16, bottom: 16 },
				scrollBeyondLastLine: false,
				automaticLayout: false,
				tabSize: 4,
				wordWrap: 'off',
				renderLineHighlight: 'line',
				cursorBlinking: 'smooth',
				smoothScrolling: true,
				fixedOverflowWidgets: true
			});

			editor!.onDidChangeModelContent(() => {
				if (isUpdatingFromProp) return;
				isUpdatingFromEditor = true;
				rewardFunctionCode = editor!.getValue();
				isUpdatingFromEditor = false;
			});

			editor!.onDidBlurEditorText(() => {
				if (rewardFunctionCode.trim().length > 0 && !isValidating) {
					validateCode(false);
				}
			});

			resizeObserver = new ResizeObserver(() => {
				if (editor) editor.layout();
			});
			resizeObserver.observe(editorContainer);
		});
	});

	onDestroy(() => {
		resizeObserver?.disconnect();
		editor?.dispose();
		editor = null;
	});

	// Sync external prop changes into the editor
	$: if (editor && !isUpdatingFromEditor) {
		const current = editor.getValue();
		if (rewardFunctionCode !== current) {
			isUpdatingFromProp = true;
			editor.setValue(rewardFunctionCode);
			isUpdatingFromProp = false;
		}
	}

	// Validation state
	let isValidating = false;
	let validationResult: any = null;
	let showTestPanel = false;

	// View mode: 0 = Table View, 1 = JSON View
	let viewModeIndex = 0;
	$: advancedTestMode = viewModeIndex === 1;
	let modeSwitchWarning: string | null = null;

	// ─── Test case data model ───
	// Each test case stores both structured fields (for Table View) and a JSON
	// string (for JSON View). The JSON string is the source of truth for the API.
	// When switching modes we sync between the two representations.

	interface TestCase {
		id: number;
		json: string;
		// Table View fields (kept in sync with json)
		data_source: string;
		solution_str: string;
		ground_truth: string;
		// Result fields (populated after validation)
		reward: number | null;
		rewardError: string | null;
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
			json: JSON.stringify(
				{ data_source, solution_str, ground_truth, extra_info: {}, ...extra },
				null,
				2
			),
			reward: null,
			rewardError: null
		};
	}

	let testCases: TestCase[] = [];
	let nextId = 1;
	let isLoadingDataset = false;
	let datasetLoaded = false;

	// Load test cases from dataset, parsed data, or fall back to empty
	$: if (datasetId && !datasetLoaded) {
		loadDatasetTestCases(datasetId);
	}
	$: if (!datasetId && parsedData.length > 0 && !datasetLoaded) {
		loadParsedDataTestCases(parsedData);
	}
	$: if (!datasetId && parsedData.length === 0 && !datasetLoaded) {
		testCases = [makeTestCase(1, '', '', '')];
		nextId = 2;
		datasetLoaded = true;
	}

	function makeWrongAnswer(groundTruth: string): string {
		const num = parseFloat(groundTruth);
		if (!isNaN(num)) {
			return `The answer is #### ${num + 10}`;
		}
		return 'I am not sure about the answer.';
	}

	async function loadDatasetTestCases(id: string) {
		isLoadingDataset = true;
		try {
			const dataset = await api.getDataset(id);
			const allRows = dataset?.train_data || [];
			if (allRows.length === 0) {
				testCases = [makeTestCase(1, '', '', '')];
				nextId = 2;
				return;
			}

			// Take up to 5 rows, split into positive (first 3) and negative (last 2)
			const rows = allRows.slice(0, 5);
			const positiveCount = Math.min(3, rows.length);
			const negativeCount = Math.min(2, rows.length - positiveCount);
			const positiveRows = rows.slice(0, positiveCount);
			const negativeRows = rows.slice(positiveCount, positiveCount + negativeCount);

			// Generate LLM solutions for positive cases
			let llmSolutions: string[] = [];
			try {
				const prompts = positiveRows
					.map((row: Record<string, any>) => row.prompt)
					.filter((p: any) => Array.isArray(p) && p.length > 0);
				if (prompts.length > 0) {
					const result = await api.generateTestSolutions(prompts);
					llmSolutions = result?.solutions || [];
				}
			} catch {
				// LLM failed — will fall back to placeholders below
			}

			const cases: TestCase[] = [];

			// Positive cases (LLM-generated solution_str)
			positiveRows.forEach((row: Record<string, any>, i: number) => {
				const dataSource = row.data_source || '';
				const groundTruth =
					row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : '';
				const solutionStr =
					llmSolutions[i] || (groundTruth ? `The answer is #### ${groundTruth}` : '');
				const extraInfo = row.extra_info || {};
				cases.push(
					makeTestCase(i + 1, dataSource, solutionStr, groundTruth, { extra_info: extraInfo })
				);
			});

			// Negative cases (deliberately wrong solution_str)
			negativeRows.forEach((row: Record<string, any>, i: number) => {
				const dataSource = row.data_source || '';
				const groundTruth =
					row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : '';
				const solutionStr = makeWrongAnswer(groundTruth);
				const extraInfo = row.extra_info || {};
				cases.push(
					makeTestCase(positiveCount + i + 1, dataSource, solutionStr, groundTruth, {
						extra_info: extraInfo
					})
				);
			});

			testCases = cases;
			nextId = cases.length + 1;
		} catch {
			testCases = [makeTestCase(1, '', '', '')];
			nextId = 2;
		} finally {
			isLoadingDataset = false;
			datasetLoaded = true;
		}
	}

	async function loadParsedDataTestCases(data: Record<string, any>[]) {
		isLoadingDataset = true;
		try {
			const allRows = data;
			if (allRows.length === 0) {
				testCases = [makeTestCase(1, '', '', '')];
				nextId = 2;
				return;
			}

			const rows = allRows.slice(0, 5);
			const positiveCount = Math.min(3, rows.length);
			const negativeCount = Math.min(2, rows.length - positiveCount);
			const positiveRows = rows.slice(0, positiveCount);
			const negativeRows = rows.slice(positiveCount, positiveCount + negativeCount);

			let llmSolutions: string[] = [];
			try {
				const prompts = positiveRows
					.map((row: Record<string, any>) => row.prompt)
					.filter((p: any) => Array.isArray(p) && p.length > 0);
				if (prompts.length > 0) {
					const result = await api.generateTestSolutions(prompts);
					llmSolutions = result?.solutions || [];
				}
			} catch {
				// LLM failed — will fall back to placeholders below
			}

			const cases: TestCase[] = [];

			positiveRows.forEach((row: Record<string, any>, i: number) => {
				const dataSource = row.data_source || '';
				const groundTruth =
					row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : '';
				const solutionStr =
					llmSolutions[i] || (groundTruth ? `The answer is #### ${groundTruth}` : '');
				const extraInfo = row.extra_info || {};
				cases.push(
					makeTestCase(i + 1, dataSource, solutionStr, groundTruth, { extra_info: extraInfo })
				);
			});

			negativeRows.forEach((row: Record<string, any>, i: number) => {
				const dataSource = row.data_source || '';
				const groundTruth =
					row.reward_model?.ground_truth != null ? String(row.reward_model.ground_truth) : '';
				const solutionStr = makeWrongAnswer(groundTruth);
				const extraInfo = row.extra_info || {};
				cases.push(
					makeTestCase(positiveCount + i + 1, dataSource, solutionStr, groundTruth, {
						extra_info: extraInfo
					})
				);
			});

			testCases = cases;
			nextId = cases.length + 1;
		} catch {
			testCases = [makeTestCase(1, '', '', '')];
			nextId = 2;
		} finally {
			isLoadingDataset = false;
			datasetLoaded = true;
		}
	}

	/** Sync simple fields → JSON string (called reactively on field changes). */
	function syncFieldsToJson(tc: TestCase) {
		let parsed: Record<string, any>;
		try {
			parsed = JSON.parse(tc.json);
		} catch {
			parsed = { extra_info: {} };
		}
		parsed.data_source = tc.data_source;
		parsed.solution_str = tc.solution_str;
		parsed.ground_truth = tc.ground_truth;
		// Inject reward result if available
		if (tc.reward !== null) {
			parsed._reward = Number(tc.reward.toFixed(3));
		} else {
			delete parsed._reward;
		}
		tc.json = JSON.stringify(parsed, null, 2);
	}

	/** Sync JSON string → simple fields (called when switching to simple mode). */
	function syncJsonToFields(tc: TestCase) {
		try {
			const parsed = JSON.parse(tc.json);
			tc.data_source = typeof parsed.data_source === 'string' ? parsed.data_source : '';
			tc.solution_str = typeof parsed.solution_str === 'string' ? parsed.solution_str : '';
			tc.ground_truth = parsed.ground_truth != null ? String(parsed.ground_truth) : '';
		} catch {
			// Leave fields as-is if JSON is invalid
		}
	}

	function addTestCase() {
		testCases = [...testCases, makeTestCase(nextId++, '', '', '')];
		requestAnimationFrame(() => {
			const el = document.querySelector('.test-cases-scroll');
			if (el) el.scrollTop = el.scrollHeight;
		});
	}

	function removeTestCase(id: number) {
		if (testCases.length > 1) {
			testCases = testCases.filter((tc) => tc.id !== id);
		}
	}

	function parseTestCases(): Record<string, any>[] | null {
		// Sync simple fields into JSON before parsing (no-op in JSON View mode)
		if (!advancedTestMode) {
			testCases.forEach(syncFieldsToJson);
		}
		try {
			return testCases.map((tc) => {
				const parsed = JSON.parse(tc.json);
				delete parsed._reward; // Strip computed field before sending to API
				return parsed;
			});
		} catch (e) {
			return null;
		}
	}

	// ─── Simple/Advanced mode helpers ───

	const STANDARD_KEYS = new Set([
		'data_source',
		'solution_str',
		'ground_truth',
		'extra_info',
		'_reward'
	]);

	function hasExtraKeys(tc: TestCase): string[] {
		try {
			const parsed = JSON.parse(tc.json);
			return Object.keys(parsed).filter((k) => !STANDARD_KEYS.has(k));
		} catch {
			return [];
		}
	}

	function canSwitchToSimple(): { ok: boolean; error?: string } {
		for (let i = 0; i < testCases.length; i++) {
			try {
				JSON.parse(testCases[i].json);
			} catch {
				return { ok: false, error: `Case ${i + 1} has invalid JSON. Fix it in JSON mode first.` };
			}
		}
		return { ok: true };
	}

	function handleViewModeChange() {
		modeSwitchWarning = null;

		if (viewModeIndex === 0) {
			// Switching to Table View
			const check = canSwitchToSimple();
			if (!check.ok) {
				modeSwitchWarning = check.error!;
				viewModeIndex = 1; // Stay on JSON View
				return;
			}
			testCases.forEach(syncJsonToFields);
			testCases = testCases;

			const allExtras = testCases.flatMap((tc, i) => {
				const extras = hasExtraKeys(tc);
				return extras.length > 0 ? [`Case ${i + 1}: ${extras.join(', ')}`] : [];
			});
			if (allExtras.length > 0) {
				modeSwitchWarning = `Custom fields preserved but hidden: ${allExtras.join(
					'; '
				)}. Use JSON View to edit them.`;
			}
		} else {
			// Switching to JSON View — sync simple fields → JSON first
			testCases.forEach(syncFieldsToJson);
			testCases = testCases;
		}
	}

	// ─── Validation ───

	async function validateCode(runTest: boolean = false) {
		isValidating = true;
		validationResult = null;

		let testInputs: Record<string, any>[] | undefined = undefined;
		if (runTest) {
			const parsed = parseTestCases();
			if (!parsed) {
				validationResult = {
					success: false,
					syntax_errors: [],
					security_issues: [],
					validation: {
						syntax_valid: true,
						security_valid: true,
						function_found: true,
						function_signature_valid: true
					},
					test_result: {
						executed: false,
						error: 'Invalid JSON in test cases. Please check your test input syntax.',
						results: []
					}
				};
				isValidating = false;
				return;
			}
			testInputs = parsed;
		}

		try {
			validationResult = await api.validateRewardFunction(
				rewardFunctionCode,
				rewardFunctionName,
				runTest,
				testInputs
			);
		} catch (e) {
			validationResult = {
				success: false,
				syntax_errors: ['Failed to reach validation server'],
				security_issues: [],
				validation: {
					syntax_valid: false,
					security_valid: false,
					function_found: false,
					function_signature_valid: false
				},
				test_result: null
			};
		}
		isValidating = false;
	}

	// Track whether all test cases passed
	$: allTestsPassed =
		validationResult?.success === true &&
		validationResult?.test_result?.executed === true &&
		validationResult?.test_result?.results?.length > 0 &&
		validationResult.test_result.results.every((r: any) => !r.error);

	// Per-case runtime errors (e.g. NameError raised inside the reward function)
	interface TestCaseError {
		index: number;
		error: string;
	}
	$: testCaseErrors = (
		validationResult?.success === true &&
		validationResult?.test_result?.executed === true &&
		Array.isArray(validationResult?.test_result?.results)
			? (validationResult.test_result.results as any[])
					.map((r: any, i: number): TestCaseError | null =>
						r?.error ? { index: i + 1, error: String(r.error) } : null
					)
					.filter((x: TestCaseError | null): x is TestCaseError => x !== null)
			: []
	) as TestCaseError[];
	$: hasTestCaseErrors = testCaseErrors.length > 0;
	$: uniqueTestCaseErrors = Array.from(new Set(testCaseErrors.map((e: TestCaseError) => e.error)));

	// Reset validation when code or function name changes
	let prevCode = rewardFunctionCode;
	let prevName = rewardFunctionName;
	$: if (rewardFunctionCode !== prevCode || rewardFunctionName !== prevName) {
		validationResult = null;
		allTestsPassed = false;
		prevCode = rewardFunctionCode;
		prevName = rewardFunctionName;
	}

	// Distribute validation results into individual test cases
	$: if (validationResult?.test_result?.results) {
		const results = validationResult.test_result.results;
		testCases = testCases.map((tc, i) => {
			const result = results[i];
			if (result) {
				tc.reward = result.error ? null : result.return_value;
				tc.rewardError = result.error || null;
			} else {
				tc.reward = null;
				tc.rewardError = null;
			}
			return tc;
		});
		// Re-sync JSON to include reward if in JSON View
		if (advancedTestMode) {
			testCases.forEach(syncFieldsToJson);
			testCases = testCases;
		}
	}

	// Clear rewards when validation is reset
	$: if (!validationResult) {
		testCases = testCases.map((tc) => {
			tc.reward = null;
			tc.rewardError = null;
			return tc;
		});
	}

	// Collapsed "how it works" state
	let showHowItWorks = false;
</script>

<Grid noGutter fullWidth>
	<Row>
		<Column>
			<div class="step-header">
				<div>
					<h4 class="step-title">Define Reward Function</h4>
					<p class="step-subtitle">
						Write a Python function that scores model responses during online RL training.
					</p>
				</div>
			</div>
		</Column>
	</Row>

	<!-- Code editor (full width) -->
	<Row style="margin-top: var(--cds-spacing-05, 1rem);">
		<Column lg={16} md={8} sm={4}>
			<div class="editor-wrapper">
				<div class="editor-tab-bar">
					<span class="editor-tab">reward_function.py</span>
				</div>
				<div class="code-editor-container" bind:this={editorContainer}></div>
			</div>
			<!-- Action bar below editor -->
			<div class="action-bar" style="margin-top: var(--cds-spacing-03, 0.5rem);">
				<div class="action-bar-left">
					{#if !(validationResult && !validationResult.success) && !hasTestCaseErrors}
						<Button
							kind={showTestPanel ? 'ghost' : 'secondary'}
							size="small"
							icon={ListBoxes}
							disabled={isValidating || isLoadingDataset || rewardFunctionCode.trim().length === 0}
							on:click={() => {
								showTestPanel = !showTestPanel;
							}}
						>
							{showTestPanel ? 'Hide Test Cases' : 'Show Test Cases'}
						</Button>
					{/if}
					{#if !(validationResult && !validationResult.success)}
						<Button
							kind="primary"
							size="small"
							icon={Play}
							disabled={isValidating || isLoadingDataset || rewardFunctionCode.trim().length === 0}
							on:click={() => validateCode(true)}
						>
							Run
						</Button>
					{/if}
				</div>
				<div class="action-bar-right">
					{#if isLoadingDataset}
						<InlineLoading description="Generating test cases..." />
					{:else if isValidating}
						<InlineLoading description="Validating..." />
					{/if}
					{#if allTestsPassed}
						<Tag type="green" size="sm" icon={Checkmark}>Tests passed</Tag>
					{/if}
				</div>
			</div>
		</Column>
	</Row>

	<!-- Test cases panel (below editor, hidden on validation error or per-case runtime error) -->
	{#if showTestPanel && !(validationResult && !validationResult.success) && !hasTestCaseErrors}
		<Row style="margin-top: var(--cds-spacing-03, 0.5rem);">
			<Column lg={16} md={8} sm={4}>
				<!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
				<div
					class="test-panel"
					role="region"
					aria-label="Test cases"
					transition:slide={{ duration: 200 }}
					on:keydown={(e) => {
						if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
							e.preventDefault();
							validateCode(true);
						}
					}}
				>
					<div class="test-panel-header">
						<div class="test-panel-header-row">
							<p class="test-panel-title">Test Cases</p>
							<div class="view-mode-switcher">
								<ContentSwitcher
									size="sm"
									bind:selectedIndex={viewModeIndex}
									on:change={handleViewModeChange}
								>
									<Switch text="Table View" />
									<Switch text="JSON View" />
								</ContentSwitcher>
							</div>
						</div>
						<p class="test-panel-desc">
							{#if advancedTestMode}
								JSON kwargs passed to your function.
							{:else}
								Fill in the fields your function expects.
							{/if}
						</p>
					</div>

					<div class="test-cases-scroll">
						{#if modeSwitchWarning}
							<InlineNotification
								kind={modeSwitchWarning.includes('invalid JSON') ? 'error' : 'warning'}
								title=""
								subtitle={modeSwitchWarning}
								hideCloseButton={false}
								lowContrast
								on:close={() => {
									modeSwitchWarning = null;
								}}
							/>
						{/if}

						{#each testCases as testCase, i (testCase.id)}
							<div class="test-case-block">
								<div class="test-case-top-row">
									<span class="test-case-label">Case {i + 1}</span>
									{#if testCases.length > 1}
										<button
											class="test-case-remove"
											on:click={() => removeTestCase(testCase.id)}
											title="Remove test case"
										>
											<TrashCan size={16} />
										</button>
									{/if}
								</div>

								{#if advancedTestMode}
									<textarea
										class="test-case-input"
										bind:value={testCase.json}
										spellcheck="false"
										rows={6}
										placeholder={'{\n  "data_source": "...",\n  "solution_str": "..."\n}'}
									></textarea>
								{:else}
									<div class="simple-fields">
										<TextArea
											labelText="Expected answer (ground_truth)"
											rows={2}
											bind:value={testCase.ground_truth}
											placeholder="Ground truth value"
										/>
										<TextArea
											labelText="Model response (solution_str)"
											rows={2}
											bind:value={testCase.solution_str}
											placeholder="The model's generated response..."
										/>
										<!-- Inline reward result (Table View) -->
										{#if testCase.reward !== null}
											<div class="inline-reward">
												<span class="inline-reward-label">Reward</span>
												<code class="inline-reward-value">{testCase.reward.toFixed(3)}</code>
											</div>
										{:else if testCase.rewardError}
											<div class="inline-reward inline-reward-error">
												<span class="inline-reward-label">Error</span>
												<span class="inline-reward-error-text">{testCase.rewardError}</span>
											</div>
										{/if}
									</div>
								{/if}
							</div>
						{/each}
					</div>

					<div class="test-panel-footer">
						<Button kind="ghost" size="small" icon={Add} on:click={addTestCase}>Add Case</Button>
					</div>
				</div>
			</Column>
		</Row>
	{/if}

	<!-- Per-case runtime errors (code validated OK, but raised at exec time) -->
	{#if hasTestCaseErrors}
		<Row style="margin-top: 0.5rem;">
			<Column>
				{#if uniqueTestCaseErrors.length === 1}
					<InlineNotification
						kind="error"
						title="Test Execution Error"
						subtitle={`${uniqueTestCaseErrors[0]} (affected ${testCaseErrors.length} of ${validationResult.test_result.results.length} cases)`}
						hideCloseButton
						lowContrast
					/>
				{:else}
					<InlineNotification
						kind="error"
						title="Test Execution Errors"
						subtitle={testCaseErrors.map((e) => `Case ${e.index}: ${e.error}`).join(' • ')}
						hideCloseButton
						lowContrast
					/>
				{/if}
			</Column>
		</Row>
	{/if}

	<!-- Validation errors only -->
	{#if validationResult && !validationResult.success}
		<Row style="margin-top: 0.5rem;">
			<Column>
				{#if validationResult.syntax_errors?.length > 0}
					<InlineNotification
						kind="error"
						title="Syntax Error"
						subtitle={validationResult.syntax_errors.join('; ')}
						hideCloseButton
						lowContrast
					/>
				{/if}
				{#if validationResult.security_issues?.length > 0}
					<InlineNotification
						kind="error"
						title="Security Issues"
						subtitle={validationResult.security_issues.join('; ')}
						hideCloseButton
						lowContrast
						style="margin-top: 0.5rem;"
					/>
				{/if}
				{#if validationResult.test_result?.error}
					<InlineNotification
						kind="error"
						title="Test Execution Error"
						subtitle={validationResult.test_result.error}
						hideCloseButton
						lowContrast
						style="margin-top: 0.5rem;"
					/>
				{/if}
				{#if !validationResult.validation?.function_found}
					<InlineNotification
						kind="warning"
						title="Function not found"
						subtitle="Function '{rewardFunctionName}' was not found in the code."
						hideCloseButton
						lowContrast
						style="margin-top: 0.5rem;"
					/>
				{:else if !validationResult.validation?.function_signature_valid}
					<InlineNotification
						kind="warning"
						title="Invalid signature"
						subtitle="The function should accept at least 2 parameters: (data_source, solution_str)"
						hideCloseButton
						lowContrast
						style="margin-top: 0.5rem;"
					/>
				{/if}
			</Column>
		</Row>
	{/if}

	<!-- stdout output from test (only when validation succeeded) -->
	{#if validationResult?.success && validationResult?.test_result?.stdout}
		<Row style="margin-top: 0.5rem;">
			<Column>
				<div class="stdout-wrapper">
					<span class="stdout-label">stdout</span>
					<pre class="stdout-output">{validationResult.test_result.stdout}</pre>
				</div>
			</Column>
		</Row>
	{/if}

	<!-- Collapsible "How it works" -->
	<!-- <Row style="margin-top: 0.75rem;">
		<Column>
			<button class="how-it-works-toggle" on:click={() => showHowItWorks = !showHowItWorks}>
				<span class="how-it-works-arrow" class:open={showHowItWorks}>&#9654;</span>
				How does this work?
			</button>
			{#if showHowItWorks}
				<p class="how-it-works-body">
					During training, the model generates responses to prompts from your dataset. Your reward function scores each response. The model learns to produce higher-scoring responses over time.
				</p>
			{/if}
		</Column>
	</Row> -->
</Grid>

<style>
	/* ─── Step header ─── */
	.step-header {
		display: flex;
		align-items: flex-start;
		gap: 0.75rem;
	}

	.step-title {
		margin: 0;
		font-size: 1.125rem;
		font-weight: 600;
		color: var(--cds-text-01, #161616);
	}

	.step-subtitle {
		margin: 0.25rem 0 0 0;
		font-size: 0.875rem;
		color: var(--cds-text-02, #525252);
		line-height: 1.4;
	}

	/* ─── Action bar ─── */
	.action-bar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.action-bar-left {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}

	.action-bar-right {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-left: auto;
	}

	.action-bar :global(.bx--tag) {
		margin: 0;
	}

	/* ─── Editor ─── */
	.editor-wrapper {
		border: 1px solid var(--cds-border-subtle, #e0e0e0);
		overflow: hidden;
	}

	.editor-tab-bar {
		display: flex;
		align-items: center;
		padding: 0 1rem;
		height: 2rem;
		background: var(--cds-background-inverse, #262626);
		border-bottom: 1px solid rgba(255, 255, 255, 0.08);
	}

	.editor-tab {
		font-size: 0.6875rem;
		font-family: 'IBM Plex Mono', monospace;
		color: var(--cds-text-on-color, #f4f4f4);
		opacity: 0.65;
		letter-spacing: 0.02em;
	}

	.code-editor-container {
		width: 100%;
		height: 560px;
		overflow: hidden;
		background: #1e1e1e;
	}

	/* ─── Test panel ─── */
	.test-panel {
		background: var(--cds-layer-01, #f4f4f4);
		border: 1px solid var(--cds-border-subtle, #e0e0e0);
		display: flex;
		flex-direction: column;
	}

	.test-panel-header {
		padding: 0.75rem 0.75rem 0.5rem;
		border-bottom: 1px solid var(--cds-border-subtle, #e0e0e0);
	}

	.test-panel-header-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.25rem;
	}

	/* ─── View mode switcher ─── */
	.view-mode-switcher {
		flex: 0 0 auto;
	}

	.view-mode-switcher :global(.bx--content-switcher) {
		height: 2rem;
		min-height: 2rem;
	}

	.view-mode-switcher :global(.bx--content-switcher-btn) {
		font-size: 0.75rem;
		padding: 0 0.75rem;
		min-height: 2rem;
		min-width: 5.5rem;
		white-space: nowrap;
		overflow: visible;
	}

	.test-panel-title {
		font-size: 0.8125rem;
		font-weight: 600;
		margin: 0;
		color: var(--cds-text-01, #161616);
	}

	.test-panel-desc {
		font-size: 0.6875rem;
		color: var(--cds-text-02, #525252);
		margin: 0.25rem 0 0 0;
		line-height: 1.35;
	}

	.test-cases-scroll {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		flex: 1;
		overflow-y: auto;
		padding: 0.5rem 0.75rem;
		max-height: 300px;
	}

	.test-case-block {
		display: flex;
		flex-direction: column;
	}

	.test-case-top-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 0.25rem;
	}

	.test-case-label {
		font-size: 0.6875rem;
		font-weight: 600;
		color: var(--cds-text-02, #525252);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.test-case-input {
		width: 100%;
		padding: 0.375rem 0.5rem;
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.6875rem;
		line-height: 1.4;
		color: var(--cds-text-01, #161616);
		background: var(--cds-field-01, #ffffff);
		border: 1px solid var(--cds-border-strong, #8d8d8d);
		border-radius: 0;
		resize: vertical;
		white-space: pre;
		tab-size: 2;
		box-sizing: border-box;
	}

	.test-case-input:focus {
		outline: 2px solid var(--cds-focus, #0f62fe);
		outline-offset: -2px;
	}

	.test-case-remove {
		background: none;
		border: none;
		padding: 2px;
		cursor: pointer;
		color: var(--cds-text-02, #525252);
		display: flex;
		align-items: center;
		transition: color 120ms;
	}

	.test-case-remove:hover {
		color: var(--cds-support-error, #da1e28);
	}

	/* ─── Simple mode fields ─── */
	.simple-fields {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	@media (min-width: 66rem) {
		.simple-fields {
			flex-direction: row;
			align-items: stretch;
			gap: 1rem;
		}
		.simple-fields > :global(.bx--form-item) {
			flex: 1;
		}
		.simple-fields > .inline-reward {
			flex: 0 0 auto;
		}
	}

	.simple-fields :global(.bx--form-item) {
		margin-bottom: 0;
	}

	.simple-fields :global(.bx--label) {
		font-size: 0.6875rem;
		margin-bottom: 0.125rem;
	}

	.test-panel-footer {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem;
		border-top: 1px solid var(--cds-border-subtle, #e0e0e0);
	}

	/* ─── Inline reward display ─── */
	.inline-reward {
		display: flex;
		flex-direction: column;
		justify-content: center;
		gap: 0.25rem;
		padding: 0.375rem 0.5rem;
		background: var(--cds-layer-accent-01, #e0e0e0);
		border-left: 3px solid var(--cds-support-success, #24a148);
		font-size: 0.75rem;
		min-width: 5.5rem;
	}

	.inline-reward-error {
		border-left-color: var(--cds-support-error, #da1e28);
		background: var(--cds-support-error-inverse, #fff1f1);
	}

	.inline-reward-label {
		font-weight: 600;
		color: var(--cds-text-02, #525252);
		font-size: 0.6875rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.inline-reward-value {
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.8125rem;
		font-weight: 600;
		color: var(--cds-text-01, #161616);
	}

	.inline-reward-error-text {
		font-size: 0.75rem;
		color: var(--cds-support-error, #da1e28);
		word-break: break-word;
	}

	/* ─── Stdout ─── */
	.stdout-wrapper {
		background: var(--cds-background-inverse, #262626);
		padding: 0.625rem 0.75rem;
		border: 1px solid rgba(255, 255, 255, 0.06);
	}

	.stdout-label {
		font-size: 0.6875rem;
		font-family: 'IBM Plex Mono', monospace;
		color: var(--cds-text-on-color, #f4f4f4);
		opacity: 0.5;
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	.stdout-output {
		font-family: 'IBM Plex Mono', monospace;
		font-size: 0.75rem;
		color: var(--cds-text-on-color, #f4f4f4);
		margin: 0.375rem 0 0 0;
		white-space: pre-wrap;
		max-height: 120px;
		overflow-y: auto;
	}

	/* ─── How it works toggle ─── */
	.how-it-works-toggle {
		background: none;
		border: none;
		padding: 0;
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
		cursor: pointer;
		display: flex;
		align-items: center;
		gap: 0.375rem;
	}

	.how-it-works-toggle:hover {
		color: var(--cds-text-01, #161616);
	}

	.how-it-works-arrow {
		font-size: 0.5rem;
		transition: transform 150ms ease;
		display: inline-block;
	}

	.how-it-works-arrow.open {
		transform: rotate(90deg);
	}

	.how-it-works-body {
		margin: 0.5rem 0 0 0;
		font-size: 0.8125rem;
		color: var(--cds-text-02, #525252);
		line-height: 1.5;
		max-width: 56rem;
	}
</style>
