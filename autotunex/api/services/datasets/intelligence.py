# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dataset intelligence: LLM-driven parsing-strategy generation and column-mapping
suggestion. Independent of storage and the database. Moved verbatim from
dataset_service.Dataset.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class DatasetIntelligence:
    @staticmethod
    def _type_to_str(t) -> str:
        """Convert a Python type/typing annotation to a clean JSON-friendly string."""
        origin = getattr(t, "__origin__", None)
        args = getattr(t, "__args__", None)
        if origin is not None and args:
            origin_name = getattr(origin, "__name__", str(origin))
            args_str = ", ".join(DatasetIntelligence._type_to_str(a) for a in args)
            return f"{origin_name}[{args_str}]"
        if isinstance(t, type):
            return t.__name__
        return str(t)

    @staticmethod
    def _select_llm_backend():
        """Resolve the LLM endpoint used by dataset intelligence.

        Prefers LiteLLM (LITELLM_URL / LITELLM_API_KEY / LITELLM_MODEL); falls
        back to a generic OpenAI-compatible endpoint (OPENAI_BASE_URL /
        OPENAI_API_KEY / OPENAI_MODEL). Both speak the OpenAI
        ``/v1/chat/completions`` contract, so callers build a single request
        shape. Returns ``(chat_completions_url, headers, model)``.

        Raises ValueError if neither backend is configured.
        """
        litellm_url = os.getenv("LITELLM_URL")
        litellm_api_key = os.getenv("LITELLM_API_KEY")
        if litellm_url and litellm_api_key:
            model = os.getenv("LITELLM_MODEL", "claude-sonnet-4-6")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {litellm_api_key}",
            }
            return f"{litellm_url}/v1/chat/completions", headers, model

        openai_base = os.getenv("OPENAI_BASE_URL")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_base and openai_key:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}",
            }
            return f"{openai_base}/v1/chat/completions", headers, model

        raise ValueError(
            "No LLM backend configured for dataset intelligence. Set "
            "LITELLM_URL + LITELLM_API_KEY (optionally LITELLM_MODEL), or "
            "OPENAI_BASE_URL + OPENAI_API_KEY (optionally OPENAI_MODEL)."
        )

    def get_autotune_dataset_types(self) -> dict:
        # Lazy import: the autotune training core is an optional IBM dependency,
        # absent in a credential-free install. Import it only when dataset-type
        # discovery is actually requested.
        from autotune.utils import get_autotune_dataset_types

        dataset_types = get_autotune_dataset_types()
        # Convert Python type objects to clean string representations for JSON serialization
        for dtype_val in dataset_types.values():
            for col_val in dtype_val.get("columns", {}).values():
                if "type" in col_val:
                    col_val["type"] = self._type_to_str(col_val["type"])
        return dataset_types

    async def generate_parsing_strategy(
        self, sample: List[Any], format: str, custom_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Generate an intelligent parsing strategy using LLM to transform raw data
        into input-output pairs.

        Args:
            sample: Sample data to analyze (10-20 examples)
            format: Original file format (jsonl, json, csv, txt, xml)
            custom_prompt: Optional custom prompt to override default LLM instructions

        Returns:
            Dictionary containing parsing strategy with type, fields, patterns, etc.
        """
        try:
            # First, try to detect if it's already structured
            if self._has_io_structure(sample):
                return self._create_direct_mapping_strategy(sample)

            # If not structured, use LLM to generate strategy
            strategy = await self._generate_llm_strategy(
                sample, format, custom_prompt=custom_prompt
            )
            logger.debug("parsing strategy: %s", strategy)
            return strategy

        except Exception as e:
            logger.error(f"Error generating parsing strategy: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate parsing strategy: {str(e)}"
            )

    def _has_io_structure(self, sample: List[Any]) -> bool:
        """Check if sample data already has input-output structure"""
        if not sample or len(sample) == 0:
            return False

        # Check first few items for input/output-like keys
        for item in sample[:5]:
            if not isinstance(item, dict):
                return False

            has_input = any(
                key in item for key in ["input", "prompt", "question", "text"]
            )
            has_output = any(
                key in item
                for key in ["output", "response", "answer", "completion", "target"]
            )

            if not (has_input and has_output):
                return False

        return True

    def _create_direct_mapping_strategy(self, sample: List[Dict]) -> Dict[str, Any]:
        """Create a direct mapping strategy for structured data"""
        first_item = sample[0]

        # Find input field
        input_field = None
        for key in ["input", "prompt", "question", "text"]:
            if key in first_item:
                input_field = key
                break

        # Find output field
        output_field = None
        for key in ["output", "response", "answer", "completion", "target"]:
            if key in first_item:
                output_field = key
                break

        return {
            "type": "direct_mapping",
            "description": f"Direct field mapping from '{input_field}' to input and '{output_field}' to output",
            "input_field": input_field,
            "output_field": output_field,
        }

    async def _generate_llm_strategy(
        self, sample: Any, format: str, max_retries: int = 5, custom_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Enhanced LLM strategy generation with better prompting and error handling.
        Handles both string (raw text) and array samples intelligently.
        Validates generated strategy and retries if parsing fails.

        Args:
            sample: Sample data to analyze
            format: Original file format
            max_retries: Maximum number of LLM calls if validation fails (default: 3)
            custom_prompt: Optional custom prompt to override default instructions
        """
        # Handle both string (raw text) and array samples
        if isinstance(sample, str):
            sample_str = sample  # Use raw text directly for better context
            is_raw_text = True
        else:
            sample_str = json.dumps(
                sample[:20] if len(sample) > 20 else sample, indent=2
            )[:2000]
            is_raw_text = False
        # Retry loop - attempt to generate and validate strategy
        last_error = None
        previous_failures = []  # Track previous attempts for feedback

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"LLM strategy generation attempt {attempt + 1}/{max_retries}"
                )

                # Include feedback from previous failures in the prompt
                feedback = None
                if previous_failures:
                    feedback = previous_failures[-1]  # Most recent failure

                strategy = await self._call_llm_for_strategy(
                    sample_str, format, is_raw_text, custom_prompt, feedback
                )

                # Validate the strategy works on the sample
                validation_result = self._validate_strategy_on_sample(strategy, sample)

                if (
                    validation_result["success"]
                    and validation_result["parsed_count"] > 0
                ):
                    logger.info(
                        f"✅ Strategy validated successfully! Parsed {validation_result['parsed_count']} records"
                    )
                    logger.info(
                        f"Sample results: {validation_result['sample_results'][:2]}"
                    )
                    return strategy
                else:
                    error_msg = f"Strategy validation failed: {', '.join(validation_result.get('errors', ['Unknown error']))}"
                    logger.warning(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                    last_error = error_msg

                    # Store failure details for next retry
                    previous_failures.append(
                        {
                            "attempt": attempt + 1,
                            "strategy": strategy,
                            "errors": validation_result.get("errors", []),
                            "parsed_count": validation_result.get("parsed_count", 0),
                        }
                    )

                    if attempt < max_retries - 1:
                        logger.info(
                            f"Retrying with LLM (attempt {attempt + 2}/{max_retries}) with error feedback..."
                        )
                        continue

            except Exception as e:
                error_msg = f"Error in attempt {attempt + 1}: {str(e)}"
                logger.error(error_msg)
                last_error = error_msg

                if attempt < max_retries - 1:
                    logger.info(
                        f"Retrying after error (attempt {attempt + 2}/{max_retries})..."
                    )
                    continue

        # All retries exhausted
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate valid parsing strategy after {max_retries} attempts. Last error: {last_error}",
        )

    def _validate_js_regex_compatibility(
        self, input_pattern: str, output_pattern: str
    ) -> Dict[str, Any]:
        """
        Validate that regex patterns are compatible with JavaScript regex engine.
        Python and JavaScript regex have different syntax and features.

        Returns:
            Dict with 'compatible' (bool) and 'errors' (list of error messages)
        """
        errors = []

        # Check for Python-specific inline flags that don't work in JavaScript
        # JavaScript doesn't support inline flags like (?m), (?s), (?x), (?i) etc.
        for pattern_name, pattern in [
            ("input_pattern", input_pattern),
            ("output_pattern", output_pattern),
        ]:
            if not pattern:
                continue

            # Check for inline flag modifiers (Python-style)
            if (
                "(?m)" in pattern
                or "(?s)" in pattern
                or "(?x)" in pattern
                or "(?i)" in pattern
            ):
                errors.append(
                    f"{pattern_name} contains inline flags like (?m), (?s), (?i), (?x) which are not supported in JavaScript. "
                    f"Use /pattern/gm flags instead or remove the inline modifier."
                )

            # Check for named groups (?P<name>...) - JavaScript uses (?<name>...)
            if "(?P<" in pattern:
                errors.append(
                    f"{pattern_name} uses Python named groups (?P<name>...). "
                    f"JavaScript uses (?<name>...) syntax instead."
                )

            # Check for conditional patterns (?(id)yes|no) - not supported in JavaScript
            if "(?(" in pattern:
                errors.append(
                    f"{pattern_name} contains conditional patterns (?(id)yes|no) which are not supported in JavaScript."
                )

            # Check for \A and \Z (Python-style anchors) - JavaScript uses ^ and $
            if "\\A" in pattern or "\\Z" in pattern:
                errors.append(
                    f"{pattern_name} uses \\A or \\Z anchors (Python-style). "
                    f"Use ^ and $ instead for JavaScript compatibility."
                )

            # Check for lookbehind with variable length (not supported in older JS)
            # This is a simplified check - just warn about lookbehind
            if "(?<=" in pattern or "(?<!" in pattern:
                # Lookbehind is supported in modern JavaScript, but let's validate it's not variable-length
                import re

                try:
                    # Try to detect variable-length lookbehind patterns (very basic check)
                    if re.search(r"\(\?<[=!][^)]*[+*{]", pattern):
                        errors.append(
                            f"{pattern_name} may contain variable-length lookbehind which has limited support in JavaScript. "
                            f"Ensure the lookbehind pattern has fixed length."
                        )
                except Exception as e:
                    logger.warning(e)
                    pass

        return {"compatible": len(errors) == 0, "errors": errors}

    def _validate_strategy_on_sample(
        self, strategy: Dict[str, Any], sample: Any
    ) -> Dict[str, Any]:
        """
        Validate that a parsing strategy actually works on the sample data.
        Returns validation result with success status and parsed records.
        """
        try:
            results = []

            if strategy.get("type") == "direct_mapping":
                if isinstance(sample, list):
                    for item in sample[:20]:  # Test first 20
                        if not isinstance(item, dict):
                            continue
                        input_val = item.get(strategy.get("input_field", ""), "")
                        output_val = item.get(strategy.get("output_field", ""), "")
                        if input_val and output_val:
                            results.append(
                                {"input": str(input_val), "output": str(output_val)}
                            )

            elif strategy.get("type") == "regex":
                import re

                text = sample if isinstance(sample, str) else json.dumps(sample)

                input_pattern = strategy.get("input_pattern", "")
                output_pattern = strategy.get("output_pattern", "")

                if input_pattern and output_pattern:
                    # Validate JavaScript compatibility before testing
                    js_validation = self._validate_js_regex_compatibility(
                        input_pattern, output_pattern
                    )
                    if not js_validation["compatible"]:
                        return {
                            "success": False,
                            "parsed_count": 0,
                            "sample_results": [],
                            "errors": [
                                f"JavaScript regex incompatibility: {', '.join(js_validation['errors'])}"
                            ],
                        }

                    try:
                        input_matches = list(
                            re.finditer(input_pattern, text, re.DOTALL)
                        )
                        output_matches = list(
                            re.finditer(output_pattern, text, re.DOTALL)
                        )

                        for i in range(
                            min(len(input_matches), len(output_matches), 20)
                        ):
                            try:
                                input_val = (
                                    input_matches[i].group(1)
                                    if input_matches[i].lastindex
                                    else input_matches[i].group(0)
                                )
                                output_val = (
                                    output_matches[i].group(1)
                                    if output_matches[i].lastindex
                                    else output_matches[i].group(0)
                                )

                                if input_val and output_val:
                                    results.append(
                                        {
                                            "input": input_val.strip(),
                                            "output": output_val.strip(),
                                        }
                                    )
                            except Exception as e:
                                logger.debug(f"Error extracting match {i}: {str(e)}")
                                continue
                    except re.error as e:
                        return {
                            "success": False,
                            "parsed_count": 0,
                            "sample_results": [],
                            "errors": [f"Invalid regex pattern: {str(e)}"],
                        }

            return {
                "success": len(results) > 0,
                "parsed_count": len(results),
                "sample_results": results[:5],
                "errors": (
                    []
                    if len(results) > 0
                    else ["No records could be parsed with this strategy"]
                ),
            }

        except Exception as e:
            logger.error(f"Strategy validation error: {str(e)}")
            return {
                "success": False,
                "parsed_count": 0,
                "sample_results": [],
                "errors": [str(e)],
            }

    async def _call_llm_for_strategy(
        self,
        sample_str: str,
        format: str,
        is_raw_text: bool,
        custom_prompt: str = None,
        previous_failure: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Make a single call to the LLM to generate a parsing strategy.
        Extracted from _generate_llm_strategy for retry logic.

        Args:
            custom_prompt: Additional user instructions to guide the LLM (integrated into system prompt)
            previous_failure: Details about the previous failed attempt (for retry feedback)
        """
        # Build the system prompt with standard instructions
        prompt = f"""Analyze this data sample and create a parsing strategy to convert it into input-output pairs for machine learning training.

File format: {format}
Sample data ({"raw text" if is_raw_text else "structured data"}):
{sample_str}

IMPORTANT INSTRUCTIONS:
1. For TEXT files with repeating patterns, use "regex" type with global patterns
2. For structured files (JSON/CSV), use "direct_mapping" type  
3. Ensure regex patterns capture the ENTIRE input and output blocks, not just fragments
4. Test patterns mentally - they should work across multiple occurrences in the text
5. For text files, patterns must use non-greedy matching and proper lookaheads
6. The input_pattern should capture what goes INTO the model (questions, prompts, tasks)
7. The output_pattern should capture what comes OUT of the model (answers, completions, results)

🚨 CRITICAL - JAVASCRIPT REGEX COMPATIBILITY:
- DO NOT use inline flags like (?m), (?s), (?i), (?x) - these are Python-only
- DO NOT use (?P<name>...) for named groups - JavaScript uses (?<name>...)
- DO NOT use \\A or \\Z anchors - use ^ and $ instead
- DO NOT use conditional patterns (?(id)yes|no) - not supported in JavaScript
- Lookaheads (?=...) and (?!...) are OK
- Lookbehinds (?<=...) and (?<!...) are OK but keep them simple (fixed length)
- Use standard JavaScript regex syntax - the patterns will run in a browser!"""

        # Add feedback from previous failed attempt if available
        if previous_failure:
            prompt += f"""

⚠️ PREVIOUS ATTEMPT FAILED - LEARN FROM THIS:
Attempt #{previous_failure["attempt"]} failed with these errors:
{chr(10).join(f"  - {error}" for error in previous_failure["errors"])}

Previous strategy that FAILED:
  Type: {previous_failure["strategy"].get("type", "unknown")}
  {f"Input Pattern: {previous_failure['strategy'].get('input_pattern', 'N/A')}" if previous_failure["strategy"].get("type") == "regex" else ""}
  {f"Output Pattern: {previous_failure['strategy'].get('output_pattern', 'N/A')}" if previous_failure["strategy"].get("type") == "regex" else ""}
  {f"Input Field: {previous_failure['strategy'].get('input_field', 'N/A')}" if previous_failure["strategy"].get("type") == "direct_mapping" else ""}
  {f"Output Field: {previous_failure['strategy'].get('output_field', 'N/A')}" if previous_failure["strategy"].get("type") == "direct_mapping" else ""}
  Records Parsed: {previous_failure["parsed_count"]} (FAILED - need > 0)

🔧 ADJUST YOUR STRATEGY:
- If regex patterns failed, try different patterns with proper capture groups
- If no matches found, check if patterns are too specific or too broad
- Ensure patterns work across ALL occurrences in the sample, not just the first one
- Test mentally: would your pattern capture the examples you see?"""

        # Add custom user instructions if provided
        if custom_prompt:
            prompt += f"""

ADDITIONAL USER INSTRUCTIONS:
{custom_prompt}

⚠️ Follow the user instructions above carefully while maintaining the JSON response format below."""

        # Add standard task and output format
        prompt += """

Your task:
1. Identify what parts of the data should be the "input" (question, prompt, context, task description)
2. Identify what parts should be the "output" (answer, response, completion, result)
3. Determine the best extraction method (direct field mapping, regex, or transformation)
4. Provide a confidence score (0-1) for how well you think this strategy will work

Respond with a JSON object containing:
- type: "direct_mapping", "regex", or "transformation"
- description: Brief explanation of the strategy
- input_field: Field path for input (for direct_mapping only)
- output_field: Field path for output (for direct_mapping only)
- input_pattern: Regex pattern for input (for regex type only)
- output_pattern: Regex pattern for output (for regex type only)
- confidence: Number between 0 and 1 indicating confidence in this strategy
- sample_extraction: Object with "input_example" and "output_example" strings

Example response for CSV with "Question" and "Answer" columns:
{{
  "type": "direct_mapping",
  "description": "Map Question column to input and Answer column to output",
- sample_extraction: Object with "input_example" and "output_example" strings

EXAMPLE for structured data (JSON/CSV):
{{
  "type": "direct_mapping",
  "description": "Map Question field to input and Answer field to output",
  "input_field": "Question",
  "output_field": "Answer",
  "confidence": 0.95,
  "sample_extraction": {{
    "input_example": "What is AI?",
    "output_example": "Artificial Intelligence is..."
  }}
}}

Only generate required JSON nothing else and no explanation.
"""
        # Call the configured LLM backend (LiteLLM or OpenAI-compatible).
        try:
            import re

            import requests

            url, headers, model = self._select_llm_backend()
            logger.info(f"Calling LLM ({model}) to generate parsing strategy...")

            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            logger.debug(f"Calling chat-completions endpoint at {url}")

            response = requests.post(
                url,
                json=request_body,
                headers=headers,
                timeout=120,
            )

            response.raise_for_status()
            result = response.json()

            # Extract the text response. The prompt asks for JSON only; we parse
            # it directly first and fall back to extraction below if needed.
            llm_response = result["choices"][0]["message"]["content"]
            logger.debug("LLM prompt: %s", prompt)
            logger.debug("LLM response: %s", llm_response)
            logger.info(f"Full LLM Response (first 1000 chars): {llm_response[:1000]}")

            # Try to parse the response as JSON directly first
            try:
                strategy = json.loads(llm_response)
                # Validate it has the expected structure
                if "type" in strategy and "description" in strategy:
                    logger.info(
                        f"Successfully generated parsing strategy: {strategy.get('type', 'unknown')}"
                    )
                    return strategy
                else:
                    raise ValueError("JSON missing required fields")
            except (json.JSONDecodeError, ValueError):
                logger.warning("Direct JSON parse failed, attempting extraction...")

                # Fallback: Extract JSON from response if there's extra text
                # Look for all JSON objects and validate them against our schema
                cleaned_response = re.sub(r"```json\s*|\s*```", "", llm_response)

                # Find ALL complete JSON objects
                found_strategies = []
                start_idx = 0

                while True:
                    start_idx = cleaned_response.find("{", start_idx)
                    if start_idx < 0:
                        break

                    brace_count = 0
                    in_string = False
                    escape_next = False

                    for i in range(start_idx, len(cleaned_response)):
                        char = cleaned_response[i]

                        if escape_next:
                            escape_next = False
                            continue

                        if char == "\\":
                            escape_next = True
                            continue

                        if char == '"' and not escape_next:
                            in_string = not in_string

                        if not in_string:
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = cleaned_response[
                                        start_idx : i + 1
                                    ].strip()
                                    # Try to parse this JSON
                                    try:
                                        candidate = json.loads(json_str)
                                        # Check if it matches our schema (has type and description)
                                        if (
                                            isinstance(candidate, dict)
                                            and "type" in candidate
                                            and "description" in candidate
                                        ):
                                            if candidate.get("type") in [
                                                "direct_mapping",
                                                "regex",
                                                "transformation",
                                            ]:
                                                found_strategies.append(candidate)
                                                logger.info(
                                                    f"Found valid strategy candidate: {candidate.get('type')}"
                                                )
                                    except json.JSONDecodeError:
                                        pass
                                    start_idx = i + 1
                                    break
                    else:
                        # Reached end without finding closing brace
                        break

                if found_strategies:
                    # Return the last valid strategy found (most complete)
                    strategy = found_strategies[-1]
                    logger.info(
                        f"Successfully extracted parsing strategy: {strategy.get('type', 'unknown')}"
                    )
                    logger.info(f"Full strategy: {json.dumps(strategy, indent=2)}")
                    return strategy

                # If we get here, extraction failed
                logger.error(
                    "Could not extract valid parsing strategy JSON from LLM response"
                )
                logger.error(f"Full response: {llm_response}")
                raise ValueError(
                    "LLM response did not contain valid parsing strategy JSON"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API request failed: {str(e)}")
            raise HTTPException(
                status_code=503, detail=f"LLM service unavailable: {str(e)}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Invalid response from LLM service: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error generating parsing strategy: {str(e)}"
            )

    async def suggest_column_mapping(
        self,
        sample_data: List[Dict[str, Any]],
        column_names: List[str],
        column_samples: Dict[str, List[str]],
        target_dataset_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze sample data against supported dataset types and suggest
        the best column mapping, dataset type, and algorithm.

        Args:
            sample_data: First 5-10 rows from the uploaded dataset
            column_names: List of column names in the dataset
            column_samples: Dict mapping column name to list of sample values
            target_dataset_type: If provided, force mapping to this specific dataset type
                instead of auto-detecting the best match

        Returns:
            Dictionary with dataset_type, algorithm, confidence, column_mapping, reasoning
        """
        try:
            # Fetch supported dataset types
            dataset_types = self.get_autotune_dataset_types()
            dataset_types_str = json.dumps(dataset_types, indent=2)

            # Prepare sample data string (truncate values for prompt efficiency)
            truncated_samples = []
            for row in sample_data[:8]:
                truncated_row = {}
                for k, v in row.items():
                    val_str = str(v) if v is not None else ""
                    truncated_row[k] = val_str[:200] if len(val_str) > 200 else val_str
                truncated_samples.append(truncated_row)
            sample_str = json.dumps(truncated_samples, indent=2)[:3000]

            # Build column info
            col_info = {}
            for col in column_names:
                samples = column_samples.get(col, [])
                col_info[col] = [str(s)[:100] for s in samples[:3]]
            col_info_str = json.dumps(col_info, indent=2)

            # Algorithm mapping for each dataset type
            algorithm_map = {
                "dataset_type_a": "lora",
                "dataset_type_b": "dpo",
                "dataset_type_c": "kto",
                "dataset_type_d": "grpo",
            }
            algorithm_map_str = json.dumps(algorithm_map, indent=2)

            # Build task instructions based on whether a target type is specified
            if target_dataset_type and target_dataset_type in dataset_types:
                target_algorithm = algorithm_map.get(target_dataset_type, "lora")
                task_instructions = f"""YOUR TASK:
1. The user has chosen to use dataset type "{target_dataset_type}" (algorithm: {target_algorithm})
2. You MUST map the user's columns to the required columns of "{target_dataset_type}", even if the data appears to better fit a different type
3. For each required column of {target_dataset_type}, find the user column that is the closest semantic match
4. Provide a confidence score (0-1) for the overall match and for each individual column mapping
5. Set dataset_type to "{target_dataset_type}" and algorithm to "{target_algorithm}"

IMPORTANT:
- Match based on CONTENT and SEMANTICS, not just column names
- If a column contains question/instruction text, it's likely an input/prompt column
- If a column contains response/answer text, it's likely an output/completion column
- Use your best judgment to find the closest semantic match for each required column
- Only map columns that have a genuine semantic match. If no user column is a reasonable match for a required column, omit that required column from the column_mapping entirely. Do NOT force placeholder mappings."""
            else:
                task_instructions = """YOUR TASK:
1. Analyze the user's dataset columns and sample values
2. Determine which dataset type (dataset_type_a, dataset_type_b, dataset_type_c, or dataset_type_d) best matches this data
3. For each required column of the matching dataset type, identify which user column should map to it
4. Provide a confidence score (0-1) for the overall match and for each individual column mapping

IMPORTANT:
- Match based on CONTENT and SEMANTICS, not just column names
- If a column contains question/instruction text, it's likely an input/prompt column
- If a column contains response/answer text, it's likely an output/completion column
- If data has chosen/rejected pairs, it's preference data (dataset_type_b)
- If data has binary labels with completions, it's KTO data (dataset_type_c)
- If data has structured prompts with role/content dicts, it's RL data (dataset_type_d)"""

            prompt = f"""You are a dataset analysis assistant for an LLM fine-tuning platform.

SUPPORTED DATASET TYPES:
{dataset_types_str}

ALGORITHM MAPPING (dataset_type -> recommended algorithm):
{algorithm_map_str}

USER'S DATASET:
Column names: {json.dumps(column_names)}

Column samples (column_name -> sample values):
{col_info_str}

Sample rows:
{sample_str}

{task_instructions}

Respond with ONLY a JSON object in this exact format:
{{
  "dataset_type": "dataset_type_a",
  "dataset_type_desc": "Description of the matched type",
  "algorithm": "lora",
  "confidence": 0.9,
  "column_mapping": {{
    "required_col_name": {{"source_column": "user_col_name", "confidence": 0.95}}
  }},
  "reasoning": "Brief explanation of why this mapping was chosen"
}}

All fields are required. Do not include any text outside the JSON object."""

            import requests

            url, headers, model = self._select_llm_backend()
            logger.info(f"Calling LLM ({model}) to suggest column mapping...")
            request_body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = requests.post(
                url,
                json=request_body,
                headers=headers,
                timeout=120,
            )

            response.raise_for_status()
            result = response.json()
            logger.debug(f"Full LLM response keys: {list(result.keys())}")
            logger.debug(
                f"Choices: {json.dumps(result.get('choices', []), indent=2)[:1000]}"
            )
            llm_response = result["choices"][0]["message"]["content"]
            logger.info(f"LLM column mapping suggestion: {llm_response[:1000]}")

            if not llm_response or llm_response.strip() in ("", "{}"):
                raise ValueError(
                    f"LLM returned empty response. Full API result: {json.dumps(result)[:500]}"
                )

            # Strip markdown code fences if present
            cleaned = llm_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]  # remove first line
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            suggestion = json.loads(cleaned)

            # Add dataset_type_desc if not provided by LLM
            if "dataset_type_desc" not in suggestion:
                dtype = suggestion.get("dataset_type", "")
                if dtype in dataset_types:
                    suggestion["dataset_type_desc"] = dataset_types[dtype].get(
                        "desc", ""
                    )

            return suggestion

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed for column mapping: {str(e)}")
            raise HTTPException(
                status_code=503, detail=f"LLM service unavailable: {str(e)}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM mapping response: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Invalid response from LLM: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error suggesting column mapping: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to suggest column mapping: {str(e)}"
            )
