# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any, Dict

import models as api
from auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from services.reward_validation import (
    DEFAULT_TEST_INPUTS,
    _analyze_security,
    _check_syntax,
    _execute_safely,
    _validate_function,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/generate-test-solutions",
    tags=["Reward Function"],
    summary="Generate solution strings from prompts using LLM",
    response_description="Generated solution strings for reward function test cases",
)
async def generate_test_solutions(
    request: api.GenerateTestSolutionsRequest,
    auth_user: api.AuthUser = Depends(get_current_user),
):
    """Generate solution strings by sending VERL prompts to the LLM."""
    import requests as http_requests

    litellm_url = os.getenv("LITELLM_URL")
    litellm_api_key = os.getenv("LITELLM_API_KEY")
    litellm_model = os.getenv("LITELLM_MODEL", "aws/claude-sonnet-4-6")

    if not litellm_url or not litellm_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM service not configured (LITELLM_URL, LITELLM_API_KEY)",
        )

    solutions = []
    for messages in request.prompts:
        try:
            response = http_requests.post(
                f"{litellm_url}/v1/chat/completions",
                json={
                    "model": litellm_model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {litellm_api_key}",
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            solution = result["choices"][0]["message"]["content"]
            solutions.append(solution)
        except Exception as e:
            logger.warning(f"LLM generation failed for a prompt: {e}")
            solutions.append("")

    return api.GenerateTestSolutionsResponse(solutions=solutions)


@router.post(
    "/api/reward-function/validate",
    tags=["Reward Function"],
    summary="Validate and optionally test a reward function",
    response_description="Validation results with optional test execution output",
)
async def validate_reward_function(
    request: Dict[str, Any],
    auth_user: api.AuthUser = Depends(get_current_user),
):
    """
    Validate a Python reward function for syntax, security, and correctness.

    **Request Body:**
    - **code** (str): Python source code of the reward function
    - **function_name** (str): Name of the function to validate (default: compute_score)
    - **test_execution** (bool): Whether to execute with test inputs (default: false)
    - **test_inputs** (dict): Optional {data_source, solution_str} for test execution

    **Returns:**
    - **success**: Overall validation passed
    - **validation**: Detailed check results (syntax, security, function, signature)
    - **security_issues**: List of security violations found
    - **syntax_errors**: List of syntax/function errors
    - **test_result**: Execution result if test_execution was true
    """
    code = request.get("code", "")
    function_name = request.get("function_name", "compute_score")
    test_execution = request.get("test_execution", False)
    test_inputs = request.get("test_inputs", DEFAULT_TEST_INPUTS)

    if not code or not code.strip():
        return {
            "success": False,
            "validation": {
                "syntax_valid": False,
                "security_valid": False,
                "function_found": False,
                "function_signature_valid": False,
            },
            "security_issues": [],
            "syntax_errors": ["Code cannot be empty"],
            "test_result": None,
        }

    if len(code) > 50000:
        return {
            "success": False,
            "validation": {
                "syntax_valid": False,
                "security_valid": False,
                "function_found": False,
                "function_signature_valid": False,
            },
            "security_issues": [],
            "syntax_errors": ["Code exceeds maximum allowed size (50KB)"],
            "test_result": None,
        }

    # Phase 1: Syntax check
    syntax_valid, syntax_errors, tree = _check_syntax(code)
    if not syntax_valid:
        return {
            "success": False,
            "validation": {
                "syntax_valid": False,
                "security_valid": False,
                "function_found": False,
                "function_signature_valid": False,
            },
            "security_issues": [],
            "syntax_errors": syntax_errors,
            "test_result": None,
        }

    # Phase 2: Security analysis
    security_issues = _analyze_security(tree)
    security_valid = len(security_issues) == 0

    # Phase 3: Function validation
    function_found, signature_valid, fn_errors = _validate_function(tree, function_name)

    # Phase 4: Test execution (only if all checks pass)
    test_result = None
    if test_execution and security_valid and function_found and signature_valid:
        test_result = await run_in_threadpool(
            _execute_safely, code, function_name, test_inputs
        )
    elif test_execution:
        test_result = {"executed": False, "error": "Cannot execute: validation failed"}

    overall_success = (
        syntax_valid and security_valid and function_found and signature_valid
    )
    if test_result and test_result.get("error"):
        overall_success = False

    return {
        "success": overall_success,
        "validation": {
            "syntax_valid": syntax_valid,
            "security_valid": security_valid,
            "function_found": function_found,
            "function_signature_valid": signature_valid,
        },
        "security_issues": security_issues,
        "syntax_errors": syntax_errors + fn_errors,
        "test_result": test_result,
    }
