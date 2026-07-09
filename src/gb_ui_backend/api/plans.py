"""Flight plans API — proxies MCP plan_list and plan_get tools via Streamable HTTP."""

from __future__ import annotations

import json
import logging
from itertools import count
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Header, HTTPException

from gb_ui_backend.config import get_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics/plans")

_req_counter = count(1)


async def _mcp_call_tool(
    gbmcp_url: str,
    token: Optional[str],
    tool_name: str,
    arguments: dict,
) -> Any:
    """Call an MCP tool via the Streamable HTTP transport using httpx."""
    base_headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    session_id: Optional[str] = None

    async def _post(body: dict) -> Optional[dict]:
        nonlocal session_id
        headers = dict(base_headers)
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", gbmcp_url, json=body, headers=headers
            ) as resp:
                if not resp.is_success:
                    await resp.aread()
                    resp.raise_for_status()

                if sid := resp.headers.get("mcp-session-id"):
                    session_id = sid

                # 202 = notification accepted, no body expected
                if resp.status_code == 202:
                    await resp.aread()
                    return None

                ct = resp.headers.get("content-type", "")
                if "text/event-stream" in ct:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                break
                            try:
                                parsed = json.loads(payload)
                                if "result" in parsed or "error" in parsed:
                                    return parsed
                            except json.JSONDecodeError:
                                pass
                    return None
                else:
                    raw = await resp.aread()
                    return json.loads(raw)

    req_id = next(_req_counter)

    # 1. Initialize session
    await _post(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "gb-ui", "version": "0.1.0"},
            },
        }
    )

    # 2. Initialized notification (no id = notification, server may return 202)
    await _post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # 3. Tool call
    tool_req_id = next(_req_counter)
    response = await _post(
        {
            "jsonrpc": "2.0",
            "id": tool_req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
    )

    if response is None:
        raise HTTPException(status_code=502, detail="MCP server returned no response")
    if "error" in response:
        raise HTTPException(status_code=502, detail=f"MCP error: {response['error']}")

    result = response.get("result", {})
    for item in result.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            return json.loads(item["text"])
    return {}


def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    lower = authorization.lower()
    if lower.startswith("bearer "):
        return authorization[7:]
    return authorization


@router.get("")
async def list_plans(authorization: Optional[str] = Header(None)):
    config = get_config()
    if not config.gbmcp_url:
        raise HTTPException(status_code=503, detail="GB_UI_GBMCP_URL is not configured")

    token = _extract_token(authorization)
    try:
        data = await _mcp_call_tool(config.gbmcp_url, token, "plan_list", {"limit": 50})
        plans = data.get("plans", [])
        total = data.get("total_count", len(plans))
        return {"plans": plans, "total": total}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("plan_list failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch plans: {exc}"
        ) from exc


@router.get("/{plan_id}")
async def get_plan(plan_id: str, authorization: Optional[str] = Header(None)):
    config = get_config()
    if not config.gbmcp_url:
        raise HTTPException(status_code=503, detail="GB_UI_GBMCP_URL is not configured")

    token = _extract_token(authorization)
    try:
        data = await _mcp_call_tool(
            config.gbmcp_url, token, "plan_get", {"plan_id": plan_id}
        )
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("plan_get(%s) failed: %s", plan_id, exc)
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch plan: {exc}"
        ) from exc
