"""
AI analysis prompt templates.

Ported from gb_dashboard/services/ai_prompts.py.
GitHub prompt hot-reload and IBM-specific repo references abstracted.
Prompts are loaded from ai_system_prompts.md in this directory.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2"
MAX_PROMPT_TOKENS = 100_000
CHARS_PER_TOKEN = 4


def _parse_prompts_from_content(content: str) -> Dict[str, str]:
    prompts = {}
    sections = re.split(r"^## (\w+)\s*$", content, flags=re.MULTILINE)
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            name = sections[i].strip().lower()
            text = sections[i + 1].strip()
            text = text.lstrip("---").rstrip("---").strip()
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*", "", text)
            if name and text:
                prompts[name] = text
    return prompts


def _load_system_prompts() -> Dict[str, str]:
    prompt_file = Path(__file__).parent / "ai_system_prompts.md"
    if not prompt_file.exists():
        logger.warning("System prompts file not found: %s — AI analysis will be unavailable", prompt_file)
        return {}
    try:
        content = prompt_file.read_text()
        prompts = _parse_prompts_from_content(content)
        logger.info("Loaded %d system prompts from %s", len(prompts), prompt_file.name)
        return prompts
    except Exception as e:
        logger.error("Error loading system prompts: %s", e)
        return {}


SYSTEM_PROMPTS: Dict[str, str] = _load_system_prompts()
_last_prompt_refresh: Optional[datetime] = None


async def reload_prompts_from_url(url: str, token: str = "") -> bool:
    """Fetch latest system prompts from a URL (e.g. raw GitHub content).

    Args:
        url: Direct URL to the ai_system_prompts.md file.
        token: Optional Bearer token for private repos.
    """
    global SYSTEM_PROMPTS, _last_prompt_refresh
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            new_prompts = _parse_prompts_from_content(resp.text)
            if new_prompts and new_prompts != SYSTEM_PROMPTS:
                SYSTEM_PROMPTS.clear()
                SYSTEM_PROMPTS.update(new_prompts)
                _last_prompt_refresh = datetime.now(timezone.utc)
                logger.info("Reloaded %d prompts from %s", len(new_prompts), url)
                return True
    except Exception as e:
        logger.warning("Failed to reload prompts from %s: %s", url, e)
    return False


def get_system_prompt(analysis_type: str) -> str:
    if analysis_type not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown analysis type: {analysis_type}. Available: {list(SYSTEM_PROMPTS.keys())}")
    return SYSTEM_PROMPTS[analysis_type]


def determine_analysis_type(status: str) -> str:
    s = (status or "").upper()
    if s in ("FAILED", "ERROR", "DELETED"):
        return "failure"
    if s in ("RUNNING", "WARMINGUP"):
        return "health"
    if s in ("PENDING", "SUSPENDED", "QUEUED"):
        return "scheduling"
    if s in ("SUCCESS", "COMPLETED", "SUCCEEDED"):
        return "performance"
    return "failure"


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _truncate_logs(logs: str, max_chars: int,
                   keep_start: int = 50, keep_end: int = 200) -> str:
    if len(logs) <= max_chars:
        return logs
    lines = logs.split("\n")
    if len(lines) <= keep_start + keep_end:
        return logs[:max_chars] + "\n... [truncated]"
    truncated = (
        "\n".join(lines[:keep_start])
        + f"\n\n... [{len(lines) - keep_start - keep_end} lines truncated] ...\n\n"
        + "\n".join(lines[-keep_end:])
    )
    if len(truncated) > max_chars:
        end_text = "\n".join(lines[-keep_end:])
        if len(end_text) > max_chars:
            end_text = end_text[-(max_chars - 100):]
        truncated = f"... [truncated, showing last {keep_end} lines] ...\n\n" + end_text
    return truncated


def _clean_status_message(msg: str, include_stack_traces: bool = False) -> str:
    if not msg:
        return ""
    if include_stack_traces:
        result = msg.replace("```", "").strip()
        return result[:2000] + "..." if len(result) > 2000 else result
    lines = msg.split("\n")
    cleaned = []
    in_stack = in_details = False
    user_exceptions = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Exception:", "Error:")) and "gbserver" not in stripped.lower():
            user_exceptions.append(stripped)
        elif "Exception:" in stripped and "/app/src/gbserver/" not in line:
            idx = stripped.find("Exception:")
            exc = stripped[idx:]
            if "gbserver" not in exc.lower():
                user_exceptions.append(exc)
    for line in lines:
        if "<details>" in line.lower():
            in_details = True; continue
        if "</details>" in line.lower():
            in_details = False; continue
        if in_details:
            continue
        if "Full Stack Trace" in line or "Traceback (most recent call last)" in line:
            in_stack = True; continue
        if in_stack:
            if line.strip() == "" or (line.strip() and not line.startswith((" ", "|"))):
                in_stack = False
            else:
                continue
        if any(x in line for x in ("/app/src/gbserver/", "gbserver.types.errors",
                                     "File \"/", "| ")):
            continue
        if line.strip() == "```":
            continue
        if line.strip():
            cleaned.append(line.strip())
    for exc in user_exceptions:
        if exc not in cleaned:
            cleaned.insert(0, exc)
    result = " ".join(cleaned)
    if "workload failed:" in result.lower():
        idx = result.lower().find("workload failed:")
        result = result[idx:].split(".")[0] + "."
    return result[:500] + "..." if len(result) > 500 else result


@dataclass
class BuildContext:
    build_id: str
    build_name: str
    status: str
    failure_reason: Optional[str] = None
    failure_message: Optional[str] = None
    total_cpu: Optional[str] = None
    total_memory: Optional[str] = None
    total_gpu: int = 0
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_str: Optional[str] = None
    k8s_resources: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    pod_logs: Optional[str] = None
    step_logs: Dict[str, str] = field(default_factory=dict)
    clusterqueue_capacity_gpu: int = 0
    clusterqueue_usage_gpu: int = 0
    clusterqueue_pending_workloads: int = 0
    gbserver_events: List[Dict[str, Any]] = field(default_factory=list)
    gbserver_status_msgs: List[Dict[str, Any]] = field(default_factory=list)
    user_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "build_id": self.build_id, "build_name": self.build_name, "status": self.status,
            "failure_reason": self.failure_reason, "failure_message": self.failure_message,
            "total_cpu": self.total_cpu, "total_memory": self.total_memory,
            "total_gpu": self.total_gpu, "created_at": self.created_at,
            "finished_at": self.finished_at, "duration_str": self.duration_str,
            "k8s_resources": self.k8s_resources, "events": self.events,
            "pod_logs": self.pod_logs, "step_logs": self.step_logs,
            "clusterqueue_capacity_gpu": self.clusterqueue_capacity_gpu,
            "clusterqueue_usage_gpu": self.clusterqueue_usage_gpu,
            "clusterqueue_pending_workloads": self.clusterqueue_pending_workloads,
            "gbserver_events": self.gbserver_events,
            "gbserver_status_msgs": self.gbserver_status_msgs,
        }


def _has_useful_logs(ctx: BuildContext) -> bool:
    if ctx.step_logs:
        total = sum(len(v.split("\n")) for v in ctx.step_logs.values() if v)
        if total > 5:
            return True
    return bool(ctx.pod_logs and len(ctx.pod_logs.split("\n")) > 5)


def format_user_prompt(context: BuildContext, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    lines = [
        f"Build: {context.build_name} ({context.build_id[:8]})",
        f"Status: {context.status}",
    ]
    if context.failure_reason:
        lines.append(f"Failure Reason: {context.failure_reason}")
    if context.failure_message:
        lines.append(f"Failure Message: {context.failure_message}")
    if context.created_at:
        created = context.created_at.replace("T", " ").split("+")[0].split(".")[0]
        lines.append(f"Started: {created} UTC")
    if context.duration_str:
        lines.append(f"Duration: {context.duration_str}")
    resources = []
    if context.total_cpu: resources.append(f"{context.total_cpu} CPU")
    if context.total_memory: resources.append(f"{context.total_memory} memory")
    if context.total_gpu: resources.append(f"{context.total_gpu} GPU")
    if resources:
        lines.append(f"Resources: {', '.join(resources)}")

    if context.gbserver_status_msgs:
        lines += ["", "Backend Status Messages:"]
        has_logs = _has_useful_logs(context)
        for msg in context.gbserver_status_msgs:
            lines.append(f"  - [{msg.get('entity','')}] {msg.get('name','unknown')}: {msg.get('status','')}")
            sm = _clean_status_message(msg.get("status_msg", ""), include_stack_traces=not has_logs)
            if sm:
                lines.append(f"    Message: {sm}")

    if context.gbserver_events:
        err_events = [e for e in context.gbserver_events
                      if e.get("type") == "MESSAGE_EVENT" and e.get("level") == "ERROR"]
        if err_events:
            lines += ["", "Backend Error Messages:"]
            for ev in err_events[:20]:
                ts = ev.get("created_time", "")
                ts = f"[{ts.split('T')[1].split('.')[0]}] " if ts and "T" in ts else ""
                lines.append(f"  - {ts}{ev.get('message','')[:500]}")

    if context.k8s_resources:
        lines += ["", "K8s Resources:"]
        for res in context.k8s_resources[:10]:
            line = f"  - {res.get('kind','Unknown')}/{res.get('name','unknown')}: {res.get('status','unknown')}"
            if res.get("failure_reason"):
                line += f" ({res['failure_reason']})"
            lines.append(line)

    PREEMPTION_REASONS = {"Preempted", "Evicted", "Killing", "OOMKilling", "OOMKilled", "NodeNotReady"}
    important = [e for e in context.events
                 if e.get("type") == "Warning" or e.get("reason") in PREEMPTION_REASONS]
    if important:
        lines += ["", "K8s Events:"]
        for ev in important[:15]:
            ts = ev.get("last_timestamp", "")
            ts = f"[{ts.replace('T',' ').split('+')[0].split('.')[0]}] " if ts else ""
            reason = ev.get("reason", "Unknown")
            prefix = "⚠️ PREEMPTION: " if reason in PREEMPTION_REASONS else ""
            obj = f" ({ev.get('object_name','')})" if ev.get("object_name") else ""
            lines.append(f"  - {prefix}{ts}{reason}{obj}: {ev.get('message','')[:200]}")

    if context.clusterqueue_capacity_gpu > 0:
        lines += ["", "ClusterQueue Status:",
                  f"  - Capacity: {context.clusterqueue_capacity_gpu} GPU",
                  f"  - In Use: {context.clusterqueue_usage_gpu} GPU",
                  f"  - Pending Workloads: {context.clusterqueue_pending_workloads}"]

    tokens_used = _estimate_tokens("\n".join(lines))
    log_budget = max(10000, min((max_tokens - tokens_used) * CHARS_PER_TOKEN - 1000, 200_000))

    if context.step_logs:
        lines += ["", "Pod Logs by Step:"]
        per_step = log_budget // max(len([v for v in context.step_logs.values() if v]), 1)
        for step_name, step_log in context.step_logs.items():
            if step_log:
                lines.append(f"\n=== Step: {step_name} ===")
                lines.append(_truncate_logs(step_log, per_step))
    elif context.pod_logs:
        lines += ["", "Pod Logs (recent):"]
        lines.append(_truncate_logs(context.pod_logs, log_budget))

    result = "\n".join(lines)
    if _estimate_tokens(result) > max_tokens:
        max_chars = max_tokens * CHARS_PER_TOKEN
        result = result[:max_chars] + "\n\n... [truncated]"
    return result


@dataclass
class KnowledgeBaseEntry:
    meta_id: int
    build_id: str
    source: str
    error_category_1: Optional[str] = None
    error_category_2: Optional[str] = None
    error_category_3: Optional[str] = None
    error_category_4: Optional[str] = None
    root_cause: Optional[str] = None
    summary: Optional[str] = None
    suggested_action: Optional[str] = None
    human_solution: Optional[str] = None
    feedback_rating: Optional[int] = None
    feedback_helpful: Optional[bool] = None
    upvotes: int = 0
    downvotes: int = 0
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_type": "build_analysis",
            "meta_id": self.meta_id,
            "error_categories": " > ".join(filter(None, [
                self.error_category_1, self.error_category_2,
                self.error_category_3, self.error_category_4,
            ])) or None,
            "root_cause": self.root_cause,
            "human_solution": self.human_solution,
            "suggested_action": self.suggested_action,
            "feedback_rating": self.feedback_rating,
            "upvotes": self.upvotes,
        }


def format_phase2_prompt(root_cause: str, error_messages: List[str],
                          error_categories: List[str],
                          knowledge_base: List[KnowledgeBaseEntry]) -> str:
    lines = ["=== CURRENT ISSUE ===", f"Root Cause: {root_cause}"]
    if error_messages:
        lines += ["", "Error Messages:"]
        for m in error_messages[:5]:
            lines.append(f"  - {m[:200]}")
    if error_categories:
        cat_str = " > ".join(filter(None, error_categories))
        if cat_str:
            lines.append(f"\nError Category: {cat_str}")
    lines += ["", "=== KNOWLEDGE BASE ===", f"({len(knowledge_base)} entries)", ""]
    for entry in knowledge_base:
        d = entry.to_dict()
        lines.append(f"--- Analysis {d['meta_id']} ---")
        if d.get("error_categories"):
            lines.append(f"Categories: {d['error_categories']}")
        if d.get("root_cause"):
            lines.append(f"Root Cause: {d['root_cause'][:300]}")
        if d.get("human_solution"):
            lines.append(f"Human Solution: {d['human_solution'][:500]}")
        elif d.get("suggested_action"):
            lines.append(f"AI Suggestion: {d['suggested_action'][:300]}")
        lines.append("")
    return "\n".join(lines)
