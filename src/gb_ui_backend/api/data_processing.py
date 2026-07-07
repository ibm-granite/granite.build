"""Data processing pipeline API endpoints.

Provides:
  GET /api/analytics/data-processing/lineage        — build-derived DAG (no COS required)
  GET /api/analytics/data-processing/recent-datasets
  POST /api/analytics/data-processing/scan-prefixes
  DELETE /api/analytics/data-processing/scan-prefixes
  GET /api/analytics/data-processing/node-counts    — COS-dependent (graceful 200 if unconfigured)
  GET /api/analytics/data-processing/pipeline-status — COS-dependent
  GET /api/analytics/data-processing/report          — COS-dependent (requires vendored megatron CLI)

Ported from gb_dashboard/src/gb_dashboard/api/data_processing.py.
"""
from __future__ import annotations

import json
import logging
import re

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from gb_ui_backend.config import get_config
from gb_ui_backend.services.gbserver_source import get_gbserver_source

logger = logging.getLogger(__name__)

_cos_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cos-io")

# User-added COS scan prefixes (in-memory, not persisted)
_extra_scan_prefixes: list[str] = []

router = APIRouter(prefix="/api/analytics/data-processing", tags=["data-processing"])

# ---------------------------------------------------------------------------
# Regex patterns (ported verbatim from gb_dashboard)
# ---------------------------------------------------------------------------

_RE_MEGATRON_PATH   = re.compile(r"--megatron_path\s+(\S+)")
_RE_ARROW_PATH      = re.compile(r"--arrow_path\s+(\S+)")
_RE_MERGED_TEXT     = re.compile(r"--merged_text_path\s+(\S+)")
_RE_MERGED_BIN      = re.compile(r"--merged_bin_path\s+(\S+)")
_RE_OUTPUT_FOLDER   = re.compile(r"--output_folder\s+(\S+)")
_RE_INPUT_FOLDER    = re.compile(r"--input_folder\s+(\S+)")
_RE_IS_MEGATRON     = re.compile(r"run_cos_pipeline\.py")
_RE_IS_TOKENIZATION = re.compile(r"tokenization2arrow")

_TYPE_COLUMNS: dict[str, int] = {
    "parquet": 0, "arrow": 1, "megatron": 2, "merged_text": 3, "merged_bin": 3
}


def _extract_dp_paths(yaml_content: str) -> Optional[dict[str, Any]]:
    """Extract data processing COS paths from a build YAML string."""
    has_megatron = _RE_IS_MEGATRON.search(yaml_content)
    has_tokenization = _RE_IS_TOKENIZATION.search(yaml_content)

    if has_megatron and has_tokenization:
        result: dict[str, Any] = {"type": "e2e"}
    elif has_megatron:
        result = {"type": "megatron"}
    elif has_tokenization:
        result = {"type": "tokenization"}
    else:
        return None

    if has_megatron:
        m = _RE_MEGATRON_PATH.search(yaml_content)
        if m:
            result["megatron_path"] = m.group(1).rstrip("\"'")
        m = _RE_ARROW_PATH.search(yaml_content)
        if m:
            result["arrow_path"] = m.group(1).rstrip("\"'")
        m = _RE_MERGED_TEXT.search(yaml_content)
        if m:
            result["merged_text_path"] = m.group(1).rstrip("\"'")
        m = _RE_MERGED_BIN.search(yaml_content)
        if m:
            result["merged_bin_path"] = m.group(1).rstrip("\"'")

    if has_tokenization:
        m = _RE_OUTPUT_FOLDER.search(yaml_content)
        if m and "arrow_path" not in result:
            result["arrow_path"] = m.group(1).rstrip("\"'")
        m = _RE_INPUT_FOLDER.search(yaml_content)
        if m:
            result["parquet_path"] = m.group(1).rstrip("\"'")

    if "megatron_path" in result or "arrow_path" in result:
        return result
    return None


def _normalize_path(p: str) -> str:
    return p.strip().rstrip("/").lower()


def _short_name(path: str) -> str:
    for s in reversed(path.strip().rstrip("/").split("/")):
        if s:
            return s
    return path


# ---------------------------------------------------------------------------
# Dataset scanning (async — uses GbserverSource)
# ---------------------------------------------------------------------------

async def _scan_datasets_async(days: int) -> tuple[list[dict], int, int, Optional[str]]:
    """Scan gbserver builds for data processing datasets.

    Returns (datasets, scanned_count, matched_count, warning_or_None).
    """
    source = get_gbserver_source()
    if not source:
        logger.debug("scan-datasets: GbserverSource not configured (set GB_UI_GBSERVER_DB_URL)")
        return [], 0, 0, "GB_UI_GBSERVER_DB_URL is not configured — cannot scan builds for DP datasets."

    try:
        builds, db_warning = await source.list_builds_for_dp_scan(days_back=days, limit=10000)
    except Exception as exc:
        msg = (
            f"DB query failed — ensure GB_UI_GBSERVER_DB_URL points to gbserver's own database "
            f"(schema: {get_config().gbserver_db_schema}). Error: {exc}"
        )
        logger.warning("scan-datasets: %s", msg)
        return [], 0, 0, msg

    datasets: dict[str, dict] = {}
    scanned = 0
    matched = 0

    # Diagnostic: log all build names and look for any DP-related patterns
    DP_KEYWORDS = ("run_cos_pipeline", "tokenization2arrow", "dpk", "megatron", "arrow_path",
                   "output_folder", "parquet", "tokenize", "cos_pipeline")
    build_names = [b.get("name", "?")[:40] for b in builds]
    logger.info("scan-datasets days=%d: fetched builds = %s", days, build_names)
    for b in builds:
        yaml = b.get("yaml_content") or ""
        found = [kw for kw in DP_KEYWORDS if kw.lower() in yaml.lower()]
        if found:
            sample = yaml[:300].replace("\n", " | ")
            logger.info("  DP keyword(s) %s in build %s (%s): %s",
                        found, b["uuid"][:8], b.get("name", "?")[:30], sample)

    for build in builds:
        yaml_content = build.get("yaml_content")
        if not yaml_content:
            continue

        scanned += 1
        paths = _extract_dp_paths(yaml_content)
        if not paths:
            continue

        matched += 1

        # Use updated_time as the primary timestamp so the time-window filter
        # reflects when the build was last active, not when it was started.
        def _to_utc(raw: Any) -> Optional[datetime]:
            if isinstance(raw, datetime):
                return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            return None

        bt = _to_utc(build.get("updated_time")) or _to_utc(build.get("created_time"))

        arrow = paths.get("arrow_path", "")
        megatron = paths.get("megatron_path", "")
        group_key = _normalize_path(arrow) if arrow else _normalize_path(megatron)
        if not group_key:
            continue

        build_entry: dict[str, Any] = {
            "uuid": build["uuid"],
            "name": build.get("name", ""),
            "username": build.get("username", ""),
            "status": build.get("status", "unknown"),
            "created_time": str(bt) if bt else "",
            "type": paths["type"],
        }

        if group_key not in datasets:
            datasets[group_key] = {
                "arrow_path": arrow,
                "megatron_path": megatron,
                "parquet_path": paths.get("parquet_path", ""),
                "merged_text_path": paths.get("merged_text_path", ""),
                "merged_bin_path": paths.get("merged_bin_path", ""),
                "short_name": _short_name(megatron or arrow),
                "latest_build_time": str(bt) if bt else "",
                "builds": [],
            }
        else:
            ds = datasets[group_key]
            if megatron and not ds["megatron_path"]:
                ds["megatron_path"] = megatron
                ds["short_name"] = _short_name(megatron)
            for pkey in ("parquet_path", "merged_text_path", "merged_bin_path"):
                if paths.get(pkey) and not ds[pkey]:
                    ds[pkey] = paths[pkey]
            if bt and (not ds["latest_build_time"] or str(bt) > ds["latest_build_time"]):
                ds["latest_build_time"] = str(bt)

        datasets[group_key]["builds"].append(build_entry)

    result = sorted(datasets.values(), key=lambda d: d["latest_build_time"], reverse=True)
    for ds in result:
        ds["build_count"] = len(ds["builds"])
        ds["builds"].sort(key=lambda b: b.get("created_time", ""), reverse=True)
        if ds["builds"]:
            latest = ds["builds"][0]
            ds["latest_build_id"] = latest["uuid"]
            ds["latest_build_status"] = latest["status"]
        else:
            ds["latest_build_id"] = None
            ds["latest_build_status"] = None
        ds["name"] = ds["short_name"]

    logger.info(
        "scan-datasets days=%d: fetched=%d scanned=%d matched=%d datasets=%d%s",
        days, len(builds), scanned, matched, len(result),
        f" [warning: {db_warning}]" if db_warning else "",
    )
    return result, scanned, matched, db_warning


# ---------------------------------------------------------------------------
# Lineage graph construction (ported verbatim from gb_dashboard)
# ---------------------------------------------------------------------------

def _build_lineage_graph(datasets: list[dict]) -> dict:
    """Build lineage graph (nodes + edges) from dataset list."""
    nodes: dict[str, dict] = {}
    edges: dict[tuple, dict] = {}
    node_counter = 0
    edge_counter = 0

    def get_or_create_node(path: str, node_type: str) -> str:
        nonlocal node_counter
        norm = _normalize_path(path)
        if norm in nodes:
            return nodes[norm]["id"]
        nid = f"n{node_counter}"
        node_counter += 1
        nodes[norm] = {
            "id": nid,
            "type": node_type,
            "path": path,
            "short_name": _short_name(path),
            "column": _TYPE_COLUMNS.get(node_type, 0),
        }
        return nid

    def get_or_create_edge(src_path: str, tgt_path: str) -> dict:
        nonlocal edge_counter
        key = (_normalize_path(src_path), _normalize_path(tgt_path))
        if key not in edges:
            eid = f"e{edge_counter}"
            edge_counter += 1
            edges[key] = {
                "id": eid,
                "source": nodes[key[0]]["id"],
                "target": nodes[key[1]]["id"],
                "builds": [],
            }
        return edges[key]

    for ds in datasets:
        parquet = ds.get("parquet_path", "")
        arrow = ds.get("arrow_path", "")
        megatron = ds.get("megatron_path", "")
        merged_text = ds.get("merged_text_path", "")
        merged_bin = ds.get("merged_bin_path", "")

        if parquet:      get_or_create_node(parquet, "parquet")
        if arrow:        get_or_create_node(arrow, "arrow")
        if megatron:     get_or_create_node(megatron, "megatron")
        if merged_text:  get_or_create_node(merged_text, "merged_text")
        if merged_bin:   get_or_create_node(merged_bin, "merged_bin")

        if parquet and arrow:           get_or_create_edge(parquet, arrow)
        if arrow and megatron:          get_or_create_edge(arrow, megatron)
        if megatron and merged_text:    get_or_create_edge(megatron, merged_text)
        if megatron and merged_bin:     get_or_create_edge(megatron, merged_bin)

        for build in ds.get("builds", []):
            btype = build["type"]
            if btype in ("tokenization", "e2e") and parquet and arrow:
                get_or_create_edge(parquet, arrow)["builds"].append(build)
            if btype in ("megatron", "e2e"):
                if arrow and megatron:
                    get_or_create_edge(arrow, megatron)["builds"].append(build)
                if megatron and merged_text:
                    get_or_create_edge(megatron, merged_text)["builds"].append(build)
                if megatron and merged_bin:
                    get_or_create_edge(megatron, merged_bin)["builds"].append(build)

    # Cross-dataset linking: link arrow→megatron nodes with the same short name
    arrow_by_name: dict[str, str] = {}
    megatron_by_name: dict[str, str] = {}
    for norm, node in nodes.items():
        if node["type"] == "arrow":
            arrow_by_name[node["short_name"]] = norm
        elif node["type"] == "megatron":
            megatron_by_name[node["short_name"]] = norm

    for name, arrow_norm in arrow_by_name.items():
        if name in megatron_by_name:
            meg_norm = megatron_by_name[name]
            key = (arrow_norm, meg_norm)
            if key not in edges:
                eid = f"e{edge_counter}"
                edge_counter += 1
                edges[key] = {
                    "id": eid,
                    "source": nodes[arrow_norm]["id"],
                    "target": nodes[meg_norm]["id"],
                    "builds": [],
                }

    for edge in edges.values():
        if edge["builds"]:
            edge["builds"].sort(key=lambda b: b.get("created_time", ""), reverse=True)
            latest = edge["builds"][0]
            edge["label"] = latest.get("name") or latest.get("uuid", "")[:8]
            edge["status"] = latest.get("status", "unknown")
        else:
            edge["label"] = ""
            edge["status"] = "unknown"

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/lineage")
async def get_lineage(days: int = Query(1, ge=1, le=30)) -> JSONResponse:
    """Return lineage DAG (nodes + edges + datasets) from recent DP builds."""
    datasets, scanned, matched, warning = await _scan_datasets_async(days)
    graph = _build_lineage_graph(datasets)
    result: dict[str, Any] = {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "datasets": datasets,
        "scanned": scanned,
        "matched": matched,
        "days": days,
    }
    if warning:
        result["warning"] = warning
    return JSONResponse(result)


@router.get("/recent-datasets")
async def recent_datasets(days: int = Query(1, ge=1, le=30)) -> JSONResponse:
    """Scan recent builds and return data processing datasets."""
    datasets, scanned, matched, warning = await _scan_datasets_async(days)
    resp: dict[str, Any] = {"datasets": datasets, "scanned": scanned, "matched": matched, "days": days}
    if warning:
        resp["warning"] = warning
    return JSONResponse(resp)


@router.post("/scan-prefixes")
async def add_scan_prefix(prefix: str = Query(..., min_length=1)) -> JSONResponse:
    """Add a COS scan prefix (invalidates the lineage cache)."""
    norm = prefix.strip().rstrip("/")
    if norm and norm not in _extra_scan_prefixes:
        _extra_scan_prefixes.append(norm)
    return JSONResponse({"prefixes": list(_extra_scan_prefixes)})


@router.delete("/scan-prefixes")
async def remove_scan_prefix(prefix: str = Query(..., min_length=1)) -> JSONResponse:
    """Remove a COS scan prefix (invalidates the lineage cache)."""
    norm = prefix.strip().rstrip("/")
    if norm in _extra_scan_prefixes:
        _extra_scan_prefixes.remove(norm)
    return JSONResponse({"prefixes": list(_extra_scan_prefixes)})


# ---------------------------------------------------------------------------
# COS-dependent endpoints (graceful 200 with error field when unconfigured)
# ---------------------------------------------------------------------------

def _cos_configured() -> bool:
    cfg = get_config()
    return bool(cfg.cos_endpoint and cfg.cos_access_key and cfg.cos_secret_key)


def _make_s3_client():
    cfg = get_config()
    try:
        import boto3
        from botocore.config import Config as BotoConfig
        return boto3.client(
            "s3",
            endpoint_url=cfg.cos_endpoint,
            aws_access_key_id=cfg.cos_access_key,
            aws_secret_access_key=cfg.cos_secret_key,
            config=BotoConfig(signature_version="s3v4"),
        )
    except ImportError:
        return None


@router.get("/node-counts")
async def get_node_counts(
    paths: str = Query(..., description="JSON array of {id, path}"),
) -> JSONResponse:
    """Return COS object counts per node (requires COS credentials + boto3)."""
    if not _cos_configured():
        return JSONResponse({"error": "not_configured", "counts": {}})

    import asyncio

    try:
        path_list: list[dict] = json.loads(paths)
    except Exception:
        return JSONResponse({"error": "invalid_paths", "counts": {}})

    cfg = get_config()
    bucket = cfg.cos_bucket

    def _count(path_info: dict) -> tuple[str, int]:
        s3 = _make_s3_client()
        if not s3:
            return path_info["id"], 0
        try:
            key = path_info["path"].strip().strip("/")
            if key.startswith(bucket + "/"):
                key = key[len(bucket) + 1:]
            count = 0
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=key + "/", Delimiter="/", MaxKeys=1000):
                count += page.get("KeyCount", 0)
            return path_info["id"], count
        except Exception as exc:
            logger.debug("node-counts %s: %s", path_info.get("id"), exc)
            return path_info["id"], 0

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(_cos_executor, _count, p) for p in path_list[:20]]
    )
    return JSONResponse({"counts": dict(results)})


@router.get("/pipeline-status")
async def get_pipeline_status(
    paths: str = Query(..., description="JSON array of {id, path, build_id?}"),
) -> JSONResponse:
    """Return pipeline stage summaries for megatron nodes (requires COS)."""
    if not _cos_configured():
        return JSONResponse({"error": "not_configured", "statuses": {}})

    import asyncio

    try:
        path_list: list[dict] = json.loads(paths)
    except Exception:
        return JSONResponse({"error": "invalid_paths", "statuses": {}})

    cfg = get_config()
    bucket = cfg.cos_bucket

    def _read_summary(path_info: dict) -> tuple[str, Optional[dict]]:
        s3 = _make_s3_client()
        if not s3:
            return path_info["id"], None
        try:
            key = path_info["path"].strip().strip("/")
            if key.startswith(bucket + "/"):
                key = key[len(bucket) + 1:]
            build_id = path_info.get("build_id", "")
            summary_key = f"{key.rstrip('/')}/pipeline/{build_id}/pipeline_summary.json"
            resp = s3.get_object(Bucket=bucket, Key=summary_key)
            return path_info["id"], json.loads(resp["Body"].read().decode("utf-8"))
        except Exception:
            return path_info["id"], None

    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *[loop.run_in_executor(_cos_executor, _read_summary, p) for p in path_list[:20]]
    )
    statuses = {nid: summary for nid, summary in results if summary is not None}
    return JSONResponse({"statuses": statuses})


@router.get("/report")
async def get_pipeline_report(
    megatron_path: str = Query(..., min_length=1),
    arrow_path: Optional[str] = Query(None),
    parquet_path: Optional[str] = Query(None),
    include_p1: bool = Query(True),
    include_tokens: bool = Query(False),
) -> JSONResponse:
    """Fetch full S1-S5 pipeline validation report from COS (requires vendored megatron CLI)."""
    if not _cos_configured():
        return JSONResponse({"error": "not_configured"})

    try:
        from dashboard.dashboard_data import run_full_report  # type: ignore[import]
    except ImportError:
        return JSONResponse({"error": "megatron_cli_not_available"})

    import asyncio

    cfg = get_config()

    def _run() -> dict:
        try:
            return run_full_report(  # type: ignore[no-untyped-call]
                megatron_path=megatron_path,
                endpoint=cfg.cos_endpoint,
                bucket=cfg.cos_bucket,
                arrow_path=arrow_path,
                parquet_path=parquet_path,
                include_p1=include_p1,
                include_tokens=include_tokens,
            )
        except Exception as exc:
            return {"error": str(exc)}

    loop = asyncio.get_event_loop()
    report = await loop.run_in_executor(_cos_executor, _run)
    return JSONResponse(content=json.loads(json.dumps(report, default=str)))
