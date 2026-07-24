# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# api/tests/test_cors_expose_headers.py
# Asserts the CORS middleware exposes the tus headers the browser client must read.
# Does not start a server; inspects the configured middleware kwargs.
import importlib


def test_cors_exposes_tus_headers():
    server = importlib.import_module("server")
    expose = None
    for mw in server.app.user_middleware:
        if "CORSMiddleware" in str(mw.cls):
            expose = mw.kwargs.get("expose_headers")
    assert expose is not None, "CORSMiddleware has no expose_headers"
    # Assert the FULL tus header set the browser client must read — a missing
    # one silently breaks resume (the client can't read Location/Upload-Offset).
    for h in (
        "Location",
        "Upload-Offset",
        "Upload-Length",
        "Tus-Resumable",
        "Tus-Version",
        "Tus-Extension",
        "Tus-Max-Size",
        "Upload-Expires",
        "Upload-Metadata",
    ):
        assert h in expose, f"expose_headers missing {h}"
