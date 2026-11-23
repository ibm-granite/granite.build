# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`autotunex.core.urls.split_origin`."""

from __future__ import annotations

import pytest

from autotunex.core.urls import split_origin


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://ui.example.com/x?y#z", "https://ui.example.com"),
        ("https://UI.Example.com", "https://ui.example.com"),
        ("https://ui.example.com:443", "https://ui.example.com"),
        ("http://ui.example.com:80", "http://ui.example.com"),
        ("https://ui.example.com:8443", "https://ui.example.com:8443"),
        ("http://[0:0::1]:5173/", "http://[::1]:5173"),
        ("http://[::FFFF:7F00:1]", "http://[::ffff:7f00:1]"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("https://my_host.example.com", "https://my_host.example.com"),
    ],
)
def test_split_origin_serializes_the_origin_the_way_a_browser_does(url: str, origin: str) -> None:
    split = split_origin(url)

    assert split is not None
    assert split[0] == origin


def test_split_origin_returns_the_parsed_url() -> None:
    """The caller redirects to the URL that was validated, not a second parse."""
    split = split_origin("https://ui.example.com/a\tb")

    assert split is not None
    assert split[1].geturl() == "https://ui.example.com/ab"


@pytest.mark.parametrize(
    "url",
    [
        # Not an absolute http(s) URL with a host.
        "ui.example.com",
        "/path",
        "//ui.example.com",
        "ftp://ui.example.com",
        "javascript:alert(1)",
        "https://",
        # Does not parse.
        "https://[",
        "https://ui.example.com:99999",
        # Userinfo.
        "https://user@ui.example.com",
        "https://user:pw@ui.example.com",
        # A host no browser origin can equal.
        "https://dash board.invalid",
        "https://dash%20board.invalid",
        "https://dash\\board.invalid",
        "https://dash^board.invalid",
        "http://[fe80::1%25eth0]",
        "http://[v1.fe]",
    ],
)
def test_split_origin_rejects_a_url_with_no_browser_origin(url: str) -> None:
    assert split_origin(url) is None
