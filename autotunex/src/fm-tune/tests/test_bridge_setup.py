"""Tests for autotune.bridge_setup — the opt-in gate for AutoTuneX bridge logging."""

from autotune.bridge_setup import resolve_bridge_settings


def test_disabled_when_url_absent():
    enabled, base_url = resolve_bridge_settings(None)
    assert enabled is False
    assert base_url is None


def test_enabled_when_url_present():
    enabled, base_url = resolve_bridge_settings("https://my-bridge.example.com")
    assert enabled is True
    assert base_url == "https://my-bridge.example.com"


def test_trailing_slash_preserved_for_caller():
    # The helper does not normalize; AutoTuneXAPI handles rstrip internally.
    enabled, base_url = resolve_bridge_settings("https://x/")
    assert enabled is True
    assert base_url == "https://x/"


def test_empty_string_is_disabled():
    # An empty --autotunex_server_url="" is treated as "not provided".
    enabled, base_url = resolve_bridge_settings("")
    assert enabled is False
    assert base_url is None
