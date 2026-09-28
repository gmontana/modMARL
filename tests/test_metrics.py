from __future__ import annotations

from modmarl.common.metrics import success_count, success_indicator


def test_success_helpers_handle_missing_success() -> None:
    info = {"success": None}
    assert success_count(info) == 0
    assert success_indicator(info) == "n/a"


def test_success_helpers_handle_boolean_success() -> None:
    assert success_count({"success": True}) == 1
    assert success_count({"success": False}) == 0
    assert success_indicator({"success": True}) == 1
    assert success_indicator({"success": False}) == 0
