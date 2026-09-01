"""Tests for real-data OHT runner checkpoint handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

RUNNER = Path(__file__).with_name("run_oht_n_pro_real.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_oht_n_pro_real", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_data_runner_removes_stale_default_checkpoint(tmp_path: Path) -> None:
    runner = _load_runner()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stale checkpoint")
    keep = tmp_path / "config_protrider_stability_run.yaml"
    keep.write_text("out_dir: test\n", encoding="utf-8")

    assert runner.remove_stale_default_checkpoint(tmp_path) is True

    assert not checkpoint.exists()
    assert keep.exists()


def test_real_data_runner_noops_without_stale_checkpoint(tmp_path: Path) -> None:
    runner = _load_runner()

    assert runner.remove_stale_default_checkpoint(tmp_path) is False
