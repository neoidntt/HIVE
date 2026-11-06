"""Saved settings and run records: a result must be reproducible from its record."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hive import provenance
from hive.config import HiveConfig
from hive.pipeline import analyze_folder

from helpers import quick_config, write_synthetic_folder


def test_config_survives_a_json_round_trip():
    cfg = HiveConfig(seed=7, extra_channels=["GFP"], conditions={"a.tif": "ctrl"})
    cfg.preprocess.tophat_radius = (2.0, 25.0, 25.0)       # JSON turns tuples into lists
    cfg.export.imagej_rois = False
    again = HiveConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    assert isinstance(again.preprocess.tophat_radius, tuple)


def test_unknown_settings_are_refused_not_ignored():
    with pytest.raises(ValueError, match="bogus"):
        HiveConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match=r"arbor\.nonsense"):
        HiveConfig.from_dict({"arbor": {"nonsense": 1}})


def test_an_older_record_without_new_sections_gets_defaults():
    old = HiveConfig().to_dict()
    for key in ("export", "large_image", "extra_channels", "conditions"):
        del old[key]
    assert HiveConfig.from_dict(old) == HiveConfig()


def test_invalid_values_in_a_record_are_still_rejected():
    bad = HiveConfig().to_dict()
    bad["voxel_size_xy"] = -1
    with pytest.raises(ValueError, match="Voxel sizes"):
        HiveConfig.from_dict(bad)


def test_a_run_writes_its_record_and_reloading_reproduces_it_exactly(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    cfg = quick_config(tmp_path, seed=11)
    first = analyze_folder(cfg)

    record = provenance.read_run_record(cfg.output_folder)      # a folder is accepted
    assert record.config == cfg
    assert record.record["seed"] == 11
    assert record.record["hive_release"] == "0.0.4"
    assert "numpy" in record.record["environment"]["packages"]
    assert len(record.record["inputs"]) == 3 and record.record["inputs"][0]["sha256"]

    rerun_cfg = record.config
    rerun_cfg.output_folder = str(tmp_path / "rerun")
    second = analyze_folder(rerun_cfg)
    pd.testing.assert_frame_equal(first["measurements"], second["measurements"])
    pd.testing.assert_frame_equal(first["summary"], second["summary"])


def test_a_bare_settings_file_without_the_record_wrapper_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(HiveConfig(seed=3).to_dict()))
    assert provenance.read_run_record(path).config.seed == 3


def test_bad_files_fail_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        provenance.read_run_record(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        provenance.read_run_record(broken)


def test_library_changes_since_the_run_are_reported():
    record = {"environment": {"python": "3.0.0", "packages": {"numpy": "0.0.1", "ghost": "1"}}}
    lines = provenance.environment_differences(record)
    assert any(l.startswith("python") for l in lines)
    assert any("numpy: saved 0.0.1" in l for l in lines)
    assert any("ghost" in l and "not installed" in l for l in lines)
    assert provenance.environment_differences(
        {"environment": {"packages": provenance.package_versions()}}) == []


def test_large_inputs_are_not_hashed_past_the_budget(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 100)
    spent = [50]
    assert provenance.file_fingerprint(f, spent)["sha256"] is None
    assert provenance.file_fingerprint(f)["sha256"]
