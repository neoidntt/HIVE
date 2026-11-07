"""A folder of whole stacks: one after another, a bad file does not stop the rest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from hive.pipeline import analyze_stacks
from hive_cli import main

from helpers import blob_volume, quick_config


def _folder(tmp_path):
    folder = tmp_path / "stacks"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", blob_volume(seed=1))
    (folder / "broken.tif").write_bytes(b"this is not a tiff")
    tifffile.imwrite(folder / "c.tif", blob_volume(seed=2))
    return folder


def test_every_stack_gets_its_own_folder_and_a_bad_file_is_reported_not_fatal(tmp_path):
    folder = _folder(tmp_path)
    cfg = quick_config(tmp_path, input_path=str(folder), output_folder=str(tmp_path / "out"), mode="3d")
    seen = []
    out = analyze_stacks(cfg, progress=lambda fraction, message: seen.append(fraction))
    assert [s["name"] for s in out["stacks"]] == ["a.tif", "c.tif"]
    assert [name for name, _ in out["errors"]] == ["broken.tif"]
    assert all(s["objects"] == 9 for s in out["stacks"])
    for name in ("a", "c"):
        assert (tmp_path / "out" / name / "HIVE_3D_measurements.csv").exists()
    combined = pd.read_csv(tmp_path / "out" / "combined_3d_measurements.csv")
    assert sorted(combined["image"].unique()) == ["a.tif", "c.tif"] and len(combined) == 18
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert cfg.input_path == str(folder)                        # the caller's settings are untouched


def test_the_command_line_flag_runs_each_file_as_a_stack(tmp_path, capsys):
    folder = _folder(tmp_path)
    code = main(["--mode", "3d", "--each-file", "--input", str(folder), "--output",
                 str(tmp_path / "cli"), "--quiet"])
    text = capsys.readouterr()
    assert code == 0 and "2 stack(s)" in text.out and "broken.tif" in text.err
    assert (tmp_path / "cli" / "combined_3d_summary.csv").exists()
