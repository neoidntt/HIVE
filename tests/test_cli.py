"""Command-line additions: settings reload, channels, conditions, inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hive_cli import main

from helpers import write_ome, write_synthetic_folder


def _two_channel_folder(folder):
    from hive.synthetic import make_synthetic_2d
    folder.mkdir()
    for name, image in make_synthetic_2d(0):
        stack = np.stack([image, image * 0.5]).astype(np.float32)
        write_ome(folder / name.replace(".tif", ".ome.tif"), stack, "CYX", ["DAPI", "GFP"], xy=0.5)


def test_inspect_lists_channels_and_pixel_size(tmp_path, capsys):
    path = write_ome(tmp_path / "a.ome.tif", np.zeros((2, 8, 8), np.uint16), "CYX", ["DAPI", "GFP"])
    assert main(["--inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "DAPI" in out and "GFP" in out and "0.5 um" in out and "ome-tiff" in out


def test_a_run_can_be_repeated_exactly_from_its_record(tmp_path, capsys):
    write_synthetic_folder(tmp_path / "in")
    args = ["--input", str(tmp_path / "in"), "--no-arbor", "--no-clahe", "--quiet",
            "--min-size", "30", "--seed", "9"]
    assert main(args + ["--output", str(tmp_path / "first")]) == 0
    assert main(["--settings", str(tmp_path / "first"), "--input", str(tmp_path / "in"),
                 "--output", str(tmp_path / "second"), "--min-size", "5", "--quiet"]) == 0
    assert "ignored" in capsys.readouterr().err               # --min-size was not allowed to change it
    first = pd.read_csv(tmp_path / "first" / "combined_measurements.csv")
    second = pd.read_csv(tmp_path / "second" / "combined_measurements.csv")
    pd.testing.assert_frame_equal(first, second)


def test_named_channels_extra_channels_and_conditions_work_together(tmp_path):
    _two_channel_folder(tmp_path / "in")
    (tmp_path / "c.csv").write_text("image,condition\nsynthetic_dim.ome.tif,low\n"
                                    "synthetic_reference.ome.tif,low\nsynthetic_bright.ome.tif,high\n")
    code = main(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"),
                 "--channel", "DAPI", "--extra-channels", "GFP", "--conditions",
                 str(tmp_path / "c.csv"), "--no-arbor", "--quiet"])
    assert code == 0
    table = pd.read_csv(tmp_path / "out" / "combined_measurements.csv")
    assert "coloc_GFP_pearson" in table.columns and len(table) == 18
    assert (tmp_path / "out" / "group_comparison" / "group_statistics.csv").exists()


def test_an_unknown_channel_is_reported_with_the_choices(tmp_path, capsys):
    _two_channel_folder(tmp_path / "in")
    main(["--input", str(tmp_path / "in"), "--output", str(tmp_path / "o"), "--channel", "TRITC",
          "--quiet"])
    assert "DAPI" in capsys.readouterr().err


def test_bad_settings_files_give_a_message_not_a_traceback(tmp_path, capsys):
    (tmp_path / "bad.json").write_text('{"nonsense": 1}')
    assert main(["--settings", str(tmp_path / "bad.json")]) == 1
    assert "nonsense" in capsys.readouterr().err
    assert main(["--settings", str(tmp_path / "missing.json")]) == 1


def test_baseline_preview_picture_is_written(tmp_path):
    write_synthetic_folder(tmp_path / "in")
    picture = tmp_path / "p.png"
    assert main(["--preview-baseline", str(tmp_path / "in" / "synthetic_dim.tif"), str(picture),
                 "--channel", "gray"]) == 0
    assert picture.stat().st_size > 5000
