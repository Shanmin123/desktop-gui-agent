"""scripts/probe_model.py 里换算口径和抽样的两个小函数。"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("probe_model", ROOT / "scripts" / "probe_model.py")
probe_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_model)


def test_to_norm_divides_by_the_size_of_each_space():
    size, resized = (1920, 1080), (1504, 848)
    assert probe_model.to_norm((752, 424), "pixel", size, resized) == (0.5, 0.5)
    assert probe_model.to_norm((500, 250), "rel1000", size, resized) == (0.5, 0.25)
    assert probe_model.to_norm((960, 540), "original", size, resized) == (0.5, 0.5)


def test_spread_samples_across_the_whole_set():
    rows = list(range(334))
    got = probe_model.spread(rows, 40)
    assert len(got) == 40 and got[0] == 0 and got[-1] > 320
    assert probe_model.spread(rows[:5], 40) == rows[:5]
