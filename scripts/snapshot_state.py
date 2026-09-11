"""记录跑实验前的桌面状态，跑完对一遍，确认没留下东西。

live 实验会切分辨率、开关程序、写文件。分辨率靠 display.resolution 自己还原，
但程序和窗口不会自己退出——之前就出现过实验结束后留着记事本和浏览器窗口。
跑之前存一份，跑之后 --check 对比，差异一眼看得到。

用法：
    python scripts/snapshot_state.py --save     # 实验前
    python scripts/snapshot_state.py --check    # 实验后
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.tasks import EDITOR_NAME, browser_pids, pids_of, window_titles

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "logs" / "_state_before.json"


def snapshot() -> dict:
    from gui_agent import display

    state = {
        "browsers": sorted(browser_pids()),
        "editors": sorted(pids_of(EDITOR_NAME)),
        "titles": sorted(window_titles()),
    }
    try:
        state["resolution"] = list(display.current_resolution())
        state["scaling"] = display.scaling_percent()
    except Exception as e:  # 非 Windows 或取不到时不该影响进程和窗口的记录
        state["display_error"] = f"{type(e).__name__}: {e}"
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    now = snapshot()

    if args.save:
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(now, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已记录：分辨率 {now.get('resolution')} 缩放 {now.get('scaling')}%，"
              f"浏览器进程 {len(now['browsers'])} 个，编辑器 {len(now['editors'])} 个，"
              f"窗口 {len(now['titles'])} 个")
        print(f"存到 {SNAPSHOT}")
        return

    if not SNAPSHOT.is_file():
        raise SystemExit(f"没有 {SNAPSHOT}，实验前先跑一次 --save")
    before = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    clean = True
    for key, label in [("resolution", "分辨率"), ("scaling", "系统缩放")]:
        if before.get(key) != now.get(key):
            clean = False
            print(f"{label}没还原：{before.get(key)} -> {now.get(key)}")

    for key, label in [("browsers", "浏览器进程"), ("editors", "编辑器进程"), ("titles", "窗口")]:
        added = sorted(set(now.get(key, [])) - set(before.get(key, [])))
        if added:
            clean = False
            print(f"多出来的{label}（{len(added)} 个）：{added[:8]}")

    print("桌面状态和实验前一致" if clean else "\n上面这些要手工收拾掉")


if __name__ == "__main__":
    main()
