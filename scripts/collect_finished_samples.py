"""造「任务已经做完」的训练样本：程序化把任务做到终态，截图，回答 finished。

为什么要这个：真机跑 25 个任务时，24 个跑到步数上限才停，只有 1 个主动给出 finished。
查下来 ScreenAgent test 的 353 步里 finished 的真值一条都没有，交付配方的 375 条动作样本里
也是 0 条——模型没见过「做完了该收尾」的例子，自然不会收尾。

这里不靠人工标注：任务的终态本来就能用程序造出来（写文件、打开对应的程序），造完截一张图，
配上「已执行」里那几步，回答就是 finished。动作只由本脚本发出，不调模型、不点鼠标。

用法（会短暂占用桌面：开记事本 / 计算器 / 浏览器，跑完自己关掉）：
    python scripts/collect_finished_samples.py --dry-run     # 只列要造哪些，不动桌面
    python scripts/collect_finished_samples.py               # 采集，约 3 分钟
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent import tasks as T  # noqa: E402
from gui_agent.chain import render_target_prompt  # noqa: E402
from gui_agent.perception import Perception, imwrite  # noqa: E402
from gui_agent.schema import Action, ScreenState, Step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "finetune_finished"
SHOTS = OUT / "shots"

# 终态怎么造：写文件的直接写，然后用记事本把结果打开——这就是「做完了」时屏幕上的样子。
# 历史写成几条与任务相符的动作，和推理时喂进去的格式一致（gui_agent/agent.format_history）。
CASES = [
    ("type_and_save", "用记事本新建文件，输入「你好」，保存为 {s}/output.txt",
     [("click", "记事本的新建"), ("type", "你好"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "output.txt", "你好\n")),
    ("write_three_lines", "用记事本新建文件，写三行：苹果、香蕉、橙子，每行一个，保存为 {s}/fruits.txt",
     [("type", "苹果"), ("type", "香蕉"), ("type", "橙子"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "fruits.txt", "苹果\n香蕉\n橙子\n")),
    ("append_line_in_place", "用记事本打开 {s}/todo.txt，在末尾另起一行写「买牛奶」，直接保存",
     [("click", "文件末尾"), ("type", "买牛奶"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "todo.txt", "买面包\n买牛奶\n")),
    ("delete_log_line", "用记事本打开 {s}/app.log，删掉含有 DEBUG 的那一行，保存",
     [("click", "DEBUG 那一行"), ("hotkey", "shift+home"), ("hotkey", "delete"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "app.log", "INFO 启动完成\nINFO 收到第一个请求\n")),
    ("replace_all_text", "用记事本把 {s}/draft.txt 里的「草稿」全部替换成「终稿」，然后保存",
     [("hotkey", "ctrl+h"), ("type", "终稿"), ("click", "全部替换"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "draft.txt", "第一版终稿\n第二版终稿\n第三版终稿\n")),
    ("copy_between_files", "把 {s}/source.txt 里的文字复制到新文件 {s}/copy.txt 并保存",
     [("hotkey", "ctrl+a"), ("hotkey", "ctrl+c"), ("hotkey", "ctrl+v"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "copy.txt", "要复制的内容\n")),
    ("calculate_to_file", "用计算器算出 128 × 256 的结果，把结果写进 {s}/calc.txt 并保存",
     [("click", "计算器的等于"), ("hotkey", "ctrl+c"), ("hotkey", "ctrl+v"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "calc.txt", "32768\n")),
    ("extract_meeting_time", "打开 {s}/meeting.txt，找到会议开始的时间，把这个时间写进 {s}/answer.txt 并保存",
     [("click", "会议时间那一行"), ("type", "14:30"), ("hotkey", "ctrl+s")],
     lambda s: _write_and_open(s, "answer.txt", "14:30\n")),
    ("append_and_save_as", "用记事本打开 {s}/sample.txt，在末尾加一行「已审阅」，另存为 {s}/reviewed.txt",
     [("type", "已审阅"), ("hotkey", "ctrl+shift+s"), ("type", "reviewed.txt"), ("click", "保存按钮")],
     lambda s: _write_and_open(s, "reviewed.txt", "示例文本\n已审阅\n")),
    ("open_calculator", "打开计算器", [("hotkey", "win+r"), ("type", "calc"), ("hotkey", "enter")],
     lambda s: _open_calculator()),
]


def _write_and_open(scratch: Path, name: str, text: str):
    path = scratch / name
    path.write_text(text, encoding="utf-8")
    T._open_notepad(path)
    return f"记事本打开 {name}"


def _open_calculator():
    subprocess.Popen(["calc.exe"], shell=True)
    return "计算器已打开"


def _history(steps) -> list:
    """把 (动作, 内容) 列表变成 Step，渲染进提示词的「已执行」。"""
    out = []
    for kind, text in steps:
        act = (Action(type=kind, point=(0.42, 0.37)) if kind in ("click", "left_double", "right_single")
               else Action(type=kind, text=text))
        out.append(Step(ScreenState(width=0, height=0), act, ok=True, changed=True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只列要造哪些样本，不动桌面")
    ap.add_argument("--wait", type=float, default=1.8, help="打开程序后等几秒再截图")
    args = ap.parse_args()

    scratch = T._scratch()
    if args.dry_run:
        for tid, instruction, steps, _ in CASES:
            print(f"{tid:<22} 历史 {len(steps)} 步  {instruction.format(s=scratch)[:70]}")
        print(f"\n共 {len(CASES)} 条，采集时会打开记事本 / 计算器，跑完自动关掉")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)
    perception = Perception()
    rows = []
    try:
        for tid, instruction, steps, make in CASES:
            T._close_notepad()
            note = make(scratch)
            time.sleep(args.wait)
            state, model_img = perception.perceive(run_ocr=False)
            shot = SHOTS / f"{tid}.png"
            imwrite(str(shot), model_img)
            text = instruction.format(s=scratch)
            rows.append({
                "kind": "action",
                "source": "finished_synth",
                "image": str(shot),
                "prompt": render_target_prompt(text, _history(steps)),
                "response": json.dumps({"thought": "要求的结果已经在屏幕上了，任务完成",
                                        "action": {"type": "finished"}}, ensure_ascii=False),
            })
            print(f"  {tid:<22} {note}", flush=True)
    finally:
        T._close_notepad()
        perception.close()

    (OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} 条 finished 样本已写到 {OUT / 'train.jsonl'}，截图在 {SHOTS}")


if __name__ == "__main__":
    main()
