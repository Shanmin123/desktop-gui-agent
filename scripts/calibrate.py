"""感知与控制联调：测量坐标定位的端到端误差。

对应大纲第 2 周第 4 项。流程是

    截图 -> OCR 得到元素框 -> 取中心点(归一化) -> 反归一化成像素
    -> 移动鼠标 -> 读回光标实际位置 -> 比对

只移动鼠标，不点击。跑完把鼠标放回原位，把鼠标甩到屏幕左上角可强制中断。

用法：
    python scripts/calibrate.py            # 默认取 9 个分散的元素
    python scripts/calibrate.py -n 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.control import Controller, PyAutoGUIBackend
from gui_agent.perception import Perception, annotate, imwrite


def pick_spread(elements, n):
    """按 3x3 网格分桶轮流取 n 个元素，让取样点分散到屏幕各处。"""
    buckets = {}
    for e in elements:
        cx, cy = e.center()
        buckets.setdefault((min(2, int(cx * 3)), min(2, int(cy * 3))), []).append(e)

    picked, keys = [], sorted(buckets)
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop(0))
    return picked


def move_and_verify(backend, want, retries: int = 3, settle: float = 0.05):
    """移动到目标像素并确认落点，返回 (实际落点, 重试次数)。

    光标可能被其他程序移动，一次读数对不上不代表坐标算错了，先重试再判定。
    """
    got = None
    for i in range(retries):
        backend.move(*want)
        time.sleep(settle)
        got = backend.position()
        if got == want:
            return got, i
    return got, retries


def detect_interference(backend, seconds: float = 0.6) -> bool:
    """不动鼠标，看它自己会不会动。会动就说明有别的程序在抢光标。"""
    a = backend.position()
    time.sleep(seconds)
    return a != backend.position()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=9, help="测试多少个元素")
    ap.add_argument("--delay", type=float, default=3.0, help="开始前等待秒数")
    args = ap.parse_args()

    print(f"即将接管鼠标 {args.delay} 秒后开始，只移动不点击，结束后放回原位。")
    print("需要中断就把鼠标甩到屏幕左上角。\n")
    time.sleep(args.delay)

    perception = Perception()
    controller = Controller(backend=PyAutoGUIBackend())
    origin = controller.backend.position()

    print("截图并识别中……")
    state, small = perception.perceive()
    perception.close()
    print(f"截图 {state.width}x{state.height}，识别到 {len(state.elements)} 个元素\n")

    if not state.elements:
        print("屏幕上没识别到文字，换个有文字的界面再跑。")
        return

    targets = pick_spread(state.elements, args.n)
    errors, retried = [], 0
    print(f"{'编号':<6}{'目标像素':<18}{'实际落点':<18}{'误差':<8}{'重试':<6}文字")
    for e in targets:
        want = controller.to_pixel(e.center())
        got, tries = move_and_verify(controller.backend, want)
        err = max(abs(want[0] - got[0]), abs(want[1] - got[1]))
        errors.append(err)
        retried += bool(tries)
        text = e.text[:20].replace("\n", " ")
        print(f"{e.id:<6}{str(want):<18}{str(got):<18}{err:<8}{tries:<6}{text}")

    interfered = detect_interference(controller.backend)
    controller.backend.move(*origin)

    worst = max(errors)
    print(f"\n共测 {len(errors)} 个点，最大误差 {worst} px，平均 {sum(errors)/len(errors):.2f} px")
    if worst <= 2:
        print("坐标链路正常。" + (f"（其中 {retried} 个点重试过）" if retried else ""))
    elif interfered or retried:
        print(
            "误差来自外部干扰：有别的程序在移动光标，或测试期间碰了鼠标。\n"
            "坐标换算本身是否正确，请在没有干扰时重跑，"
            "或直接跑 pytest 里的坐标换算测试。"
        )
    else:
        print("误差偏大且无干扰迹象，检查缩放系数或多显示器设置。")

    out = Path(__file__).resolve().parents[1] / "logs" / "calibrate.png"
    imwrite(str(out), annotate(small, targets))
    print(f"被测元素已标注在 {out}")


if __name__ == "__main__":
    main()
