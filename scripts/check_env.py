"""开发环境与基础工具库检查。

对应大纲第 1 周第 4 项：搭建 Python 开发环境，配置 CUDA 加速，测试基础工具库。

用法：
    python scripts/check_env.py           # 快速检查，不下载 OCR 模型
    python scripts/check_env.py --ocr     # 额外实测 OCR（首次会下载模型权重）
"""

import argparse
import importlib
import platform
import sys
import time

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'OK ' if ok else 'FAIL'}] {name:<22} {detail}")


def check_python():
    v = sys.version_info
    record("Python", v >= (3, 10), f"{platform.python_version()}（大纲要求 3.10+）")


def check_torch():
    try:
        import torch
    except Exception as e:
        record("PyTorch", False, str(e))
        return
    ver = torch.__version__
    major_minor = tuple(int(x) for x in ver.split("+")[0].split(".")[:2])
    record("PyTorch", major_minor >= (2, 2), f"{ver}（大纲要求 2.2+）")

    cuda = torch.cuda.is_available()
    if cuda:
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        record("CUDA", True, f"{name}，显存 {total:.1f}GB（当前空闲 {free:.1f}GB）")
        # 实际跑一次矩阵乘，确认不只是能识别到卡
        try:
            a = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (a @ a).sum().item()
            torch.cuda.synchronize()
            record("CUDA 计算", True, f"2048x2048 matmul 耗时 {(time.perf_counter()-t0)*1000:.1f} ms")
            del a
            torch.cuda.empty_cache()
        except Exception as e:
            record("CUDA 计算", False, str(e))
    else:
        record("CUDA", False, "不可用，微调需改用 Colab")


def check_screenshot():
    """截图：mss 与 pyautogui 各测一次，并比对两者的屏幕尺寸。"""
    mss_size = None
    try:
        import mss
        import numpy as np

        with mss.mss() as sct:
            mon = sct.monitors[1]
            t0 = time.perf_counter()
            for _ in range(10):
                shot = sct.grab(mon)
            dt = (time.perf_counter() - t0) / 10 * 1000
            img = np.array(shot)
            mss_size = (img.shape[1], img.shape[0])
            record("mss 截图", True, f"{mss_size[0]}x{mss_size[1]}，平均 {dt:.1f} ms/帧")
    except Exception as e:
        record("mss 截图", False, str(e))

    try:
        import pyautogui

        pag_size = tuple(pyautogui.size())
        record("pyautogui", True, f"报告屏幕尺寸 {pag_size[0]}x{pag_size[1]}")

        if mss_size and pag_size != mss_size:
            ratio = mss_size[0] / pag_size[0]
            record(
                "坐标系一致性",
                False,
                f"两者不一致，比例 {ratio:.2f}（系统缩放），控制模块需做坐标换算",
            )
        elif mss_size:
            record("坐标系一致性", True, "截图尺寸与控制尺寸一致")
    except Exception as e:
        record("pyautogui", False, str(e))


def check_simple_imports():
    libs = [
        ("pynput", "pynput"),
        ("OpenCV", "cv2"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("datasets", "datasets"),
        ("langchain", "langchain"),
        ("matplotlib", "matplotlib"),
    ]
    for label, mod in libs:
        try:
            m = importlib.import_module(mod)
            record(label, True, getattr(m, "__version__", ""))
        except Exception as e:
            record(label, False, type(e).__name__ + ": " + str(e)[:60])


def check_ocr(run_ocr):
    for label, mod in [("easyocr", "easyocr"), ("paddleocr", "paddleocr")]:
        try:
            importlib.import_module(mod)
            record(label, True, "导入成功" + ("" if run_ocr else "（未实测，加 --ocr 实测）"))
        except Exception as e:
            record(label, False, type(e).__name__ + ": " + str(e)[:60])

    if not run_ocr:
        return

    # 实测：截当前屏幕，跑一次识别，看能不能读出文字
    try:
        import easyocr
        import mss
        import numpy as np

        with mss.mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
        reader = easyocr.Reader(["ch_sim", "en"], gpu=True)
        t0 = time.perf_counter()
        out = reader.readtext(img)
        dt = time.perf_counter() - t0
        sample = out[0][1] if out else "(无)"
        record("easyocr 实测", True, f"识别 {len(out)} 处文字，耗时 {dt:.1f}s，例：{sample}")
    except Exception as e:
        record("easyocr 实测", False, type(e).__name__ + ": " + str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", action="store_true", help="实测 OCR（首次会下载模型）")
    args = ap.parse_args()

    print(f"系统：{platform.platform()}\n")
    check_python()
    check_torch()
    check_screenshot()
    check_simple_imports()
    check_ocr(args.ocr)

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n合计 {len(results)} 项，通过 {len(results)-len(failed)} 项。")
    if failed:
        print("未通过：" + "、".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
