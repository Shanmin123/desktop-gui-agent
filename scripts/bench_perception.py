"""实测感知模块各环节耗时，并输出一张带边界框的截图用于人工核对定位。

用法：
    python scripts/bench_perception.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_agent.perception import Perception, annotate, benchmark, imwrite


def main() -> None:
    print("正在实测，OCR 全图那一轮较慢，请稍候……\n")
    stats = benchmark(n=5)
    width = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"  {k:<{width}} : {v}")

    out_dir = Path(__file__).resolve().parents[1] / "logs"
    out_dir.mkdir(exist_ok=True)

    # OCR 跑原图，框画在缩放图上，靠归一化坐标对齐
    p = Perception()
    state, small = p.perceive()
    p.close()

    path = out_dir / "annotated.png"
    imwrite(str(path), annotate(small, state.elements))
    print(f"\n带框截图已存到 {path}，共标出 {len(state.elements)} 个元素。")


if __name__ == "__main__":
    main()
