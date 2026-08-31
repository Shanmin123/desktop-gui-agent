"""数据集预处理：把公开 grounding 数据转成本项目的格式。

对应大纲第 3 周第 1 项。

只写清单，不另存图片：图片留在 HuggingFace 缓存里，训练时按 (数据集, 下标) 取。

输出：
    data/grounding/train.jsonl    ShowUI-desktop 划分出的训练集
    data/grounding/val.jsonl      ShowUI-desktop 划分出的验证集
    data/grounding/test.jsonl     ScreenSpot 的桌面分片，做定位精度评测
    data/grounding/samples/*.png  抽样画框，人工核对坐标约定是否一致

用法：
    python scripts/prepare_data.py
    python scripts/prepare_data.py --check-only    # 只抽样画框，不写清单
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from datasets import load_dataset

from gui_agent.perception import annotate, imwrite
from gui_agent.schema import Element

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "grounding"

# ScreenSpot 里属于桌面的来源。剩下的 ios / android / web 不是本项目的场景。
DESKTOP_SOURCES = {"windows", "macos"}


def meta_rows(ds) -> list:
    """只取元数据，跳过图片列。直接索引 `ds[i]` 会连带解码图片。"""
    cols = [c for c in ds.column_names if c != "image"]
    return ds.select_columns(cols).to_list()


def to_record(sample, source: str, index: int) -> dict:
    """统一成一条清单记录。

    两个数据集的 bbox 都已经是归一化的 [x1,y1,x2,y2]，和 schema.py 的约定一致，
    不需要转换。point 缺失时取 bbox 中心。
    """
    x1, y1, x2, y2 = [float(v) for v in sample["bbox"]]
    point = sample.get("point")
    if point is None:
        point = [(x1 + x2) / 2, (y1 + y2) / 2]

    return {
        "source": source,
        "index": index,
        "instruction": sample["instruction"],
        "bbox": [x1, y1, x2, y2],
        "point": [float(point[0]), float(point[1])],
        "platform": sample.get("data_source", "desktop"),
        "element_type": sample.get("data_type") or sample.get("type", ""),
    }


def safe_name(text: str, limit: int = 20) -> str:
    """把指令截短并清成合法文件名。

    Windows 下冒号会被当成 NTFS 数据流分隔符，需要和其他非法字符一并替换。
    """
    cleaned = "".join("_" if c in '<>:"/\\|?*' or ord(c) < 32 else c for c in text[:limit])
    return cleaned.strip(" .") or "unnamed"


def dump_samples(ds, records, out_dir: Path, n: int = 6, tag: str = "") -> None:
    """抽几条把框画到原图上，人工确认坐标约定与本项目一致。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in random.sample(records, min(n, len(records))):
        img = np.array(ds[r["index"]]["image"].convert("RGB"))[:, :, ::-1]  # RGB->BGR
        e = Element(id=0, bbox=tuple(r["bbox"]), text=r["instruction"])
        name = f"{tag}_{r['index']}_{safe_name(r['instruction'])}.png"
        imwrite(str(out_dir / name), annotate(img, [e]))


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只抽样画框")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # --- 评测集：ScreenSpot 的桌面部分 ---
    print("加载 ScreenSpot ……")
    ss = load_dataset("rootsautomation/ScreenSpot", split="test")
    test = [
        to_record(row, "screenspot", i)
        for i, row in enumerate(meta_rows(ss))
        if row["data_source"] in DESKTOP_SOURCES
    ]
    print(f"  全量 {len(ss)}，桌面分片 {len(test)}")

    # --- 训练集：ShowUI-desktop ---
    print("加载 ShowUI-desktop ……")
    su = load_dataset("showlab/ShowUI-desktop", split="train")
    train_all = [to_record(row, "showui", i) for i, row in enumerate(meta_rows(su))]
    random.shuffle(train_all)
    n_val = int(len(train_all) * args.val_ratio)
    val, train = train_all[:n_val], train_all[n_val:]
    print(f"  全量 {len(train_all)}，训练 {len(train)}，验证 {len(val)}")

    # --- 抽样核对坐标约定 ---
    sample_dir = OUT / "samples"
    dump_samples(ss, test, sample_dir, tag="screenspot")
    dump_samples(su, train, sample_dir, tag="showui")
    print(f"\n抽样画框已存到 {sample_dir}，请人工确认框落在指令描述的控件上。")

    if args.check_only:
        return

    for name, recs in [("train", train), ("val", val), ("test", test)]:
        write_jsonl(OUT / f"{name}.jsonl", recs)
        print(f"  {name}.jsonl  {len(recs)} 条")

    # --- 统计 ---
    print("\n评测集按平台和控件类型分布：")
    dist = {}
    for r in test:
        dist[(r["platform"], r["element_type"])] = dist.get((r["platform"], r["element_type"]), 0) + 1
    for k in sorted(dist):
        print(f"  {k[0]:<10}{k[1]:<8}{dist[k]}")


if __name__ == "__main__":
    main()
