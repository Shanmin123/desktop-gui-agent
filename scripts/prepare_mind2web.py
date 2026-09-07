"""把 Multimodal-Mind2Web 转成本项目的任务拆解语料。

对应大纲第 3 周第 1 项。

    python scripts/prepare_mind2web.py            # 首次会下载 13.6 GB
    python scripts/prepare_mind2web.py --limit 2  # 只读前 2 个分片，先确认跑得通

数据里一行是一步操作，同一个 `annotation_id` 的若干行组成一条完整任务。每行都带
截图、原始 HTML 和候选元素，整份 13.6 GB；本脚本只读元数据那几列，不解码截图，
所以很快。截图留在 HuggingFace 缓存里，训练时按行号取。

这是 Web 域的数据，不用于桌面定位训练，取它的两样东西：
  1. 任务 -> 步骤序列，用作任务拆解能力的语料
  2. 操作类型分布，对照桌面端动作空间的覆盖情况

输出：
    data/mind2web/tasks.jsonl   一条任务一行，带完整步骤序列
    data/mind2web/steps.jsonl   一步一行，带该步的操作类型和目标描述
"""

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mind2web"
REPO = "osunlp/Multimodal-Mind2Web"

# 只读这几列，跳过 screenshot / raw_html / cleaned_html / 候选元素
COLUMNS = [
    "annotation_id", "action_uid", "operation", "website", "domain", "subdomain",
    "confirmed_task", "action_reprs", "target_action_index", "target_action_reprs",
]


def snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=REPO, repo_type="dataset", max_workers=8))


def parse_op(raw: str) -> tuple:
    """operation 是一段 JSON 字符串，取出操作类型和输入内容。"""
    try:
        d = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return "", ""
    return d.get("original_op") or d.get("op") or "", d.get("value") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只读前 N 个分片")
    ap.add_argument("--split", default="train", help="train / test_task / test_website / test_domain")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = sorted((snapshot() / "data").glob(f"{args.split}-*.parquet"))
    if not files:
        raise SystemExit(f"没有 {args.split} 分片")
    if args.limit:
        files = files[: args.limit]
    print(f"{args.split}：{len(files)} 个分片")

    steps, ops, doms = [], Counter(), Counter()
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=COLUMNS):
            for r in batch.to_pylist():
                op, value = parse_op(r.get("operation"))
                ops[op] += 1
                doms[r.get("domain") or ""] += 1
                steps.append({
                    "source": "mind2web",
                    "annotation_id": r.get("annotation_id"),
                    "action_uid": r.get("action_uid"),
                    "task": r.get("confirmed_task") or "",
                    "step_index": r.get("target_action_index"),
                    "operation": op,
                    "value": value,
                    "target": r.get("target_action_reprs") or "",
                    "website": r.get("website") or "",
                    "domain": r.get("domain") or "",
                    "subdomain": r.get("subdomain") or "",
                    "all_steps": list(r.get("action_reprs") or []),
                })

    # 同一个 annotation_id 的若干步组成一条任务
    tasks = OrderedDict()
    for s in steps:
        tasks.setdefault(s["annotation_id"], {
            "source": "mind2web",
            "annotation_id": s["annotation_id"],
            "task": s["task"],
            "website": s["website"],
            "domain": s["domain"],
            "subdomain": s["subdomain"],
            "steps": s["all_steps"],
            "n_steps": len(s["all_steps"]),
        })

    print(f"\n步数 {len(steps)}，任务 {len(tasks)} 条")
    lens = [t["n_steps"] for t in tasks.values()]
    if lens:
        print(f"每条任务 {min(lens)}~{max(lens)} 步，平均 {sum(lens)/len(lens):.1f} 步")

    print("\n操作类型：")
    for k, v in ops.most_common():
        print(f"  {k or '(空)':<10} {v}")
    print("\n领域：")
    for k, v in doms.most_common():
        print(f"  {k or '(空)':<14} {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("steps", steps), ("tasks", list(tasks.values()))]:
        p = OUT / f"{name}.jsonl"
        # all_steps 每行都重复一份完整序列，写 steps 时去掉，任务那份里有
        payload = [{k: v for k, v in r.items() if k != "all_steps"} for r in rows]
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload),
                     encoding="utf-8")
        print(f"\n{p}  {len(payload)} 条")


if __name__ == "__main__":
    main()
