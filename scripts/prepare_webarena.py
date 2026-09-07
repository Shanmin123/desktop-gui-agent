"""把 WebArena 的任务定义转成本项目的格式。

对应大纲第 3 周第 1 项。

WebArena 不是截图数据集，是一套要用 docker 自建的网站环境加任务定义。仓库里
`config_files/test.raw.json` 是 812 条任务规格，没有截图，所以做不了多模态微调
数据。能用的是两样：

1. 812 条真实的 GUI 任务指令，以及它们的参数化模板。大纲第 7 周要「设计包含 20 个
   不同难度桌面任务的测试集」，这批指令是现成的参照。
2. 三种程序化验收方式（string_match / url_match / program_html）。本项目的验收
   条件也是程序判断，这里能对照着补。

用法：
    python scripts/prepare_webarena.py            # 自动下载 test.raw.json
    python scripts/prepare_webarena.py --src 本地/test.raw.json
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "webarena"
RAW_URL = "https://raw.githubusercontent.com/web-arena-x/webarena/main/config_files/test.raw.json"


def load(src: Path) -> list:
    if not src.is_file():
        src.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 {RAW_URL} ……")
        urllib.request.urlretrieve(RAW_URL, src)
    return json.loads(src.read_text(encoding="utf-8"))


def to_record(task: dict) -> dict:
    ev = task.get("eval") or {}
    return {
        "source": "webarena",
        "task_id": task.get("task_id"),
        "instruction": task.get("intent", ""),
        "template": task.get("intent_template", ""),
        "template_id": task.get("intent_template_id"),
        "params": task.get("instantiation_dict") or {},
        "sites": task.get("sites") or [],
        "requires_login": bool(task.get("require_login")),
        "check_types": ev.get("eval_types") or [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT / "test.raw.json"))
    args = ap.parse_args()

    tasks = load(Path(args.src))
    records = [to_record(t) for t in tasks]
    print(f"任务 {len(records)} 条")

    sites = Counter(s for r in records for s in r["sites"])
    checks = Counter(c for r in records for c in r["check_types"])
    print("\n站点：")
    for k, v in sites.most_common():
        print(f"  {k:<16} {v}")
    print("\n验收方式：")
    for k, v in checks.most_common():
        print(f"  {k:<16} {v}")
    print(f"\n指令模板 {len({r['template_id'] for r in records})} 个，"
          f"需要登录的 {sum(r['requires_login'] for r in records)} 条")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "tasks.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                 encoding="utf-8")
    print(f"\n{p}  {len(records)} 条")
    print("没有截图，不能作为多模态微调数据；用作第 7 周测试集设计和验收条件的参照。")


if __name__ == "__main__":
    main()
