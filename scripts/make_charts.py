"""从 logs/ 里的评测结果画报告用的图，存到 docs/figures/。

对应大纲第 7 周交付的「性能分析可视化图表」。缺哪份日志就跳过哪张图，不补数。

  screenspot_by_category.png   ScreenSpot 桌面 334 条，按平台和元素类型分组的定位准确率
  screenspot_vs_tokens.png     视觉 token 预算和定位准确率的关系
  screenagent_metrics.png      ScreenAgent test 353 条：Op.F1（macro/micro）与 Step SR（≤0.10/≤0.14）
  prompt_variants.png          两段式提示词变体（tune_prompt.py --mode two_stage）
  training_loss.png            两个基座上交付配方的训练 loss
  suite_by_level.png           25 个任务评测集按难度档的成功率（跑过 live 才有）

按应用拆分的图由 scripts/analyze_by_app.py 画（screenagent_by_app.png）。

用法：
    python scripts/make_charts.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "docs" / "figures"

# 指标一律用 eval_screenagent.py 里的实现，别在这儿另算一套
_spec = importlib.util.spec_from_file_location("eval_screenagent", ROOT / "scripts" / "eval_screenagent.py")
_E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_E)

SCREENAGENT_METRICS = ["Op.F1 macro", "Op.F1 micro", "Step SR ≤0.10", "Step SR ≤0.14"]

SCREENSPOT = [
    ("基座", ["grounding_base.json"]),
    ("交付配置", ["grounding_q25_lm.json"]),
    ("加对齐层（对照）", ["grounding_q25_proj.json"]),
]
TOKEN_CURVES = {
    "Qwen2.5-VL-3B 基座": [(320, "grounding_base_320.json"), (640, "grounding_base_640.json"),
                          (1280, "grounding_base.json"), (1920, "grounding_base_1920.json")],
}
SCREENAGENT = [
    ("基座", ["screenagent_q25_base_2s.json"]),
    ("交付配置", ["screenagent_q25_lm_2s.json"]),
    ("加对齐层（对照）", ["screenagent_q25_proj_2s.json"]),
]
PROMPTS = [("基座", "prompt_q35_2s.json")]
# 两条曲线是导师方法论里的两档冻结策略，不是同一配方的两次迭代
TRAINING = [("交付配置：只训语言模型 LoRA", "train_q25_lm.json"),
            ("对照：加对齐层全量训练", "train_q25_proj.json")]


def load(name: str):
    try:
        return json.loads((LOGS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def joint_accuracy(d: dict):
    """类型对且（坐标类动作）距离 ≤ 0.10 的比例；老日志没存这个字段，从逐条记录里算。"""
    if d.get("joint_accuracy") is not None:
        return d["joint_accuracy"]
    cases = d.get("cases") or []
    if not cases:
        return None
    ok = sum(1 for c in cases if c["gt"] == c["pred"] and (c.get("dist") is None or c["dist"] <= 0.10))
    return ok / d["n"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _bars(plt, groups, series, title, ylabel, path):
    """groups：横轴分组名；series：[(图例名, [每组的值或 None])]。

    每组四根柱子以上时，横着写的百分比会叠在一起、图例会盖住柱子：数字改竖着写，图例放到图下面。
    """
    n = max(len(series), 1)
    crowded = n >= 4
    fig, ax = plt.subplots(figsize=(max(9, 0.35 * n * len(groups)), 5 if crowded else 4.5))
    width = 0.8 / n
    for i, (label, values) in enumerate(series):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        ys = [v if v is not None else 0 for v in values]
        rects = ax.bar(xs, ys, width, label=label)
        for rect, v in zip(rects, values):
            if v is not None:
                ax.annotate(f"{v:.0%}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=6 if crowded else 7, rotation=90 if crowded else 0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.18 if crowded else 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if crowded:
        below = -0.22 if any("\n" in g for g in groups) else -0.12
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, below), ncol=3, frameon=False)
    else:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_screenspot(plt) -> bool:
    keys = [("总体", "总体"), ("windows", "Windows"), ("macos", "macOS"), ("text", "文本"), ("icon", "图标")]
    series = []
    for label, names in SCREENSPOT:
        d = next((x for x in (load(n) for n in names) if x), None)
        if not d:
            continue
        acc = d["accuracy"]
        series.append((label, [acc[k]["hit"] / acc[k]["total"] if k in acc else None for k, _ in keys]))
    if not series:
        return False
    _bars(plt, [g for _, g in keys], series, "ScreenSpot 桌面 334 条：定位准确率", "准确率",
          OUT / "screenspot_by_category.png")
    return True


def chart_tokens(plt) -> bool:
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    seen = set()
    for label, points in TOKEN_CURVES.items():
        xs, ys = [], []
        for tokens, name in points:
            d = load(name)
            if d:
                total = d["accuracy"]["总体"]
                xs.append(tokens)
                ys.append(total["hit"] / total["total"])
        seen.update(xs)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xscale("log", base=2)
    # 刻度直接标实测的几档，别让 matplotlib 标成 2^9、2^10
    ax.set_xticks(sorted(seen))
    ax.set_xticklabels([str(x) for x in sorted(seen)])
    ax.minorticks_off()
    ax.set_xlabel("视觉 token 上限")
    ax.set_ylabel("ScreenSpot 桌面准确率")
    ax.set_title("图片分辨率（视觉 token 预算）对定位的影响")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "screenspot_vs_tokens.png", dpi=150)
    plt.close(fig)
    return True


def action_metrics(d: dict) -> list:
    """Op.F1 macro / micro、Step SR ≤0.10 / ≤0.14，口径与 score_actions.py 和报告一致。"""
    cases = d.get("cases") or []
    if not cases:
        return [None] * 4
    f1 = _E.op_f1(cases)
    return [f1["macro_f1"], f1["micro_f1"],
            _E.step_success(cases, 0.10), _E.step_success(cases, 0.14)]


def chart_screenagent(plt) -> bool:
    series = []
    for label, names in SCREENAGENT:
        d = next((x for x in (load(n) for n in names) if x), None)
        if d:
            series.append((label, action_metrics(d)))
    if not series:
        return False
    _bars(plt, SCREENAGENT_METRICS, series, "ScreenAgent test 353 条（生成上限 256）", "比例",
          OUT / "screenagent_metrics.png")
    return True


def chart_prompts(plt) -> bool:
    """每个模型画两组柱：动作类型准确率、类型对且点准。老的一段式日志没有后者，那组柱留空。"""
    logs = [(label, load(name)) for label, name in PROMPTS]
    logs = [(label, d) for label, d in logs if d and d.get("results")]
    if not logs:
        return False
    variants = list(dict.fromkeys(v for _, d in logs for v in d["results"]))
    series = []
    for label, d in logs:
        res = d["results"]
        series.append((f"{label} 类型准确", [res.get(v, {}).get("type_accuracy") for v in variants]))
        series.append((f"{label} 类型对且点准", [res.get(v, {}).get("joint_accuracy") for v in variants]))
    _bars(plt, variants, series, f"两段式提示词变体（ScreenAgent test 前 {logs[0][1].get('n')} 条）", "比例",
          OUT / "prompt_variants.png")
    return True


def chart_training(plt) -> bool:
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for label, name in TRAINING:
        d = load(name)
        steps = (d or {}).get("steps") or []
        if steps:
            ax.plot([s["step"] for s in steps], [s["loss"] for s in steps], label=label)
            drawn = True
    if not drawn:
        plt.close(fig)
        return False
    ax.set_xlabel("更新次数")
    ax.set_ylabel("训练 loss（每 10 次更新的均值）")
    ax.set_title("交付配方（两段式动作 + 拆解，6 轮）的训练 loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_loss.png", dpi=150)
    plt.close(fig)
    return True


# 日志名到图例名：日志名带 live_ / vm_ 前缀和 _suite 后缀，图上读着费劲
SUITE_LABELS = {
    "q25_proj": "对齐层全量 + 语言模型 LoRA", "q25_lm": "只训语言模型 LoRA",
    "q25_base": "基座", "q25_2sp": "微调（上一版数据）",
}


def suite_label(stem: str) -> str:
    name = stem.replace("tasks_", "").removesuffix("_suite")
    for prefix in ("live_", "vm_"):
        name = name.removeprefix(prefix)
    return SUITE_LABELS.get(name, name)


def chart_suite(plt) -> bool:
    runs = []
    for path in sorted(LOGS.glob("tasks_*suite*.json")):
        d = load(path.name)
        records = (d or {}).get("records") or []
        if not records or not any(r.get("level") for r in records):
            continue
        label = suite_label(path.stem)
        rates = []
        for level in ("T1", "T2", "T3"):
            group = [r for r in records if r.get("level") == level]
            rates.append(sum(r["passed"] for r in group) / len(group) if group else None)
        runs.append((label, rates))
    if not runs:
        return False
    _bars(plt, ["T1", "T2", "T3"], runs, "25 个任务评测集：按难度档的成功率", "成功率",
          OUT / "suite_by_level.png")
    return True


def main() -> None:
    try:
        plt = _plt()
    except ImportError:
        raise SystemExit("没有 matplotlib：pip install matplotlib")
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("screenspot_by_category", chart_screenspot), ("screenspot_vs_tokens", chart_tokens),
                     ("screenagent_metrics", chart_screenagent), ("prompt_variants", chart_prompts),
                     ("training_loss", chart_training), ("suite_by_level", chart_suite)]:
        print(f"{name:<24} {'已生成' if fn(plt) else '缺日志，跳过'}")
    print(f"图在 {OUT}")


if __name__ == "__main__":
    main()
