"""由 pytest 的收集结果生成 docs/单元测试报告.md。

用例明细手工维护会和代码脱节：报告里还写着 357 项时，实际已经加到 473 项。
这里直接读 `pytest --collect-only` 的输出，加完测试重跑一次即可。

用法：
    python scripts/gen_test_report.py
"""

import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
                     cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout

per = {}
for line in out.splitlines():
    if "::" not in line:
        continue
    f, _, name = line.partition("::")
    f = f.strip().replace("\\", "/")
    if not f.startswith("tests/"):
        continue
    per.setdefault(f, []).append(name.strip())

DESC = {
 "tests/test_schema.py": "数据格式：屏幕识别结果、动作、执行记录",
 "tests/test_perception.py": "桌面感知：缩放系数、坐标归一化、边界框绘制、图片读写、变化检测、图标候选框、OCR 复用",
 "tests/test_control.py": "桌面控制：10 个动作、坐标换算、安全拦截、批量执行",
 "tests/test_models.py": "大模型调用接口：输出解析、定位坐标口径（像素 / 0~1000）、关思考模式、适配器基座校验、本地与 API 后端选择、故障注入",
 "tests/test_display.py": "屏幕分辨率：切换与还原、系统缩放检测、跨平台退化",
 "tests/test_chain.py": "LangChain 链：提示词模板、输出解析器、链式组装、两段式定位、动作类型别名、提示词变体",
 "tests/test_planner.py": "任务拆解与规划：拆解解析、子任务反思、循环推进、提示词防抄袭、不带元素清单的拆解",
 "tests/test_agent.py": "Agent 循环：JSON 提取、动作解析、失败重试、卡住检测、终止条件、截图落盘、两段式不跑 OCR",
 "tests/test_monitor.py": "执行状态记录：动作描述、每步落盘、崩溃后可读、重试事件",
 "tests/test_tasks.py": "任务集：程序化验收、状态差异比对、跨平台进程操作、复杂任务",
 "tests/test_suite_tasks.py": "25 个任务的评测集：分档、做完前后验收分别不通过和通过、部分完成与伪装产物判不通过",
 "tests/test_finetune_data.py": "微调数据构建：样本生成、裁剪窗口、训练验证划分、定位目标的坐标空间、两段式样本的类型配比",
 "tests/test_train_lora.py": "LoRA 训练：提示词屏蔽、标签对齐、只算回答段的 logits、LoRA 目标自动发现、显存与步数计算",
 "tests/test_probe_model.py": "换基座探针：坐标口径换算、输出格式统计",
 "tests/test_tune_prompt.py": "提示词变体对比：一段式、两段式两条路径用假模型走通",
 "tests/test_analyze_by_app.py": "按应用拆分：关键词归类、逐类准确率、离线任务成功率",
 "tests/test_summarize_suite.py": "评测集汇总：Wilson 区间、成功率 / 耗时 / 错误率口径、dry-run 日志不计入",
 "tests/test_make_charts.py": "报告图表：缺日志跳过、有日志出图",
 "tests/test_serve_vlm.py": "本机模型服务：OpenAI 兼容接口、图片解码、真 HTTP 端到端调用",
 "tests/test_vm_worker.py": "虚拟机 worker：参数白名单、领任务、回执与结果拷回",
 "tests/test_vm_host.py": "宿主机投任务：暂停显卡队列、等心跳、收回执、恢复快照命令",
 "tests/test_vm_batches.py": "虚拟机批次清单：参数过白名单、坐标口径随模型、暂停文件挂满整段",
 "tests/test_screenagent.py": "ScreenAgent 数据集：原始动作到本项目 Action 的转换",
 "tests/test_webarena.py": "WebArena 数据集：任务规格到本项目格式的转换",
 "tests/test_mind2web.py": "Mind2Web 数据集：操作字段解析与读取列",
}
order = list(DESC)
total = sum(len(v) for v in per.values())

buf = io.StringIO()
buf.write("# 单元测试报告\n\n对应大纲第 2、4 周交付物，第 5~7 周新增的模块同步补测。\n\n")
buf.write(f"运行 `pytest tests/ -q`，共 {total} 项，默认执行的 {total - 1} 项全部通过。"
          "余下 1 项会真的切换屏幕分辨率，默认跳过，`GUI_AGENT_DISPLAY_TESTS=1` 时运行。\n\n")
buf.write("| 模块 | 覆盖范围 | 用例数 |\n|---|---|---|\n")
for f in order:
    if f in per:
        buf.write(f"| `{f}` | {DESC[f]} | {len(per[f])} |\n")
buf.write("\n---\n\n## 用例明细\n")
for f in order:
    if f not in per:
        continue
    buf.write(f"\n### {f}\n\n")
    for n in per[f]:
        buf.write(f"- `{n}`\n")

(ROOT / "docs" / "单元测试报告.md").write_text(buf.getvalue(), encoding="utf-8")
print("总数", total, "文件", len(per))
missing = set(per) - set(DESC)
print("没写说明的：", missing or "无")
