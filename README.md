# 桌面 GUI 智能体

基于多模态大模型的桌面 GUI 智能体：读取屏幕内容，输出鼠标键盘操作，完成指定任务。基座是 Qwen2.5-VL-3B，
用公开 GUI 数据集做参数高效微调。

## 环境

Windows 11，Python 3.10，PyTorch 2.6 + CUDA 12.4，transformers 5.3，NVIDIA GPU 12 GB 显存
（Qwen2.5-VL-3B bf16 推理峰值约 7.6 GB，训练用 4-bit 加载）。

```bash
pip install -r requirements.txt
python scripts/check_env.py          # 检查环境与基础工具库
python scripts/check_env.py --ocr    # 额外实测 OCR（构建训练数据和一段式对照要用），首次会下载模型权重
```

## 结构

```
gui_agent/
    schema.py       屏幕识别结果、动作、执行记录三个数据格式
    perception.py   截图、多分辨率适配、屏幕变化检测；OCR 与图标候选框（构建数据、一段式对照用）
    control.py      鼠标键盘控制、坐标换算、安全限制
    models.py       大模型调用接口：本地加载（Qwen2.5-VL）与 OpenAI 兼容 API
    chain.py        LangChain 提示词模板、输出解析、两段式定位、提示词变体
    planner.py      任务拆解与子任务状态判定
    agent.py        执行循环、动作解析、错误检测与重试
    monitor.py      执行状态实时记录，每步落盘
    tasks.py        基础任务、复杂任务与程序化验收条件
    suite.py        25 个任务的评测集，分 T1/T2/T3 三档
    display.py      临时切换屏幕分辨率
scripts/
    run_agent.py           命令行入口，给一句话让智能体去做
    run_tasks.py           跑任务集，统计成功率
    check_env.py           环境检查
    bench_perception.py    感知各环节耗时实测
    calibrate.py           感知与控制联调，测坐标端到端误差
    prepare_*.py           公开数据集预处理（ScreenAgent、Mind2Web、WebArena）
    build_finetune_data.py 由预处理结果构建微调训练集与验证集
    audit_finetune_data.py 微调数据集审计，查静默的错标、重复、泄漏
    train_lora.py          LoRA 微调
    probe_model.py         换基座前实测定位坐标口径和输出格式
    eval_grounding.py      UI 元素定位精度评测
    eval_screenagent.py    动作生成评测
    eval_plan.py           任务拆解质量评测
    eval_perception.py     感知模块的命中率与速度评测
    tune_prompt.py         提示词变体对比
    analyze_by_app.py      动作生成按应用拆分，离线任务成功率
    summarize_suite.py     任务日志汇总：成功率、平均执行时间、错误率
    make_charts.py         评测结果画图
    serve_vlm.py           本机模型服务（OpenAI 兼容），给虚拟机里的智能体调用
    vm/                    虚拟机评测：客户机 worker、宿主机投任务、批次清单
    snapshot_state.py      记录并核对实验前后的桌面状态
    gen_test_report.py     由 pytest 收集结果生成单元测试报告
    md2pdf.py              文档转 PDF，支持合并多份
tests/              单元测试
docs/               调研报告、环境配置、各周实验报告、系统全面评估报告、虚拟机评测指南、项目构建说明
```

## 设计要点

**两段式定位，执行时不跑 OCR。** 模型先说要操作哪个控件，再用定位提示词让它在截图上框出这个控件，框中心
换算成坐标。读字和找控件都交给模型，没有文字的图标也能点。OCR 只用于构建训练数据和复现一段式对照。

**坐标一律归一化到 0~1。** 截图端按自己的分辨率归一化，控制端按自己的分辨率反归一化，两边不需要知道对方的
尺寸，也不受系统 DPI 缩放影响。

**定位坐标口径按基座登记。** Qwen2.5-VL 回的是缩放后图片的像素值。口径由
预训练决定、提示词改不了，换基座先用 `scripts/probe_model.py` 实测再登记。

**动作格式沿用 UI-TARS 的桌面子集**（arXiv:2501.12326 Table 1），10 个动作：click、left_double、
right_single、drag、scroll、type、hotkey、wait、finished、call_user。

**任务验收对比执行前后的状态。** 只看当前状态会把「本来就是这样」判成成功，成功率会虚高。

**容错分三层。** 单步故障先重试；动作执行成功但界面没变，说明点空了，把这件事喂回历史让模型换目标；连续几步
重复同一动作且界面一直没变，判定卡住并停下。

**图片读写统一走 `perception.imwrite` / `imread`。** `cv2.imwrite` 在非 ASCII 路径下返回 False 但不抛异常，
文件不会写出来。

## 运行

默认基座是 Qwen2.5-VL-3B，默认走两段式。

```bash
python scripts/run_agent.py "打开计算器" --adapter checkpoints/q25_proj          # dry-run，只打印动作
python scripts/run_agent.py "打开计算器" --adapter checkpoints/q25_proj --live   # 真的操作桌面

python scripts/run_tasks.py --adapter checkpoints/q25_proj                                   # 基础任务集，dry-run
python scripts/run_tasks.py --adapter checkpoints/q25_proj --set suite --live --repeat 3     # 25 个任务评测集
```

可选开关：`--plan` 先拆解子任务，`--one-stage` 走一段式（提示词带 OCR 元素清单，只用于对照），
`--resolution 1280x720` 临时切分辨率（结束后还原），`--model` 换基座，
`--api-base` 走 OpenAI 兼容接口。

默认 dry-run。`--live` 会真实操作桌面，开始前有倒计时，鼠标甩到屏幕左上角可强制中断。真机跑之前按
`docs/项目构建说明.md` 第七节的步骤来：确认没有聊天软件和可见浏览器窗口、存桌面状态、最小化窗口，跑完恢复
并核对。要无人值守连着跑几批，用虚拟机：`python scripts/vm/run_batches.py`，准备步骤见 `docs/虚拟机评测指南.md`。

### 微调权重

权重只能挂回训练它的那个基座，挂错会直接报错。ScreenAgent test 353 步（生成上限 256）与 ScreenSpot 桌面 334 条：

| 权重 | Op.F1 macro | Op.F1 micro | Step SR ≤0.10 | Step SR ≤0.14 | ScreenSpot 桌面 |
|---|---|---|---|---|---|
| 不挂（基座） | 10.7% | 28.0% | 13.0% | 13.6% | 71.6% |
| **`checkpoints/q25_proj`（交付）** | **25.6%** | **40.2%** | **21.2%** | **23.5%** | 61.7% |
| `checkpoints/q25_proj`（对照，只训语言模型） | 24.5% | 39.7% | 19.0% | 21.0% | 70.7% |

Op.F1 是七类动作的操作 F1（macro 按类平均、micro 按步平均，键盘动作要求输入内容也对），
Step SR 要求动作类型对、坐标类动作点在真值给定距离以内（0.14 对应屏幕对角线的 14%）。交付权重在 ScreenSpot 上低于基座，但流程内的点击精度更高：坐标类动作里类型判断正确的 123 步中
56 步落在真值 0.10 以内（45.5%），中位距离 0.127，基座是 41.8% 和 0.230。两者分布不同，
取舍依据见 `docs/微调方案设计.md` 第五节。指标口径和完整结果见 `docs/系统全面评估报告.md`，配方的取舍见 `docs/微调方案设计.md`。

真机 25 个任务的评测集跑法见 `scripts/run_tasks.py --suite`，成功率按交付配方定下来后复测，
结果与 Wilson 95% 置信区间记在 `docs/系统全面评估报告.md`。

## 测试

```bash
pytest tests/ -q
```

测试用 `RecordingBackend` 替换真实的鼠标键盘后端，不会操作桌面。
