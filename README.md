# 桌面 GUI 智能体

基于多模态大模型的桌面 GUI 智能体：读取屏幕内容，输出鼠标键盘操作，完成指定任务。

## 环境

Windows 11，Python 3.10，PyTorch 2.6 + CUDA 12.4，NVIDIA GPU（12 GB 显存可跑 3B 级模型）。

```bash
pip install -r requirements.txt
python scripts/check_env.py          # 检查环境与基础工具库
python scripts/check_env.py --ocr    # 额外实测 OCR，首次会下载模型权重
```

## 结构

```
gui_agent/
    schema.py       屏幕识别结果、动作、执行记录三个数据格式
    perception.py   截图、多分辨率适配、OCR、UI 元素识别、图标候选框
    control.py      鼠标键盘控制、坐标换算、安全限制
    models.py       大模型调用接口，本地部署与 API 两种后端
    chain.py        LangChain 提示词模板、输出解析、提示词变体
    planner.py      任务拆解与子任务状态判定
    agent.py        执行循环、动作解析、错误检测与重试
    monitor.py      执行状态实时记录，每步落盘
    tasks.py        基础任务、复杂任务与程序化验收条件
    display.py      临时切换屏幕分辨率
scripts/
    run_agent.py           命令行入口，给一句话让智能体去做
    run_tasks.py           跑任务集，统计成功率
    check_env.py           环境检查
    bench_perception.py    感知各环节耗时实测
    calibrate.py           感知与控制联调，测坐标端到端误差
    prepare_data.py        公开数据集预处理
    build_finetune_data.py 由预处理结果构建微调训练集与验证集
    train_lora.py          LoRA 微调
    audit_finetune_data.py 微调数据集审计，查静默的错标、重复、泄漏
    eval_grounding.py      UI 元素定位精度评测
    eval_screenagent.py    动作生成评测
    eval_perception.py     感知模块的命中率与速度评测
    eval_plan.py           任务拆解质量评测
    tune_prompt.py         提示词变体对比
    snapshot_state.py      记录并核对实验前后的桌面状态
    gen_test_report.py     由 pytest 收集结果生成单元测试报告
    md2pdf.py              文档转 PDF，支持合并多份
tests/              单元测试
docs/               调研报告、环境配置文档、实验报告
```

## 设计要点

**坐标一律归一化到 0~1。** 截图端按自己的分辨率归一化，控制端按自己的分辨率反归一化，两边不需要知道对方的尺寸，也不受系统 DPI 缩放影响。

**动作格式沿用 UI-TARS 的桌面子集**（arXiv:2501.12326 Table 1），10 个动作：click、left_double、right_single、drag、scroll、type、hotkey、wait、finished、call_user。

**OCR 跑原始分辨率，缩放只用于模型输入。** 在缩放图上跑 OCR 会明显掉识别率，实测数据见 `docs/环境配置文档.md`。

**任务验收对比执行前后的状态。** 只看当前状态会把「本来就是这样」判成成功，成功率会虚高。

**容错分三层。** 单步故障先重试；动作执行成功但界面没变，说明点空了，把这件事喂回历史让模型换目标；连续几步重复同一动作且界面一直没变，判定卡住并停下。

**图片读写统一走 `perception.imwrite` / `imread`。** `cv2.imwrite` 在非 ASCII 路径下返回 False 但不抛异常，文件不会写出来。

## 运行

```bash
python scripts/run_agent.py "打开计算器"          # dry-run，只打印动作
python scripts/run_agent.py "打开计算器" --live   # 真的操作桌面

python scripts/run_tasks.py                      # 跑基础任务集，dry-run
python scripts/run_tasks.py --live --repeat 3    # 真实执行，每个任务跑三次
python scripts/run_tasks.py --live --set complex --plan   # 多步任务，先拆解再执行
```

可选开关：`--locate-target` 两段式定位，`--plan` 先拆解子任务，`--cache-ocr` 屏幕没变时
复用上一次 OCR，`--cv-elements` 用 OpenCV 补图标候选框，`--resolution 1280x720` 临时切
分辨率（结束后还原），`--adapter checkpoints/lora_2sp` 挂上微调权重。

### 微调权重

**权重和提示词是配套的，挂错路径会掉十几个点**，ScreenAgent test 353 条上实测：

| 权重 | 配套路径 | 动作类型准确 | 键盘召回 | 距离 ≤0.10 | 类型对且点得准 |
|---|---|---|---|---|---|
| 不挂 | 一段式 | 42.2% | 46.6% | 25.4% | 18.1% |
| `checkpoints/lora_v3` | 一段式（不加 `--locate-target`） | 40.8% | 65.4% | **44.7%** | 26.6% |
| `checkpoints/lora_2sp` | **两段式（要加 `--locate-target`）** | **54.1%** | **81.6%** | 37.3% | **31.4%** |

```bash
python scripts/run_tasks.py --live --adapter checkpoints/lora_2sp --locate-target
```

`lora_2sp` 训的是动作决策和任务拆解，定位（`vlm.locate`）这一段没训过，走基座自带的
像素 `bbox_2d`：ScreenSpot 上 68.3%，基座 71.6%——图标那一类掉得多一些（47.9% 对 55.7%），
因为训练里的控件名全取自 OCR 文字。动作选对换定位差一点，按端到端的用法划算，
但不是没有代价。训练与评测的完整对照见 `docs/第3周实验报告.md`。

默认 dry-run。`--live` 会真实操作桌面，开始前有倒计时，鼠标甩到屏幕左上角可强制中断。

## 测试

```bash
pytest tests/ -q
```

测试用 `RecordingBackend` 替换真实的鼠标键盘后端，不会操作桌面。
