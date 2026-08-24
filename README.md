# 桌面 GUI 智能体

基于多模态大模型的桌面 GUI 智能体，能读取屏幕内容并执行鼠标键盘操作完成指定任务。

大模型 AI Agent 算法岗线上实习项目，周期 4 周（2026-08-24 至 2026-09-20）。

## 环境

Windows 11 + Anaconda base（Python 3.10.9），PyTorch 2.6.0+cu124，NVIDIA RTX 4080 Laptop 12GB。

```bash
pip install -r requirements.txt
```

装完后跑一次环境检查：

```bash
python scripts/check_env.py
```

加 `--ocr` 会额外实测一次 OCR 识别，首次运行需要下载模型权重。

## 目录

```
gui_agent/
    schema.py       屏幕识别结果、动作、执行记录三个数据格式
    perception.py   截图、多分辨率适配、OCR、UI 元素识别
    control.py      鼠标键盘控制、坐标换算
    models.py       大模型调用接口
    agent.py        任务规划、动作解析、结果反馈
scripts/            可执行脚本
tests/              单元测试
docs/               调研报告、环境配置文档、周实验报告
```

## 进度

| 周 | 日期 | 内容 |
|---|---|---|
| 1 | 8/24 – 8/30 | 技术调研、环境搭建、桌面感知与控制模块 |
| 2 | 8/31 – 9/6 | 数据集处理、Agent 框架、端到端原型 v1.0 |
| 3 | 9/7 – 9/13 | LoRA 微调、容错与监控、v2.0 |
| 4 | 9/14 – 9/20 | 20 任务全面评估、技术报告、演示视频 |
