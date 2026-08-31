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
    perception.py   截图、多分辨率适配、OCR、UI 元素识别
    control.py      鼠标键盘控制、坐标换算、安全限制
    models.py       大模型调用接口，本地部署与 API 两种后端
    agent.py        任务规划、动作解析、结果反馈
    display.py      临时切换屏幕分辨率
scripts/
    check_env.py         环境检查
    bench_perception.py  感知各环节耗时实测
    calibrate.py         感知与控制联调，测坐标端到端误差
    prepare_data.py      公开数据集预处理
    eval_grounding.py    UI 元素定位精度评测
    md2pdf.py            文档转 PDF，支持合并多份
tests/              单元测试
docs/               调研报告、环境配置文档、实验报告
```

## 设计要点

**坐标一律归一化到 0~1。** 截图端按自己的分辨率归一化，控制端按自己的分辨率反归一化，两边不需要知道对方的尺寸，也不受系统 DPI 缩放影响。

**动作格式沿用 UI-TARS 的桌面子集**（arXiv:2501.12326 Table 1），10 个动作：click、left_double、right_single、drag、scroll、type、hotkey、wait、finished、call_user。

**OCR 跑原始分辨率，缩放只用于模型输入。** 在缩放图上跑 OCR 会明显掉识别率，实测数据见 `docs/环境配置文档.md`。

**图片读写统一走 `perception.imwrite` / `imread`。** `cv2.imwrite` 在非 ASCII 路径下返回 False 但不抛异常，文件不会写出来。

## 测试

```bash
pytest tests/ -q
```

测试用 `RecordingBackend` 替换真实的鼠标键盘后端，不会操作桌面。
