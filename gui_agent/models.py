"""大模型调用接口：本地部署与 API 两种方式，提示词模板。

对应大纲第 3 周第 4 项。

两种后端共用一个 `ask(image, prompt) -> str` 接口，上层不关心模型跑在哪。

定位坐标的口径随模型而定：Qwen2.5-VL 回的是 smart_resize 之后的像素值，归一化要用同一个
smart_resize 算出的尺寸；Qwen3.5 回的是 0~1000 的相对值。口径登记在
COORD_SPACE_BY_MODEL_TYPE，正确性由 scripts/eval_grounding.py 验证。
"""

from __future__ import annotations

import base64
import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np

# 主基座 Qwen3.5-4B；上一代 Qwen2.5-VL-3B-Instruct 留作对照，用 --model 指定。
# 两代的差别（切块系数、思考模式、定位坐标口径）在本文件里按模型处理，脚本不用分支。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

# 让模型只回坐标，不要解释。要求 JSON 是因为比自由文本好解析。
#
# 两套口径：
#   像素框    Qwen2.5-VL 预训练时就是这么输出的，基座模型零样本走这条
#   归一化点  微调后走这条。OS-Atlas、SeeClick 的动作空间都用 [0,1] 的比例值，
#             和分辨率无关——像素框那套在训练和推理的 max_pixels 不一致时，
#             目标框会整体偏掉（实测差 39%，ScreenSpot 从 71.6% 掉到 30.2%）
GROUNDING_PROMPT = (
    "请在截图中找到「{instruction}」对应的界面元素，"
    "只返回一个 JSON 对象，格式为 {{\"bbox_2d\": [x1, y1, x2, y2]}}，"
    "坐标为图片中的像素值。不要输出任何其他内容。"
)

# 生成长度上限。原来是 128，而微调样本里 4~5% 的回答本身就超过 128 token
# （thought 写得长的那些），生成到一半被截断，JSON 收不了尾，评测里记成解析失败。
# 两段式那轮 353 条里有 24 条是这么丢的。
MAX_NEW_TOKENS = 256

GROUNDING_PROMPT_NORM = (
    "请在截图中找到「{instruction}」对应的界面元素，"
    "只返回一个 JSON 对象，格式为 {{\"point\": [x, y]}}，"
    "x 和 y 是 0 到 1 之间的小数，表示该位置在图片宽和高上的比例。"
    "不要输出任何其他内容。"
)


def add_backend_args(ap) -> None:
    """给命令行脚本加上选后端的参数。"""
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true", help="本地加载时用 4bit 量化")
    ap.add_argument("--api-base", default=None,
                    help="走 OpenAI 兼容 API，给出 base_url；不给则本地加载")
    ap.add_argument("--api-key", default=None,
                    help="API 密钥，默认读环境变量 OPENAI_API_KEY")
    ap.add_argument("--api-qwen", action="store_true",
                    help="服务端跑的是 Qwen2.5-VL，坐标按 smart_resize 尺寸归一化")
    ap.add_argument("--api-coord-space", default=None, choices=["pixel", "rel1000"],
                    help="服务端模型回的定位坐标口径：pixel 是 smart_resize 之后的像素值（Qwen2.5-VL），"
                         "rel1000 是 0~1000 的相对值（Qwen3.5）。都不给时按原图尺寸算")
    ap.add_argument("--adapter", default=None,
                    help="LoRA 权重目录，如 checkpoints/lora。给了就在基座上挂适配器")


def load_vlm(args):
    """按命令行参数选后端。

    对应大纲第 3 周第 4 项「支持开源多模态模型的本地部署与 API 调用」。
    """
    if not args.api_base:
        return LocalQwenVL(args.model, load_in_4bit=args.load_in_4bit,
                           adapter=getattr(args, "adapter", None))

    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("走 API 需要 --api-key，或设环境变量 OPENAI_API_KEY")
    coord_space = getattr(args, "api_coord_space", None)
    px = ((256 * 28 * 28, 1280 * 28 * 28) if args.api_qwen or coord_space == "pixel"
          else (None, None))
    return OpenAICompatVLM(base_url=args.api_base, api_key=key, model=args.model,
                           min_pixels=px[0], max_pixels=px[1], coord_space=coord_space)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError("JPEG 编码失败")
    return buf.tobytes()


# --------------------------------------------------------------------------
# 输出解析
# --------------------------------------------------------------------------


def parse_box(text: str) -> Optional[Tuple[float, float, float, float]]:
    """从模型输出里抠出一个框。

    Qwen 系列可能回三种形式，都兼容：
      {"bbox_2d": [x1,y1,x2,y2]}
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>
      裸的 [x1, y1, x2, y2]
    """
    m = re.search(r'"bbox_2d"\s*:\s*\[([^\]]+)\]', text)
    if not m:
        m = re.search(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)", text)
        if m:
            return tuple(float(v) for v in m.groups())
        m = re.search(r"\[\s*([\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+)\s*\]", text)
    if not m:
        return None
    # 逐个匹配完整的数，不能用 [\d.]+ 笼统地抓：那样 "1..2" 会送进 float() 抛异常，
    # 科学计数法 "1e3" 会被拆成 1 和 3
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", m.group(1))]
    return tuple(nums[:4]) if len(nums) >= 4 else None


def parse_norm_point(text: str) -> Optional[Tuple[float, float]]:
    """从模型输出里抠出一个归一化的点。

    收两种形式：{"point": [x, y]} 和裸的 [x, y]。值必须落在 0~1，
    超出范围说明模型退回了像素口径，这时返回 None 交给上层按解析失败处理，
    不能硬当成比例——那会把点压到左上角。
    """
    m = re.search(r'"point"\s*:\s*\[([^\]]+)\]', text)
    if not m:
        m = re.search(r"\[\s*([\d.]+\s*,\s*[\d.]+)\s*\]", text)
    if not m:
        return None
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", m.group(1))]
    if len(nums) != 2 or not all(0.0 <= v <= 1.0 for v in nums):
        return None
    return nums[0], nums[1]


def box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


# --------------------------------------------------------------------------
# 本地部署
# --------------------------------------------------------------------------


# 定位坐标的口径：模型回的坐标除以什么才是 0~1。
#   pixel    缩放后图片上的像素值，Qwen2.5-VL 预训练就是这样
#   rel1000  0~1000 的相对值，Qwen3.5 是这样
# 口径由预训练定死，提示词改不动。新模型先用 scripts/probe_model.py 量出来再登记；
# 没登记的一律按 pixel 处理并提示。
# Qwen3.5-4B 实测（ScreenSpot 抽 40 条）：按 1000 换算命中 32 条，按缩放后像素 6 条、
# 按原图像素 3 条；提示词写不写「像素值」、用中文还是英文，三组结果一样。
COORD_SPACE_BY_MODEL_TYPE = {"qwen2_5_vl": "pixel", "qwen3_5": "rel1000"}
COORD_SPACES = ("pixel", "rel1000")


def strip_thinking(text: str) -> str:
    """去掉思考段。推理时已关掉思考模式，正常不会出现；出现了也不能挡住后面的 JSON。"""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*", "", text, flags=re.S).strip()


def check_adapter_base(adapter: str, model_id: str) -> None:
    """适配器只能挂回训练它的那个基座。挂错了不一定报错，结果却全是错的。"""
    import json

    cfg = os.path.join(adapter, "adapter_config.json")
    if not os.path.isfile(cfg):
        return
    with open(cfg, encoding="utf-8") as f:
        base = json.load(f).get("base_model_name_or_path") or ""
    name = lambda x: x.replace("\\", "/").rstrip("/").split("/")[-1]
    if base and name(base) != name(model_id):
        raise ValueError(f"适配器 {adapter} 是在 {base} 上训的，不能挂到 {model_id} 上（加 --model {base}）")


class LocalQwenVL:
    """本地跑 Qwen 系列视觉语言模型：Qwen2.5-VL、Qwen3.5。

    两代的差别都从模型自己的配置里读，不写死：
      切块系数  Qwen2.5-VL 是 14×2=28，Qwen3.5 是 16×2=32
      思考模式  Qwen3.5 的对话模板默认先思考再回答，这里一律关掉
    图片上限用视觉 token 数给（max_tokens），像素 = token 数 × 系数²，
    两个模型同一个数就是同一份 token 预算。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda",
        load_in_4bit: bool = False,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        adapter: Optional[str] = None,
        norm_coords: bool = False,
        min_tokens: int = 256,
        max_tokens: int = 1280,
        coord_space: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if adapter:
            check_adapter_base(adapter, model_id)
        if coord_space is not None and coord_space not in COORD_SPACES:
            raise ValueError(f"没有 {coord_space!r} 这种坐标口径，可选 {COORD_SPACES}")

        self.norm_coords = norm_coords
        self.torch = torch
        self.model_id = model_id
        self.adapter = adapter

        ip = AutoProcessor.from_pretrained(model_id).image_processor
        self.factor = int(getattr(ip, "patch_size", 14) or 14) * int(getattr(ip, "merge_size", 2) or 2)
        self.min_pixels = min_pixels if min_pixels is not None else min_tokens * self.factor ** 2
        self.max_pixels = max_pixels if max_pixels is not None else max_tokens * self.factor ** 2
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

        kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs.pop("dtype")

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        gen = self.model.generation_config
        if gen.pad_token_id is None:
            # Qwen3.5 的生成配置里没写 pad，generate 每调一次就提示一遍再拿 eos 顶上。
            # 这里照同样的规则先设好，输出不变，日志里不再刷屏
            eos = gen.eos_token_id
            gen.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        if coord_space is None:
            model_type = getattr(self.model.config, "model_type", "")
            coord_space = COORD_SPACE_BY_MODEL_TYPE.get(model_type)
            if coord_space is None:
                print(f"注意：{model_type} 的定位坐标口径还没实测登记，暂按 pixel 处理")
                coord_space = "pixel"
        self.coord_space = coord_space
        if adapter:
            # 挂 LoRA 权重。微调前后必须用同一套评测脚本，差别只在有没有这一步。
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def resized_size(self, height: int, width: int) -> Tuple[int, int]:
        """模型实际看到的尺寸 (高, 宽)。"""
        from qwen_vl_utils.vision_process import smart_resize

        return smart_resize(
            height, width, factor=getattr(self, "factor", 28),
            min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )

    def coord_size(self, height: int, width: int) -> Tuple[int, int]:
        """模型回的坐标要除以的数 (高方向, 宽方向)，与 resized_size 同序。

        pixel 口径就是缩放后的尺寸；rel1000 口径两边都是 1000。
        """
        if getattr(self, "coord_space", "pixel") == "rel1000":
            return 1000, 1000
        return self.resized_size(height, width)

    def ask(self, image: np.ndarray, prompt: str,
            max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}],
            }
        ]
        # 关掉思考模式：Qwen3.5 的模板默认先写一段思考，JSON 会被挤到后面甚至截断。
        # Qwen2.5-VL 的模板不认这个参数，渲染出来的文本不变。
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt").to(
            self.model.device
        )
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return strip_thinking(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0])

    def locate(self, image: np.ndarray, instruction: str) -> Optional[Tuple[float, float]]:
        """给一句话，返回归一化的点击点，找不到返回 None。

        norm_coords=True 时直接问 0~1 的比例值，不经过换算。否则问框，框中心除以
        coord_size：pixel 口径是缩放后的尺寸，rel1000 口径是 1000。
        """
        if getattr(self, "norm_coords", False):
            raw = self.ask(image, GROUNDING_PROMPT_NORM.format(instruction=instruction))
            return parse_norm_point(raw)
        raw = self.ask(image, GROUNDING_PROMPT.format(instruction=instruction))
        box = parse_box(raw)
        if box is None:
            return None
        h, w = image.shape[:2]
        rh, rw = self.coord_size(h, w)
        cx, cy = box_center(box)
        return min(max(cx / rw, 0.0), 1.0), min(max(cy / rh, 0.0), 1.0)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


class OpenAICompatVLM:
    """走 OpenAI 兼容格式的 API。

    DashScope、火山方舟、以及大多数自建推理服务都提供这个格式的端点，
    换服务只要改 base_url 和 model。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        coord_space: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.coord_space = coord_space

    def ask(self, image: np.ndarray, prompt: str,
            max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        b64 = base64.b64encode(encode_jpeg(image)).decode()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def resized_size(self, height: int, width: int) -> Tuple[int, int]:
        """模型实际看到的尺寸，它输出的坐标就在这个尺寸的像素空间里。

        默认按原图尺寸算。服务端跑的是 Qwen 系列时，构造时传 min_pixels /
        max_pixels，这里按同一个 smart_resize 还原它的坐标空间。

        smart_resize 会把边长取整到 28 的倍数，超出像素上限时还会整体缩小。图在
        上限之内时按原图尺寸归一化只差 1~2%，超出就不止：1920×1080 送进去模型看到
        的是 1316×728，偏差 30% 以上。
        """
        if self.min_pixels is None or self.max_pixels is None:
            return height, width

        from qwen_vl_utils.vision_process import smart_resize

        return smart_resize(
            height, width, factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def coord_size(self, height: int, width: int) -> Tuple[int, int]:
        """服务端模型回的坐标要除以的数 (高方向, 宽方向)。rel1000 口径两边都是 1000，其余同 resized_size。"""
        if getattr(self, "coord_space", None) == "rel1000":
            return 1000, 1000
        return self.resized_size(height, width)

    def locate(self, image: np.ndarray, instruction: str) -> Optional[Tuple[float, float]]:
        """给一句话，返回归一化的点击点，找不到返回 None。

        norm_coords=True 时直接问 0~1 的比例值，不经过像素换算。像素那条路要拿
        预测框除以 smart_resize 后的尺寸，训练和推理的 max_pixels 一旦不同就整体
        偏掉；比例值与分辨率无关，OS-Atlas 和 SeeClick 用的都是这套。
        """
        if getattr(self, "norm_coords", False):
            raw = self.ask(image, GROUNDING_PROMPT_NORM.format(instruction=instruction))
            return parse_norm_point(raw)
        raw = self.ask(image, GROUNDING_PROMPT.format(instruction=instruction))
        box = parse_box(raw)
        if box is None:
            return None
        h, w = image.shape[:2]
        rh, rw = self.coord_size(h, w)
        cx, cy = box_center(box)
        return min(max(cx / rw, 0.0), 1.0), min(max(cy / rh, 0.0), 1.0)


class FlakyVLM:
    """按给定比例把模型输出换成垃圾，用来测容错（大纲第 6 周第 2 项）。

    真实故障（显存不足、模型偶尔吐不出 JSON、截图失败）出现得不规律，等它自己发生
    没法做对照。这里按固定比例主动制造同一类故障：输出变成不可解析的文本，Agent
    那一步就会解析失败，走到重试逻辑上。

    seed 固定，同一比例下每次注入的位置一样，开关重试两次跑才可比。
    """

    def __init__(self, vlm, rate: float = 0.3, seed: int = 0) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"故障比例要在 0~1 之间，收到 {rate}")
        import random

        self.vlm = vlm
        self.rate = rate
        self._rng = random.Random(seed)
        self.injected = 0
        self.calls = 0

    def ask(self, image, prompt: str, **kw) -> str:
        self.calls += 1
        if self._rng.random() < self.rate:
            self.injected += 1
            return "（注入的故障：这不是 JSON）"
        return self.vlm.ask(image, prompt, **kw)

    def __getattr__(self, name):
        # resized_size、locate 这些照常转给真模型，只有 ask 被换掉
        return getattr(self.vlm, name)
