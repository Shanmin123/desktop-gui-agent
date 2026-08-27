"""大模型调用接口：本地部署与 API 两种方式，提示词模板。

对应大纲第 3 周第 4 项。

两种后端共用一个 `ask(image, prompt) -> str` 接口，上层不关心模型跑在哪。

坐标处理是这里最容易出错的地方。Qwen2.5-VL 输出的坐标不是原图像素，而是它内部
smart_resize 之后的像素。所以要用同一个 smart_resize 算出实际送进去的尺寸，再拿它
归一化。这个约定不靠猜，用 ScreenSpot 的真值验证过，见 scripts/eval_grounding.py。
"""

from __future__ import annotations

import base64
import json
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# 让模型只回坐标，不要解释。要求 JSON 是因为比自由文本好解析。
GROUNDING_PROMPT = (
    "请在截图中找到「{instruction}」对应的界面元素，"
    "只返回一个 JSON 对象，格式为 {{\"bbox_2d\": [x1, y1, x2, y2]}}，"
    "坐标为图片中的像素值。不要输出任何其他内容。"
)


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
    nums = [float(v) for v in re.findall(r"-?[\d.]+", m.group(1))]
    return tuple(nums[:4]) if len(nums) >= 4 else None


def box_center(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


# --------------------------------------------------------------------------
# 本地部署
# --------------------------------------------------------------------------


class LocalQwenVL:
    """本地跑 Qwen2.5-VL。

    3B 在 12GB 显存上 bf16 直接装得下；7B 需要 load_in_4bit=True。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "cuda",
        load_in_4bit: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        kwargs = {"dtype": torch.bfloat16, "device_map": device}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs.pop("dtype")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def resized_size(self, height: int, width: int) -> Tuple[int, int]:
        """模型实际看到的尺寸。它输出的坐标就在这个尺寸的像素空间里。"""
        from qwen_vl_utils.vision_process import smart_resize

        return smart_resize(
            height, width, factor=28, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )

    def ask(self, image: np.ndarray, prompt: str, max_new_tokens: int = 128) -> str:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt").to(
            self.model.device
        )
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def locate(self, image: np.ndarray, instruction: str) -> Optional[Tuple[float, float]]:
        """给一句话，返回归一化的点击点，找不到返回 None。"""
        raw = self.ask(image, GROUNDING_PROMPT.format(instruction=instruction))
        box = parse_box(raw)
        if box is None:
            return None
        h, w = image.shape[:2]
        rh, rw = self.resized_size(h, w)
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

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def ask(self, image: np.ndarray, prompt: str, max_new_tokens: int = 128) -> str:
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

    def locate(self, image: np.ndarray, instruction: str) -> Optional[Tuple[float, float]]:
        """API 侧不知道服务端怎么缩放的，按原图尺寸归一化。

        自己先把图缩到目标尺寸再送，这样这里的假设才成立。
        """
        raw = self.ask(image, GROUNDING_PROMPT.format(instruction=instruction))
        box = parse_box(raw)
        if box is None:
            return None
        h, w = image.shape[:2]
        cx, cy = box_center(box)
        return min(max(cx / w, 0.0), 1.0), min(max(cy / h, 0.0), 1.0)
