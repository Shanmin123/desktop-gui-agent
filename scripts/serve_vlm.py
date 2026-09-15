"""在宿主机上用显卡跑模型，按 OpenAI 兼容格式提供接口，给虚拟机里的智能体调用。

live 实验放进虚拟机后，虚拟机没有显卡，模型只能留在宿主机：这里把 LocalQwenVL 包成
/v1/chat/completions，虚拟机里的 run_tasks.py 走现成的 API 后端（OpenAICompatVLM）
来调，智能体的鼠标键盘动作只落在虚拟机桌面上。

只实现智能体用到的那部分格式：最后一条 user 消息里一张 data:image/...;base64 截图加
一段文字，max_tokens 当生成上限。显卡一次只跑一个请求。

默认只监听 127.0.0.1。VirtualBox 的 NAT 网络里，客户机访问 10.0.2.2 会转到宿主机的
本机回环地址，不用对局域网开放端口。

用法（宿主机）：
    python scripts/serve_vlm.py --model Qwen/Qwen3.5-4B --adapter checkpoints/q35_2sp
虚拟机里：
    python scripts/run_tasks.py --live --set suite --locate-target ^
        --api-base http://10.0.2.2:8000/v1 --api-key local --api-coord-space rel1000
"""

from __future__ import annotations

import argparse
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def decode_image(url: str) -> np.ndarray:
    """data:image/...;base64,xxx → BGR 数组。"""
    if not url.startswith("data:image") or "," not in url:
        raise ValueError("图片要写成 data:image/...;base64,... 的形式")
    raw = base64.b64decode(url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败")
    return img


def split_message(messages: list) -> Tuple[Optional[np.ndarray], str]:
    """取最后一条 user 消息里的图片和文字。"""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return None, content
        image, texts = None, []
        for part in content or []:
            if part.get("type") == "image_url":
                image = decode_image(part["image_url"]["url"])
            elif part.get("type") == "text":
                texts.append(part.get("text", ""))
        return image, "\n".join(texts)
    raise ValueError("请求里没有 user 消息")


def create_app(vlm, model_name: str):
    from fastapi import FastAPI, HTTPException

    app = FastAPI()
    lock = threading.Lock()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        try:
            image, prompt = split_message(body.get("messages") or [])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        if image is None:
            raise HTTPException(status_code=400, detail="请求里没有截图")
        max_tokens = int(body.get("max_tokens") or 256)
        with lock:
            t = time.perf_counter()
            text = vlm.ask(image, prompt, max_new_tokens=max_tokens)
            seconds = time.perf_counter() - t
        now = int(time.time())
        return {
            "id": f"chatcmpl-{now}-{threading.get_ident()}",
            "object": "chat.completion",
            "created": now,
            "model": body.get("model") or model_name,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_s": round(seconds, 3),
        }

    return app


def main() -> None:
    from gui_agent.models import DEFAULT_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="挂 LoRA 权重，如 checkpoints/q35_2sp")
    ap.add_argument("--max-pixels", type=int, default=1280, help="图片上限，单位视觉 token 数")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    import uvicorn

    from gui_agent.models import LocalQwenVL

    vlm = LocalQwenVL(args.model, adapter=args.adapter, max_tokens=args.max_pixels)
    name = args.model.rstrip("/").split("/")[-1]
    if args.adapter:
        name += "+" + Path(args.adapter).name
    print(f"模型 {name} 已加载，坐标口径 {vlm.coord_space}；虚拟机里加 --api-coord-space {vlm.coord_space}")
    uvicorn.run(create_app(vlm, name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
