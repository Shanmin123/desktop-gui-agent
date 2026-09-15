"""宿主机模型服务：请求解析、接口格式，以及和智能体的 API 后端能不能直接对上。"""

import base64
import importlib.util
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("serve_vlm", ROOT / "scripts" / "serve_vlm.py")
serve_vlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_vlm)


class StubVLM:
    """记下收到的图片尺寸、提示词和生成上限，回一段固定文字。"""

    def __init__(self, reply='{"action": {"type": "wait"}}'):
        self.reply = reply
        self.calls = []

    def ask(self, image, prompt, max_new_tokens=256):
        self.calls.append((image.shape, prompt, max_new_tokens))
        return self.reply


def _data_url(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _body(img, text="下一步做什么", max_tokens=128):
    return {"model": "stub", "max_tokens": max_tokens, "messages": [{
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": _data_url(img)}},
                    {"type": "text", "text": text}],
    }]}


def test_decode_image_round_trips_the_size():
    img = np.full((40, 60, 3), 128, dtype=np.uint8)
    assert serve_vlm.decode_image(_data_url(img)).shape == (40, 60, 3)


def test_decode_image_rejects_plain_urls():
    with pytest.raises(ValueError):
        serve_vlm.decode_image("https://example.com/a.png")


def test_split_message_uses_the_last_user_message():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    messages = [{"role": "system", "content": "你是助手"}, _body(img, "第一条")["messages"][0],
                _body(img, "第二条")["messages"][0]]
    image, text = serve_vlm.split_message(messages)
    assert image.shape == (10, 10, 3) and text == "第二条"


def test_split_message_without_user_raises():
    with pytest.raises(ValueError):
        serve_vlm.split_message([{"role": "system", "content": "x"}])


def test_chat_endpoint_passes_image_prompt_and_token_limit():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    vlm = StubVLM()
    client = TestClient(serve_vlm.create_app(vlm, "stub"))
    img = np.zeros((72, 128, 3), dtype=np.uint8)
    resp = client.post("/v1/chat/completions", json=_body(img, "打开记事本", max_tokens=64))
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == vlm.reply
    assert vlm.calls == [((72, 128, 3), "打开记事本", 64)]


def test_chat_endpoint_rejects_requests_without_a_screenshot():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(serve_vlm.create_app(StubVLM(), "stub"))
    resp = client.post("/v1/chat/completions",
                       json={"messages": [{"role": "user", "content": "只有文字"}]})
    assert resp.status_code == 400


def test_agent_api_backend_talks_to_the_server_end_to_end():
    """虚拟机里用的就是 OpenAICompatVLM，这里起一个真的 HTTP 服务让它调一次。"""
    uvicorn = pytest.importorskip("uvicorn")
    from gui_agent.models import OpenAICompatVLM

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    vlm = StubVLM('{"bbox_2d": [0, 0, 1000, 1000]}')
    server = uvicorn.Server(uvicorn.Config(serve_vlm.create_app(vlm, "stub"),
                                           host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started

        client = OpenAICompatVLM(base_url=f"http://127.0.0.1:{port}/v1", api_key="local",
                                 model="stub", coord_space="rel1000")
        screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert client.ask(screen, "你好", max_new_tokens=32) == vlm.reply
        assert client.locate(screen, "整块屏幕") == pytest.approx((0.5, 0.5))
        assert vlm.calls[0][2] == 32
    finally:
        server.should_exit = True
        thread.join(timeout=10)
