# -*- coding: utf-8 -*-
"""
core/llm.py —— 本地大模型推理封装（Ollama）
=============================================

职责：
    对 Ollama 的两个核心端点做薄封装，供 agent / 生成 / 评测共用：
      * /api/chat     —— 多轮对话（带消息列表）
      * /api/generate —— 单轮补全（带 prompt）

设计目标（对应简历「本地推理 / 落地」）：
    1. 零第三方依赖：只用标准库 urllib，配合 requests 可选加速；
    2. 优雅降级：Ollama 不可用 / 模型未拉取时，给出「ollama pull ...」修复指引，
       而不是抛裸异常，方便现场演示时快速定位；
    3. 统一错误语义：网络/超时/模型缺失分别报错，便于上层（agent）做兜底策略。

面试可讲点：
    - 企业本地部署时，模型服务与业务服务解耦：这里只认 HTTP 端点，
      换 vLLM / TGI / 自研推理网关时只改一个 URL，业务代码零改动。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import CONFIG


class LLMError(RuntimeError):
    """所有 LLM 调用异常的基类，附带机器可读的 reason。"""

    def __init__(self, message: str, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    """用标准库 POST JSON；requests 可用时用 requests（支持流式与重试）。

    统一收口：所有 HTTP 错误都转成带上下文的 LLMError，调用方无需关心底层库。
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        # 连接拒绝大概率是 Ollama 没启动；404 大概率是模型没 pull
        raise LLMError(
            f"无法连接 Ollama（{url}）：{e}。请确认已运行 `ollama serve`，"
            f"并已执行 `ollama pull {CONFIG['chat_model']}`。",
            reason="connection",
        ) from e


def chat(messages: list[dict], temperature: float | None = None,
         json_mode: bool = False) -> str:
    """多轮对话：messages 形如 [{"role":"user","content":"..."}]。

    :param messages:     对话消息列表
    :param temperature:  采样温度；None 用全局默认
    :param json_mode:    True 时要求模型只输出合法 JSON（用于「规划」步骤的决策）
    :return:             模型回复文本（可能是一段 JSON 字符串）
    """
    payload: dict = {
        "model": CONFIG["chat_model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": CONFIG["temperature"] if temperature is None else temperature,
        },
    }
    # Ollama 的 JSON 模式：强制结构化输出，减少「规划器返回非 JSON」的概率
    if json_mode:
        payload["format"] = "json"

    body = _post_json(f"{CONFIG['ollama_url']}/api/chat", payload)
    msg = body.get("message") or {}
    text = msg.get("content", "")
    if not text:
        raise LLMError(f"Ollama 返回空回复：{body}", reason="empty")
    return text.strip()


def generate(prompt: str, temperature: float | None = None,
             json_mode: bool = False) -> str:
    """单轮补全：给定 prompt，返回模型续写文本。

    与 chat() 的区别：不带历史，适合「改写/摘要/打标签」这类一次性任务，
    能少传很多上下文，节省首 Token 延迟（呼应简历 KV Cache / 前缀复用思路）。
    """
    payload: dict = {
        "model": CONFIG["chat_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": CONFIG["temperature"] if temperature is None else temperature,
        },
    }
    if json_mode:
        payload["format"] = "json"

    body = _post_json(f"{CONFIG['ollama_url']}/api/generate", payload)
    text = body.get("response", "")
    if not text:
        raise LLMError(f"Ollama 返回空回复：{body}", reason="empty")
    return text.strip()


def chat_json(messages: list[dict], temperature: float | None = None) -> dict:
    """chat() 的 JSON 便捷版：直接把回复解析成 dict，解析失败抛错。

    用于 agent 的「规划 / 工具选择」步骤：要求模型输出 {"action":..., "args":...}。
    """
    raw = chat(messages, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # 即便开了 JSON 模式仍可能夹带解释文字，这里做一层容错抽取
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"模型未返回合法 JSON：{raw[:200]}", reason="bad_json") from e
