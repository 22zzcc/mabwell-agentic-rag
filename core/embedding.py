# -*- coding: utf-8 -*-
"""
core/embedding.py —— 向量化（Embedding）封装
=============================================

职责：
    把文本编码成稠密向量，供稠密检索做语义匹配。

设计目标：
    1. 默认走本地 Ollama（mxbai-embed-large，1024 维），离线可跑；
    2. 生产可无缝切到 BGE（简历「BGE 微调」）：
         - sentence-transformers 加载本地微调后的 BGE 权重（见 core/finetune_bge.py）；
         - 或 `ollama pull bge-m3` 后把 EMBED_MODEL 改成 bge-m3；
    3. 提供纯 Python 的余弦相似度 + 批量矩阵运算（numpy 可用时加速）。

面试可讲点：
    - 为何用「同库同服务」的 embedding：企业数据不出内网，避免第三方 API；
    - embedding 质量决定检索上限，所以简历里专门做了 BGE 领域微调。
"""
from __future__ import annotations

import math
import urllib.error
import urllib.request
import json

from .config import CONFIG

try:  # numpy 可选：有则加速矩阵余弦，无则退化为纯 Python（保证能跑）
    import numpy as _np
except Exception:  # pragma: no cover - 环境缺 numpy 时的兜底
    _np = None


class EmbeddingError(RuntimeError):
    pass


def embed(texts: list[str]) -> list[list[float]]:
    """批量向量化。返回与 texts 等长的 1024 维向量列表。

    走 Ollama /api/embeddings（官方批量端点）。
    """
    url = f"{CONFIG['ollama_url']}/api/embeddings"
    out: list[list[float]] = []
    for text in texts:
        payload = {"model": CONFIG["embed_model"], "prompt": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise EmbeddingError(
                f"embedding 调用失败（{url}）：{e}。请确认 Ollama 已启动并已 "
                f"`ollama pull {CONFIG['embed_model']}`。" ) from e
        vec = body.get("embedding")
        if not vec:
            raise EmbeddingError(f"embedding 返回为空：{body}")
        out.append([float(x) for x in vec])
    return out


def embed_one(text: str) -> list[float]:
    """单条向量化便捷版。"""
    return embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度（范围 -1~1，越大越相似）。

    纯 Python 实现，避免依赖 numpy；numpy 的批量版见 cosine_matrix。
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def cosine_matrix(query_vec: list[float], doc_vecs: list[list[float]]) -> list[float]:
    """查询向量 与 一组文档向量 的余弦相似度数组。

    numpy 可用时走矩阵运算（快几十倍），否则逐条计算。
    """
    if _np is not None and doc_vecs:
        q = _np.asarray(query_vec, dtype="float32")
        d = _np.asarray(doc_vecs, dtype="float32")
        # 先做 L2 归一化，再点积即余弦；一次广播完成，避免逐条 Python 循环
        q = q / (_np.linalg.norm(q) + 1e-12)
        d = d / (_np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        return (d @ q).tolist()
    return [cosine(query_vec, v) for v in doc_vecs]
