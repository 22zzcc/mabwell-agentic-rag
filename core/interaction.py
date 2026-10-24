# -*- coding: utf-8 -*-
"""
core/interaction.py —— 交互式记忆（「这个问题你上次也提过相似的」）
=====================================================================

对应自我介绍「在压缩上下文的同时，让 LLM 与提问者交互——这个问题你上次也提过
相似的，鼓励提问者进行多维度思考」。

这是「记忆是纽带」在问答层的一个具体落点：Agent 不只回答问题，还**记得你问过什么**，
遇到相似问题时主动提示历史，鼓励从不同角度再想——从「一次性问答」变成「陪伴式交互」。
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _similarity(a: str, b: str) -> float:
    """相似度：优先用本地 embedding 余弦；不可用则退化为 token Jaccard。"""
    try:
        from .embedding import embed_one, cosine
        return cosine(embed_one(a), embed_one(b))
    except Exception:
        from .vector_store import tokenize
        ta, tb = set(tokenize(a)), set(tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


class ConversationMemory:
    """对话历史记忆：记录问答，找相似历史问题，生成「多维度思考」提示。"""

    def __init__(self, path: str = "data/history.json",
                 similarity_threshold: float = 0.75):
        self.path = Path(path)
        self.threshold = similarity_threshold
        self.history: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def add(self, question: str, answer: str) -> None:
        self.history.append({"question": question, "answer": answer[:200],
                             "ts": time.time()})
        self.save()

    def find_similar(self, question: str, top_k: int = 1) -> list[dict]:
        """返回与当前问题最相似的历史问答（超过阈值）。"""
        scored = sorted(
            ((_similarity(question, h["question"]), h) for h in self.history),
            key=lambda x: x[0], reverse=True)
        return [h for s, h in scored[:top_k] if s >= self.threshold]

    def nudge(self, question: str) -> str:
        """生成「多维度思考」提示；无相似历史则返回空串。"""
        sim = self.find_similar(question)
        if not sim:
            return ""
        h = sim[0]
        return (f"💡 你上次也提过相似的问题：「{h['question'][:40]}」，"
                f"当时的结论是「{h['answer'][:40]}」。这次不妨从不同角度补充或推翻它。")
