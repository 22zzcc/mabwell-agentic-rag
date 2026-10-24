# -*- coding: utf-8 -*-
"""
core/memory.py —— 长期记忆反馈模块（Critic + badcase 收集 + 自我迭代）
========================================================================

用户诉求：在 RAG 与 Agent（主流程）之外，设一个**独立的长期记忆反馈模块**，
类似 PPO 里的「评论家（critic）」，但比普通 critic 多一个能力——**badcase 收取**。

职责：
    1. judge()    —— 评论家：给一轮问答打分（规则 + 可选 LLM），同时发现 issue；
    2. collect()  —— 把 judge 发现的 badcase（上下文自动 + 对话反馈）持久化进长期记忆；
    3. feedback_to_datasets() —— 把记忆里的 badcase 路由成数据集：
         检索类坏例 → 三元组（LoRA）；反馈类坏例 → 偏好对（PPO/DPO）；
    4. self_iterate() —— 自我迭代：badcase → 数据集 → 回归集，形成闭环。

「长期记忆」的意义：badcase 不是一次性丢给评测的日志，而是**跨会话沉淀的资产**——
用户上次纠正过的错误，会变成下一次微调的训练数据；这就是「越用越准」的自我迭代。

评论家 vs 普通 PPO critic 的区别（面试可讲）：
    - PPO 的 critic 只对「当前 rollout」给价值估计，用完即弃；
    - 本模块的 critic 额外把低分样本**落库成 badcase**，反哺 LoRA/PPO 的训练数据，
      把「在线信号」变成了「离线训练资产」。
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from .badcase import Badcase, collect_context_badcases, collect_feedback, RegressionSet
from .config import CONFIG

# 各类 issue 的扣分权重（rule-based 评论家的打分依据）
_ISSUE_PENALTY = {
    "empty_context": 1.0,
    "hallucination": 0.6,
    "recall_miss": 0.5,
    "citation_fail": 0.4,
    "low_relevance": 0.3,
    "user_correction": 0.5,
}


class Memory:
    """长期记忆持久化：badcase + 反馈的落盘存储（JSON，跨会话保留）。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else Path(CONFIG["index_dir"]).parent / "memory.json"
        self.badcases: list[dict] = []
        self.feedback: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.badcases = data.get("badcases", [])
            self.feedback = data.get("feedback", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"badcases": self.badcases, "feedback": self.feedback},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    def add_badcase(self, bc: Badcase) -> None:
        self.badcases.append(bc.to_dict())

    def add_feedback(self, question: str, answer: str, fb: str) -> None:
        self.feedback.append({"question": question, "answer": answer,
                              "feedback": fb, "ts": time.time()})

    def stats(self) -> dict:
        issues = Counter(b.get("issue") for b in self.badcases)
        return {"n_badcases": len(self.badcases),
                "n_feedback": len(self.feedback),
                "by_issue": dict(issues)}


class Critic:
    """评论家：打分 + 发现 issue + 落库 + 反哺数据集。"""

    def __init__(self, index, memory: Memory | None = None, use_llm: bool = False):
        self.index = index
        self.memory = memory or Memory()
        self.regression = RegressionSet()
        self.use_llm = use_llm

    # ---- 评论家打分 ----
    def judge(self, question: str, answer: str, citations: list[str],
              hits: list, golden: dict | None = None) -> dict:
        """给一轮问答打分（1.0 满分，按发现的 issue 扣分），并返回 issues 列表。

        规则层（离线）：复用 badcase 的上下文检测（引用越界/零召回/低相关/漏召回）；
        LLM 层（可选）：额外判幻觉（答案是否用了资料外内容）。
        """
        cases = collect_context_badcases(question, answer, citations, hits, golden)
        issues = [c.issue for c in cases]
        score = 1.0 - sum(_ISSUE_PENALTY.get(i, 0.3) for i in issues)
        score = max(0.0, round(score, 2))

        # 可选 LLM 幻觉判定
        if self.use_llm and citations:
            if self._llm_hallucinate(question, answer):
                issues.append("hallucination")
                score = max(0.0, round(score - _ISSUE_PENALTY["hallucination"], 2))
        return {"score": score, "issues": issues}

    def _llm_hallucinate(self, question: str, answer: str) -> bool:
        """LLM 判幻觉：答案是否完全无法被引用片段支持（简化版，返回 True/False）。"""
        try:
            from .llm import generate
            prompt = (f"判断下面回答是否只能由问题推得、而资料无法支持（是=幻觉）：\n"
                      f"问题：{question}\n回答：{answer}\n只回答 是/否：")
            return "是" in generate(prompt)
        except Exception:
            return False

    # ---- badcase 收取（落库长期记忆） ----
    def collect(self, question: str, answer: str, citations: list[str],
                hits: list, golden: dict | None = None) -> list[Badcase]:
        """judge 出的 issue 落成 badcase 存进长期记忆，并加入回归集。"""
        cases = collect_context_badcases(question, answer, citations, hits, golden)
        for c in cases:
            self.memory.add_badcase(c)
            self.regression.add(c)
        return cases

    def collect_user_feedback(self, question: str, answer: str,
                              feedback: str) -> Badcase:
        """对话反馈：用户纠正 → badcase 落库。"""
        bc = collect_feedback(question, answer, feedback)
        self.memory.add_badcase(bc)
        self.regression.add(bc)
        return bc

    # ---- 反哺数据集 ----
    def feedback_to_datasets(self) -> dict:
        """把长期记忆里的 badcase 路由成训练数据。

        路由规则（明确分工，宁缺毋滥）：
          - 反馈类坏例（有用户纠正/更好答案）→ 偏好对（PPO/DPO）；
          - 检索类坏例 → 已在 collect() 时进入回归集（eval 用）；
            需要 LoRA 三元组时，用 datasets.build_triples(index, golden)，
            其正例来自黄金集、负例来自 BM25 难负例——这就是「基于 BM25 badcase 造数据」。
        """
        from .datasets import build_preference_pairs
        pairs = build_preference_pairs(self.memory.badcases)
        return {"preference_pairs": pairs, "triples": []}

    # ---- 自我迭代：闭环演示 ----
    def self_iterate(self, question: str, answer: str, citations: list[str],
                     hits: list, golden: dict | None = None) -> dict:
        """一次完整闭环：judge → collect → 反哺数据集 → 回归集统计。"""
        verdict = self.judge(question, answer, citations, hits, golden)
        self.collect(question, answer, citations, hits, golden)
        feedback = self.feedback_to_datasets()
        self.memory.save()
        return {
            "verdict": verdict,
            "new_badcases": len(self.memory.badcases),
            "preference_pairs": len(feedback["preference_pairs"]),
            "regression": self.regression.stats(),
            "memory": self.memory.stats(),
        }
