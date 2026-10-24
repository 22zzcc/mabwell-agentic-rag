# -*- coding: utf-8 -*-
"""
core/badcase.py —— badcase 收集（上下文自动 + 对话反馈）+ 分类 + 回归集
=========================================================================

对应简历「偏好数据构建：收集 300+ 条维修员反馈」+ 用户新增诉求：
badcase 不只来自「离线评测」，还有两个在线来源——

    1. 基于上下文（collect_context_badcases）：
       分析「检索上下文 / 引用 / 答案」自动发现坏例，无需用户反馈；
    2. 基于对话反馈（collect_feedback）：
       用户在对话里点 👎 / 给纠正，即把该轮沉淀成 badcase。

这些 badcase 三处复用（数据闭环的核心）：
    - 回归集   → 修好后必须重新验证（保证不再犯）；
    - 三元组   → LoRA 微调（教模型「别这么检索/别这么答」）；
    - 偏好对   → PPO/DPO（教模型「这样更好」）。

「三个 W」方法论（来自 harbor-deep，面试可讲）：
    What（现象：可复现的事实）→ Why（根因：检索/生成/数据/评测）→ Way（修复方案 + 回归验证）。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

# 根因分类 → (Why 根因, Way 修复方向)；可加 LLM 细分
_ROOT_CAUSE = {
    "citation_fail": {"why": "生成：引用越界/错引，答案标注了不存在的 [n]",
                      "way": "强化引用接地校验，越界即降级重答"},
    "recall_miss": {"why": "检索：漏召回，相关片段未进候选",
                    "way": "扩大召回 或 补 BGE 三元组微调"},
    "low_relevance": {"why": "检索：召回噪声，候选与问题不相关",
                      "way": "优化混合检索权重/阈值"},
    "empty_context": {"why": "检索：零召回，语料缺失或 query 措辞不匹配",
                      "way": "补语料 或 改写 query"},
    "hallucination": {"why": "生成：幻觉，用了资料外内容",
                      "way": "强化引用约束 + 降低温度"},
    "user_correction": {"why": "生成：用户纠正，答案方向或结论错误",
                        "way": "转偏好对喂 PPO/DPO"},
}


@dataclass
class Badcase:
    """一条 badcase：What/Why/Way 三要素 + 反馈路由标记。"""
    question: str
    answer: str = ""
    source: str = "context"          # context=上下文自动 / feedback=对话反馈
    issue: str = "low_relevance"     # 问题类型（见 _ROOT_CAUSE）
    citations: list = field(default_factory=list)
    user_feedback: str = ""          # 用户纠正/更好答案（feedback 来源）
    why: str = ""                    # 根因分类
    way: str = ""                    # 修复方案
    status: str = "open"             # open / regression / fixed
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        if not self.why:
            cause = _ROOT_CAUSE.get(self.issue, {"why": "其他", "way": "人工复核"})
            self.why = cause["why"]
            self.way = cause["way"]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# ① 基于上下文的 badcase 收集（自动，无需用户反馈）
# ---------------------------------------------------------------------------
def collect_context_badcases(question: str, answer: str, citations: list[str],
                             hits: list, golden: dict | None = None,
                             min_dense_score: float = 0.2) -> list[Badcase]:
    """分析「检索上下文 + 引用 + 答案」，自动发现坏例。

    检测三类（均可离线）：
      - citation_fail ：答案里出现越界的 [n]（引用编号 > 实际来源数）；
      - empty_context ：检索零召回；
      - low_relevance：召回片段与问题的稠密相关分过低（噪声）；
      - recall_miss  ：有 golden 时，召回未命中期望来源。
    """
    cases: list[Badcase] = []

    # 1) 引用越界：答案里 [n] 的 n 超过来源数
    import re
    if citations:
        n_src = len(citations)
        for m in re.findall(r"\[(\d+)\]", answer):
            if int(m) > n_src:
                cases.append(Badcase(question, answer, issue="citation_fail",
                                     citations=citations))
                break
    elif answer.strip():
        cases.append(Badcase(question, answer, issue="empty_context"))

    # 2) 零召回
    if not hits:
        cases.append(Badcase(question, answer, issue="empty_context"))
        return cases

    # 3) 低相关（用召回时的 dense 分数判断噪声）
    low = [h for h in hits if getattr(h, "dense_score", 1.0) < min_dense_score]
    if len(low) >= max(2, len(hits) // 2):
        cases.append(Badcase(question, answer, issue="low_relevance", citations=citations))

    # 4) 漏召回（有 golden 才判）
    if golden and golden.get("relevant"):
        rel = golden["relevant"]
        if not any(rel in h.meta.get("heading_path", "") or rel in h.text for h in hits):
            cases.append(Badcase(question, answer, issue="recall_miss", citations=citations))

    return cases


# ---------------------------------------------------------------------------
# ② 基于对话反馈的 badcase 收集
# ---------------------------------------------------------------------------
def collect_feedback(question: str, answer: str, user_feedback: str,
                     citations: list | None = None) -> Badcase:
    """用户给纠正/点 👎 时，把该轮对话沉淀成 badcase。

    :param user_feedback: 用户给的更好答案或「哪里错了」的说明；
                          作为 preference pair 的 chosen 侧。
    """
    bc = Badcase(question=question, answer=answer, source="feedback",
                 issue="user_correction", citations=citations or [],
                 user_feedback=user_feedback)
    return bc


# ---------------------------------------------------------------------------
# ③ 分类（三个 W）—— 已在 Badcase.__post_init__ 内完成，这里提供便捷入口
# ---------------------------------------------------------------------------
def classify(badcase: Badcase) -> dict:
    """返回 (What, Why, Way) 结构化结果。"""
    return {
        "what": f"问题「{badcase.question[:40]}…」答出：{badcase.answer[:80]}…",
        "why": badcase.why,
        "way": badcase.way,
        "issue": badcase.issue,
    }


# ---------------------------------------------------------------------------
# ④ 回归集：坏例一律入集，修好必须回归验证
# ---------------------------------------------------------------------------
class RegressionSet:
    """回归集管理：fix 只有「回归通过」这一条路能到 fixed（面试讲点）。"""

    def __init__(self):
        self.items: list[Badcase] = []

    def add(self, bc: Badcase) -> None:
        self.items.append(bc)

    def verify(self, bc_id: str, passed: bool) -> None:
        """回归验证：通过才置 fixed，否则保持 open。"""
        for bc in self.items:
            if bc.id == bc_id:
                bc.status = "fixed" if passed else "open"

    def stats(self) -> dict:
        from collections import Counter
        return {"total": len(self.items),
                "by_status": dict(Counter(b.status for b in self.items)),
                "by_issue": dict(Counter(b.issue for b in self.items))}
