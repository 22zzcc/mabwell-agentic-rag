# -*- coding: utf-8 -*-
"""
core/diagnoser.py —— 多因聚合诊断模块（Diagnoser）
=====================================================

这是本项目「最值得深度思考」的一个设计取舍（面试可重点讲），一句话概括：

    「一个故障/生物现象往往由多种原因造成。与其追求唯一答案（高精度、低召回），
      不如主动降低搜索精度、扩大召回，把多种可能原因都捞进来，再交给 LLM 聚类成
      「原因清单」罗列给提问者。提问者大多不是完全不会，而是缺一个提示——所以
      牺牲一点点搜索精度，反而让召回更全、方案罗列更全、用户体验上升。」

为什么这个 trade-off 成立？
    - 事实类问题（「主要终点是什么」）→ 答案是唯一的，precision 优先，top-1 精确；
    - 诊断类问题（「为什么纯度偏低」）→ 原因是集合，recall 优先，宁可多捞几个
      候选让 LLM 罗列，也不能只给一个可能错的唯一答案。

因此 Diagnoser 与 AgenticRAG（core/agent.py）是**两种模式**，互补：
    agent.run()      精确问答：少量召回 → 给一个带引用的结论；
    diagnoser.diagnose() 诊断问答：大量召回 → LLM 聚合 → 罗列多个可能原因。

实现三步（对应「集成问题 → 下属多个原因」）：
    ① 多面召回   —— 原始问题 + 多个「角度」子查询，用更大的 top_k 召回候选池；
    ② LLM 聚合   —— 把候选聚类成「可能原因/方案」清单，每个原因带引用 [n]；
    ③ 结构化输出 —— 原因清单 + 排查建议，让提问者逐条对照。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CONFIG
from .llm import chat, chat_json, generate
from .vector_store import Hit, HybridIndex

# 诊断模式的召回放大倍数：相对精确问答(默认5)放大 3 倍，召回更多候选
_RECALL_TOP_K = int(CONFIG["top_k"]) * 3

# 多角度召回的子查询规划 prompt：要求「从不同原因方向」拆解，而非「精简」
_FACET_PROMPT = """你是诊断类问题的检索规划器。用户的描述可能由多种原因造成。
请从 2~3 个不同的原因方向拆成子查询，每个子查询针对一类可能原因（用中文、贴近资料措辞）。
严格输出 JSON：{{"facets": ["子查询1", "子查询2", "子查询3"]}}

用户问题：{question}
JSON："""

# LLM 聚合 prompt：把宽召回的片段聚类成「可能原因清单」，而非唯一答案
_AGGREGATE_PROMPT = """你是生物制药领域的诊断助手。用户的描述可能由多种原因造成。
请**只依据**下面资料片段，罗列出**所有可能的原因/方案**，并遵守：
1. 每个原因写成一条，格式：`N. **原因/方案标题**：一句话说明 [依据编号]`；
2. 尽量把资料里提到的不同原因都列全，不要只挑一个；资料里没提到的不要编；
3. 最后给 1~2 句「排查/下一步建议」；
4. 用中文，引用编号用 [n] 形式，如 [1][2]。

资料片段：
{context}

用户问题：{question}
诊断清单："""


@dataclass
class DiagnoseResult:
    """一次诊断问答的完整结果。"""
    question: str
    answer: str = ""               # LLM 聚合后的「原因清单」文本
    facets: list[str] = field(default_factory=list)   # 多角度子查询
    citations: list[str] = field(default_factory=list)  # 去重后的参考来源
    recall_count: int = 0          # 召回候选数（直观展示「扩大召回」）
    use_llm: bool = True


class Diagnoser:
    """多因聚合诊断器：与 AgenticRAG 互补的「高召回 + 罗列式」问答模式。"""

    def __init__(self, index: HybridIndex, use_llm: bool = True):
        self.index = index
        self.use_llm = use_llm

    # ---- ① 多面召回：故意放大召回，牺牲一点精度换更全的原因覆盖 ----
    def _facets(self, question: str) -> list[str]:
        """用 LLM 把问题拆成「不同原因方向」的子查询（多角度召回）。"""
        if not self.use_llm:
            return []
        try:
            msg = [{"role": "user", "content": _FACET_PROMPT.format(question=question)}]
            obj = chat_json(msg)
            facets = [s for s in obj.get("facets", []) if isinstance(s, str) and s]
            return facets[:3]
        except Exception:
            return []  # LLM 不可用 → 只用原始问题一路召回

    def _broad_recall(self, question: str, facets: list[str]) -> tuple[list[Hit], list[str]]:
        """扩大召回：原始问题 + 各角度子查询，都用更大的 top_k。

        关键点：这里**故意**放大 top_k（默认 5 → 15），并合并多路结果去重，
        让「多种可能原因」尽量都进入候选池——这正是「降精度换召回」的落点。
        """
        seen: dict[str, Hit] = {}
        for q in [question] + facets:
            for h in self.index.search(q, top_k=_RECALL_TOP_K):
                seen.setdefault(h.text, h)      # 同片段去重，跨路合并
        hits = list(seen.values())
        citations = [h.meta.get("heading_path", "") or h.meta.get("doc", "")
                     for h in hits]
        return hits, citations

    # 聚合上下文上限：召回可以很宽（recall 优先），但喂给 LLM 的候选设个上限，
    # 既保证「多种原因都进清单」，又避免本地慢模型因上下文过长而超时。
    _AGGREGATE_TOP_K = 15

    # ---- ② LLM 聚合：把候选聚类成「可能原因清单」 ----
    def _aggregate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "（知识库中未检索到相关内容）"

        # 取召回里最靠前的 _AGGREGATE_TOP_K 条做聚合（召回仍全量计数，见 recall_count）
        top = hits[:_AGGREGATE_TOP_K]
        blocks = []
        for i, h in enumerate(top, start=1):
            src = h.meta.get("heading_path", "") or h.meta.get("doc", "")
            blocks.append(f"[{i}] {src}\n    {h.text}")
        context = "\n\n".join(blocks)

        if not self.use_llm:
            # 快速降级：不生成，直接把召回片段按标题罗列（同样体现「罗列」语义）
            lines = [f"- {h.meta.get('entry', h.meta.get('chapter',''))}："
                     f"{h.text[:80]}" for h in top[:8]]
            return "（离线快速模式，未调用生成模型）资料中可能相关的原因/条目：\n" + "\n".join(lines)

        try:
            return chat([{"role": "user", "content":
                          _AGGREGATE_PROMPT.format(context=context, question=question)}])
        except Exception as e:
            # 生成失败时明确报错，绝不把「检索到内容」误报成「未检索到」
            return f"（聚合生成失败：{type(e).__name__}。可改用 --no-llm 快速罗列召回条目）"

    # ---- 总入口 ----
    def diagnose(self, question: str) -> DiagnoseResult:
        """多因诊断主流程：多面召回 → LLM 聚合 → 原因清单。"""
        res = DiagnoseResult(question=question, use_llm=self.use_llm)

        res.facets = self._facets(question)
        hits, res.citations = self._broad_recall(question, res.facets)
        res.recall_count = len(hits)
        res.answer = self._aggregate(question, hits)
        return res
