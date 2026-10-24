# -*- coding: utf-8 -*-
"""
core/diagnoser.py —— 多因聚合诊断模块（Diagnoser）
=====================================================

核心设计（面试可重点讲，这是一套「重写 + 精准检索」的架构，不是「宽召回 + 混合搜索」）：

    多因诊断**放在问题重写阶段**，而不是检索阶段：
        一个故障/生物现象往往由多种原因造成，所以先把「原始问题」重写成 N 个
        「原因向子查询」；而树状 chunk 的每个节点本来就是一个「可能原因/条目」，
        于是每个子查询直接命中对应节点——**检索根本不费力**；最后只要做一层
        「相关性剔除 + rerank」的轻量后处理，把噪声去掉、重排一下即可。

因此 Diagnoser 与 AgenticRAG（core/agent.py）是两种模式：
    agent.run()       精确问答：问题 → 一次精准检索 → 唯一结论；
    diagnoser.diagnose() 诊断问答：问题 → 重写成多因子查询 → 精准检索命中各原因节点
                          → 相关性剔除 + rerank → 罗列多个可能原因。

四步：
    ① 问题重写   —— 把多因诊断放到这里：LLM 拆成 2~3 个「原因向子查询」；
    ② 精准检索   —— 树状 chunk 每个节点=一个原因，子查询直接命中节点（top_k 很小）；
    ③ 相关性剔除 + rerank —— 轻量后处理：滤噪声、按语义相关分重排；
    ④ LLM 罗列   —— 把命中块聚类成「可能原因清单」，每个带引用 [n]。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CONFIG
from .llm import chat, chat_json
from .vector_store import Hit, HybridIndex

# 精准检索的 top_k：树状 chunk 每个节点=一个原因，重写后的子查询直接命中节点，
# 不需要宽召回（对比精确问答 top_k=5，这里只需更小的精准命中）。
_PRECISE_TOP_K = 3

# 相关性剔除阈值：语义相关分(dense)低于此值的块视为噪声（重写+树状检索后噪声本就不多）
_MIN_DENSE_SCORE = 0.1

# 问题重写 prompt：把多因诊断放到这里，拆成「原因向子查询」
_REWRITE_PROMPT = """你是诊断类问题的问题重写器。用户的描述可能由多种原因造成。
请把问题重写成 2~3 个「原因向子查询」，每个子查询针对一类可能原因（用中文、贴近资料措辞、
尽量与文档条目标题对齐）。严格输出 JSON：{{"sub_queries": ["子查询1", "子查询2", "子查询3"]}}

用户问题：{question}
JSON："""

# LLM 罗列 prompt：把命中的原因块聚类成「可能原因清单」
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
    answer: str = ""               # LLM 罗列后的「原因清单」文本
    sub_queries: list[str] = field(default_factory=list)  # 问题重写后的原因向子查询
    citations: list[str] = field(default_factory=list)    # 去重后的参考来源
    hit_count: int = 0             # 精准检索命中块数
    use_llm: bool = True


class Diagnoser:
    """多因聚合诊断器：问题重写 → 精准检索 → 相关性剔除/rerank → 罗列。"""

    def __init__(self, index: HybridIndex, use_llm: bool = True):
        self.index = index
        self.use_llm = use_llm

    # ---- ① 问题重写：多因诊断发生在这里 ----
    def _rewrite(self, question: str) -> list[str]:
        """把原始问题重写成「原因向子查询」（LLM）；不可用则退回原问题单路。"""
        if not self.use_llm:
            return [question]
        try:
            msg = [{"role": "user", "content": _REWRITE_PROMPT.format(question=question)}]
            obj = chat_json(msg)
            subs = [s for s in obj.get("sub_queries", []) if isinstance(s, str) and s]
            return subs[:3] or [question]
        except Exception:
            return [question]

    # ---- ② 精准检索：树状 chunk 每个节点=一个原因，子查询直接命中 ----
    def _retrieve(self, sub_queries: list[str]) -> list[Hit]:
        """对每个「原因向子查询」做精准检索（小 top_k），跨路去重。

        关键点：因为树状 chunk 的每个节点就是一个原因/条目，重写后的子查询能
        直接命中对应节点，所以 top_k 只需很小（_PRECISE_TOP_K=3），检索不费力。
        """
        seen: dict[str, Hit] = {}
        for q in sub_queries:
            for h in self.index.search(q, top_k=_PRECISE_TOP_K):
                seen.setdefault(h.text, h)      # 同片段去重
        return list(seen.values())

    # ---- ③ 相关性剔除 + rerank（轻量后处理）----
    def _filter_rerank(self, hits: list[Hit]) -> list[Hit]:
        """滤掉语义相关分过低的噪声块，再按相关分重排。

        重写 + 树状检索之后噪声本就不多，这一步只是「稍微」做个相关性剔除 + rerank，
        不必上重型混合搜索/交叉编码器，所以速度快。
        """
        if not hits:
            return hits
        hits = sorted(hits, key=lambda h: h.dense_score, reverse=True)  # rerank
        kept = [h for h in hits if h.dense_score >= _MIN_DENSE_SCORE]   # 相关性剔除
        return kept or hits[:1]  # 全被滤掉时至少保留最相关的一条，避免空结果

    # ---- ④ LLM 罗列：聚类成原因清单 ----
    def _aggregate(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            return "（知识库中未检索到相关内容）"
        blocks = []
        for i, h in enumerate(hits, start=1):
            src = h.meta.get("heading_path", "") or h.meta.get("doc", "")
            blocks.append(f"[{i}] {src}\n    {h.text}")
        context = "\n\n".join(blocks)

        if not self.use_llm:
            lines = [f"- {h.meta.get('entry', h.meta.get('chapter',''))}：{h.text[:80]}"
                     for h in hits[:8]]
            return "（离线快速模式，未调用生成模型）资料中可能相关的原因/条目：\n" + "\n".join(lines)
        try:
            return chat([{"role": "user", "content":
                          _AGGREGATE_PROMPT.format(context=context, question=question)}])
        except Exception as e:
            return f"（聚合生成失败：{type(e).__name__}。可改用 --no-llm 快速罗列）"

    # ---- 总入口 ----
    def diagnose(self, question: str) -> DiagnoseResult:
        """多因诊断主流程：问题重写 → 精准检索 → 相关性剔除/rerank → 罗列。"""
        res = DiagnoseResult(question=question, use_llm=self.use_llm)

        res.sub_queries = self._rewrite(question)          # ① 多因诊断放在重写
        hits = self._retrieve(res.sub_queries)             # ② 精准检索（不费力）
        hits = self._filter_rerank(hits)                   # ③ 相关性剔除 + rerank
        res.hit_count = len(hits)
        res.citations = [h.meta.get("heading_path", "") or h.meta.get("doc", "")
                         for h in hits]
        res.answer = self._aggregate(question, hits)       # ④ 罗列
        return res
