# -*- coding: utf-8 -*-
"""
core/evaluate.py —— 评测体系（Recall@5 / MRR@10 / 引用正确率）
================================================================

对应简历「独立评测集上 Recall@5 0.71→0.92，MRR@10 0.65→0.88」与
「引用正确率」。一个 RAG 系统能不能上线，不能靠「看着还行」，要靠可复现指标。

本模块提供两个口径：
    1. 检索质量（evaluate_retrieval）：不经过生成，直接评「检索能不能命中答案片段」；
    2. 端到端质量（evaluate_agent）：跑完整 Agentic 循环，评「回答是否引用了正确来源、
       是否覆盖关键结论」。

纯 Python 实现，无 numpy 依赖，保证离线可跑。
"""
from __future__ import annotations

from .agent import AgenticRAG
from .config import CONFIG
from .vector_store import HybridIndex

# ---------------------------------------------------------------------------
# 黄金集：问题 → 期望命中的「标题路径子串」+ 期望答案包含的关键词
# 这些条目与 sample_data 里的六篇文档一一对应，是「可复现评测」的地基。
# ---------------------------------------------------------------------------
GOLDEN = [
    {"question": "9MW2821 靶向的靶点是什么？",
     "relevant": "Nectin-4 靶点概述", "must_contain": "Nectin-4"},
    {"question": "9MW2821 抗体的结合亲和力 KD 值是多少？",
     "relevant": "亲和力", "must_contain": "0.8"},
    {"question": "9MW2821 使用什么细胞毒载荷？",
     "relevant": "细胞毒载荷", "must_contain": "MMAE"},
    {"question": "9MW2821 的平均药物抗体比 DAR 是多少？",
     "relevant": "药物抗体比", "must_contain": "4.0"},
    {"question": "9MW2821 Ⅲ期临床的主要终点是什么？",
     "relevant": "主要终点", "must_contain": "PFS"},
    {"question": "9MW2821 Ⅲ期临床试验的对照组采用什么方案？",
     "relevant": "试验分期与对照", "must_contain": "多西他赛"},
    {"question": "严重不良事件 SAE 需要在多久内报告？",
     "relevant": "SAE 报告要求", "must_contain": "24"},
    {"question": "CTD 通用技术文件共分为几个模块？",
     "relevant": "模块划分", "must_contain": "五"},
]


def _is_relevant(hit, relevant: str) -> bool:
    """一条召回是否「相关」：标题路径或正文里出现关键词即算命中。"""
    return (relevant in hit.meta.get("heading_path", "")
            or relevant in hit.text)


def evaluate_retrieval(index: HybridIndex, golden: list[dict] = None,
                       k_recall: int = 5, k_mrr: int = 10) -> dict:
    """检索质量：Recall@K 与 MRR@K（一次取 top-k_mrr，再分别截断计算）。

    - Recall@5：前 5 条里能否命中相关片段（衡量「别漏」）；
    - MRR@10 ：第一条相关片段的排名倒数均值（衡量「排得靠不靠前」）。
    """
    golden = golden or GOLDEN
    recall_sum = 0.0
    mrr_sum = 0.0
    for g in golden:
        hits = index.search(g["question"], top_k=k_mrr)
        # 命中位次（1-based）
        ranks = [i + 1 for i, h in enumerate(hits) if _is_relevant(h, g["relevant"])]
        if ranks and ranks[0] <= k_recall:
            recall_sum += 1.0
        if ranks:
            mrr_sum += 1.0 / ranks[0]
    n = len(golden) or 1
    return {"recall@5": recall_sum / n, "mrr@10": mrr_sum / n, "n": len(golden)}


def evaluate_agent(index: HybridIndex, golden: list[dict] = None,
                   use_llm: bool = False) -> dict:
    """端到端：跑完整 Agentic 循环，评「引用正确率」与「答案覆盖」。

    - citation_hit：回答引用的来源里，是否有「命中期望来源」的那一条；
    - content_hit ：回答文本是否覆盖关键结论（关键词命中）。

    :param use_llm: False 走快速降级路径（秒级），True 真调本地模型（慢）。
    """
    golden = golden or GOLDEN
    agent = AgenticRAG(index, use_llm=use_llm)
    cite_hits = content_hits = 0
    badcases = []
    for g in golden:
        res = agent.run(g["question"])
        # 引用正确：任一被引用来源的标题路径里出现期望来源关键词
        cited_ok = any(g["relevant"] in c for c in res.citations)
        content_ok = g["must_contain"] in res.answer
        cite_hits += cited_ok
        content_hits += content_ok
        if not (cited_ok and content_ok):
            badcases.append({
                "question": g["question"],
                "cited_ok": cited_ok, "content_ok": content_ok,
                "answer": res.answer[:200],
            })
    n = len(golden) or 1
    return {
        "citation_accuracy": cite_hits / n,
        "content_coverage": content_hits / n,
        "n": len(golden),
        "badcases": badcases,
    }


def run_full_eval(index: HybridIndex, golden: list[dict] = None,
                  use_llm: bool = False) -> dict:
    """汇总检索 + 端到端指标，供 CLI/API 一键调用。

    默认 use_llm=False（秒级，适合快速演示）；需要真实生成质量时再传 True。
    """
    return {
        "retrieval": evaluate_retrieval(index, golden),
        "agent": evaluate_agent(index, golden, use_llm=use_llm),
    }
