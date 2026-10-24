#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/demo.py —— 端到端演示（一键跑通 ingest → ask → eval）
================================================================

用法：python scripts/demo.py
它会：1) 建库  2) 连问几个问题  3) 输出评测指标。
适合面试现场快速演示，或第一次拿到代码时验证环境。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import ingest_directory, load_index
from core.agent import AgenticRAG
from core.evaluate import run_full_eval

QUESTIONS = [
    "9MW2821 靶向什么靶点，结合亲和力是多少？",
    "9MW2821 的Ⅲ期临床主要终点和对照方案分别是什么？",
    "严重不良事件需要在多久内上报？",
]


def main() -> None:
    print("① 建库（解析 sample_data → Document-Tree 分块 → 混合索引）")
    index = ingest_directory("sample_data")
    print(f"   索引规模：{len(index)} 个块\n")

    print("② Agentic 问答（默认快速降级路径，秒级；改 use_llm=True 走完整 LLM）")
    agent = AgenticRAG(index, use_llm=False)
    for q in QUESTIONS:
        res = agent.run(q)
        print("=" * 70)
        print(f"Q: {q}")
        print(f"  子查询: {'；'.join(res.sub_queries) if res.sub_queries else '-'}")
        print(f"A: {res.answer}")
        print()

    print("③ 评测")
    result = run_full_eval(index)
    print(f"  Recall@5 = {result['retrieval']['recall@5']:.3f}")
    print(f"  MRR@10   = {result['retrieval']['mrr@10']:.3f}")
    print(f"  引用正确率 = {result['agent']['citation_accuracy']:.3f}")
    print(f"  答案覆盖   = {result['agent']['content_coverage']:.3f}")


if __name__ == "__main__":
    main()
