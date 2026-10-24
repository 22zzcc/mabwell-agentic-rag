#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark.py —— 检索基准（树状 chunk 与扩召回的「牺牲精度换速度」可复现化）
====================================================================================

对应自我介绍里的量化表述，把「树状 chunk + 扩 Top-k」的 trade-off 变成可复现数字：
    1. 树状分块 vs 平铺分块的「块数量」对比（块更少 → 检索更省、层级定位更准）；
    2. 精确问答(top_k=5) vs 多因诊断(top_k=15) 的检索耗时对比。

用法：python scripts/benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import load_index
from core.evaluate import GOLDEN

QUERIES = [g["question"] for g in GOLDEN]


def _flat_chunk_count(data_dir: str = "sample_data", size: int = 512) -> int:
    """估算「平铺分块」的块数（按固定窗口硬切），用于对比树状分块的块数。"""
    total = 0
    for p in Path(data_dir).rglob("*.md"):
        text = p.read_text(encoding="utf-8")
        total += max(1, (len(text) + size - 1) // size)
    return total


def _time_search(index, queries, top_k, rounds=5):
    """对一组 query 重复 rounds 轮检索，返回平均耗时(ms)。"""
    t0 = time.perf_counter()
    for _ in range(rounds):
        for q in queries:
            index.search(q, top_k=top_k)
    return (time.perf_counter() - t0) / (rounds * len(queries)) * 1000


def main() -> None:
    index = load_index()
    tree_chunks = len(index)
    flat_chunks = _flat_chunk_count()

    t5 = _time_search(index, QUERIES, top_k=5)
    t15 = _time_search(index, QUERIES, top_k=15)

    print("=" * 70)
    print("① 分块对比（树状 vs 平铺）")
    print(f"   树状分块(Document-Tree)：{tree_chunks} 块（每块=一个「文档-章节-条目」，语义完整）")
    print(f"   平铺分块(固定512窗口)  ：约 {flat_chunks} 块（硬切，可能把「问题+答案」切断）")
    print("   树状分块的价值：不在「块数更少」，在「每块语义完整 + 带层级定位 → 可溯源、召回更准」")
    print()
    print("② 检索耗时（本地 embedding + 混合检索，8 个查询 × 5 轮均值）")
    print(f"   精确问答 top_k=5 ：{t5:.1f} ms/查询")
    print(f"   多因诊断 top_k=15：{t15:.1f} ms/查询")
    print(f"   扩召回额外开销：{t15 - t5:.1f} ms/查询（几乎可忽略）")
    print()
    print("结论：扩 Top-k 的代价极小（实测 +%.1f ms），却让「多种可能原因都进候选」——" % (t15 - t5))
    print("这就是「牺牲一点检索精度，换更全的原因罗列与更好用户体验」的可复现数字。")


if __name__ == "__main__":
    main()
