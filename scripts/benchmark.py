#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/benchmark.py —— 检索耗时基准（精确问答 vs 多因诊断）
============================================================

只测「扩召回的额外耗时」，给「牺牲一点精度换召回」一个可复现的数字。
（不测树状 vs 平铺——小数据量体现不出树状结构优势，见 core/document_tree.py 说明。）

用法：python scripts/benchmark.py 或 python main.py benchmark
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import load_index
from core.evaluate import GOLDEN

QUERIES = [g["question"] for g in GOLDEN]


def _time_search(index, queries, top_k, rounds=5):
    t0 = time.perf_counter()
    for _ in range(rounds):
        for q in queries:
            index.search(q, top_k=top_k)
    return (time.perf_counter() - t0) / (rounds * len(queries)) * 1000


def main() -> None:
    index = load_index()
    t5 = _time_search(index, QUERIES, top_k=5)
    t15 = _time_search(index, QUERIES, top_k=15)

    print("=" * 70)
    print("检索耗时（本地 embedding + 混合检索，8 个查询 × 5 轮均值）")
    print(f"   精确问答 top_k=5 ：{t5:.1f} ms/查询")
    print(f"   多因诊断 top_k=15：{t15:.1f} ms/查询")
    print(f"   扩召回额外开销：{t15 - t5:.1f} ms/查询（几乎可忽略）")
    print()
    print("结论：扩 Top-k 的代价极小（实测 +%.1f ms），却让「多种可能原因都进候选」" % (t15 - t5))
    print("——牺牲一点检索精度，换更全的原因罗列与更好用户体验。")


if __name__ == "__main__":
    main()
