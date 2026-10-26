#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_tree.py —— 问题精度 tree chunk vs 普通平铺 chunk 对比
======================================================================

对比三件事：
    ① 块数（问题类聚合 → 块更少）；
    ② 检索速度（块更少 → 向量比较更少 → 更快）；
    ③ 原因覆盖完整度（命中一块 = 拿到整类原因）。

牺牲的是什么：**检索精度 / 定位粒度（precision / granularity）**——
一个 chunk 塞了整类原因，命中后拿到的是「整类」，无法精确定位到某个具体条目。
这是「用粗粒度换快召回 + 完整召回」。

用法：python scripts/compare_tree.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vector_store import HybridIndex
from core.question_tree import build_question_tree_chunks
from core.evaluate import GOLDEN

QUERIES = [g["question"] for g in GOLDEN]
MULTI_CAUSE_Q = "抗体 SEC-HPLC 纯度偏低可能有哪些原因？"
GROUND_TRUTH_CAUSES = ["聚集体", "电荷异质性", "片段与降解", "工艺相关杂质", "分析方法"]


def _build_flat_index(data_dir="sample_data", size=512, overlap=64) -> HybridIndex:
    idx = HybridIndex()
    for p in Path(data_dir).rglob("*.md"):
        text = p.read_text(encoding="utf-8")
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size].strip()
            if len(chunk) < 40:
                continue
            idx.add(chunk, {"doc": p.stem, "heading_path": p.stem, "chunk_index": i})
    idx.build_dense()
    idx.bm25.finalize()
    return idx


def _build_tree_index() -> HybridIndex:
    """问题类 chunk 不能超过 embedding context（mxbai 512 token），这里截断到 400 字符。"""
    idx = HybridIndex()
    for c in build_question_tree_chunks("sample_data"):
        idx.add(c["text"][:400], c["meta"])   # 截断到 embedding context 内
    idx.build_dense()
    idx.bm25.finalize()
    return idx


def _time(index, queries, rounds=5) -> float:
    t0 = time.perf_counter()
    for _ in range(rounds):
        for q in queries:
            index.search(q, top_k=5)
    return (time.perf_counter() - t0) / (rounds * len(queries)) * 1000


def _cause_coverage(index, query, top_k) -> int:
    hits = index.search(query, top_k=top_k)
    hit_text = " ".join(h.text for h in hits)
    return sum(1 for c in GROUND_TRUTH_CAUSES if c in hit_text)


def main() -> None:
    flat = _build_flat_index()
    tree = _build_tree_index()

    print("=" * 70)
    print("① 块数对比")
    print(f"   普通平铺 chunk（固定512窗口）：{len(flat)} 块")
    print(f"   问题精度 tree chunk（2层问题类）：{len(tree)} 块")
    print(f"   块数减少：{1 - len(tree) / len(flat):.0%}")

    print("=" * 70)
    print("② 检索速度（8 查询 × 5 轮均值，top_k=5）")
    tf = _time(flat, QUERIES)
    tt = _time(tree, QUERIES)
    print(f"   普通平铺：{tf:.2f} ms/查询")
    print(f"   tree chunk：{tt:.2f} ms/查询")
    print(f"   提速：{(tf - tt) / tf:.0%}（块更少 → 向量比较更少）")

    print("=" * 70)
    print("③ 多因诊断的原因覆盖完整度（top_k=1 单块命中）")
    cf = _cause_coverage(flat, MULTI_CAUSE_Q, top_k=1)
    ct = _cause_coverage(tree, MULTI_CAUSE_Q, top_k=1)
    print(f"   普通平铺 top_k=1：命中 {cf}/{len(GROUND_TRUTH_CAUSES)} 个原因（一块只有一个碎片）")
    print(f"   tree chunk top_k=1：命中 {ct}/{len(GROUND_TRUTH_CAUSES)} 个原因（一块=整类）")

    print("=" * 70)
    print("结论：tree chunk 牺牲「定位粒度/检索精度」，换「块更少→检索更快 + 一块拿全整类」。")


if __name__ == "__main__":
    main()
