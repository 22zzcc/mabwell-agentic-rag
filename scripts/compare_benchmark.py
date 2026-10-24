#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_benchmark.py —— 三要素 vs 基线 效果对比
==========================================================

跑几轮，给「多因诊断 / 树状 chunk / 长期记忆反馈」三个思路各一组具体数字：

    ① 多因诊断：top_k=5（精确） vs top_k=15（多因）的「原因召回数」；
    ② 树状 chunk：Document-Tree 分块 vs 平铺固定窗口分块的 Recall@5 / MRR@10；
    ③ 长期记忆反馈：喂几个坏例，看评论家收集的 badcase / 偏好对 / 三元组产出。

用法：python scripts/compare_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import load_index
from core.vector_store import HybridIndex
from core.evaluate import evaluate_retrieval, GOLDEN

# ① 多因诊断：黄金「原因」= 纯度偏低文档的 5 个根因章节
MULTI_CAUSE_Q = "抗体 SEC-HPLC 纯度偏低可能有哪些原因？"
GROUND_TRUTH_CAUSES = ["聚集体", "电荷异质性", "片段与降解", "工艺相关杂质", "分析方法"]


def count_causes_hit(index, query, top_k) -> int:
    hits = index.search(query, top_k=top_k)
    hit_text = " ".join(h.text for h in hits)
    return sum(1 for c in GROUND_TRUTH_CAUSES if c in hit_text)


# ② 树状 chunk vs 平铺 chunk
def build_flat_index(data_dir="sample_data", size=512, overlap=64) -> HybridIndex:
    """按固定窗口硬切（无标题树）建一个平铺索引，用于对比。"""
    idx = HybridIndex()
    for p in Path(data_dir).rglob("*.md"):
        text = p.read_text(encoding="utf-8")
        title = p.stem
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size].strip()
            if len(chunk) < 40:
                continue
            idx.add(chunk, {"doc": title, "chapter": title, "entry": title,
                            "heading_path": title, "chunk_index": i // (size - overlap)})
    idx.build_dense()
    idx.bm25.finalize()   # 关键：建库后必须 finalize 才算平均长度与 IDF
    return idx


def main() -> None:
    tree = load_index()

    # ---- ① 多因诊断 ----
    print("=" * 70)
    print("① 多因诊断：精确问答 vs 扩召回 的原因覆盖")
    c5 = count_causes_hit(tree, MULTI_CAUSE_Q, top_k=5)
    c15 = count_causes_hit(tree, MULTI_CAUSE_Q, top_k=15)
    print(f"   top_k=5 （精确问答）  ：命中 {c5}/{len(GROUND_TRUTH_CAUSES)} 个可能原因")
    print(f"   top_k=15（多因诊断）  ：命中 {c15}/{len(GROUND_TRUTH_CAUSES)} 个可能原因")

    # ---- ② 树状 chunk vs 平铺 ----
    print("=" * 70)
    print("② 树状 chunk vs 平铺 chunk（黄金集 8 问）")
    flat = build_flat_index()
    tree_retr = evaluate_retrieval(tree, GOLDEN)
    flat_retr = evaluate_retrieval(flat, GOLDEN)
    print(f"   树状(Document-Tree)：Recall@5={tree_retr['recall@5']:.3f}  MRR@10={tree_retr['mrr@10']:.3f}  块数={len(tree)}")
    print(f"   平铺(固定512窗口)   ：Recall@5={flat_retr['recall@5']:.3f}  MRR@10={flat_retr['mrr@10']:.3f}  块数={len(flat)}")
    # 树状的价值：不是关键词召回率（平铺块大反而更高），而是「可溯源 + 语义完整」
    tree_traceable = sum(1 for c in tree.chunks if " / " in c["meta"].get("heading_path", ""))
    flat_traceable = sum(1 for c in flat.chunks if " / " in c["meta"].get("heading_path", ""))
    print(f"   可溯源到条目的块占比：树状 {tree_traceable}/{len(tree)}，平铺 {flat_traceable}/{len(flat)}")
    print("   → 树状牺牲了 %.3f 的关键词召回，换取「答案可精确定位到文档-章节-条目」+ 块语义完整" % (
        flat_retr['recall@5'] - tree_retr['recall@5']))

    # ---- ③ 长期记忆反馈 ----
    print("=" * 70)
    print("③ 长期记忆反馈：坏例 → 训练数据 产出")
    from core.memory import Critic, Memory
    critic = Critic(tree, Memory(path="data/bench_memory.json"))
    # 喂 3 个坏例（零召回 / 引用越界 / 用户纠正）
    critic.collect("9MW2821 的亲和力是多少？", "资料中未提及", citations=[], hits=[])
    critic.collect("9MW2821 靶向什么？", "靶向 HER2 [99]", citations=["x"], hits=tree.search("9MW2821 靶向什么", 5))
    critic.collect_user_feedback("9MW2821 的载荷是什么？", "载荷是 MMAE？", "载荷是 MMAE（单甲基奥瑞他汀E）")
    fb = critic.feedback_to_datasets()
    from core.datasets import build_triples
    triples = build_triples(tree, GOLDEN)
    print(f"   喂入坏例 3 条 → 收集 badcase {len(critic.memory.badcases)} 条")
    print(f"   其中用户纠正 → 偏好对 {len(fb['preference_pairs'])} 条（喂 PPO/RLHF）")
    print(f"   BM25 难负例 → 三元组 {len(triples)} 条（喂 LoRA）")
    print(f"   记忆统计：{critic.memory.stats()}")


if __name__ == "__main__":
    main()
