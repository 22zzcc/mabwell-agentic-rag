#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_benchmark.py —— 效果实测（多因诊断 + 长期记忆反馈）
====================================================================

跑几轮，给两个「站得住」的思路各一组具体数字：

    ① 多因诊断：top_k=5（精确） vs top_k=15（多因）的「原因召回数」；
    ② 长期记忆反馈：喂几个坏例，看评论家收集的 badcase / 偏好对 / 三元组产出。

（树状 chunk 的优势在大数据量下才体现，小 MVP 不测，见 core/document_tree.py。）

用法：python scripts/compare_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingest import load_index
from core.datasets import build_triples
from core.evaluate import GOLDEN

# ① 多因诊断：黄金「原因」= 纯度偏低文档的 5 个根因章节
MULTI_CAUSE_Q = "抗体 SEC-HPLC 纯度偏低可能有哪些原因？"
GROUND_TRUTH_CAUSES = ["聚集体", "电荷异质性", "片段与降解", "工艺相关杂质", "分析方法"]


def count_causes_hit(index, query, top_k) -> int:
    hits = index.search(query, top_k=top_k)
    hit_text = " ".join(h.text for h in hits)
    return sum(1 for c in GROUND_TRUTH_CAUSES if c in hit_text)


def main() -> None:
    tree = load_index()

    # ---- ① 多因诊断 ----
    print("=" * 70)
    print("① 多因诊断：精确问答 vs 扩召回 的原因覆盖")
    c5 = count_causes_hit(tree, MULTI_CAUSE_Q, top_k=5)
    c15 = count_causes_hit(tree, MULTI_CAUSE_Q, top_k=15)
    print(f"   top_k=5 （精确问答）  ：命中 {c5}/{len(GROUND_TRUTH_CAUSES)} 个可能原因")
    print(f"   top_k=15（多因诊断）  ：命中 {c15}/{len(GROUND_TRUTH_CAUSES)} 个可能原因")

    # ---- ② 长期记忆反馈 ----
    print("=" * 70)
    print("② 长期记忆反馈：坏例 → 训练数据 产出")
    from core.memory import Critic, Memory
    critic = Critic(tree, Memory(path="data/bench_memory.json"))
    critic.collect("9MW2821 的亲和力是多少？", "资料中未提及", citations=[], hits=[])
    critic.collect("9MW2821 靶向什么？", "靶向 HER2 [99]", citations=["x"],
                   hits=tree.search("9MW2821 靶向什么", 5))
    critic.collect_user_feedback("9MW2821 的载荷是什么？", "载荷是 MMAE？",
                                 "载荷是 MMAE（单甲基奥瑞他汀E）")
    fb = critic.feedback_to_datasets()
    triples = build_triples(tree, GOLDEN)
    print(f"   喂入坏例 3 条 → 收集 badcase {len(critic.memory.badcases)} 条")
    print(f"   其中用户纠正 → 偏好对 {len(fb['preference_pairs'])} 条（喂 PPO/RLHF）")
    print(f"   BM25 难负例 → 三元组 {len(triples)} 条（喂 LoRA）")
    print(f"   记忆统计：{critic.memory.stats()}")


if __name__ == "__main__":
    main()
