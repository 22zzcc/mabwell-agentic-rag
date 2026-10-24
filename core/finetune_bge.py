# -*- coding: utf-8 -*-
"""
core/finetune_bge.py —— BGE 领域微调管线（三元组 + 对比损失）
==============================================================

对应简历「BGE 微调：构建领域检索三元组微调 BGE-base，适配设备名、故障码与症状描述；
Recall@5 由 0.71 升至 0.92，MRR@10 由 0.65 升至 0.88」。

在生物制药场景里，「设备名/故障码/症状」对应为：靶点名（Nectin-4）、药物代号（9MW2821）、
临床术语（PFS/ORR/SAE）与自然语言症状/终点描述。通用 BGE 对这些企业内部黑话召回差，
所以要做**领域三元组微调**。

管线分三步（都可独立运行）：
    1. build_triples()   —— 用「黄金问答 + 难负例挖掘」构造 (query, pos, neg) 三元组；
    2. train()           —— 多负例对比损失（InfoNCE/MNR）微调 BGE（需 torch）；
    3. evaluate_before_after() —— 训练前后检索指标对比，复现简历中的提升。

说明：本文件重依赖（torch / sentence-transformers）全部惰性导入；不装也能跑
三元组构造与评测（纯 Python），只有真正 `train()` 时才要求装 torch。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import CONFIG
from .vector_store import HybridIndex


@dataclass
class Triple:
    """一条训练三元组：query 与 positive 相关、与 negative 不相关。"""
    query: str
    positive: str
    negative: str


def build_triples(golden: list[dict], index: HybridIndex,
                  hard_negative_pool: int = 10) -> list[Triple]:
    """把「黄金问答对」转成三元组，并用 BM25 做难负例挖掘。

    为什么用「难负例」？—— 随机负例太容易区分，模型学不到判别力；
    取 BM25 召回里「和 query 词面相关但并非答案」的片段当负例，
    才能逼模型学会「语义相关 vs 真相关」的精细差别（面试重点）。

    :param golden: [{"question":..., "positive_path": 黄金片段标题路径子串}]
    :param index:  已建索引（提供 BM25 负例挖掘）
    """
    triples: list[Triple] = []
    for g in golden:
        q = g["question"]
        # 正例：命中「标题路径包含 positive_path」的片段
        positives = [c["text"] for c in index.chunks
                     if g["positive_path"] in c["meta"].get("heading_path", "")]
        if not positives:
            continue
        pos = positives[0]
        # 负例：BM25 召回的 top 片段里，排除正例后取最靠前的（即「难」负例）
        bm25_hits = index.bm25.search(q, hard_negative_pool)
        for doc_id, _ in bm25_hits:
            cand = index.chunks[doc_id]["text"]
            if cand != pos:
                triples.append(Triple(query=q, positive=pos, negative=cand))
                break
    return triples


def train(triples: list[Triple], base_model: str = "BAAI/bge-base-zh-v1.5",
          output_dir: str = "models/bge-base-bio", epochs: int = 2,
          batch_size: int = 8, lr: float = 2e-5) -> None:
    """用多负例对比损失（Multiple Negatives Ranking Loss）微调 BGE。

    数学本质：对每个三元组，最大化 query 与 positive 的余弦、最小化与 in-batch
    所有 negative（含其它样本的 positive 当隐式负例）的余弦。BGE 官方即用此损失。

    :param base_model: 基座（BAAI/bge-base-zh-v1.5 是中文检索的通用基座）
    :param output_dir: 微调后权重的保存目录（推理时用 sentence-transformers 加载）
    """
    try:
        import torch  # noqa: F401
        from sentence_transformers import (SentenceTransformer,  # type: ignore
                                           InputExample, losses)
        from torch.utils.data import DataLoader  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "BGE 微调需要 torch + sentence-transformers：\n"
            "  pip install torch sentence-transformers\n"
            "（国内可先 export HF_ENDPOINT=https://hf-mirror.com）") from e

    model = SentenceTransformer(base_model)
    # 组装训练样本：pos/neg 作为 in-batch 负例
    examples = [InputExample(texts=[t.query, t.positive, t.negative]) for t in triples]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    # MNR Loss：正例对拉近、其它对推远；temperature 0.05 是常用值
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        optimizer_params={"lr": lr},
        output_path=output_dir,
        show_progress_bar=True,
    )
    print(f"[finetune_bge] 微调完成，权重已保存到 {output_dir}")


def evaluate_before_after(index: HybridIndex, golden: list[dict],
                          old: dict, new: dict) -> dict:
    """打印训练前后指标对比（复现简历里的 Recall@5 / MRR@10 提升表）。

    实际项目里：old 用通用 BGE 检索、new 用微调后 BGE 检索，
    二者在同一黄金集上算指标，得到「0.71→0.92 / 0.65→0.88」这类对比。
    """
    from .evaluate import evaluate_retrieval  # 复用同一套指标口径，避免口径漂移
    if not old:
        old = evaluate_retrieval(index, golden)
    print("指标对比：")
    print(f"  Recall@5 : {old['recall@5']:.2f} -> {new['recall@5']:.2f}")
    print(f"  MRR@10   : {old['mrr@10']:.2f} -> {new['mrr@10']:.2f}")
    return {"before": old, "after": new}
