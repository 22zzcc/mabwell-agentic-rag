# -*- coding: utf-8 -*-
"""
core/datasets.py —— 数据集构建中枢（eval / LoRA / PPO 三用）
=============================================================

对应简历「偏好数据构建」+ 用户新增诉求：把 RAG 里的数据**有组织地**转成三类数据集，
分别喂给评测、LoRA 微调、PPO/DPO 强化：

    1. 检索评测集  build_retrieval_set()    —— 黄金问答 + BM25 难负例（distractor）；
    2. 三元组      build_triples()           —— (query, pos, neg)，喂 LoRA 对比学习；
    3. 偏好对      build_preference_pairs()  —— (prompt, chosen, rejected)，喂 PPO/DPO。

关键思路（用户点出）：既然检索召回里已经能靠 BM25 拿到「词面相关但非答案」的坏例，
那就**基于这些坏例反向构造数据集**——它们既是检索评测的干扰项，也是 LoRA 的难负例，
一份数据三处复用，形成「数据 → 评测 → 微调 → 强化」的统一供给。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG
from .vector_store import HybridIndex


@dataclass
class Triple:
    """三元组：query 与 positive 相关、与 negative 不相关（LoRA 用）。"""
    query: str
    positive: str
    negative: str


@dataclass
class PreferencePair:
    """偏好对：同一个 prompt 下 chosen 优于 rejected（PPO/DPO 用）。"""
    prompt: str
    chosen: str
    rejected: str


# ---------------------------------------------------------------------------
# ① 检索评测集：黄金问答 + BM25 难负例
# ---------------------------------------------------------------------------
def build_retrieval_set(index: HybridIndex, golden: list[dict],
                        hard_neg_k: int = 10) -> dict:
    """把「黄金问答对」扩展成检索评测集，并标注 BM25 难负例。

    每个样本 = {question, relevant_ids, hard_negatives, must_contain}：
      - relevant_ids    ：命中「相关子串」的块 id（正例，评测时算 Recall/MRR）；
      - hard_negatives  ：BM25 召回里「词面相关但非正例」的块 id（干扰项）——
                          它们就是 badcase 的来源，也被复用为 LoRA 难负例。
    """
    samples = []
    for g in golden:
        q = g["question"]
        rel = g.get("relevant", "")
        relevant_ids = [c["id"] for c in index.chunks
                        if rel in c["meta"].get("heading_path", "") or rel in c["text"]]
        # BM25 难负例：BM25 top 里排除正例后的块
        hard_neg = [doc_id for doc_id, _ in index.bm25.search(q, hard_neg_k)
                    if doc_id not in relevant_ids]
        samples.append({
            "question": q,
            "relevant_ids": relevant_ids,
            "hard_negatives": hard_neg,
            "must_contain": g.get("must_contain", ""),
        })
    return {"version": 1, "built_at": time.time(), "n": len(samples), "samples": samples}


# ---------------------------------------------------------------------------
# ② 三元组（LoRA）
# ---------------------------------------------------------------------------
def build_triples(index: HybridIndex, golden: list[dict],
                  hard_negative_pool: int = 10) -> list[Triple]:
    """从黄金问答 + BM25 难负例挖矿构造 (query, pos, neg) 三元组（喂 LoRA）。

    与 finetune_bge.build_triples 同源，此处收口到「数据集中枢」统一管理。
    """
    triples: list[Triple] = []
    for g in golden:
        q = g["question"]
        rel = g.get("relevant", "")
        positives = [c["text"] for c in index.chunks
                     if rel in c["meta"].get("heading_path", "") or rel in c["text"]]
        if not positives:
            continue
        pos = positives[0]
        for doc_id, _ in index.bm25.search(q, hard_negative_pool):
            cand = index.chunks[doc_id]["text"]
            if cand != pos:
                triples.append(Triple(query=q, positive=pos, negative=cand))
                break
    return triples


# ---------------------------------------------------------------------------
# ③ 偏好对（PPO/DPO）
# ---------------------------------------------------------------------------
def build_preference_pairs(badcases: list[dict]) -> list[PreferencePair]:
    """从 badcase/反馈构造偏好对 (prompt, chosen, rejected)。

    规则：带「用户纠正/正确答案」的 badcase → chosen=用户给的更好答案，
    rejected=当时的差答案；无更好答案的自动 badcase 不强行造偏好对（宁缺毋滥）。
    """
    pairs: list[PreferencePair] = []
    for b in badcases:
        chosen = b.get("user_feedback") or b.get("gold_answer")
        if not chosen:
            continue
        pairs.append(PreferencePair(
            prompt=b.get("question", ""),
            chosen=chosen,
            rejected=b.get("answer", ""),
        ))
    return pairs


# ---------------------------------------------------------------------------
# ④ DatasetHub：版本化存取
# ---------------------------------------------------------------------------
class DatasetHub:
    """数据集统一管理：注册/保存/加载/版本化，供 eval / LoRA / PPO 共用。"""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else Path(CONFIG["index_dir"]).parent / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, data) -> Path:
        """保存一份数据集（版本号 = 时间戳，可追溯）。"""
        path = self.root / f"{name}.json"
        payload = {"name": name, "saved_at": time.time(), "data": data}
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                        encoding="utf-8")
        return path

    def load(self, name: str) -> dict:
        path = self.root / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"数据集不存在：{path}。请先 build_* 再 save。")
        return json.loads(path.read_text(encoding="utf-8"))["data"]

    def list(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
