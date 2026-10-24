# -*- coding: utf-8 -*-
"""
core/vector_store.py —— 混合检索（BM25 稀疏 + 稠密向量 + RRF 融合）
====================================================================

这是「普通 RAG」和「能打的 RAG」差距最大的地方（对应简历
「Milvus + BGE 稠密 / BM25 稀疏混合检索，RRF(k=60) 融合」）。

为什么需要混合检索？
    - 稠密向量擅长「语义相似」（「降糖」↔「血糖」），但遇到精确术语
      （靶点名、药名、批号、故障码）容易漏召回；
    - BM25 擅长「精确词匹配」（「9MW2821」「Nectin-4」这类专有名词），
      但换一种说法就匹配不到；
    - 两者互补，用 RRF（Reciprocal Rank Fusion）把各自的排序合并，
      得到既懂「同义词」又不丢「专有名词」的最终召回。

为了「离线零依赖可跑」：
    - 稠密侧用本地 Ollama embedding + numpy 余弦（core/embedding.py）；
    - 稀疏侧手写了一个经典 BM25（纯 Python），不依赖 rank-bm25；
    - 持久化：向量存 .npz（numpy 二进制），块与 BM25 索引存 .json，
      保证「一次建库、多次复用」（企业内网常见诉求）。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG
from .embedding import cosine_matrix, embed

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# ---------------------------------------------------------------------------
# 分词：中英混排的轻量方案
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """轻量分词：中文按「字 + 相邻双字」切（近似词边界），英文/数字整词小写。

    说明：正规做法是结巴分词 / 医学词表；这里为了零依赖用「字级 bigram」
    逼近中文词边界，对专有名词（药名/靶点）召回已足够，且方便面试讲清原理。
    """
    tokens: list[str] = []
    text = text.lower()
    # 英文/数字词直接入列
    tokens.extend(_WORD_RE.findall(text))
    # 中文部分：单字 + 相邻双字（bigram）
    cjk_runs = _CJK_RE.findall(text)
    tokens.extend(cjk_runs)
    for i in range(len(cjk_runs) - 1):
        tokens.append(cjk_runs[i] + cjk_runs[i + 1])
    return tokens


# ---------------------------------------------------------------------------
# BM25 稀疏检索（纯 Python 经典实现）
# ---------------------------------------------------------------------------
class BM25:
    """经典 Okapi BM25。公式：

        score(D,Q) = Σ_{t∈Q} IDF(t) · [ tf(t,D)·(k1+1) ] / [ tf(t,D) + k1·(1-b+b·|D|/avgdl) ]

      - IDF 抑制「到处都出现的词」，凸显「少见而重要的词」；
      - k1 控制词频饱和（出现很多次 ≠ 相关很多倍）；
      - b  控制文档长度归一化强度（长文档不被天然偏袒）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len: list[int] = []          # 每篇文档的 token 数
        self.avgdl: float = 0.0               # 平均文档长度
        self.tf: dict[str, dict[int, int]] = {}   # term -> {doc_id: 词频}
        self.df: dict[str, int] = {}          # term -> 出现在多少篇文档
        self._idf: dict[str, float] = {}      # 缓存 idf

    def add(self, doc_id: int, tokens: list[str]) -> None:
        """向索引加入一篇文档（token 已分好）。"""
        self.doc_len.append(len(tokens))
        seen = set()
        for t in tokens:
            # 注意：先取出 term 对应的词频表再累加，避免 setdefault 与取值顺序问题
            doc_freq = self.tf.setdefault(t, {})
            doc_freq[doc_id] = doc_freq.get(doc_id, 0) + 1
            if t not in seen:                 # df 只计「出现过」，不重复累加
                self.df[t] = self.df.get(t, 0) + 1
                seen.add(t)

    def finalize(self) -> None:
        """建库完成后调用：算平均长度并预计算 IDF。"""
        n = len(self.doc_len)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self.df.items()
        }

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        """单查询对单文档打分。"""
        if not self.doc_len:
            return 0.0
        dl = self.doc_len[doc_id]
        score = 0.0
        for t in query_tokens:
            if t not in self.tf or doc_id not in self.tf[t]:
                continue
            f = self.tf[t][doc_id]
            idf = self._idf.get(t, 0.0)   # 查询里的 OOV 词：idf 视为 0，不参与打分
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """返回 top_k 个 (doc_id, bm25_score)。"""
        qt = tokenize(query)
        scored = [(i, self.score(qt, i)) for i in range(len(self.doc_len))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 检索结果与混合索引
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    """一条召回结果：块 + 融合得分 + 各来源得分（便于讲「为什么召回了它」）。"""
    chunk_id: int
    text: str
    meta: dict
    score: float            # RRF 融合分
    dense_score: float = 0.0
    bm25_score: float = 0.0


def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion：合并多个排序，doc 最终分 = Σ 1/(k + rank)。

    rank 从 1 开始。RRF 的好处：不需要各检索器的分数量纲一致，天然鲁棒。
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridIndex:
    """把「块 + BM25 + 稠密向量」打包，对外只暴露 add/search/save/load。"""

    def __init__(self):
        self.chunks: list[dict] = []          # 每块: {id, text, meta}
        self.bm25 = BM25()
        self.dense: list[list[float]] = []    # 与 chunks 对齐的向量（仅内存；持久化到 npz）

    # ---- 建库 ----
    def add(self, text: str, meta: dict) -> None:
        """加入一个块。注意：embedding 在 save 时统一做（批量更快）。

        检索侧的关键设计：BM25 索引「标题路径 + 正文」，而不是只索正文——
        否则「只有序列、不含实体名」的块（如 VH/VL 序列）无法被「9MW2821 序列」
        这类查询命中；稠密侧仍只用正文做语义向量。
        """
        doc_id = len(self.chunks)
        self.chunks.append({"id": doc_id, "text": text, "meta": meta})
        index_text = " ".join(filter(None, [meta.get("heading_path", ""), text]))
        self.bm25.add(doc_id, tokenize(index_text))

    def build_dense(self, batch_size: int = 8) -> None:
        """对全部块批量向量化（Ollama embedding）。"""
        texts = [c["text"] for c in self.chunks]
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            vecs.extend(embed(texts[i:i + batch_size]))
        self.dense = vecs

    # ---- 检索 ----
    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        """混合检索主入口：稠密 top-N + BM25 top-N → RRF 融合 → 截断 top_k。"""
        top_k = top_k or CONFIG["top_k"]
        n = len(self.chunks)
        if n == 0:
            return []

        cand = CONFIG["rrf_candidates"]

        # 1) 稠密召回：查询向量与全部块做余弦
        if self.dense:
            from .embedding import embed_one
            qv = embed_one(query)
            sims = cosine_matrix(qv, self.dense)
            dense_rank = sorted(range(n), key=lambda i: sims[i], reverse=True)[:cand]
            dense_scores = {i: sims[i] for i in dense_rank}
        else:
            dense_rank, dense_scores = [], {}

        # 2) BM25 召回
        bm25_hits = self.bm25.search(query, cand)
        bm25_rank = [i for i, _ in bm25_hits]
        bm25_scores = {i: s for i, s in bm25_hits}

        # 3) RRF 融合
        fused = _rrf_fuse([dense_rank, bm25_rank], k=CONFIG["rrf_k"])
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

        hits: list[Hit] = []
        for doc_id, score in ranked:
            c = self.chunks[doc_id]
            hits.append(Hit(
                chunk_id=doc_id,
                text=c["text"],
                meta=c["meta"],
                score=score,
                dense_score=dense_scores.get(doc_id, 0.0),
                bm25_score=bm25_scores.get(doc_id, 0.0),
            ))
        return hits

    # ---- 持久化 ----
    def save(self, index_dir: str | Path) -> None:
        """把索引落盘：chunks+BM25 存 JSON，向量存 npz。"""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        # 落盘前务必先 finalize：计算 avgdl 与 IDF（否则 _idf 为空）
        self.bm25.finalize()

        # BM25 索引可 JSON 化
        tf = {t: {str(d): f for d, f in docmap.items()} for t, docmap in self.bm25.tf.items()}
        payload = {
            "chunks": self.chunks,
            "bm25": {
                "k1": self.bm25.k1, "b": self.bm25.b,
                "doc_len": self.bm25.doc_len, "avgdl": self.bm25.avgdl,
                "tf": tf, "df": self.bm25.df, "idf": self.bm25._idf,
            },
        }
        (index_dir / "index.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if self.dense and _np is not None:
            _np.savez_compressed(index_dir / "dense.npz",
                                 dense=_np.asarray(self.dense, dtype="float32"))

    @classmethod
    def load(cls, index_dir: str | Path) -> "HybridIndex":
        """从磁盘恢复索引（含向量）。"""
        index_dir = Path(index_dir)
        idx = cls()
        payload = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
        idx.chunks = payload["chunks"]
        b = payload["bm25"]
        idx.bm25.k1, idx.bm25.b = b["k1"], b["b"]
        idx.bm25.doc_len = b["doc_len"]
        idx.bm25.avgdl = b["avgdl"]
        idx.bm25.tf = {t: {int(d): f for d, f in docmap.items()} for t, docmap in b["tf"].items()}
        idx.bm25.df = b["df"]
        idx.bm25._idf = b["idf"]

        npz = index_dir / "dense.npz"
        if npz.exists() and _np is not None:
            idx.dense = _np.load(npz)["dense"].tolist()
        return idx

    def __len__(self) -> int:
        return len(self.chunks)
