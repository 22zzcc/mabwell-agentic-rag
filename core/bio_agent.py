# -*- coding: utf-8 -*-
"""
core/bio_agent.py —— 基于生物制药的领域 Agent（BioAgent）
===========================================================

核心定位（面试最该讲的一句话）：
    「生物制药公司不缺『会写代码的 agent』，缺的是『会做科学的 agent』。
      同一个 agent harness（ReAct + 工具 + RAG + 上下文管理），把工具从
      Read/Write/Bash 换成『蛋白理化计算 / 演化保守性分析 / 知识检索』，
      就变成了：发一句话 → 结合本地数据库 → 推导药性特性。」

这正好回答了「为什么 coding agent 对药企不是重点，但 agent 开发是重点」：
    - coding-agent-mvp 证明的是「agent harness 能力」（工具/权限/上下文/路由）；
    - 本模块把同一套 harness 的「工具层」换成生物制药领域工具，
      让 Agent 干的是「科学推理」而不是「改代码」。

BioAgent 的能力（一个请求 → 三步）：
    request: "分析 9MW2821 抗体的演化表征与药性特性"
      ① 检索     —— RAG 取「既有结果」：已知亲和力 KD、热稳定性 Tm、序列与演化注释；
      ② 计算     —— 领域工具：蛋白理化性质（MW/pI/GRAVY/氨基酸组成）、
                   演化保守性（位点熵 → 保守区 vs 可变区 = 演化趋向）；
      ③ 推导     —— LLM 把「数据库既有结果 + 计算特征」综合成药性推导，带引用。

所有计算工具纯 Python 实现（不依赖 Biopython），离线可跑；LLM 推导可选降级。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import CONFIG
from .llm import chat
from .vector_store import HybridIndex

# ---------------------------------------------------------------------------
# ① 蛋白理化性质计算（纯 Python，对标 ExPASy ProtParam 的简化版）
# ---------------------------------------------------------------------------
# 20 种标准氨基酸
AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_RE = re.compile(rf"[{AA}]{{{15},}}")   # 提取序列：>=15 个连续标准氨基酸

# 残基平均分子量（Da，含侧链；肽键脱水后另减 18.0153*(n-1)）
_AA_MW = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
    "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
    "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
    "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
}
# Kyte-Doolittle 疏水性标度（正值疏水、负值亲水）
_AA_HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}
# 可解离基团 pKa（用于等电点 pI 计算）
_PKA_NTERM, _PKA_CTERM = 9.69, 2.34
_PKA_POS = {"K": 10.53, "R": 12.48, "H": 6.04}
_PKA_NEG = {"D": 3.86, "E": 4.25, "C": 8.33, "Y": 10.07}


def extract_sequences(text: str) -> list[str]:
    """从文档里提取氨基酸序列（>=15 个连续标准氨基酸的片段）。

    这体现了「结合我造出的数据库」——序列就存在本地文档里，Agent 自己抽出来算。
    """
    return list(dict.fromkeys(_AA_RE.findall(text)))  # 去重保序


def protein_properties(seq: str) -> dict:
    """单条蛋白序列的理化性质：分子量 MW、等电点 pI、疏水性 GRAVY、氨基酸组成。

    这些是「基本蛋白分子模型」的第一批可计算特性，也是药性推导的输入。
    """
    seq = "".join(c for c in seq.upper() if c in AA)
    n = len(seq)
    if n == 0:
        return {}

    comp = {aa: seq.count(aa) for aa in AA if seq.count(aa)}
    # 分子量 = Σ残基质量 - 18.0153*(n-1)（每形成一个肽键脱一分子水）
    mw = sum(_AA_MW[aa] for aa in seq) - 18.0153 * (n - 1)
    # GRAVY = Σ疏水值 / 长度
    gravy = sum(_AA_HYDRO[aa] for aa in seq) / n
    # 等电点 pI：净电荷为零的 pH（二分法）
    pi = _compute_pi(seq)
    return {
        "length": n, "mw_da": round(mw, 1), "pi": round(pi, 2),
        "gravy": round(gravy, 3), "composition": comp,
    }


def _net_charge(seq: str, ph: float) -> float:
    """某 pH 下的净电荷（Henderson-Hasselbalch 近似）。"""
    charge = 0.0
    charge += 1.0 / (1.0 + 10 ** (ph - _PKA_NTERM))          # N 端（+）
    charge -= 1.0 / (1.0 + 10 ** (_PKA_CTERM - ph))          # C 端（-）
    for aa in seq:
        if aa in _PKA_POS:
            charge += 1.0 / (1.0 + 10 ** (ph - _PKA_POS[aa]))   # 碱性侧链（+）
        elif aa in _PKA_NEG:
            charge -= 1.0 / (1.0 + 10 ** (_PKA_NEG[aa] - ph))   # 酸性侧链（-）
    return charge


def _compute_pi(seq: str) -> float:
    """二分法求净电荷≈0 的 pH 值（等电点）。"""
    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# ② 演化保守性计算（演化表征 + 演化趋向）
# ---------------------------------------------------------------------------
def conservation_profile(sequences: list[str]) -> dict:
    """对一组**等长对齐**序列计算每个位点的保守性（Shannon 熵 → 保守分数）。

    演化表征：每个位点的熵/保守分数序列；
    演化趋向：保守位点（熵低，功能约束强）vs 可变位点（熵高，正在演化/适应）。
    抗体里典型结果：FR 框架区保守、CDR 决定区可变——这直接对应「结合特异性在演化」。

    :param sequences: 等长序列列表（不等长时截断到最短）
    """
    if not sequences:
        return {}
    L = min(len(s) for s in sequences)
    seqs = [s[:L] for s in sequences]
    n = len(seqs)
    hmax = math.log2(min(20, n)) if n > 1 else 1.0

    profile = []
    for i in range(L):
        col = [s[i] for s in seqs]
        counts = Counter(col)
        # Shannon 熵 H = -Σ p·log2 p
        h = -sum((c / n) * math.log2(c / n) for c in counts.values())
        cons = 1 - h / hmax          # 1=完全保守，0=完全随机
        profile.append({"pos": i + 1, "consensus": max(counts, key=counts.get),
                        "entropy": round(h, 3), "conservation": round(cons, 3)})

    conserved = [p["pos"] for p in profile if p["conservation"] >= 0.8]
    variable = [p["pos"] for p in profile if p["conservation"] <= 0.5]
    return {
        "length": L, "n_seqs": n,
        "conserved_positions": conserved,
        "variable_positions": variable,
        "profile": profile,
    }


def _group_aligned(sequences: list[str], min_seqs: int = 3) -> list[str]:
    """从提取到的序列里，找出「等长且 ≥min_seqs 条」的那一组（用于保守性分析）。"""
    by_len: dict[int, list[str]] = {}
    for s in sequences:
        by_len.setdefault(len(s), []).append(s)
    for length in sorted(by_len, reverse=True):
        group = by_len[length]
        if len(group) >= min_seqs:
            return group
    return []


# ---------------------------------------------------------------------------
# ③ 药性推导 prompt（LLM 综合「既有结果 + 计算特征」）
# ---------------------------------------------------------------------------
_SYNTH_PROMPT = """你是生物制药研发分析助手。请**只依据**下面两部分信息，推导该蛋白分子的药性特性：
A) 数据库既有结果（带来源，可引用）；
B) 计算得到的理化/演化特征（分子量、pI、疏水性、保守区/可变区等）。

要求：
1. 用中文给出 3~5 条「特性推导」，每条说明「依据什么」（既有结果标 [n]，计算特征直接引用数值）；
2. 把演化趋向与药性关联起来（如：可变区→结合特异性/免疫原性风险，保守区→结构稳定骨架）；
3. 不编造资料里没有的数据，不确定的明确说「需进一步实验验证」。

[A] 数据库既有结果：
{rag_context}

[B] 计算特征：
{computed}

用户请求：{request}
药性推导："""


# ---------------------------------------------------------------------------
# ④ BioAgent：一个请求 → 检索 + 计算 + 推导
# ---------------------------------------------------------------------------
@dataclass
class BioResult:
    request: str
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    properties: list[dict] = field(default_factory=list)
    conservation: dict = field(default_factory=dict)
    use_llm: bool = True


class BioAgent:
    """生物制药领域 Agent：复用 RAG + 领域工具，把自然语言请求变成药性推导。"""

    def __init__(self, index: HybridIndex, use_llm: bool = True):
        self.index = index
        self.use_llm = use_llm

    def run(self, request: str) -> BioResult:
        res = BioResult(request=request, use_llm=self.use_llm)

        # ① 检索：多面召回，取「既有结果」并抽取序列（结合本地数据库）。
        #    只用原请求检索，会漏掉「只有序列、语义与请求不直接相关」的序列块；
        #    所以补「氨基酸序列」「VH VL 可变区」两个领域角度，把序列块也捞进来。
        hits = self._multi_recall(request)
        rag_ctx, citations = self._format_rag(hits)
        res.citations = citations
        full_text = "\n".join(h.text for h in hits)
        res.sequences = extract_sequences(full_text)

        # ② 计算：蛋白理化性质 + 演化保守性
        for s in sorted(res.sequences, key=len, reverse=True)[:2]:   # 最长两条（VH/VL）
            res.properties.append(protein_properties(s))
        aligned = _group_aligned(res.sequences)
        if aligned:
            res.conservation = conservation_profile(aligned)

        # ③ 推导：LLM 综合（可降级为模板）
        res.answer = self._synthesize(request, rag_ctx, res)
        return res

    def _multi_recall(self, request: str) -> list:
        """多面召回：原请求 + 序列/可变区两个领域角度，用更大 top_k 合并去重。"""
        top_k = int(CONFIG["top_k"]) * 3
        queries = [request,
                   f"{request} 氨基酸序列",
                   f"{request} VH VL 可变区"]
        seen: dict[str, object] = {}
        for q in queries:
            for h in self.index.search(q, top_k=top_k):
                seen.setdefault(h.text, h)
        return list(seen.values())

    def _format_rag(self, hits) -> tuple[str, list[str]]:
        blocks, citations = [], []
        for i, h in enumerate(hits, start=1):
            src = h.meta.get("heading_path", "") or h.meta.get("doc", "")
            blocks.append(f"[{i}] {src}\n    {h.text}")
            citations.append(src)
        return "\n\n".join(blocks), citations

    def _synthesize(self, request: str, rag_ctx: str, res: BioResult) -> str:
        computed = {
            "理化性质": res.properties,
            "演化保守性": {k: v for k, v in res.conservation.items() if k != "profile"}
            if res.conservation else None,
        }
        if not self.use_llm:
            # 快速降级：直接把计算特征罗列出来（同样体现「推导」的输入）
            lines = [f"- 序列数 {len(res.sequences)} 条"]
            for p in res.properties:
                lines.append(f"- 分子量 {p.get('mw_da')} Da / pI {p.get('pi')} / "
                             f"GRAVY {p.get('gravy')}")
            if res.conservation:
                lines.append(f"- 保守位点 {len(res.conservation['conserved_positions'])} 个 / "
                             f"可变位点 {len(res.conservation['variable_positions'])} 个")
            return "（离线快速模式，未调用生成模型）计算特征如下：\n" + "\n".join(lines)
        try:
            return chat([{"role": "user", "content": _SYNTH_PROMPT.format(
                rag_context=rag_ctx, computed=str(computed), request=request)}])
        except Exception as e:
            return f"（推导生成失败：{type(e).__name__}。可改用 --no-llm 查看计算特征）"
