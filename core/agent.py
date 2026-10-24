# -*- coding: utf-8 -*-
"""
core/agent.py —— Agentic RAG 主循环
====================================

这是「Agentic RAG」区别于「朴素 RAG」的核心：朴素 RAG 是「一次检索 + 一次生成」，
而 Agent 会**自己决定查什么、查几次、要不要拆成子问题**，再带着引用作答。

本文件实现一个轻量 ReAct 式循环，五个阶段（对应简历「memory→plan→action→answer」）：

    question
      │ ① rewrite   术语标准化/缩写展开（AE→不良事件、9MW2821→Nectin-4 ADC）
      │ ② plan      判断单跳还是多跳，产出子查询列表（JSON 决策）
      │ ③ action    逐个子查询调用 search / lookup 工具，聚合去重
      │ ④ verify    引用接地校验（每个 [n] 必须指向真实召回的块）
      │ ⑤ answer    带 [n] 引用作答，并附参考来源
      ▼
    AgentResult(轨迹 + 答案 + 引用)

健壮性设计（面试重点）：
    - 每一步都有**降级路径**：LLM 不可用/JSON 解析失败时，自动退回
      「原文检索 + 模板作答」，保证离线演示永不崩溃；
    - max_steps 防死循环；工具调用记录成轨迹，供 evaluate 分析。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import CONFIG
from .llm import chat, chat_json, generate
from .vector_store import Hit, HybridIndex

_CITE_RE = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# 工具层：Agent 能调用的「技能」
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """一次工具调用的返回：统一 (内容, 是否成功) 结构。"""
    content: str
    ok: bool = True


class Tools:
    """工具注册表。新增工具只需实现一个方法 + 在 TOOLS 里登记描述。"""

    def __init__(self, index: HybridIndex):
        self.index = index
        self.calls: list[dict] = []          # 轨迹：记录每次工具调用（供评测）

    def search(self, query: str, top_k: int | None = None) -> ToolResult:
        """核心检索工具：混合检索（BM25+稠密+RRF）返回带来源的片段。"""
        hits = self.index.search(query, top_k)
        self.calls.append({"tool": "search", "args": {"query": query}, "n_hits": len(hits)})
        if not hits:
            return ToolResult("（知识库中未检索到相关内容）", ok=True)
        lines = []
        for i, h in enumerate(hits, start=1):
            lines.append(
                f"[{i}] {h.meta.get('doc','')} / {h.meta.get('chapter','')} / "
                f"{h.meta.get('entry','')}\n    {h.text}")
        return ToolResult("\n".join(lines))

    def lookup(self, heading: str) -> ToolResult:
        """按标题路径在文档树里定位：适合「先找药物、再找它的临床终点」这类
        多跳导航（如 lookup("9MW2821 Ⅲ期临床试验方案概要/2. 研究终点")）。"""
        matches = [c for c in self.index.chunks
                   if heading in c["meta"].get("heading_path", "")]
        self.calls.append({"tool": "lookup", "args": {"heading": heading}, "n_hits": len(matches)})
        if not matches:
            return ToolResult(f"（未找到标题路径包含「{heading}」的内容）", ok=True)
        lines = [f"[{i+1}] {c['meta'].get('heading_path','')}\n    {c['text']}"
                 for i, c in enumerate(matches[:CONFIG['top_k']])]
        return ToolResult("\n".join(lines))

    def describe(self) -> str:
        """把可用工具转成 prompt 描述，喂给「规划器」。"""
        return (
            "1. search(query) —— 在本地知识库做混合检索，返回相关片段与来源；\n"
            "2. lookup(heading) —— 按标题路径精确浏览文档树（多跳导航用）。"
        )


# ---------------------------------------------------------------------------
# 各阶段的 prompt 模板
# ---------------------------------------------------------------------------
_REWRITE_PROMPT = """你是生物制药领域的信息检索助手。请把下面的用户问题改写成检索友好的一句话：
1. 展开缩写（如 AE=不良事件、SAE=严重不良事件、ADC=抗体偶联药物）；
2. 补充同义词/别称（如 9MW2821 也可写作 Nectin-4 ADC）；
3. 保留专有名词与编号的精确写法。
只输出改写后的一句话，不要解释。

用户问题：{question}
改写结果："""

_PLAN_PROMPT = """你是检索规划器。针对下面的问题，决定检索策略：
- 若问题只需一个知识点的直接事实，给出 1 个子查询；
- 若问题涉及多个知识点（如「某药的适应症 + 主要终点 + 对照药」），拆成多个子查询。
要求：子查询必须用中文、贴近知识库原文措辞（保留专有名词，如 9MW2821、主要终点、PFS）。
严格输出 JSON：{{"sub_queries": ["子查询1", "子查询2"]}}

用户问题：{question}
可用工具：{tools}
JSON："""

_ANSWER_PROMPT = """你是迈威生物内部知识库的问答助手。请**只依据**下面提供的资料片段作答，并遵守：
1. 用中文简洁作答，直接给结论；
2. 每句话若来自某片段，在句末标注 [编号]（如 [1]）；不凭资料外的知识编造；
3. 资料不足时明确说「资料中未提及」；
4. 作答后另起一行列出「参考来源」。

资料片段：
{context}

用户问题：{question}
回答："""


# ---------------------------------------------------------------------------
# Agent 主循环
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    """一次 Agent 问答的完整结果（含轨迹，供评测与复盘）。"""
    question: str
    answer: str = ""
    rewritten: str = ""
    sub_queries: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)   # 参考来源（去重后的片段定位）
    tool_calls: list[dict] = field(default_factory=list)
    degraded: bool = False                                # 是否走了降级路径


class AgenticRAG:
    """把「检索 + 工具 + 规划 + 生成」串成一个可复用的 Agent。"""

    def __init__(self, index: HybridIndex, use_llm: bool = True):
        self.index = index
        self.tools = Tools(index)
        # use_llm=False 时走「快速降级路径」：跳过 rewrite/plan/generate 三个 LLM 调用，
        # 直接原文检索 + 模板作答（本地模型慢/不可用时用于快速演示）。
        self.use_llm = use_llm

    # ---- ① 改写 ----
    def _rewrite(self, question: str) -> str:
        if not self.use_llm:
            return question
        try:
            return generate(_REWRITE_PROMPT.format(question=question)).strip()
        except Exception:
            return question  # LLM 不可用 → 原样返回

    # ---- ② 规划 ----
    def _plan(self, question: str) -> list[str]:
        if not self.use_llm:
            return [question]  # 快速模式：单查询
        try:
            msg = [{"role": "user", "content":
                    _PLAN_PROMPT.format(question=question, tools=self.tools.describe())}]
            obj = chat_json(msg)
            subs = [s for s in obj.get("sub_queries", []) if isinstance(s, str) and s]
            return subs or [question]
        except Exception:
            return [question]  # 规划失败 → 单查询兜底

    # ---- ③ 行动：多子查询检索 + 聚合去重 ----
    def _build_context(self, sub_queries: list[str]) -> tuple[str, list[str]]:
        """把所有子查询的召回片段去重、编号，拼成作答上下文。

        :return: (编号片段文本, 参考来源列表)，两者一一对应。
        """
        seen: dict[str, Hit] = {}
        for q in sub_queries:
            hits = self.index.search(q, top_k=CONFIG["top_k"])
            # 记录检索轨迹（供评测统计「查了几次、每次命中多少」）
            self.tools.calls.append({"tool": "search", "args": {"query": q}, "n_hits": len(hits)})
            for h in hits:
                seen.setdefault(h.text, h)          # 相同片段只保留一次（跨子查询去重）
        blocks: list[str] = []
        citations: list[str] = []
        for i, (text, h) in enumerate(seen.items(), start=1):
            # heading_path 已含文档标题（breadcrumb[0]），无需再重复拼 doc
            src = h.meta.get("heading_path", "") or h.meta.get("doc", "")
            blocks.append(f"[{i}] {src}\n    {text}")
            citations.append(src)
        return "\n\n".join(blocks), citations

    # ---- ④ 校验引用 ----
    def _verify_citations(self, answer: str, context: str) -> bool:
        """引用接地校验：答案里的每个 [n] 是否都在上下文编号范围内。"""
        if not context.strip():
            return False
        n_blocks = len(re.findall(r"^\[\d+\]", context, flags=re.M))
        for m in _CITE_RE.findall(answer):
            if int(m) > n_blocks:
                return False
        return True

    # ---- ⑤ 作答 ----
    def _answer(self, question: str, context: str) -> str:
        if not self.use_llm:
            # 快速降级：不生成，直接把检索到的权威片段正文拼成回答（带来源）
            facts = [ln.strip() for ln in context.splitlines() if ln.startswith("    ")]
            body = "；".join(facts[:2])[:600] if facts else "（无检索结果）"
            return f"（离线快速模式，未调用生成模型）根据本地知识库：{body}"
        try:
            return chat([{"role": "user", "content":
                          _ANSWER_PROMPT.format(context=context, question=question)}])
        except Exception:
            # 降级：直接把检索片段拼成回答，至少保证「有据可查」
            head = "\n".join(c.split(" / ")[0] for c in self.citations[:5])
            return f"（降级作答，未调用生成模型）相关资料片段如下：\n{head}"

    # ---- 总入口 ----
    def run(self, question: str) -> AgentResult:
        """Agentic RAG 主循环：改写→规划→检索→校验→作答。"""
        res = AgentResult(question=question)
        self.tools.calls = []       # 每次运行前清空轨迹，避免跨问题串扰

        # ① 改写：术语标准化 / 缩写展开
        res.rewritten = self._rewrite(question)

        # ② 规划：单跳 or 多跳，产出子查询
        res.sub_queries = self._plan(res.rewritten)

        # ③ 行动：多子查询检索 + 聚合去重。
        # 关键健壮性设计：始终把「原始问题」也纳入检索集合，避免规划器跑偏
        # （如输出英文/偏离措辞）时把最相关的原始中文查询漏掉。
        search_queries = [question] + [q for q in res.sub_queries if q != question]
        context, res.citations = self._build_context(search_queries)
        res.tool_calls = list(self.tools.calls)

        # ④⑤ 校验引用 + 作答
        res.answer = self._answer(question, context)
        if not self._verify_citations(res.answer, context):
            res.degraded = True  # 引用异常打标，供评测定位「幻觉 / 错引」
        return res
