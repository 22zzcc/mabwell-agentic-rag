# -*- coding: utf-8 -*-
"""
core/document_tree.py —— 文档→章节→条目 三级层级还原 + 分块
=========================================================

这是本项目和「普通 RAG」拉开差距的第一个关键点（对应简历
「恢复文档—章节—故障条目 三级层级，Document-Tree 分块并关联所属章节」）。

为什么要做层级分块，而不是整篇平均切？
    - 企业文档（药典/申报资料/批记录）是强结构化的：一个「条目」往往自包含
      一个完整知识点（某靶点的 KD 值、某终点的定义、某 SOP 的上报时限）；
    - 若按固定窗口硬切，会把「问题+答案」切到不同块里，检索召回时上下文残缺，
      生成时容易拼错/漏答；
    - 保留「文档→章节→条目」的 breadcrumb，回答时可以精确定位原文出处
      （面试官很看重「答案可溯源 / 引用正确率」）。

关键设计（与多因诊断一体，不是两个独立技巧）：
    树状 chunk 是**基于多因/问题精度思路构建**的——每个叶子节点对应一个「可能原因/条目」，
    因此多因诊断时树结构本身就完成了「跨节点多面召回」，几乎不需要额外的混合搜索
    去补召回。这一优势在小数据量上体现不出来，当知识库规模变大（上千个条目）时，
    树状的层级导航 + 语义完整性会明显优于平铺硬切。

chunk 大小的两个边界（面试被问「chunk 为什么这么大」时的答案）：
    - 下限 = 语义完整性：一个知识点不能切碎，否则召回残缺、答案拼错；
    - 上限 = embedding 模型的 context 窗口（如 mxbai-embed-large 是 512 token）：
      块超过 context 会直接报错（实测整篇 ~2000 字就超了），所以「问题类」chunk 的
      粒度只能在这两个边界之间选——这正是「语义完整 ↔ 定位精度」trade-off 的落点。

实现思路：
    1. 用 Markdown 标题层级（#/##/###/####）还原树结构；
    2. 每个节点记录：标题、层级、父节点、自身正文；
    3. 分块时以「叶子正文」为单位做句界感知切割，并把祖先面包屑写入元数据，
       使每块都自带「文档/章节/条目」定位信息。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG

# 匹配 Markdown 标题：# 数量决定层级
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# 句界：中文句末标点 / 英文句末标点 / 换行。用于「切分点回退到句子边界」。
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？!?；;](?=\s|$)|[.\n]")


@dataclass
class DocNode:
    """文档树节点：标题 + 层级 + 正文 + 子节点。"""
    title: str
    level: int                              # 0=文档根, 1=##章节, 2=###条目 ...
    parent: "DocNode | None" = None
    children: list["DocNode"] = field(default_factory=list)
    own_text: list[str] = field(default_factory=list)  # 直属该标题、且不在任何子标题下的正文行

    @property
    def breadcrumb(self) -> list[str]:
        """从根到当前节点的标题路径（不含根文档标题本身时也保留层级感）。"""
        crumbs: list[str] = []
        node: DocNode | None = self
        while node is not None:
            crumbs.append(node.title)
            node = node.parent
        return list(reversed(crumbs))


@dataclass
class Chunk:
    """一个可入库的文本块，携带完整的层级定位元数据。"""
    text: str
    doc: str            # 文档标题（对应「文档」层）
    chapter: str        # 章节标题（对应「章节」层，##）
    entry: str          # 条目标题（对应「条目」层，### 或更深）
    heading_path: str   # 完整面包屑，如「靶点研究 / 靶点选择 / Nectin-4」
    chunk_index: int    # 同一节点内的块序号


def _split_sentences(text: str, max_chars: int, overlap: int) -> list[str]:
    """句界感知分块：贪心装填到 max_chars，超限时回退到最近的句界切开，
    相邻块之间保留 overlap 个字符的上下文重叠。

    这样既保证每块「够大、语义完整」，又通过重叠避免关键句被切断。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 先按句界切成原子句子
    sentences: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in "。！？!?；;":
            sentences.append(buf)
            buf = ""
    if buf.strip():
        sentences.append(buf)

    chunks: list[str] = []
    cur = ""
    for s in sentences:
        # 当前块 + 新句子 超限且当前块非空 → 封存当前块
        if cur and len(cur) + len(s) > max_chars:
            chunks.append(cur.strip())
            # 取上一块结尾 overlap 个字符作为新块的开头（句界对齐）
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = tail
        cur += s
        # 单句过长（超过 max_chars）时按字符硬切
        while len(cur) > max_chars:
            chunks.append(cur[:max_chars].strip())
            cur = cur[max_chars - overlap:] if overlap > 0 else ""
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def parse_markdown_tree(text: str, doc_title: str) -> DocNode:
    """把 Markdown 文本解析成 DocNode 树。

    :param text:      文件全文
    :param doc_title: 文档标题（文件名或第一个 # 标题）
    """
    root = DocNode(title=doc_title, level=0)
    stack: list[DocNode] = [root]  # 用栈维护「当前标题的祖先链」
    title_skipped = False           # 文档标题(#)已作为 root.title，解析时跳过

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group(1))          # #=1, ##=2, ###=3
            title = m.group(2).strip()
            # 第一个与文档标题一致的 # 标题：它已体现在 root.title 里，
            # 跳过不建子节点，避免 breadcrumb 出现「文档标题重复」。
            if level == 1 and title == doc_title and not title_skipped:
                title_skipped = True
                continue
            node = DocNode(title=title, level=level)
            # 向上找最近的、层级比它小的节点作为父节点（还原树结构）
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else root
            node.parent = parent
            parent.children.append(node)
            stack.append(node)
        else:
            # 非标题行：归属到当前最深的标题节点
            stack[-1].own_text.append(line)

    return root


def _walk_chunk(node: DocNode, doc_title: str, chunks: list[Chunk]) -> None:
    """递归遍历树，对每个有正文的节点做句界分块，并写入层级元数据。

    层级归属规则（面包屑 crumbs = [文档标题, ##章节, ###条目, ...]）：
      chapter = crumbs[1]（最近的 ## 章节）；没有章节层时回退为文档标题；
      entry   = crumbs[-1]（最深的标题，即条目本身）；深度不足 3 时回退为章节。
    """
    crumbs = node.breadcrumb
    chapter = crumbs[1] if len(crumbs) >= 2 else doc_title
    entry = crumbs[-1] if len(crumbs) >= 3 else chapter

    text = "\n".join(node.own_text).strip()
    if text:
        for i, piece in enumerate(_split_sentences(
                text, CONFIG["chunk_size"], CONFIG["chunk_overlap"])):
            chunks.append(Chunk(
                text=piece,
                doc=doc_title,
                chapter=chapter,
                entry=entry,
                heading_path=" / ".join(crumbs),
                chunk_index=i,
            ))

    for child in node.children:
        _walk_chunk(child, doc_title, chunks)


def build_document_tree(text: str, doc_title: str) -> tuple[DocNode, list[Chunk]]:
    """入口：把一篇文档变成 (树, 分块列表)。"""
    root = parse_markdown_tree(text, doc_title)
    chunks: list[Chunk] = []
    _walk_chunk(root, doc_title, chunks)
    return root, chunks


def chunk_file(path: Path) -> list[Chunk]:
    """读取单个文件并分块；doc_title 优先取文件内第一个 # 标题，否则用文件名。"""
    text = path.read_text(encoding="utf-8")
    first = _HEADING_RE.match(text.lstrip().splitlines()[0].strip() if text.lstrip().splitlines() else "")
    title = first.group(2).strip() if first else path.stem
    _, chunks = build_document_tree(text, title)
    return chunks
