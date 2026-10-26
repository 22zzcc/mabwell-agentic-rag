# -*- coding: utf-8 -*-
"""
core/question_tree.py —— 问题精度驱动的 2 层 tree chunk
=========================================================

把「树状 chunk 基于问题精度构建」落成代码：

    一个 chunk = 一个「问题类」（2 层）
        根（第 1 层）：最低精度问题 —— 问题类标题（如「抗体纯度为什么偏低」）
        叶（第 2 层）：该类聚类出的具体条目 —— 聚集体 / 电荷异质性 / 片段降解 / ...

元数据增加两个字段：
    question_class —— 该 chunk 属于哪个「问题类」（检索时的命中钥匙）；
    precision      —— 问题精度（root=最低精度/最宽泛，leaf=高精度/最具体）。

对比普通平铺 chunk（固定窗口硬切）：这里按「问题类」聚合，一类问题一块，
块数更少、检索更快，且命中一块即拿到整类（多因诊断的 chunk 化）。
"""
from __future__ import annotations

import re
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def build_question_tree_chunks(data_dir: str | Path) -> list[dict]:
    """按「问题类」构建 2 层 tree chunk：一个「最低精度问题」= 一个 chunk。

    粒度选择（对应「问题精度」）：
        一个文档 = 一个最低精度问题（broad 问题）= 一个「问题类」chunk；
        其下的 ## 章节都是这个问题的「叶子」（具体原因/条目），并入同一个 chunk。
    这样「一类问题一块」，块数比平铺硬切少，且命中一块即拿到整类。

    返回 list[{"text":..., "meta":{question_class, precision, doc}}]。
    """
    chunks: list[dict] = []
    for path in Path(data_dir).rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        first = _HEADING_RE.match(text.lstrip().splitlines()[0].strip()) \
            if text.lstrip().splitlines() else None
        # 根 = 最低精度问题 = 文档标题（去掉编号前缀，如「01_研发_」）
        root_title = first.group(2).strip() if first else path.stem
        chunks.append(_mk(root_title, text.splitlines(), path.stem))

    return [c for c in chunks if c["text"]]


def _mk(question_class: str, lines: list[str], doc: str) -> dict:
    body = "\n".join(lines).strip()
    # 根标题不单独占一层——整个「问题类 + 它的叶子」合成一个 chunk
    return {
        "text": body,
        "meta": {
            "doc": doc,
            "question_class": question_class,
            "precision": "low",        # root = 最低精度问题
        },
    }
