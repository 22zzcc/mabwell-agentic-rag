# -*- coding: utf-8 -*-
"""
core/ingest.py —— 文档解析入库
================================

职责：
    把 sample_data（或任意目录）下的文档：
      1. 按扩展名分发解析器（.md/.txt 原生；.docx/.pdf 走可选解析，缺依赖时提示）；
      2. 用 document_tree 还原「文档→章节→条目」并分块；
      3. 灌入 HybridIndex（BM25 + 稠密向量）；
      4. 落盘索引，供 agent / evaluate 复用。

对应简历「知识库构建：解析手册、故障扫描件与历史工单，恢复三级层级」——
在生物制药场景里，这批文档就是：研发方案、临床方案、CTD 申报资料、
GMP 质量标准、药物警戒 SOP 等企业本地知识资产。
"""
from __future__ import annotations

from pathlib import Path

from .config import CONFIG
from .document_tree import chunk_file
from .vector_store import HybridIndex

# 支持的文件扩展名 → 解析方式
_TXT_EXTS = {".md", ".txt", ".markdown"}
# docx/pdf 属于「可选增强」：企业真实数据多是扫描件/Office 文档，
# 这里留出扩展点（简历「解析手册、扫描件」），MVP 阶段以 md/txt 演示。
_BIN_EXTS = {".docx", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx(path: Path) -> str:
    """解析 .docx（惰性导入 python-docx；缺依赖给安装指引）。"""
    try:
        import docx  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"解析 {path.name} 需要 python-docx：pip install python-docx") from e
    doc = docx.Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_pdf(path: Path) -> str:
    """解析 .pdf（惰性导入 pypdf；扫描件 PDF 需另接 OCR，见 README）。"""
    try:
        import pypdf  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"解析 {path.name} 需要 pypdf：pip install pypdf") from e
    reader = pypdf.PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def ingest_directory(data_dir: str | Path, index_dir: str | Path | None = None) -> HybridIndex:
    """把目录下所有受支持文档建库，返回 HybridIndex（已落盘）。

    :param data_dir:  数据目录（默认 sample_data）
    :param index_dir: 索引输出目录（默认 CONFIG['index_dir']）
    """
    data_dir = Path(data_dir)
    index_dir = Path(index_dir) if index_dir else Path(CONFIG["index_dir"])

    index = HybridIndex()
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in _TXT_EXTS | _BIN_EXTS)
    if not files:
        raise FileNotFoundError(f"目录 {data_dir} 下没有可解析的文档（支持 {_TXT_EXTS | _BIN_EXTS}）")

    for path in files:
        # 按扩展名分发
        ext = path.suffix.lower()
        if ext in _TXT_EXTS:
            text = _read_text_file(path)
        elif ext == ".docx":
            text = _read_docx(path)
        elif ext == ".pdf":
            text = _read_pdf(path)
        else:  # pragma: no cover
            continue

        # Document-Tree 分块
        for chunk in chunk_file_from_text(text, path):
            index.add(chunk["text"], chunk["meta"])

    # 批量向量化 + 落盘
    index.build_dense()
    index.save(index_dir)
    print(f"[ingest] 已索引 {len(files)} 篇文档 / {len(index)} 个块 -> {index_dir}")
    return index


def chunk_file_from_text(text: str, path: Path) -> list[dict]:
    """把单个文件文本转成 [{text, meta}]，供 ingest 与单文件调试共用。"""
    from .document_tree import build_document_tree
    import re
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
    lines = text.lstrip().splitlines()
    first = _HEADING.match(lines[0].strip()) if lines else None
    title = first.group(2).strip() if first else path.stem
    _, chunks = build_document_tree(text, title)
    return [{"text": c.text, "meta": {
        "doc": c.doc, "chapter": c.chapter, "entry": c.entry,
        "heading_path": c.heading_path, "chunk_index": c.chunk_index,
        "source_file": path.name,
    }} for c in chunks]


def load_index(index_dir: str | Path | None = None) -> HybridIndex:
    """加载已建索引；不存在时报错并提示先 ingest。"""
    index_dir = Path(index_dir) if index_dir else Path(CONFIG["index_dir"])
    if not (index_dir / "index.json").exists():
        raise FileNotFoundError(
            f"索引不存在：{index_dir}。请先运行 `python main.py ingest` 建库。")
    return HybridIndex.load(index_dir)
