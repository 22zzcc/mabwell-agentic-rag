# -*- coding: utf-8 -*-
"""
core/pdf_parser.py —— 面向企业内网 PDF 的解析与文本净化
===========================================================

企业里 PDF 是内部文件流通的**主要格式**，但直接 `extract_text()` 会引入四类噪声，
把这些噪声 embed 进向量库会直接污染检索：

    1. 水印         —— 对角线「内部资料/机密/仅供内部使用」，满页重复；
    2. 双栏排版     —— 论文/申报资料/分析报告常见，简单提取会左右栏混排、顺序错乱；
    3. 扫描件       —— 无文本层，需要 OCR；且普通 OCR 会把页眉/页码/水印一起识别；
    4. 页眉页脚/页码 —— 每页重复，是纯噪声。

本模块按「检测 → 提取 → 净化」三步处理：
    parse_pdf()
      ├─ detect_pdf_type()   数字版(有文本层) vs 扫描件(纯图)
      ├─ 数字版 → extract_layout()  按栏位/阅读顺序提取（fitz > pdfplumber > pypdf）
      ├─ 扫描件 → ocr()             可插拔 OCR 后端（rapidocr/paddleocr/tesseract）
      ├─ remove_watermark()  关键词 + 跨页重复检测，剔除水印/页眉页脚
      └─ clean_text()        页码/分隔线/乱码/单字符噪声过滤

设计原则：重依赖全部惰性导入；只装 pypdf 就能处理「数字版 PDF + 水印净化」，
装 pdfplumber/fitz 获得双栏正确顺序，装 OCR 后端才处理扫描件。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 水印/页眉页脚关键词（命中且短行即剔除）
# ---------------------------------------------------------------------------
WATERMARK_KEYWORDS = [
    "内部资料", "内部文件", "内部使用", "仅供内部", "禁止外传", "禁止复制", "禁止传播",
    "机密", "绝密", "保密", "受控文件", "商业机密", "样例", "示例", "SAMPLE",
    "CONFIDENTIAL", "INTERNAL", "DRAFT", "DO NOT COPY", "FOR INTERNAL USE",
    "COMPANY PROPRIETARY", "©", "版权所有",
]

# 页码/分隔线/噪声模式（页码覆盖 "- 12 -"、"第 3 页"、"Page 3" 三种）
_PAGE_NUM_RE = re.compile(
    r"^(?:[-—–·•\s]*\d{1,4}[-—–·•\s]*|第\s*\d{1,4}\s*页|page\s*\d{1,4})$", re.I)
_SEPARATOR_RE = re.compile(r"^[-—–_=·•*]{3,}$")
_GARBAGE_RE = re.compile(r"[□■●○◆◇□]")


# ---------------------------------------------------------------------------
# ① 类型检测：数字版 vs 扫描件
# ---------------------------------------------------------------------------
def detect_pdf_type(path: str | Path, sample_pages: int = 3,
                    min_chars_per_page: int = 30) -> str:
    """判断 PDF 是「数字版（有文本层）」还是「扫描件（纯图片）」。

    原理：数字版每页能抽出可观的文本字符；扫描件 extract_text 几乎为空。
    抽样前 sample_pages 页，若平均字符数低于阈值 → 判为扫描件。
    """
    reader = _pypdf_reader(path)
    total = 0
    n = min(sample_pages, len(reader.pages))
    for page in reader.pages[:n]:
        total += len((page.extract_text() or "").strip())
    avg = total / n if n else 0
    return "scanned" if avg < min_chars_per_page else "digital"


def _pypdf_reader(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "解析 PDF 需要 pypdf：pip install pypdf\n"
                "（双栏精确顺序建议再装 pdfplumber 或 pymupdf；扫描件另装 OCR 后端）") from e
    return PdfReader(str(path))


# ---------------------------------------------------------------------------
# ② 数字版提取（含双栏阅读顺序）
# ---------------------------------------------------------------------------
def extract_layout(path: str | Path) -> list[str]:
    """按版面顺序提取每页文本，返回 list[每页文本]。

    优先用 pymupdf(fitz) 的 block 坐标做双栏排序；其次 pdfplumber 的 word 坐标；
    都不行则退回 pypdf 整页提取（无栏位顺序，会打 warning）。
    """
    try:
        import fitz  # pymupdf
        return [_extract_page_fitz(page) for page in fitz.open(str(path))]
    except ImportError:
        pass

    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                words = [{"x0": w["x0"], "y0": w["top"], "x1": w["x1"],
                          "y1": w["bottom"], "text": w["text"]}
                         for w in page.extract_words()]
                pages.append(_join_ordered(_order_items(words)))
        return pages
    except ImportError:
        pass

    # 兜底：pypdf 整页提取（无栏位顺序）
    import warnings
    warnings.warn("未安装 pdfplumber/pymupdf，双栏 PDF 的阅读顺序可能错乱；"
                  "建议 pip install pymupdf")
    reader = _pypdf_reader(path)
    return [(p.extract_text() or "") for p in reader.pages]


def _extract_page_fitz(page) -> str:
    """pymupdf：取 block 级文本，按双栏阅读顺序拼装。"""
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if text.strip():
            blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text.strip()})
    return _join_ordered(_order_items(blocks))


def _order_items(items: list[dict]) -> list[dict]:
    """双栏检测 + 阅读顺序排序（核心算法）。

    思路：按 x 中心把条目分到左/右两半；若左右组之间确实存在「垂直空隙」
    （右组最小 x0 > 左组最大 x1），且两侧都有足够内容 → 判定为双栏，
    先读完左栏（按 y 排序）再读右栏；否则按单栏「行优先」排序。
    """
    if not items:
        return items
    page_w = max(i["x1"] for i in items)
    left = [i for i in items if (i["x0"] + i["x1"]) / 2 < page_w / 2]
    right = [i for i in items if (i["x0"] + i["x1"]) / 2 >= page_w / 2]

    if left and right:
        gap = min(i["x0"] for i in right) - max(i["x1"] for i in left)
        frac_l, frac_r = len(left) / len(items), len(right) / len(items)
        # 有正空隙 + 两侧占比都不小 → 真双栏
        if gap > 0 and frac_l > 0.15 and frac_r > 0.15:
            left.sort(key=lambda i: (i["y0"], i["x0"]))
            right.sort(key=lambda i: (i["y0"], i["x0"]))
            return left + right
    # 单栏：按行（y 分桶）再按 x
    line_h = max((i["y1"] - i["y0"]) for i in items) or 5
    items.sort(key=lambda i: (round(i["y0"] / line_h), i["x0"]))
    return items


def _join_ordered(items: list[dict]) -> str:
    """把排序后的条目按行拼成文本（同一行的词用空格连接）。"""
    if not items:
        return ""
    lines = []
    cur_y = None
    cur_line = []
    line_h = max((i["y1"] - i["y0"]) for i in items) or 5
    for it in items:
        bucket = round(it["y0"] / line_h)
        if cur_y is None:
            cur_y = bucket
        if bucket != cur_y:
            lines.append(" ".join(cur_line))
            cur_line, cur_y = [it["text"]], bucket
        else:
            cur_line.append(it["text"])
    if cur_line:
        lines.append(" ".join(cur_line))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ③ 水印净化（关键词 + 跨页重复检测）
# ---------------------------------------------------------------------------
def remove_watermark(pages_text: list[str], repeat_ratio: float = 0.5) -> list[str]:
    """剔除水印与页眉页脚：单页关键词过滤 + 跨页重复检测。

    关键洞察：水印/页眉/页脚的一个共同特征是「在绝大多数页面上逐字重复」，
    而正文不会。所以：
      1. 命中 WATERMARK_KEYWORDS 的短行 → 直接剔除；
      2. 统计每行在多少页出现，出现页数占比 >= repeat_ratio 的短行 → 视为
         页眉/页脚/水印重复噪声，一并剔除。
    """
    page_count = len(pages_text)
    if page_count == 0:
        return pages_text

    # 跨页行频次统计（先做，供第 2 条使用）
    line_pages: Counter[str] = Counter()
    for pt in pages_text:
        for ln in pt.splitlines():
            s = ln.strip()
            if s:
                line_pages[s] += 1

    cleaned: list[str] = []
    for pt in pages_text:
        kept = []
        for ln in pt.splitlines():
            s = ln.strip()
            if not s:
                kept.append(ln)
                continue
            # 1) 关键词水印（短行）
            if _is_watermark_keyword(s):
                continue
            # 2) 跨页重复（页眉/页脚/水印）
            if (len(s) < 60
                    and line_pages[s] >= max(2, int(page_count * repeat_ratio))):
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned


def _is_watermark_keyword(line: str) -> bool:
    """关键词水印判定：命中关键词且是短行（正文长句即使含词也不算）。"""
    if len(line) > 40:
        return False
    upper = line.upper()
    return any(k.upper() in upper for k in WATERMARK_KEYWORDS)


# ---------------------------------------------------------------------------
# ④ 通用文本净化：页码/分隔线/乱码/单字符
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """过滤页眉页脚之外的通用噪声：页码、分隔线、乱码符号、单字符碎片。"""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _PAGE_NUM_RE.match(s):        # 纯页码 "- 12 -"
            continue
        if _SEPARATOR_RE.match(s):       # 分隔线 "----"
            continue
        if _GARBAGE_RE.search(s) and len(s) < 20:  # 常见 OCR 乱码符号
            continue
        if len(s) <= 1:                  # 单字符碎片
            continue
        out.append(s)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# ⑤ 扫描件 OCR（可插拔后端，惰性导入）
# ---------------------------------------------------------------------------
def ocr_pdf(path: str | Path, backend: str = "auto", lang: str = "ch") -> list[str]:
    """对扫描件逐页 OCR，返回 list[每页文本]。

    backend 可选：rapidocr（推荐，onnx 无需 torch）/ paddleocr / tesseract / vision。
    auto 时按可用性依次尝试。
    """
    # 先把页面渲染成图片：pymupdf 最方便
    try:
        import fitz
        doc = fitz.open(str(path))
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            images.append(pix.tobytes("png"))
    except ImportError:
        raise RuntimeError("OCR 需要先把 PDF 渲染成图片，请安装 pymupdf：pip install pymupdf")

    backend = backend.lower()
    if backend == "auto":
        for b in ("rapidocr", "paddleocr", "tesseract", "vision"):
            try:
                return _run_ocr(images, b, lang)
            except (ImportError, RuntimeError):
                continue
        raise RuntimeError("未找到可用的 OCR 后端；请安装其一："
                           "pip install rapidocr-onnxruntime （中文推荐）")

    return _run_ocr(images, backend, lang)


def _run_ocr(images: list[bytes], backend: str, lang: str) -> list[str]:
    if backend == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        pages = []
        for img in images:
            result, _ = engine(img)
            text = "\n".join(line[1] for line in result) if result else ""
            pages.append(text)
        return pages
    if backend == "paddleocr":
        from paddleocr import PaddleOCR
        engine = PaddleOCR(use_angle_cls=True, lang=lang)
        pages = []
        for img in images:
            result = engine.ocr(img, cls=True)
            text = "\n".join(line[1][0] for line in result[0]) if result and result[0] else ""
            pages.append(text)
        return pages
    if backend == "tesseract":
        import pytesseract
        from PIL import Image
        import io
        pages = []
        for img in images:
            pages.append(pytesseract.image_to_string(
                Image.open(io.BytesIO(img)), lang="chi_sim" if lang == "ch" else "eng"))
        return pages
    if backend == "vision":
        # macOS 原生 Vision OCR（无 pip 依赖，需 macOS）
        import subprocess, tempfile, os
        pages = []
        with tempfile.TemporaryDirectory() as tmp:
            for i, img in enumerate(images):
                p = os.path.join(tmp, f"p{i}.png")
                with open(p, "wb") as f:
                    f.write(img)
                # 复用简洁的 Swift/Vision 一行脚本（见脚本注释）
                pages.append(_vision_ocr(p))
        return pages
    raise RuntimeError(f"未知 OCR 后端：{backend}")


def _vision_ocr(png_path: str) -> str:
    """macOS Vision OCR 兜底（无第三方依赖）。"""
    import subprocess
    script = '''
import Vision, AppKit, Foundation
u = Foundation.NSURL.fileURLWithPath_("%s")
r = Vision.VNRecognizeTextRequest.alloc().init()
r.setRecognitionLevel_(1)  # accurate
r.setRecognitionLanguages_(["zh-Hans", "en-US"])
h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(u, {})
h.performRequests_error_([r], None)
print("\\n".join((o.topCandidates_(1)[0].string() for o in r.results() or [])))
''' % png_path
    try:
        out = subprocess.run(["swift", "-e", script], capture_output=True, text=True, timeout=120)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ⑥ 总入口：parse_pdf
# ---------------------------------------------------------------------------
def parse_pdf(path: str | Path, ocr_backend: str = "auto", lang: str = "ch") -> str:
    """解析一个 PDF 成「净化后的纯文本」，可直接喂给 ingest 分块入库。

    流程：类型检测 → （数字版：版面提取 / 扫描件：OCR）→ 水印净化 → 通用净化。
    """
    path = Path(path)
    kind = detect_pdf_type(path)
    if kind == "digital":
        pages = extract_layout(path)
    else:
        pages = ocr_pdf(path, backend=ocr_backend, lang=lang)

    pages = remove_watermark(pages)
    text = "\n".join(pages)
    return clean_text(text)
