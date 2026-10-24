# -*- coding: utf-8 -*-
"""
core/config.py —— 全局配置装载
=================================

职责（对应简历「知识库构建 / 落地」的工程化底座）：
    1. 解析 config/.env（或项目根 .env），把敏感信息（密钥）挡在代码之外；
    2. 把所有相对路径统一解析到「项目根目录」，保证在任意 cwd 下都能跑；
    3. 暴露一组默认值，方便本地离线演示时无需任何配置即可启动。

面试可讲点：
    - 密钥不进代码 / 不进 git（.env 进 .gitignore）；
    - 配置三层优先级：环境变量 > .env 文件 > 内置默认值；
    - 面向「企业本地部署」：模型地址、embedding 模型、索引目录都可按现场环境切换。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目根目录：本文件位于 <root>/core/config.py，上溯两级即项目根。
# 所有相对路径都以它为锚点，避免「从别的目录启动就找不到文件」的坑。
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env 候选位置：优先 config/.env，其次项目根 .env
_ENV_CANDIDATES = [
    PROJECT_ROOT / "config" / ".env",
    PROJECT_ROOT / ".env",
]


def _load_dotenv() -> None:
    """极简 .env 解析器：不依赖 python-dotenv，保证零额外依赖可跑。

    只处理 KEY=VALUE 行，忽略注释(#)与空行，值不做引号剥离以外的复杂处理。
    已经存在于 os.environ 的变量**不覆盖**（环境变量优先级最高）。
    """
    for env_file in _ENV_CANDIDATES:
        if not env_file.exists():
            continue
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            # 环境变量已设置时保留其值（优先级：外部环境 > .env）
            os.environ.setdefault(key, value)


_load_dotenv()


def get_env(key: str, default: str = "") -> str:
    """读取环境变量；未设置时返回 default。

    统一入口的好处：所有模块都从这里取值，改默认值时只动一处。
    """
    return os.environ.get(key, default)


def resolve_path(p: str | Path) -> Path:
    """把「可能是相对路径」的配置项解析成绝对路径（相对项目根）。

    例：resolve_path("data/index") -> <PROJECT_ROOT>/data/index
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# 集中定义的配置项（带注释的默认值即「离线可跑」的最小配置）
# ---------------------------------------------------------------------------
CONFIG = {
    # ---- 本地模型服务（Ollama）----
    "ollama_url": get_env("OLLAMA_URL", "http://127.0.0.1:11434"),
    # 生成模型：本地 qwen3:4b；生产环境可换成 qwen2.5:14b / 32b 或 vLLM 端点
    "chat_model": get_env("CHAT_MODEL", "qwen3:4b"),
    # 向量化模型：本地 mxbai-embed-large(1024维)；简历中的 BGE-base/bge-large-zh 可经
    # sentence-transformers 或 `ollama pull bge-m3` 替换，见 core/finetune_bge.py
    "embed_model": get_env("EMBED_MODEL", "mxbai-embed-large"),

    # ---- 索引与检索 ----
    "index_dir": str(resolve_path(get_env("INDEX_DIR", "data/index"))),
    # 检索召回条数（进入重排/作答上下文前的候选数）
    "top_k": int(get_env("TOP_K", "5")),
    # RRF 融合时每个子检索器的候选规模（越大融合越稳，代价是慢一点）
    "rrf_candidates": int(get_env("RRF_CANDIDATES", "20")),
    # RRF 的常数 k（经典取 60；见 Cormack et al. 2009）
    "rrf_k": int(get_env("RRF_K", "60")),

    # ---- 分块 ----
    # 每块最大字符数（简历「分块策略」；512 + 64 重叠是经验值）
    "chunk_size": int(get_env("CHUNK_SIZE", "512")),
    "chunk_overlap": int(get_env("CHUNK_OVERLAP", "64")),

    # ---- Agent ----
    # Agent 最大行动步数，防止检索/规划死循环（简历「最大步数防死循环」）
    "max_steps": int(get_env("MAX_STEPS", "5")),
    # 生成温度：问答任务压低温度，减少幻觉、提升引用稳定性
    "temperature": float(get_env("TEMPERATURE", "0.3")),

    # ---- 评测 ----
    # 评测指标里的 K（Recall@K / MRR@K）；与简历对齐：Recall@5 / MRR@10
    "eval_k": int(get_env("EVAL_K", "5")),
}
