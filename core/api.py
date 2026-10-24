# -*- coding: utf-8 -*-
"""
core/api.py —— FastAPI 服务封装（可选依赖）
============================================

对应简历「FastAPI 封装诊断接口，支持 200+ 故障码与自然语言咨询」。
在生物制药场景里，接口从「故障码诊断」换成「知识库问答 / 文档检索」。

设计：FastAPI 是可选依赖——CLI 演示不需要它也能跑；只有启动 Web 服务时才要求安装。
这样本地演示一条命令即可，而企业部署时再 `pip install fastapi uvicorn`。
"""
from __future__ import annotations

from .config import CONFIG
from .ingest import ingest_directory, load_index
from .agent import AgenticRAG
from .evaluate import run_full_eval


def create_app():
    """构建 FastAPI 应用（惰性导入 fastapi / pydantic）。"""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as e:
        raise RuntimeError(
            "启动 Web 服务需要 fastapi + uvicorn：pip install fastapi uvicorn[standard]") from e

    app = FastAPI(title="迈威生物 Agentic RAG", version="0.1.0")

    # 进程内缓存索引，避免每次请求都重载（企业服务常见优化）
    _index_cache = {"index": None}

    def get_index():
        if _index_cache["index"] is None:
            _index_cache["index"] = load_index()
        return _index_cache["index"]

    class IngestRequest(BaseModel):
        data_dir: str = "sample_data"

    class AskRequest(BaseModel):
        question: str
        top_k: int | None = None

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        """重建索引：上传/更新本地文档后调用。"""
        idx = ingest_directory(req.data_dir)
        _index_cache["index"] = idx
        return {"chunks": len(idx), "status": "ok"}

    @app.post("/ask")
    def ask(req: AskRequest):
        """Agentic RAG 问答：返回答案 + 引用 + 轨迹。"""
        agent = AgenticRAG(get_index())
        res = agent.run(req.question)
        return {
            "question": res.question,
            "answer": res.answer,
            "sub_queries": res.sub_queries,
            "citations": res.citations,
            "degraded": res.degraded,
        }

    @app.get("/eval")
    def eval_endpoint():
        """一键评测：返回检索与端到端指标。"""
        return run_full_eval(get_index())

    return app


def run_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动服务（uvicorn）。"""
    try:
        import uvicorn  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "启动 Web 服务需要 uvicorn：pip install uvicorn[standard]") from e
    uvicorn.run(create_app(), host=host, port=port)
