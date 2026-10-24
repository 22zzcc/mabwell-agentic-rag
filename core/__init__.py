# -*- coding: utf-8 -*-
"""core 包：迈威生物 Agentic RAG 的核心模块集合。

模块职责速览（面试时可按这张表讲）：
    config        —— 配置装载（.env / 路径 / 默认值）
    llm           —— 本地大模型封装（Ollama chat/generate/JSON）
    embedding     —— 向量化 + 余弦相似度（本地 mxbai / 可切 BGE）
    document_tree —— 文档→章节→条目 三级层级还原 + 分块
    pdf_parser    —— 企业 PDF 解析（水印净化/双栏顺序/扫描件OCR/噪声过滤）
    vector_store  —— BM25 + 稠密向量 + RRF 混合检索 + 本地持久化
    ingest        —— 文档解析入库（把 sample_data 建成可检索索引）
    agent         —— Agentic 循环（改写→规划→检索→验证→作答）+ 工具
    diagnoser     —— 多因聚合诊断（降精度扩召回，LLM 聚类罗列多种可能原因）
    finetune_bge  —— BGE 领域微调管线（三元组 + 对比损失）
    evaluate      —— Recall@5 / MRR@10 / 引用正确率 + badcase
    api           —— FastAPI 服务封装（可选依赖）
"""
