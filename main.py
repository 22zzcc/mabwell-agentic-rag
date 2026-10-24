#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py —— CLI 入口
====================

用法（在项目根目录下运行）：

    python main.py ingest              # 把 sample_data 建库（需先启动 Ollama）
    python main.py ask "9MW2821 的Ⅲ期临床主要终点是什么？"
    python main.py eval                # 一键评测（Recall@5 / MRR@10 / 引用正确率）
    python main.py api                 # 启动 FastAPI 服务（可选依赖）

所有相对路径都以项目根为锚点（见 core/config.py），因此在任意 cwd 下
`python /path/to/mabwell-agentic-rag/main.py ...` 也能正常跑。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把项目根加入 sys.path，保证「从别的目录直接执行本文件」也能 import 到 core
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_ingest(args: argparse.Namespace) -> None:
    from core.ingest import ingest_directory
    ingest_directory(args.data)


def cmd_ask(args: argparse.Namespace) -> None:
    from core.ingest import load_index
    from core.agent import AgenticRAG

    index = load_index()
    agent = AgenticRAG(index, use_llm=not args.no_llm)
    res = agent.run(args.question)

    print("=" * 70)
    print(f"问题   : {res.question}")
    print(f"改写   : {res.rewritten}")
    print(f"子查询 : {'；'.join(res.sub_queries) if res.sub_queries else '-'}")
    print("-" * 70)
    print(f"回答   :\n{res.answer}")
    if res.citations:
        print("-" * 70)
        print("参考来源：")
        for i, c in enumerate(res.citations, start=1):
            print(f"  [{i}] {c}")
    if res.degraded:
        print("⚠️  引用校验未通过（可能错引/幻觉），请人工复核。")

    # 交互式记忆：记录本次问答，并在遇到相似历史问题时提示「多维度思考」
    from core.interaction import ConversationMemory
    mem = ConversationMemory()
    nudge = mem.nudge(args.question)
    mem.add(args.question, res.answer)
    if nudge:
        print("-" * 70)
        print(nudge)


def cmd_diagnose(args: argparse.Namespace) -> None:
    from core.ingest import load_index
    from core.diagnoser import Diagnoser

    index = load_index()
    diagnoser = Diagnoser(index, use_llm=not args.no_llm)
    res = diagnoser.diagnose(args.question)

    print("=" * 70)
    print(f"问题   : {res.question}")
    print(f"召回候选: {res.recall_count} 条（已放大召回，降精度换更全的原因覆盖）")
    if res.facets:
        print(f"检索角度: {'；'.join(res.facets)}")
    print("-" * 70)
    print(f"诊断清单:\n{res.answer}")
    if res.citations:
        print("-" * 70)
        print("参考来源：")
        for i, c in enumerate(res.citations[:12], start=1):
            print(f"  [{i}] {c}")


def cmd_bio(args: argparse.Namespace) -> None:
    from core.ingest import load_index
    from core.bio_agent import BioAgent

    index = load_index()
    agent = BioAgent(index, use_llm=not args.no_llm)
    res = agent.run(args.request)

    print("=" * 70)
    print(f"请求   : {res.request}")
    print(f"提取序列: {len(res.sequences)} 条")
    if res.properties:
        for p in res.properties:
            print(f"  理化性质: 长度 {p.get('length')} / 分子量 {p.get('mw_da')} Da "
                  f"/ pI {p.get('pi')} / GRAVY {p.get('gravy')}")
    if res.conservation:
        print(f"  演化保守性: 保守位点 {len(res.conservation['conserved_positions'])} 个 / "
              f"可变位点 {len(res.conservation['variable_positions'])} 个")
    print("-" * 70)
    print(f"药性推导:\n{res.answer}")
    if res.citations:
        print("-" * 70)
        print("参考来源：")
        for i, c in enumerate(res.citations[:8], start=1):
            print(f"  [{i}] {c}")


def cmd_build_data(args: argparse.Namespace) -> None:
    """构建三类数据集（检索评测集 / LoRA 三元组 / 偏好对）并版本化落盘。"""
    from core.ingest import load_index
    from core.evaluate import GOLDEN
    from core.datasets import DatasetHub, build_retrieval_set, build_triples

    index = load_index()
    hub = DatasetHub()

    retr = build_retrieval_set(index, GOLDEN)
    triples = build_triples(index, GOLDEN)
    hub.save("retrieval_eval", retr)
    hub.save("triples_lora", [t.__dict__ for t in triples])

    print("=" * 70)
    print(f"检索评测集 : {retr['n']} 条（含 BM25 难负例）")
    print(f"LoRA 三元组: {len(triples)} 条")
    print(f"落盘目录   : {hub.root}")
    print("（偏好对由对话反馈动态生成，见 `python main.py feedback`）")


def cmd_feedback(args: argparse.Namespace) -> None:
    """演示长期记忆反馈闭环：评论家打分 → badcase 落库 → 反哺数据集。"""
    from core.ingest import load_index
    from core.memory import Critic

    index = load_index()
    critic = Critic(index)

    print("=" * 70)
    print("① 评论家打分：一个「漏召回」坏例")
    q = "9MW2821 抗体的结合亲和力是多少？"
    bad_answer = "资料中未提及"
    hits = index.search(q, top_k=5)
    verdict = critic.judge(q, bad_answer, citations=[], hits=hits)
    print(f"   得分 {verdict['score']}，issues={verdict['issues']}")

    print("② badcase 落库长期记忆 + 自我迭代")
    r = critic.self_iterate(q, bad_answer, citations=[], hits=hits)
    print(f"   记忆累计 badcase {r['new_badcases']} 条；偏好对 {r['preference_pairs']} 条")
    print(f"   回归集 {r['regression']}")

    print("③ 对话反馈 → 偏好对（喂 PPO/DPO）")
    critic.collect_user_feedback(
        "9MW2821 靶向什么靶点？", "靶向 HER2", "靶向 Nectin-4（更准确）")
    fb = critic.feedback_to_datasets()
    critic.memory.save()
    print(f"   偏好对 {len(fb['preference_pairs'])} 条：")
    for p in fb["preference_pairs"]:
        print(f"     prompt  : {p.prompt}")
        print(f"     rejected: {p.rejected}")
        print(f"     chosen  : {p.chosen}")


def cmd_eval(args: argparse.Namespace) -> None:
    from core.ingest import load_index
    from core.evaluate import run_full_eval

    index = load_index()
    result = run_full_eval(index, use_llm=args.llm)
    print("=" * 70)
    print("检索质量：")
    for k, v in result["retrieval"].items():
        if k != "n":
            print(f"  {k:<12} = {v:.3f}")
    print("端到端质量：")
    for k, v in result["agent"].items():
        if k == "badcases":
            print(f"  badcases    = {len(v)} 条")
        elif k != "n":
            print(f"  {k:<12} = {v:.3f}")


def cmd_api(args: argparse.Namespace) -> None:
    from core.api import run_api
    run_api(host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="迈威生物 Agentic RAG —— 面向本地知识库的 Agentic RAG 演示")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="建库")
    p_ingest.add_argument("--data", default="sample_data", help="数据目录")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="精确问答")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("--no-llm", action="store_true",
                       help="跳过 LLM 改写/规划/生成，走快速降级路径（本地模型慢时用于演示）")
    p_ask.set_defaults(func=cmd_ask)

    p_diag = sub.add_parser("diagnose", help="多因诊断（降精度扩召回，罗列多种可能原因）")
    p_diag.add_argument("question", help="问题")
    p_diag.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 聚合，走快速降级路径")
    p_diag.set_defaults(func=cmd_diagnose)

    p_bio = sub.add_parser("bio", help="生物制药领域 Agent（蛋白演化表征 + 药性推导）")
    p_bio.add_argument("request", help="自然语言请求，如：分析 9MW2821 抗体的演化表征与药性特性")
    p_bio.add_argument("--no-llm", action="store_true",
                       help="跳过 LLM 推导，走快速降级路径")
    p_bio.set_defaults(func=cmd_bio)

    p_eval = sub.add_parser("eval", help="评测")
    p_eval.add_argument("--llm", action="store_true",
                        help="端到端评测真调本地模型（默认走快速降级路径，秒级）")
    p_eval.set_defaults(func=cmd_eval)

    p_data = sub.add_parser("build-data", help="构建数据集（检索评测集/LoRA三元组）")
    p_data.set_defaults(func=cmd_build_data)

    p_fb = sub.add_parser("feedback", help="长期记忆反馈闭环（评论家+badcase+反哺）")
    p_fb.set_defaults(func=cmd_feedback)

    def _cmd_benchmark(_args):
        from scripts.benchmark import main as bm
        bm()
    p_bm = sub.add_parser("benchmark", help="检索基准（树状chunk/扩召回的trade-off量化）")
    p_bm.set_defaults(func=_cmd_benchmark)

    p_api = sub.add_parser("api", help="启动 FastAPI 服务")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.set_defaults(func=cmd_api)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
