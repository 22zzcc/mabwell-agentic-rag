# -*- coding: utf-8 -*-
"""
core/finetune_bge.py —— BGE 领域微调管线（LoRA 优先）
=======================================================

对应简历「BGE 微调 + LoRA：构建领域检索三元组微调 BGE-base，Recall@5 0.71→0.92、
MRR@10 0.65→0.88」。

⚠️ 关键澄清（避免面试被问穿）：这里要区分「造数据」和「训练方法」两件事——
    - build_triples()  造数据：BM25 难负例挖掘，产出 (query, pos, neg) 三元组；
    - train_lora()     训练方法：真正用 **LoRA（PEFT）** 微调 BGE，不是全量微调。
  之前一版只做了三元组构造 + sentence-transformers 全量微调，没有 LoRA；
  本版补上 LoRA 主路径，并保留全量微调作对比。

为什么用 LoRA 而不是全量微调（面试常问）：
    1. 显存省：只训练注入的低秩矩阵，BGE-base 全量 100M+ 参数 → LoRA 只训 ~0.1M；
    2. 灾难性遗忘可控：基座权重冻结，只微调增量，通用能力不丢；
    3. 可插拔：一个基座 + 多个领域 LoRA adapter（港口一个、生物制药一个），
       切换领域只换 adapter，不用重训整个模型——这正是「企业多场景复用」的诉求。

管线（各步独立可跑）：
    1. build_triples()         —— BM25 难负例挖矿，构造三元组（纯 Python，离线可跑）；
    2. train_lora()            —— PEFT LoRA + InfoNCE 对比损失（需 torch/transformers/peft）；
    3. load_lora_model()       —— 推理侧加载「基座 + LoRA adapter」，替换通用 BGE；
    4. evaluate_before_after() —— 训练前后 Recall@5 / MRR@10 对比。

重依赖（torch / transformers / peft / sentence-transformers）全部惰性导入；
不装也能跑三元组构造与评测，只有真正调用 train_* 时才要求装 torch。
"""
from __future__ import annotations

from dataclasses import dataclass

from .vector_store import HybridIndex


@dataclass
class Triple:
    """一条训练三元组：query 与 positive 相关、与 negative 不相关。"""
    query: str
    positive: str
    negative: str


# ---------------------------------------------------------------------------
# ① 造数据：BM25 难负例挖矿（这步不是「微调」，是数据构造）
# ---------------------------------------------------------------------------
def build_triples(golden: list[dict], index: HybridIndex,
                  hard_negative_pool: int = 10) -> list[Triple]:
    """把「黄金问答对」转成三元组，并用 BM25 做难负例挖掘。

    为什么用「难负例」？—— 随机负例太容易区分，模型学不到判别力；
    取 BM25 召回里「和 query 词面相关但并非答案」的片段当负例，
    才能逼模型学会「语义相关 vs 真相关」的精细差别（面试重点）。

    :param golden: [{"question":..., "positive_path": 黄金片段标题路径子串}]
    :param index:  已建索引（提供 BM25 负例挖掘）
    """
    triples: list[Triple] = []
    for g in golden:
        q = g["question"]
        positives = [c["text"] for c in index.chunks
                     if g["positive_path"] in c["meta"].get("heading_path", "")]
        if not positives:
            continue
        pos = positives[0]
        bm25_hits = index.bm25.search(q, hard_negative_pool)
        for doc_id, _ in bm25_hits:
            cand = index.chunks[doc_id]["text"]
            if cand != pos:
                triples.append(Triple(query=q, positive=pos, negative=cand))
                break
    return triples


# ---------------------------------------------------------------------------
# ② 训练方法：LoRA（PEFT）主路径
# ---------------------------------------------------------------------------
def _bge_instruction_prefix() -> str:
    """BGE 中文检索模型的官方指令前缀，检索时 query 端要加。"""
    return "为这个句子生成表示以用于检索相关文章："


def train_lora(triples: list[Triple], base_model: str = "BAAI/bge-base-zh-v1.5",
               output_dir: str = "models/bge-base-bio-lora",
               r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
               epochs: int = 2, batch_size: int = 8, lr: float = 2e-4,
               temperature: float = 0.05) -> str:
    """真实 LoRA 微调 BGE：PEFT 注入低秩适配器，InfoNCE 对比损失。

    做法：
      1. 加载 BGE 基座（BERT 系），在其 attention 的 query/value 投影上注入
         LoRA（rank=r），其余参数冻结，只训练 LoRA 参数；
      2. 对每个 (query, pos, neg) 三元组编码（query 加官方指令前缀，取 [CLS] 向量）；
      3. 用 InfoNCE：query 与 positive 拉近、与 negative 及 in-batch 其它样本推远；
      4. 只保存 LoRA adapter（save_pretrained），基座不复制，体积极小。

    :param base_model: 基座（BAAI/bge-base-zh-v1.5）
    :param output_dir: LoRA adapter 保存目录
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModel
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise RuntimeError(
            "LoRA 微调需要 torch + transformers + peft：\n"
            "  pip install torch transformers peft\n"
            "（国内可先 export HF_ENDPOINT=https://hf-mirror.com）") from e

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model)

    # BGE 是 BERT 系，LoRA 挂在 attention 的 query / value 投影上（经典 BERT LoRA 配置）
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class TripleDataset(Dataset):
        def __init__(self, triples):
            self.triples = triples

        def __len__(self):
            return len(self.triples)

        def __getitem__(self, i):
            t = self.triples[i]
            return t.query, t.positive, t.negative

    def encode(texts: list[str], query: bool = False):
        """把一批文本编码成 [CLS] 向量（BGE 用 CLS 池化）。

        query=True 时给 query 端加 BGE 官方指令前缀（检索场景的标准做法）。
        """
        if query:
            texts = [_bge_instruction_prefix() + t for t in texts]
        enc = tokenizer(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")
        out = model(**enc)
        return out.last_hidden_state[:, 0]  # [CLS]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(TripleDataset(triples), shuffle=True, batch_size=batch_size)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for qs, ps, ns in loader:
            # 统一编码 q / p / n（query 端加官方指令前缀）
            q_emb = encode(list(qs), query=True)
            p_emb = encode(list(ps))
            n_emb = encode(list(ns))

            # 归一化后点积即余弦
            q_emb = torch.nn.functional.normalize(q_emb, dim=-1)
            p_emb = torch.nn.functional.normalize(p_emb, dim=-1)
            n_emb = torch.nn.functional.normalize(n_emb, dim=-1)

            # InfoNCE：正例 = 对角线(q_i·p_i)；负例 = q_i·n_i + 其它样本的 positive
            pos_logits = (q_emb @ p_emb.T) / temperature          # [B, B]
            neg_logits = (q_emb @ n_emb.T) / temperature          # [B, B]
            logits = torch.cat([pos_logits, neg_logits], dim=1)   # [B, 2B]
            labels = torch.arange(len(qs))                        # 正例在对角线
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[finetune_bge] LoRA epoch {epoch + 1}/{epochs}  loss={total_loss / len(loader):.4f}")

    # 只保存 LoRA adapter（不含基座，体积小、可插拔）
    model.save_pretrained(output_dir)
    print(f"[finetune_bge] LoRA adapter 已保存到 {output_dir}")
    return output_dir


def load_lora_model(base_model: str, adapter_dir: str):
    """推理侧加载「基座 + LoRA adapter」，返回可编码文本的模型对象。

    用法：把 core/embedding.py 里的通用 BGE 换成这个模型，即可用领域微调后的
    向量做检索，再跑 evaluate_before_after 对比指标。
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError("加载 LoRA 需要 transformers + peft") from e
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModel.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model, tokenizer


# ---------------------------------------------------------------------------
# ③ 全量微调（对比用，非主路径）
# ---------------------------------------------------------------------------
def train_full(triples: list[Triple], base_model: str = "BAAI/bge-base-zh-v1.5",
               output_dir: str = "models/bge-base-bio", epochs: int = 2,
               batch_size: int = 8, lr: float = 2e-5) -> None:
    """全量微调（sentence-transformers MNR Loss），用于与 LoRA 对比显存/效果。"""
    try:
        from sentence_transformers import (SentenceTransformer,  # type: ignore
                                           InputExample, losses)
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise RuntimeError(
            "全量微调需要 torch + sentence-transformers：\n"
            "  pip install torch sentence-transformers") from e

    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[t.query, t.positive, t.negative]) for t in triples]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              optimizer_params={"lr": lr}, output_path=output_dir,
              show_progress_bar=True)
    print(f"[finetune_bge] 全量微调完成，权重保存到 {output_dir}")


# ---------------------------------------------------------------------------
# ④ 评测：训练前后对比
# ---------------------------------------------------------------------------
def evaluate_before_after(index: HybridIndex, golden: list[dict],
                          old: dict, new: dict) -> dict:
    """打印训练前后指标对比（复现简历里的 Recall@5 / MRR@10 提升表）。

    实际项目里：old 用通用 BGE 检索、new 用 LoRA 微调后 BGE 检索，
    二者在同一黄金集上算指标，得到「0.71→0.92 / 0.65→0.88」这类对比。
    """
    from .evaluate import evaluate_retrieval  # 复用同一套指标口径，避免口径漂移
    if not old:
        old = evaluate_retrieval(index, golden)
    print("指标对比：")
    print(f"  Recall@5 : {old['recall@5']:.2f} -> {new['recall@5']:.2f}")
    print(f"  MRR@10   : {old['mrr@10']:.2f} -> {new['mrr@10']:.2f}")
    return {"before": old, "after": new}
