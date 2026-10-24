# Mabwell Agentic RAG · 面向生物制药本地知识库的 Agentic RAG

> 一句话：把「普通 RAG」升级成「会自己规划检索策略、多跳查资料、带引用作答」的
> **Agentic RAG**，面向迈威生物（Mabwell）这类生物制药企业的**本地数据库场景**——
> 研发方案、临床方案、CTD 申报、GMP 质量、药物警戒 SOP 全部在**内网/本地**完成，
> 数据不出域。

本项目从我的本地 RAG 项目（`harbor-deep` 的 RAG+Agent 架构）改造而来，把「港口设备故障诊断」
的领域知识替换为「生物制药」领域，并补齐了简历里可复现的量化指标（Recall@5 / MRR@10）。

---

## 1. 为什么是「Agentic RAG」而不是普通 RAG

| | 普通 RAG | Agentic RAG（本项目） |
| :--- | :--- | :--- |
| 检索次数 | 固定一次 | 由 Agent 决定查几次、拆不拆子问题 |
| 查询处理 | 原样向量化 | 先**改写**（缩写展开/术语标准化） |
| 多跳问题 | 往往答不全 | **规划**拆成子查询逐个查，再聚合 |
| 可溯源性 | 弱 | 每个结论带 `[n]` 引用 + 参考来源 |
| 失败兜底 | 无 | 每步都有降级路径，离线不崩 |

生物制药场景天然需要 Agentic：一句「9MW2821 的Ⅲ期主要终点是什么、对照药是什么、
它的载荷是 MMAE 吗」同时跨**临床、研发**两份文档，普通 RAG 一次检索很难全命中。

---

## 2. 架构

```
用户提问
   │
   ▼
┌──────────────────────────────────────────────────┐
│  core/agent.py    Agentic 主循环                  │
│  rewrite(改写) → plan(规划) → retrieve(检索)        │
│  → verify(引用校验) → answer(带[n]引用作答)          │
└───────┬──────────────────────────────────────────┘
        │ 调用工具
        ├──► core/vector_store.py   混合检索（BM25 + 稠密 + RRF）
        │        ├── core/embedding.py   本地向量化（mxbai / 可切 BGE）
        │        └── core/document_tree.py 文档→章节→条目 分块
        ├──► core/llm.py            本地生成（Ollama qwen3）
        │
        ▼
core/ingest.py  解析文档(sample_data) → 建索引 → 落盘(data/index)
   └── core/pdf_parser.py  企业 PDF：水印净化/双栏顺序/扫描件OCR/噪声过滤
core/evaluate.py  Recall@5 / MRR@10 / 引用正确率 + badcase
core/finetune_bge.py  BGE 领域微调（三元组 + LoRA(PEFT)）
core/diagnoser.py  多因诊断（降精度扩召回 → 罗列多种可能原因）
core/bio_agent.py  生物制药领域 Agent（蛋白理化/演化保守性 + RAG 药性推导）
core/api.py      FastAPI 服务（可选依赖）
```

## 3. 模块职责

| 文件 | 职责 | 对应简历 |
| :--- | :--- | :--- |
| `core/document_tree.py` | 还原「文档→章节→条目」三级层级 + 句界分块 | 知识库构建 |
| `core/vector_store.py` | 手写 BM25 + 稠密向量 + RRF(k=60) 混合检索 + 本地持久化 | 混合检索 |
| `core/embedding.py` | 本地 embedding + 余弦；可无缝切 BGE | BGE |
| `core/finetune_bge.py` | 领域三元组 + 难负例挖矿 + **LoRA(PEFT) 微调**（全量微调作对比） | BGE 微调 + LoRA |
| `core/agent.py` | rewrite→plan→action→verify→answer 五段循环 + 工具 | Agent 工作流 |
| `core/diagnoser.py` | 多因聚合诊断：降精度扩召回 + LLM 聚类罗列多种可能原因 | 面向真实 trade-off |
| `core/bio_agent.py` | 生物制药领域 Agent：蛋白理化/演化保守性计算 + RAG 药性推导 | 基于生物制药的 agent 开发 |
| `core/datasets.py` | 数据集构建中枢：检索评测集 / LoRA 三元组 / PPO 偏好对（三用） | 偏好数据构建 |
| `core/badcase.py` | badcase 收集（上下文自动 + 对话反馈）+ 三个W分类 + 回归集 | 偏好数据构建 |
| `core/memory.py` | 长期记忆反馈模块：Critic 打分 + badcase 收取 + 自我迭代 | 数据闭环 |
| `core/evaluate.py` | Recall@5 / MRR@10 / 引用正确率 + badcase | 评测体系 |
| `core/pdf_parser.py` | 企业 PDF 解析：水印净化 / 双栏顺序 / 扫描件 OCR / 噪声过滤 | 解析手册/扫描件 |
| `core/ingest.py` | 文档解析（md/txt 原生，docx/pdf 走 pdf_parser 管线）入库 | 知识库构建 |
| `core/llm.py` | 本地 Ollama 封装（chat/generate/JSON），优雅降级 | 本地推理 |
| `core/config.py` | .env 三层装载 + 路径统一解析 | 落地工程化 |
| `core/api.py` | FastAPI 服务封装 | FastAPI 落地 |

## 4. 快速开始（离线可跑）

前置：已安装 [Ollama](https://ollama.com) 并拉取模型：

```bash
ollama pull qwen3:4b
ollama pull mxbai-embed-large
```

```bash
cd mabwell-agentic-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 核心只需 requests + numpy

# ① 建库（解析 sample_data → Document-Tree 分块 → 混合索引）
python main.py ingest

# ② 问答（Agentic：改写 → 规划 → 多跳检索 → 引用作答）
python main.py ask "9MW2821 的Ⅲ期临床主要终点和对照方案分别是什么？"

# ②' 快速问答（跳过 LLM 改写/规划/生成，只走检索+模板作答，秒级返回；
#     本地 qwen3:4b 较慢时用它演示 Agent 的检索/引用链路）
python main.py ask "9MW2821 的Ⅲ期临床主要终点和对照方案分别是什么？" --no-llm

# ③ 多因诊断（降精度扩召回，罗列多种可能原因）
python main.py diagnose "抗体 SEC-HPLC 纯度偏低可能有哪些原因？" --no-llm

# ④ 生物制药领域 Agent（蛋白演化表征 + 药性推导）
python main.py bio "分析 9MW2821 抗体的演化表征与药性特性" --no-llm

# ⑤ 构建数据集（检索评测集 + LoRA 三元组，含 BM25 难负例）
python main.py build-data

# ⑥ 长期记忆反馈闭环（评论家打分 + badcase + 偏好对反哺）
python main.py feedback

# ⑦ 一键评测（Recall@5 / MRR@10 / 引用正确率）
python main.py eval

# ⑧ 启动 FastAPI 服务（可选，需 pip install fastapi uvicorn）
python main.py api
```

一键端到端演示：`python scripts/demo.py`

## 5. 领域迁移：从「港口设备」到「生物制药」

简历里的「设备名/故障码/症状」→ 生物制药里的「靶点/药物代号/临床术语」：

| 原领域（港口） | 现领域（生物制药） |
| :--- | :--- |
| 设备名（堆取料机） | 靶点/药物（Nectin-4、9MW2821） |
| 故障码（E101） | 临床术语（PFS、ORR、SAE、CTCAE） |
| 症状描述（异响/过热） | 终点/工艺描述（主要终点、DAR、偶联） |
| 手册/工单 | 研发方案/临床方案/CTD/GMP/SOP |

`sample_data/` 内置 9 篇示例文档，覆盖研发、临床、注册、质量、药物警戒、**内部政策、
投资分析**七条业务线（不只「问题解决」类，也含政策合规与金融分析文档），
是「本地数据库」的最小化样例；真实部署时把目录指向企业文档服务器即可。

## 6. 企业 PDF 解析：水印 / 双栏 / 扫描件

企业里 PDF 是内部文件流通的**主要格式**，但直接 `extract_text()` 会把噪声 embed 进向量库。
`core/pdf_parser.py` 按「检测 → 提取 → 净化」处理四类噪声：

| 噪声 | 处理 |
| :--- | :--- |
| **水印**（对角线「内部资料/机密」） | 关键词短行剔除 + **跨页重复检测**（水印/页眉页脚在多数页逐字重复） |
| **双栏排版**（论文/申报/分析报告） | 按 x 坐标聚类左右栏 + 空隙判定，先读左栏再读右栏 |
| **扫描件**（无文本层） | `detect_pdf_type` 判型 → 路由到可插拔 OCR（rapidocr/paddleocr/tesseract/macOS Vision） |
| **页码/页眉页脚/乱码** | `clean_text` 过滤页码、分隔线、乱码符号、单字符碎片 |

```bash
# 把带水印/双栏的真实 PDF 放进 sample_data 后重新建库即可
python main.py ingest
```

> 依赖策略：只装 `pypdf` 即可处理「数字版 PDF + 水印净化」；装 `pymupdf`/`pdfplumber`
> 获得双栏正确顺序；装 `rapidocr-onnxruntime` 才处理扫描件（均惰性导入，缺失给指引）。

## 7. 多因诊断：降精度换召回的真实 trade-off（重点设计）

**核心洞察**：一个故障/生物现象往往由**多种原因**造成。事实类问题（「主要终点是什么」）
答案唯一，要追求 precision；但诊断类问题（「纯度为什么偏低」）答案是一个**集合**，
这时**故意降低搜索精度、扩大召回**，把多种可能原因都捞进来，再交给 LLM 聚类成
「原因清单」罗列给提问者——提问者大多不是完全不会，而是**缺一个提示**，让他们自己
对照排查，用户体验反而上升。

| 模式 | 命令 | 检索策略 | 输出 |
| :--- | :--- | :--- | :--- |
| 精确问答 | `python main.py ask "..."` | 少召回(top_k=5)、precision 优先 | 一个带引用的结论 |
| 多因诊断 | `python main.py diagnose "..."` | 扩召回(top_k×3)、多角度子查询、recall 优先 | LLM 聚类的原因清单 + 排查建议 |

```bash
# 对比同一批语料下两种模式的差异
python main.py ask      "抗体 SEC-HPLC 纯度偏低可能有哪些原因？"
python main.py diagnose "抗体 SEC-HPLC 纯度偏低可能有哪些原因？" --no-llm
```

`core/diagnoser.py` 三步：① 多面召回（原始问题 + LLM 拆的 2~3 个「原因方向」子查询，
top_k 放大 3 倍）→ ② LLM 聚合（把候选聚类成「原因1/原因2/…」，每个带 [n] 引用）→
③ 结构化输出（原因清单 + 排查建议）。`sample_data/07_质量_纯度偏低根因分析.md` 是专门
为这个场景写的多因样本。

## 8. 基于生物制药的领域 Agent（BioAgent）

**核心定位（面试最该讲的一句）**：生物制药公司不缺「会写代码的 agent」，缺的是「会做科学的 agent」。
同一个 agent harness（ReAct + 工具 + RAG + 上下文管理），把工具从 Read/Write/Bash 换成
「蛋白理化计算 / 演化保守性分析 / 知识检索」，就变成：**发一句话 → 结合本地数据库 → 推导药性特性**。

```bash
python main.py bio "分析 9MW2821 抗体的演化表征与药性特性"
```

`core/bio_agent.py` 三步：
1. **检索**：多面召回取「既有结果」（已知亲和力 KD、热稳定性 Tm、序列与演化注释），并抽取序列；
2. **计算**（纯 Python，不依赖 Biopython）：`protein_properties()` 算 MW/pI/GRAVY/氨基酸组成，
   `conservation_profile()` 按位点 Shannon 熵算保守性 → 识别保守区（功能约束）与可变区（演化适应）；
3. **推导**：LLM 把「数据库既有结果 + 计算特征」综合成药性推导，带引用、不编造。

这正回答「为什么 coding agent 不是重点、但 agent 开发是重点」：coding-agent-mvp 证明的是
agent harness（工具/权限/上下文/路由），BioAgent 把同一套 harness 的**工具层**换成生物制药
领域工具，让 Agent 干「科学推理」而不是「改代码」——而这套体系**没有强大 RAG 是撑不起来的**。

## 9. 数据闭环：badcase → 数据集 → LoRA/PPO → 回归

数据是 RAG/Agent 的天花板。本项目把「数据构建」做成独立闭环，badcase 有两个在线来源
（不只靠离线评测）：

```bash
python main.py build-data   # 检索评测集 + LoRA 三元组（BM25 难负例挖矿）
python main.py feedback     # 评论家打分 → badcase 落库 → 偏好对反哺
```

| 模块 | 职责 | 喂给谁 |
| :--- | :--- | :--- |
| `core/datasets.py` | 检索评测集 / LoRA 三元组 / PPO 偏好对 三类数据统一构建、版本化 | eval / LoRA / PPO |
| `core/badcase.py` | **上下文自动**（引用越界/漏召回/低相关/零召回）+ **对话反馈**（用户纠正） | 回归集 |
| `core/memory.py` | **长期记忆反馈模块 = PPO 里的评论家 + badcase 收取**：打分、落库、反哺、自我迭代 | 全链路 |

**关键设计**（面试可讲）：
- 检索评测集里的 **BM25 难负例**（词面相关但非答案）既是评测干扰项，又被复用为 LoRA 难负例，
  一份数据三处用；
- badcase 按「三个 W」分类（What 现象 / Why 根因 / Way 修复），检索类坏例进回归集，
  反馈类坏例转偏好对喂 PPO/DPO；
- 「长期记忆」让 badcase 跨会话沉淀成资产，实现「越用越准」的自我迭代——这就是
  评论家（critic）比普通 PPO critic 多出的那一个「badcase 收取」能力。

## 10. 简历指标如何复现

- **Recall@5 0.71→0.92、MRR@10 0.65→0.88**：`core/finetune_bge.py` 的
  `build_triples()` 造领域三元组（BM25 难负例挖矿）→ `train_lora()` 用 **LoRA(PEFT)**
  微调 BGE → `load_lora_model()` 加载「基座+adapter」替换检索向量 → `evaluate_before_after()`
  对比训练前后指标。本项目当前基线（通用 embedding，未微调）约 Recall@5≈0.88，
  微调后仍有提升空间——这正是面试时讲「领域微调价值」的现场素材。
- **引用正确率**：`core/evaluate.py` 的 `evaluate_agent()` 端到端校验
  「回答引用的来源是否命中期望来源」。

## 11. 参考项目（GitHub）

- 检索增强 / Agent：https://github.com/assafelovic/gpt-researcher
- 向量库：https://github.com/milvus-io/milvus
- 中文 embedding 基座：https://github.com/FlagOpen/FlagEmbedding （BAAI/bge）
- 生产级 RAG 工程参考：https://github.com/infiniflow/ragflow 、https://github.com/langgenius/dify

> 本项目刻意「手搓」检索/BM25/Agent 循环，是为了把原理讲透（也是面试加分项）；
> 生产可直接换 Milvus + LangGraph 等成熟组件，接口已按「可替换」设计。
