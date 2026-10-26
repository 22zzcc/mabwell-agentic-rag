# 自我介绍 ↔ 代码 映射表

> 面试时你讲的每个点，都要能落到 GitHub 仓库的代码位置。这份表把自我介绍逐条对到
> `mabwell-agentic-rag` 的代码，并标注「准确说法」（修正技术硬伤，避免被问穿）。

| # | 自我介绍原话 | 对应代码 | 准确说法（面试用） |
| :--- | :--- | :--- | :--- |
| 1 | 多因诊断：一个故障/生物现象常有多个原因 | `core/diagnoser.py` | 「诊断类问题答案是集合，不是唯一」 |
| 2 | 牺牲部分检索精度，扩 Top-k | `core/diagnoser.py:_rewrite`（问题重写）+ `_retrieve`（精准检索）| ⚠️ 别再说「降低 Top-p」——Top-p 是采样参数不是检索参数。准确说法：**多因诊断放在问题重写阶段 + 树状 chunk 精准命中** |
| 3 | 树状 chunk | `core/document_tree.py`（文档→章节→条目）+ `core/question_tree.py`（问题精度 2 层 tree）| **树状 chunk 基于多因/问题精度构建**——每个节点=一个可能原因/条目，多因诊断时树结构天然做多面召回，几乎不需要额外混合搜索。优势在大数据量下体现（小 MVP 体现不出） |
| 4 | 检索耗时 | `scripts/benchmark.py`（`python main.py benchmark`） | 实测数字：精确问答 ~28ms/查询，扩召回额外开销 +1.1ms（几乎可忽略） |
| 5 | 基于上下文的长期记忆反馈模块 | `core/memory.py`（Critic）+ `core/badcase.py` | 「评论家 = PPO 里的 critic + 多一个 badcase 收取」 |
| 6 | 利用上下文收集 badcase，基于 BM25 生成问答对 | `core/datasets.py:build_triples`（BM25 难负例挖矿） | 「一份数据三处用：评测干扰项 / LoRA 难负例」 |
| 7 | 对生成 LLM 微调（PPO/RLHF） | `core/datasets.py:build_preference_pairs` + `core/finetune_bge.py:train_lora` | 诚实说法：LoRA 微调是 BGE（检索侧）；PPO/RLHF 在仓库里是**偏好对数据构造**，训练循环可接 trl |
| 8 | 「这个问题你上次也提过相似的」鼓励多维度思考 | `core/interaction.py`（ConversationMemory） | 交互式记忆：记录问答历史 + 相似问题提示 |
| 9 | 科研没有绝对错误，异常可能是新发现/不同表征 | `core/research_badcase.py`（三分类） | 科研 bad case 分三类：**确凿错误 / 可能新发现 / 不同表征**，只有「确凿错误」进负例 |
| 10 | 案例用于 PPO、RLHF、交互启发、评测数据集 | `core/datasets.py` + `core/evaluate.py` + `core/interaction.py` | 一条坏例四个去向 |

## 三个必须背牢的准确表述（修正版）

1. **多因诊断的 trade-off**：
   > 「诊断类问题答案是一个集合，我故意降精度、扩 Top-k 召回，把多种可能原因都捞进来，
   > 让提问者自己对照排查——实测扩召回的额外耗时只有 0.5ms，但原因罗列更全，用户体验上升。」

2. **树状 chunk 的价值**（与多因诊断一体）：
   > 「树状 chunk 我是基于多因思路构建的——每个节点对应一个可能原因/条目，所以多因诊断时
   > 树结构天然就做了多面召回，几乎不需要额外的混合搜索。这个优势在知识库规模大了之后
   > 才会明显，小 MVP 体现不出来。」

3. **科研 bad case 与工业 Agent 的差异**：
   > 「工业 bad case 只有对错；科研没有绝对错误——异常可能是新发现或同一结果的不同表征。
   > 所以我把 bad case 分三类，只有『确凿错误』进负例训练，『可能新发现』留待人工确认。」

## 现场演示脚本

```bash
cd mabwell-agentic-rag
python main.py diagnose "抗体 SEC-HPLC 纯度偏低有哪些原因？" --no-llm   # 多因诊断
python main.py benchmark                                              # 给 trade-off 真数字
python main.py ask "9MW2821 靶向什么靶点？" --no-llm                   # 交互式记忆（问两遍触发提示）
python main.py feedback                                               # 长期记忆反馈闭环
python scripts/compare_benchmark.py                                   # 多因诊断 + 长期记忆 的数字
python scripts/compare_tree.py                                        # 问题类 tree chunk vs 平铺 的块数对比
```

## 面试补强（7 条硬伤 → 答案）

### 数字卡（召回速度，别再"上升十几"）

| 指标 | 精准数字 |
| :--- | :--- |
| 精确问答耗时 | ~28ms/查询 |
| 多因诊断扩召回额外开销 | +1.1ms（几乎可忽略） |
| 多因诊断原因覆盖 | top_k 5→15，3/5 → 4/5 |
| 问题类 tree chunk 块数 | 10 块 vs 平铺 20 块（-50%） |
| 长期记忆产出 | 3 坏例 → 4 badcase → 1 偏好对 + 8 三元组 |

### 幻觉防控四层（制药强化版）

① 源头：引用接地校验 + Document-Tree 溯源；② 生成：prompt 硬约束「只用资料作答，不足说『资料未提及』」+ 低温；③ 后验：引用正确率进评测 + **实体归因**（答案实体必须能在引用 chunk 找到，防张冠李戴）；④ 反馈：bad case → LoRA/PPO + 科研三分类。
一句话：**「制药是强监管，幻觉=医疗风险，宁可说『资料未提及』也不编造。」**

### 港口 → 生物制药 四个共同点

强结构化私有数据（工单↔申报资料）、术语密集多因（故障码↔PFS/SAE/DAR）、强监管要溯源（安全↔GMP）、本地部署不出域。→ 那套 Agentic RAG 是**平移复用**，只换领域词典 + 知识图谱。

### 跨岗对接三件套

document tree（还原层级、可溯源）→ 多因诊断（罗列原因、让业务自己对照）→ bad case 收录（收反馈、越用越准）。三步 = 可验证可迭代的闭环，不是一次性交付。

### 术语对齐 / 三元组（别卡壳）

- **术语对齐**：把口语/缩写/别称映射到标准术语（同义词表 + embedding + LLM 改写），如 9MW2821↔Nectin-4 ADC、AE↔不良事件。
- **三元组**：(query, positive, negative)，negative = BM25 召回的「词面相关但非答案」难负例，喂 BGE 对比学习。
