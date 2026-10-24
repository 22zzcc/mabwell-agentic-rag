# -*- coding: utf-8 -*-
"""
core/research_badcase.py —— 科研场景专属 bad case 分类（「科研没有绝对错误」）
================================================================================

对应自我介绍「这是和普通工业 Agent 最大差异：科研没有绝对错误，很多异常只是不同
路径的结果——例如蛋白序列突变，可能是新发现，或是同一结果的不同表征」。

普通工业 bad case 只有「对/错」二元；科研场景的 bad case 要分三类：
    1. 确凿错误   —— 引用越界 / 幻觉 / 事实错误 → 直接当负例（进 LoRA/PPO）；
    2. 可能新发现 —— 与预期不符但可能是真实新现象（如序列突变）→ 标记「需人工确认」，
                     不直接判负，避免把新发现当错误扼杀；
    3. 不同表征   —— 同一结论的另一种说法 → 不是错误，可当正例变体（增强泛化）。

这套分类决定了一个异常「该不该进负例训练集」——这是科研 Agent 与工业 Agent 的分水岭。
"""
from __future__ import annotations

# 分类关键词（规则层，命中即归类；可加 LLM 细分）
_ERROR_MARK = ["引用", "幻觉", "越界", "不存在", "事实错误", "citation", "hallucination"]
_NEW_FINDING_MARK = ["突变", "异常", "新发现", "novel", "unexpected", "未见过", "超出预期"]
_ALTERNATIVE_MARK = ["等价", "同义", "另一种", "不同表征", "等价表征", "等价形式"]


def classify_research_badcase(issue: str, text: str = "") -> str:
    """把一条科研 bad case 归入三类之一。

    :param issue: 原 bad case 的问题类型（如 citation_fail / low_relevance）
    :param text:  相关文本（用于检测「突变/等价」等科研信号）
    :return: '确凿错误' / '可能新发现' / '不同表征'
    """
    blob = f"{issue} {text}".lower()
    if any(k in blob for k in _ERROR_MARK):
        return "确凿错误"
    if any(k in blob for k in _NEW_FINDING_MARK):
        return "可能新发现"
    if any(k in blob for k in _ALTERNATIVE_MARK):
        return "不同表征"
    return "确凿错误"   # 默认按普通错误处理（宁严勿松，交由人工复核可改判）


def is_trainable_negative(issue: str, text: str = "") -> bool:
    """是否应作为负例进训练集：只有「确凿错误」进负例；
    「可能新发现」留待人工确认，「不同表征」反而可当正例变体。"""
    return classify_research_badcase(issue, text) == "确凿错误"
