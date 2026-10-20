# -*- coding: utf-8 -*-
"""DeepSeek 生成式问答：检索 -> 构造引用上下文 -> 生成（OpenAI 兼容接口）+ 低置信度拒答。

环境变量：
  DEEPSEEK_API_KEY   必填（未配置时仅返回检索条款，不调用生成）
  DEEPSEEK_BASE_URL  默认 https://api.deepseek.com
  DEEPSEEK_MODEL     默认 deepseek-chat
"""
import os

import requests

from build_index import HybridRetriever, cite

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SYSTEM_PROMPT = (
    "你是 CCC 新能源汽车认证法规问答助手。"
    "只能依据给定的法规条款原文回答，必须逐条引用【标准号 + 条款号 + 原文】。"
    "如果给定条款不足以回答问题，直接回答“根据现有法规库，无法给出可靠答案”，绝不编造。"
)

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_with_sources(query, k=5, use_rerank=False):
    return get_retriever().search(query, k=k, use_rerank=use_rerank)


def build_context(chunks):
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        lines.append(f"[{i}] {cite(c)}\n{text}")
    return "\n\n".join(lines)


def generate(query, chunks, temperature=0.1):
    if not API_KEY:
        return None
    context = build_context(chunks)
    user_prompt = f"问题：{query}\n\n可参考法规条款：\n{context}\n\n请基于以上条款回答，并逐条给出引用。"
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def answer(query, k=5, use_rerank=False):
    chunks = retrieve_with_sources(query, k=k, use_rerank=use_rerank)
    if not chunks:
        return "根据现有法规库，无法给出可靠答案。"
    text = generate(query, chunks)
    if text is None:
        return build_context(chunks)
    return text


def main():
    import sys

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "动力电池系统需要满足哪些强制性国标"
    chunks = retrieve_with_sources(query)
    print("=" * 70)
    print("查询：", query)
    print("-" * 70)
    if API_KEY:
        print(generate(query, chunks) or "（生成失败）")
    else:
        print("（未配置 DEEPSEEK_API_KEY，仅展示检索条款）")
        print(build_context(chunks))


if __name__ == "__main__":
    main()
