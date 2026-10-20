# -*- coding: utf-8 -*-
"""Streamlit 前端：提问 -> 混合检索 -> 生成（可选）-> 引用来源展示。"""
import os

import streamlit as st

from scripts.qa_api import generate, retrieve_with_sources
from scripts.build_index import cite

st.set_page_config(page_title="CCC 法规问答", layout="wide")
st.title("CCC 新能源汽车认证法规智能问答")
st.caption("答案可溯源：标准号 + 条款号 + 原文；低置信度明确拒答。")

query = st.text_input("请输入问题", placeholder="例如：动力电池系统需要满足哪些强制性国标？")
col1, col2 = st.columns([1, 3])
with col1:
    k = st.slider("检索条款数", 1, 10, 5)
with col2:
    use_rerank = st.checkbox("启用 BGE-reranker 重排", value=False)

if st.button("提问") and query:
    with st.spinner("检索中…"):
        chunks = retrieve_with_sources(query, k=k, use_rerank=use_rerank)

    st.subheader("答案")
    if os.environ.get("DEEPSEEK_API_KEY"):
        with st.spinner("生成中…"):
            answer = generate(query, chunks)
        st.markdown(answer or "（生成失败）")
    else:
        st.info("未配置 `DEEPSEEK_API_KEY`，仅展示检索到的条款；配置后即可启用生成式回答。")

    st.subheader("引用来源")
    if not chunks:
        st.warning("未检索到相关条款。")
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        st.markdown(f"**[{i}] {cite(c)}**")
        st.caption(text[:300] + ("…" if len(text) > 300 else ""))
