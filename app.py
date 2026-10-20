# -*- coding: utf-8 -*-
"""Streamlit 前端：提问 -> 混合检索 -> 生成（可选）-> 引用来源展示。"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from qa_api import generate, get_retriever  # noqa: E402
from build_index import cite  # noqa: E402

st.set_page_config(page_title="CCC 法规问答", layout="wide")


@st.cache_resource(show_spinner=False)
def load_retriever():
    return get_retriever()


st.sidebar.title("配置")
api_key = st.sidebar.text_input(
    "DeepSeek API Key", type="password",
    value=os.environ.get("DEEPSEEK_API_KEY", ""),
)
base_url = st.sidebar.text_input(
    "API Base URL", value=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
model = st.sidebar.text_input("模型", value=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
use_rerank = st.sidebar.checkbox("启用 BGE-reranker 重排", value=True)

st.title("CCC 新能源汽车认证法规智能问答")
st.caption("答案可溯源：标准号 + 条款号 + 原文；低置信度明确拒答。")

query = st.text_input("请输入问题", placeholder="例如：动力电池系统需要满足哪些强制性国标？")
k = st.slider("检索条款数", 1, 10, 5)

if st.button("提问") and query:
    ret = load_retriever()
    with st.spinner("检索中…"):
        chunks = ret.search(query, k=k, use_rerank=use_rerank)

    st.subheader("答案")
    if api_key:
        with st.spinner("生成中…"):
            answer = generate(query, chunks, api_key=api_key, base_url=base_url, model=model)
        st.markdown(answer or "（生成失败）")
    else:
        st.info("请在左侧填入 DeepSeek API Key 以启用生成式回答；当前仅展示检索到的条款。")

    st.subheader("引用来源")
    if not chunks:
        st.warning("未检索到相关条款。")
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        st.markdown(f"**[{i}] {cite(c)}**")
        st.caption(text[:300] + ("…" if len(text) > 300 else ""))
