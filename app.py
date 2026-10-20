# -*- coding: utf-8 -*-
"""Streamlit 前端：聊天式法规问答助手。"""
import html
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from qa_api import generate, get_retriever  # noqa: E402
from build_index import cite  # noqa: E402

st.set_page_config(
    page_title="CCC 法规智能问答",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLES = [
    "动力电池系统绝缘电阻有什么要求？",
    "高压电气安全有哪些要求？",
    "首次申请整车CCC需要提交哪些资料？",
    "电动汽车交流充电桩认证规则是什么？",
]

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

.hero {
  background: linear-gradient(135deg, #0b1f2a 0%, #15364a 55%, #1f5a75 100%);
  padding: 1.5rem 1.8rem;
  border-radius: 16px;
  color: #fff;
  margin-bottom: 1rem;
  box-shadow: 0 8px 24px rgba(15, 40, 60, .15);
}
.hero .eyebrow { font-size: .72rem; letter-spacing: 3px; opacity: .75; margin-bottom: .35rem; }
.hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
.hero p { margin: .5rem 0 0; opacity: .9; font-size: .92rem; }
.tag {
  display: inline-block;
  padding: .12rem .6rem;
  border-radius: 999px;
  font-size: .72rem;
  margin: .8rem .4rem 0 0;
  background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.22);
}

.mode-badge {
  display: inline-block;
  padding: .3rem .75rem;
  border-radius: 999px;
  font-size: .8rem;
  font-weight: 600;
  margin: .2rem 0 .6rem;
}
.mode-on { background: #dcfce7; color: #15803d; }
.mode-demo { background: #fef3c7; color: #b45309; }

.feature-card {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: .8rem 1rem;
  margin: .4rem 0;
  background: #f8fafc;
}
.feature-card .icon { font-size: 1.2rem; }
.feature-card .title { font-weight: 650; color: #0f172a; margin-top: .2rem; }
.feature-card .desc { color: #64748b; font-size: .82rem; margin-top: .15rem; }

.source-card {
  border: 1px solid #e2e8f0;
  border-left: 4px solid #1f5a75;
  border-radius: 10px;
  padding: .6rem .8rem;
  margin: .55rem 0;
  background: #f8fafc;
}
.source-head { font-weight: 650; color: #0f172a; }
.source-meta { color: #64748b; font-size: .78rem; margin-top: .1rem; }
.source-text { color: #334155; font-size: .85rem; margin-top: .35rem; line-height: 1.5; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_retriever():
    return get_retriever()


def _secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def ask(question):
    st.session_state["pending_question"] = question


def render_sources(chunks):
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if len(text) > 600:
            text = text[:600] + "…"
        head = f"[{i}] {cite(c)}"
        meta = ""
        if c.get("status"):
            meta += f"🏷️ {c['status']}"
        if c.get("impl_date"):
            meta += f" · 🗓️ 实施 {c['impl_date']}"
        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-head">{html.escape(head)}</div>'
            f'<div class="source-meta">{html.escape(meta)}</div>'
            f'<div class="source-text">{html.escape(text)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def handle_query(query):
    query = (query or "").strip()
    if not query:
        return
    st.session_state.messages.append({"role": "user", "content": query})
    with st.status("正在检索法规库…", expanded=False) as status:
        ret = load_retriever()
        chunks = ret.search(query, k=k, use_rerank=use_rerank)
        if not chunks:
            answer = "根据现有法规库，未检索到相关条款，无法给出可靠答案。"
            error = None
        elif api_key:
            status.update(label="正在生成回答…")
            try:
                answer = generate(query, chunks, api_key=api_key, base_url=base_url, model=model)
                error = None
                if not answer:
                    answer = "（生成失败，请稍后重试）"
            except Exception as exc:
                answer = None
                error = f"生成失败：{exc}"
        else:
            answer = None
            error = "当前未配置 DeepSeek API Key，进入“检索演示模式”：仅展示命中的法规条款与原文引用。"
        status.update(label="完成", state="complete", expanded=False)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": chunks, "error": error}
    )
    st.rerun()


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.markdown("## ⚖️ CCC 法规助手")
    st.caption("面向认证法规工程师的智能检索问答")

    with st.expander("⚙️ 模型与检索设置", expanded=False):
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=_secret("DEEPSEEK_API_KEY", os.environ.get("DEEPSEEK_API_KEY", "")),
            help="留空则进入检索演示模式；部署到云端时由服务端密钥自动注入，访客无需填写。",
        )
        base_url = st.text_input(
            "API Base URL", value=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        model = st.text_input("模型", value=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
        use_rerank = st.checkbox("启用 BGE-reranker 重排", value=True)
        k = st.slider("检索条款数", 1, 10, 5)

    if api_key:
        st.markdown('<div class="mode-badge mode-on">● 生成式问答</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-badge mode-demo">● 检索演示模式</div>', unsafe_allow_html=True)

    st.markdown("### 💡 试试这些问题")
    for q in SAMPLES:
        st.button(q, use_container_width=True, on_click=ask, args=(q,))

    st.markdown("---")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("内容仅供内部检索参考，不替代正式认证意见。")


# ---------------- 主区 ----------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">COMPLIANCE RAG · 新能源汽车认证</div>
      <h1>CCC 法规智能问答助手</h1>
      <p>答案可溯源：标准号 + 条款号 + 原文 · 低置信度明确拒答，不编造</p>
      <div>
        <span class="tag">DeepSeek 生成</span><span class="tag">BM25 + 稠密检索</span>
        <span class="tag">BGE 重排</span><span class="tag">可溯源引用</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="feature-card"><div class="icon">🔎</div><div class="title">混合检索</div>'
            '<div class="desc">BM25 + 稠密向量 RRF 融合，命中率 hit@5 达 90%。</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="feature-card"><div class="icon">📎</div><div class="title">答案可溯源</div>'
            '<div class="desc">每条回答附标准号、条款号、页码与原文。 </div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="feature-card"><div class="icon">🛡️</div><div class="title">拒答不编造</div>'
            '<div class="desc">依据不足时明确拒答，避免幻觉回答。</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### 💡 快速开始")
        qcols = st.columns(2)
        for i, q in enumerate(SAMPLES):
            with qcols[i % 2]:
                st.button(q, use_container_width=True, on_click=ask, args=(q,))

for m in st.session_state.messages:
    if m["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant", avatar="📚"):
            if m.get("error"):
                st.info(m["error"])
            if m.get("content"):
                st.markdown(m["content"])
            sources = m.get("sources") or []
            if sources:
                with st.expander(f"📎 引用来源（{len(sources)} 条）", expanded=True):
                    render_sources(sources)

prompt = st.chat_input("输入你的法规问题，例如：动力电池系统绝缘电阻有什么要求？")
if prompt:
    ask(prompt)

pending = st.session_state.pop("pending_question", None)
if pending:
    handle_query(pending)
