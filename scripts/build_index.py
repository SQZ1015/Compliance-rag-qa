# -*- coding: utf-8 -*-
"""混合检索索引构建与查询：BM25 + BGE-M3 稠密向量 + RRF 融合（可选重排）。

- 构建：解析 knowledge_base.jsonl -> BM25（复用 retrieval_poc）-> BGE-M3 嵌入 -> Chroma
- 查询：稠密 Top-K + BM25 Top-K -> RRF 融合 -> 可选 BGE-reranker 重排
- 产物：data/processed/index/chroma/ 与 data/processed/index/bm25.pkl
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# 本机 C 盘空间不足，Windows 下模型缓存放 D 盘；可用环境变量覆盖（云端默认使用标准 HF 缓存）
if os.name == "nt":
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache\hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_poc import load_chunks, tokenize, build_index as build_bm25, bm25, cite  # noqa: E402

KB = ROOT / "data" / "processed" / "knowledge_base.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_FILE = INDEX_DIR / "bm25.pkl"

EMB_MODEL = "BAAI/bge-small-zh-v1.5"  # MVP 默认：CPU 友好；BGE-M3 已下载于 D:\hf_cache\local\bge-m3，可切换升级
RERANK_MODEL = "BAAI/bge-reranker-base"
LOCAL_MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", r"D:\hf_cache\local\bge-small-zh-v1.5"))
RERANK_LOCAL_DIR = Path(os.environ.get("RERANK_LOCAL_DIR", r"D:\hf_cache\local\bge-reranker-base"))
RRF_K = 60


def embedding_model_path():
    """本地已下载则用本地目录，否则用 HF 模型名（触发自动下载）。"""
    return str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else EMB_MODEL


def reranker_path():
    """本地已下载则用本地目录，否则用 HF 模型名。"""
    return str(RERANK_LOCAL_DIR) if RERANK_LOCAL_DIR.exists() else RERANK_MODEL


def clean_meta(chunk):
    """Chroma 元数据只接受 str/int/float/bool，此处做空值兜底。"""
    return {
        "file": str(chunk.get("file") or ""),
        "std_no": str(chunk.get("std_no") or ""),
        "name": str(chunk.get("name") or ""),
        "clause": str(chunk.get("clause") or ""),
        "page": int(chunk.get("page") or 0),
        "status": str(chunk.get("status") or ""),
        "impl_date": str(chunk.get("impl_date") or ""),
    }


def build():
    from sentence_transformers import SentenceTransformer
    import chromadb

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    ids = [f"c{i}" for i in range(len(chunks))]

    # 1) BM25（复用现有实现）
    docs, idf, avgdl = build_bm25(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_FILE.open("wb") as f:
        pickle.dump({"chunks": chunks, "docs": docs, "idf": idf, "avgdl": avgdl}, f)

    # 2) BGE-M3 稠密向量
    model = SentenceTransformer(embedding_model_path())
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

    # 3) Chroma 持久化
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection("regulations", metadata={"hnsw:space": "cosine"})
    col.upsert(
        ids=ids,
        embeddings=np.asarray(embs, dtype=np.float32).tolist(),
        documents=texts,
        metadatas=[clean_meta(c) for c in chunks],
    )
    return len(chunks), col.count()


def index_exists():
    return BM25_FILE.exists() and CHROMA_DIR.exists()


def ensure_index():
    """索引不存在时自动构建（便于全新 clone / 云端部署开箱即用）。"""
    if index_exists():
        return False
    build()
    return True


class HybridRetriever:
    """加载已构建索引，提供混合检索与可选重排。"""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb

        with BM25_FILE.open("rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.docs = data["docs"]
        self.idf = data["idf"]
        self.avgdl = data["avgdl"]
        self.chunk_to_idx = {id(c): i for i, c in enumerate(self.chunks)}

        self.model = SentenceTransformer(embedding_model_path())
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = self.client.get_collection("regulations")
        self._reranker = None

    def _dense(self, query, k):
        qemb = self.model.encode([query], normalize_embeddings=True)[0]
        res = self.col.query(
            query_embeddings=[qemb.tolist()], n_results=k, include=["documents", "distances"]
        )
        ids = res["ids"][0]
        return [self.chunks[int(i[1:])] for i in ids]

    def _bm25(self, query, k):
        scored = bm25(query, self.docs, self.idf, self.avgdl)[:k]
        return [c for _, c in scored]

    def _rerank(self, query, candidates, k):
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(reranker_path())
        pairs = [[query, c["text"]] for c in candidates]
        scores = self._reranker.predict(pairs)
        order = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in order]

    def search(self, query, k=5, use_rerank=False, dense_weight=1.0, sparse_weight=1.0):
        dense = self._dense(query, k * 4)
        sparse = self._bm25(query, k * 4)

        fused = {}
        for rank, c in enumerate(dense):
            idx = self.chunk_to_idx[id(c)]
            fused[idx] = fused.get(idx, 0.0) + dense_weight / (RRF_K + rank + 1)
        for rank, c in enumerate(sparse):
            idx = self.chunk_to_idx[id(c)]
            fused[idx] = fused.get(idx, 0.0) + sparse_weight / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        candidates = [self.chunks[i] for i, _ in ranked]
        if use_rerank and candidates:
            candidates = self._rerank(query, candidates, k)
        return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="", help="演示查询；留空则重建索引")
    parser.add_argument("--rerank", action="store_true", help="启用 BGE-reranker 重排")
    args = parser.parse_args()

    if not args.query:
        n, count = build()
        print(f"built index: chunks={n} -> chroma count={count}")
        return

    ret = HybridRetriever()
    for c in ret.search(args.query, k=5, use_rerank=args.rerank):
        text = (c.get("text") or "").replace("\n", " ").strip()
        print(f"- {cite(c)}")
        print(f"  {text[:160]}{'…' if len(text) > 160 else ''}")


if __name__ == "__main__":
    main()
