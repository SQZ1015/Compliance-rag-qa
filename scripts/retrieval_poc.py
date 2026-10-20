# -*- coding: utf-8 -*-
"""条款级 BM25 检索 POC：返回带引用的候选条款（标准号 + 条款号 + 页码 + 原文）。"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "processed" / "knowledge_base.jsonl"

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
ASCII_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    text = text.lower()
    toks = []
    for m in ASCII_RE.finditer(text):
        t = m.group(0)
        if len(t) >= 2 or t.isdigit():
            toks.append(t)
    for m in CJK_RE.finditer(text):
        seg = m.group(0)
        if len(seg) == 1:
            toks.append(seg)
        else:
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


def load_chunks():
    chunks = []
    for line in KB.open(encoding="utf-8"):
        c = json.loads(line)
        if c.get("needs_ocr") or not (c.get("text") or "").strip():
            continue
        chunks.append(c)
    return chunks


def build_index(chunks):
    docs, df = [], {}
    for c in chunks:
        tf = {}
        for t in tokenize(c["text"]):
            tf[t] = tf.get(t, 0) + 1
        docs.append({"chunk": c, "toks": tf, "len": sum(tf.values())})
        for t in tf:
            df[t] = df.get(t, 0) + 1
    n = max(len(docs), 1)
    avgdl = sum(d["len"] for d in docs) / n
    idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}
    return docs, idf, avgdl


def bm25(query, docs, idf, avgdl, k1=1.5, b=0.75):
    qtoks = tokenize(query)
    scored = []
    for d in docs:
        score = 0.0
        for t in qtoks:
            tf = d["toks"].get(t)
            if not tf:
                continue
            denom = tf + k1 * (1 - b + b * d["len"] / avgdl)
            score += idf[t] * (tf * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, d["chunk"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def cite(c):
    loc = f"第{c['page']}页" if c.get("page") else "全文"
    clause = f" {c['clause']}" if c.get("clause") else ""
    return f"{c['std_no']} {c['name']}{clause} · {loc}"


def main():
    query = " ".join(sys.argv[1:]).strip()
    chunks = load_chunks()
    docs, idf, avgdl = build_index(chunks)
    demo = [
        "动力电池系统需要满足哪些强制性国标",
        "首次申请整车CCC需要提交哪些资料",
        "电动汽车交流充电桩认证规则",
        "电动汽车用动力蓄电池安全要求",
        "高压电气安全 绝缘 耐压",
    ]
    queries = [query] if query else demo
    for q in queries:
        print("=" * 70)
        print("查询：", q)
        results = bm25(q, docs, idf, avgdl)[:5]
        if not results:
            print("（未检索到相关条款）")
            continue
        for i, (score, c) in enumerate(results, 1):
            print(f"[{i}] 相关度 {score:.3f} | {cite(c)}")
            text = (c.get("text") or "").replace("\n", " ").strip()
            print("    原文：", text[:150] + ("…" if len(text) > 150 else ""))
        print()


if __name__ == "__main__":
    main()
