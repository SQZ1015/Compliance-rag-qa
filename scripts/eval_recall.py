# -*- coding: utf-8 -*-
"""用 20 题测试集跑检索召回基线，输出 hit@1 / hit@5 预览报告。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_poc import load_chunks, build_index, bm25, cite

QUESTIONS = ROOT / "data" / "processed" / "test_questions.jsonl"
OUT = ROOT / "outputs" / "test_recall_preview.md"


def norm(s):
    return (s or "").replace(" ", "").replace("　", "").lower()


def is_hit(chunk, expected):
    hay = norm(chunk.get("std_no", "")) + " " + norm(chunk.get("name", ""))
    return any(norm(e) in hay for e in expected)


def main():
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    chunks = load_chunks()
    docs, idf, avgdl = build_index(chunks)
    lines = ["# 测试集检索召回基线", "",
             f"> 索引条款数：{len(chunks)}，测试题数：{len(questions)}", "",
             "| # | 问题 | top1 命中 | top5 命中 | Top1 来源 |",
             "| --- | --- | --- | --- | --- |"]
    hit1 = hit5 = 0
    for q in questions:
        res = bm25(q["question"], docs, idf, avgdl)[:5]
        h1 = bool(res) and is_hit(res[0][1], q["expected_sources"])
        h5 = any(is_hit(c, q["expected_sources"]) for _, c in res)
        hit1 += h1
        hit5 += h5
        top1 = cite(res[0][1]) if res else "（无结果）"
        lines.append(f"| {q['id']} | {q['question']} | {'✓' if h1 else '✗'} | "
                     f"{'✓' if h5 else '✗'} | {top1} |")
    n = len(questions)
    lines += ["", f"- **hit@1：{hit1}/{n}（{hit1/n:.0%}）**",
              f"- **hit@5：{hit5}/{n}（{hit5/n:.0%}）**", "",
              "> 说明：命中判断为 Top-K 结果的标准号或名称包含题目预期来源之一；"
              "受《CQC-C1101-2020 实施细则》等文件缺失影响，部分题目暂难命中。"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"hit@1={hit1}/{n}  hit@5={hit5}/{n} -> {OUT}")


if __name__ == "__main__":
    main()
