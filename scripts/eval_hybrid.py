# -*- coding: utf-8 -*-
"""用 20 题测试集跑混合检索（BM25 + 稠密 RRF，可选重排）召回基线，输出 hit@1 / hit@5。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_index import HybridRetriever, cite  # noqa: E402
from eval_recall import is_hit  # noqa: E402

QUESTIONS = ROOT / "data" / "processed" / "test_questions.jsonl"
OUT = ROOT / "outputs" / "hybrid_recall_preview.md"


def main(use_rerank=False):
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    ret = HybridRetriever()
    hit1 = hit5 = 0
    mode = "混合检索 + rerank" if use_rerank else "混合检索（BM25 + 稠密 RRF）"
    lines = [f"# 召回基线：{mode}", "",
             f"> 测试题数：{len(questions)}", "",
             "| # | 问题 | top1 命中 | top5 命中 | Top1 来源 |",
             "| --- | --- | --- | --- | --- |"]
    for q in questions:
        res = ret.search(q["question"], k=5, use_rerank=use_rerank)
        h1 = bool(res) and is_hit(res[0], q["expected_sources"])
        h5 = any(is_hit(c, q["expected_sources"]) for c in res)
        hit1 += h1
        hit5 += h5
        top1 = cite(res[0]) if res else "（无结果）"
        lines.append(f"| {q['id']} | {q['question']} | {'✓' if h1 else '✗'} | "
                     f"{'✓' if h5 else '✗'} | {top1} |")
    n = len(questions)
    lines += ["", f"- **hit@1：{hit1}/{n}（{hit1/n:.0%}）**",
              f"- **hit@5：{hit5}/{n}（{hit5/n:.0%}）**", "",
              "> 说明：命中判断为 Top-K 结果的标准号或名称包含题目预期来源之一。"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"hit@1={hit1}/{n}  hit@5={hit5}/{n} -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="启用 BGE-reranker 重排")
    args = parser.parse_args()
    main(use_rerank=args.rerank)
