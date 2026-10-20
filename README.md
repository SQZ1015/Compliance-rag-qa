# CCC 新能源汽车认证法规智能问答助手

面向内部法规工程师的认证法规检索与问答系统（RAG）。第一阶段聚焦 CCC 认证体系、新能源汽车整车及关键部件。

## 项目状态
- 已完成：目录初始化、15 份样本迁移与命名、元数据登记、条款级知识库、检索 POC、20 题测试集与召回基线
- 路线已确认：基于现有项目改造 + 复用开源组件（详见 `docs/项目交接与下一步.md`）
- 待办：向量检索/生成问答/前端（MVP）、扫描页 OCR、补 CQC-C1101-2020 实施细则等缺失法规

## 目录结构
```
Renzheng-QA/
├── docs/            # POC 方案、版本核对、元数据表、交接文档
├── data/
│   ├── raw/         # 原始法规 PDF / Word 样本（15 份）
│   └── processed/   # 元数据、知识库、测试集
├── scripts/         # 解析、切分、检索、评估脚本
├── outputs/         # 解析报告、检索演示、召回基线
└── README.md
```

## 关键决策
- 答案必须可溯源：引用标准号、条款号、原文
- 低置信度时明确不回答，不编造
- 法规库字段：标准/规则编号、名称、版本号、发布/实施日期、状态、代替关系、依据公告、来源机构
- 默认返回现行版本；命中被代替版本时明确提示版本状态

## 当前产出
- 知识库：`data/processed/knowledge_base.jsonl`（1408 条款块）
- 测试集：`data/processed/test_questions.jsonl`（20 题）
- 检索 POC：`scripts/retrieval_poc.py`
- 召回基线：`outputs/test_recall_preview.md`（hit@5 75%）

## 技术选型（已确认 2026-09-28）
- LLM：DeepSeek（OpenAI 兼容，可切本地 Ollama/vLLM）
- Embedding：BGE-M3 + BGE-reranker（本地）
- 向量库：Chroma（嵌入式）
- 检索：BM25 + 稠密向量 RRF 融合 + 重排
- 前端：Streamlit

## 衔接与下一步
完整交接信息见 `docs/项目交接与下一步.md`。下一步：新增 `scripts/build_index.py`（Chroma + BGE-M3 + RRF）→ `scripts/qa_api.py`（DeepSeek 生成 + 引用 + 拒答）→ Streamlit 前端 → 重测召回至 90%+ → OCR 补扫描件。
