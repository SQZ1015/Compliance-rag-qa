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

## 项目演示 / 如何运行

### 环境要求
- Python 3.10+（在 3.12 上验证）
- 混合检索需下载 BGE-M3 模型；生成式问答需自备 DeepSeek API Key

### 1. 安装依赖
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 数据说明
- 法规 PDF 原文位于 `data/raw/`，因版权未提交到仓库，需自行从官方渠道获取后放入
- 已提交 `data/processed/knowledge_base.jsonl`（1408 个条款块），可直接用于检索演示

### 3. 快速检索（纯 BM25，无需模型）
```bash
python scripts/retrieval_poc.py "动力电池系统需要满足哪些强制性国标"
```

### 4. 混合检索（BM25 + BGE-M3 + RRF）
```bash
python scripts/download_model.py            # 下载 BGE-M3；网络不佳可设 HF_DOWNLOAD_BASE=https://hf-mirror.com
python scripts/build_index.py               # 构建 BM25 + 向量索引
python scripts/build_index.py --query "电动汽车交流充电桩认证规则"
```

### 5. 生成式问答（DeepSeek，可溯源 + 低置信度拒答）
```powershell
$env:DEEPSEEK_API_KEY = "你的key"
python scripts/qa_api.py "首次申请整车CCC需要提交哪些资料"
```

### 6. Web 演示（Streamlit）
```bash
streamlit run app.py
```

### 7. 召回评估
```bash
python scripts/eval_recall.py
```

### 关键产出
- 知识库：`data/processed/knowledge_base.jsonl`
- 测试集：`data/processed/test_questions.jsonl`
- 检索演示：`outputs/retrieval_demo.md`
- 召回基线：`outputs/test_recall_preview.md`

## 衔接与下一步
完整交接信息见 `docs/项目交接与下一步.md`。下一步：新增 `scripts/build_index.py`（Chroma + BGE-M3 + RRF）→ `scripts/qa_api.py`（DeepSeek 生成 + 引用 + 拒答）→ Streamlit 前端 → 重测召回至 90%+ → OCR 补扫描件。
