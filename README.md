# SmartDocs-RAG

## 项目简介
SmartDocs-RAG 是一个本地知识库问答系统，面向文档问答场景，支持文档上传、文本切分、向量化存储、相似检索与基于检索增强生成（RAG）的答案生成。

当前版本已经从"单文件演示型原型"升级为"支持多文件知识库、持久化存储和检索阈值控制的小型问答工具"。

项目定位是个人学习与工程验证原型：用户上传资料、限定检索来源、提问并回看引用片段。当前没有真实用户试点或生产效果数据；检索相似度和引用编号校验不能证明回答事实正确。

向量库和会话保存在本地；向量化及答案生成调用所配置的外部 API，会发送文档内容或检索片段，并可能产生费用。不要用未获准外发的资料进行试验。

---

## 功能特性
- 支持上传本地 TXT / MD / PDF 文档（含文件大小校验）
- 支持多个文件合并构建同一知识库
- 支持展示文档原始内容与 PDF 页数信息
- 支持使用 `RecursiveCharacterTextSplitter` 进行标准文本切分
- 支持为 chunk 保存 `source / chunk_id / start_index / page` 等 metadata
- 支持将文档向量化后持久化保存到本地 Chroma 向量库
- 支持从侧边栏重新加载已有知识库
- 支持显示知识库更新时间、来源文件和当前版本标识
- 支持同名知识库覆盖提醒与本地删除
- 支持根据用户问题进行相似度检索
- 支持按来源文件过滤检索范围
- 支持通过相关性阈值过滤低质量检索结果
- 支持基于检索结果调用大模型生成回答
- 支持流式生成，完成后检查引用编号；失败草稿替换为明确说明且不写入会话历史
- 支持展示回答引用来源，以及按文件聚合后的参考检索片段 metadata / score
- 支持基础评估样例与简单问答评估脚本

---

## 技术栈
- Python 3.12+
- Streamlit
- LangChain（langchain-openai / langchain-community / langchain-text-splitters）
- Chroma
- pypdf
- 智谱 GLM 系列 Chat Model + Embedding-3
- python-dotenv / logging

---

## 项目结构
```
SmartDocs-RAG/
├── app.py                  # Streamlit 入口
├── requirements.txt
├── start.sh
├── .env.example
├── data/                   # 示例文档
├── evals/
│   ├── run_eval.py         # 评估脚本
│   └── sample_qa.json      # 评估样例
├── src/
│   ├── __init__.py
│   ├── config.py           # 集中配置（Settings + 单例工厂）
│   ├── loader.py           # 文档加载（TXT/MD/PDF）
│   ├── splitter.py         # 文本切分
│   ├── vectorstore.py      # Chroma 向量库 CRUD + 检索
│   ├── qa_chain.py         # LLM 问答链（ChatPromptTemplate）
│   └── logging_utils.py    # 日志工具
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_splitter.py
│   ├── test_loader.py
│   ├── test_qa_chain.py
│   └── test_vectorstore.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 快速开始

### 1. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入你的智谱 API Key
```

### 2. 安装依赖
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 启动应用
```bash
streamlit run app.py
# 或使用启动脚本
bash start.sh
```

### 4. 运行评估
```bash
python -m evals.run_eval
```

此命令会调用所配置的 Embedding 和 Chat API，可能产生费用。每次运行创建独立临时知识库，成功或异常后均尝试清理；不会使用或删除用户的 `eval_demo_kb`。

### 5. 运行测试
```bash
python -m pytest tests/ -v
```

测试使用假模型和确定性本地 Embedding，并禁止外部网络连接，不需要真实 API Key。测试包括真实 Prompt 格式化、无效引用处理、本地 Chroma 持久化与过滤、评估异常清理和 Streamlit 页面行为。

---

## 配置说明

| 环境变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `ZHIPU_API_KEY` | 是 | — | 智谱 API Key |
| `ZHIPU_BASE_URL` | 是 | — | 智谱 API 地址 |
| `ZHIPU_CHAT_MODEL` | 否 | `glm-4.7` | 聊天模型名称 |
| `ZHIPU_EMBEDDING_MODEL` | 否 | `Embedding-3` | Embedding 模型名称 |

---

## 架构设计

- **集中配置**：所有配置通过 `src/config.py` 管理，使用 `@lru_cache` 实现 Settings / Embeddings / LLM 的单例模式
- **结构化日志**：所有核心模块使用 `logging_utils.get_logger()`，记录关键操作和错误
- **输入校验**：查询长度限制（2000 字符）和文件大小限制（50MB）
- **Prompt 工程**：使用 `ChatPromptTemplate` 将系统指令与用户输入按消息角色分离
- **引用校验**：回答中的编号必须对应本次检索片段；结构化输出还要求 `used_citations` 与正文编号一致。未通过检查时返回非成功状态，不补造来源。

问答结果的 `status` 取值为 `answered`、`insufficient_context`、`invalid_response` 或 `invalid_citations`。`answered` 表示通过结构与引用编号检查，仍需用户核对引用内容是否支持答案。流式生成过程是草稿，校验在生成结束后执行。

## 验证范围与已知限制

- 2026-09-19 在 macOS / Python 3.13.12 的独立虚拟环境按 `requirements.txt` 安装，使用 Chroma 0.6.3、langchain-core 0.3.86、langchain-openai 0.3.35、Streamlit 1.64.0 完成本地离线验证：`python -m pytest tests/ -q` 为 **66 passed**，`ruff check src/ app.py evals/ tests/` 通过。依赖使用版本区间，后续安装结果可能不同。
- `evals/sample_qa.json` 的 3 条样例是问答冒烟检查，按关键词和来源判断，不是检索召回率、引用准确率或业务成功率评测。未开展本轮真实模型、成本、时延或语义正确率测量。
- 仅支持可提取文本的 PDF，不含 OCR；跨页 chunk 目前只标记起始页。
- 无用户认证和权限隔离；来源筛选只是检索范围设置。未验证恶意文档指令或历史污染防护。
- 引用检查不验证每个事实的语义支持。流式拒答仍依赖约定前缀；会话不会自动改写追问后再检索。
- 知识库版本仍使用秒级标识和绝对存储路径，没有模型兼容性校验、原子元数据更新或历史回滚。
- 当前 Chroma 集成和部分传递依赖存在弃用警告，后续升级需要重新执行离线回归与服务商适配检查。

---

## 下一步增强方向
- 增加更多评估样例与指标统计
- 增加历史版本回滚与版本差异对比
- 增加 embedding 配置兼容性检查与可靠的版本标识
- 增加无答案、冲突证据、跨页和恶意文档评估样例
- 增加多文件来源聚合下的排序控制与导出能力
