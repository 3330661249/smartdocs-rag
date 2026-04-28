# SmartDocs-RAG

## 项目简介
SmartDocs-RAG 是一个本地知识库问答系统，面向文档问答场景，支持文档上传、文本切分、向量化存储、相似检索与基于检索增强生成（RAG）的答案生成。

当前版本已经从"单文件演示型原型"升级为"支持多文件知识库、持久化存储和检索阈值控制的小型问答工具"。

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

### 5. 运行测试
```bash
python -m pytest tests/ -v
```

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

---

## 下一步增强方向
- 增加更多评估样例与指标统计
- 增加历史版本回滚与版本差异对比
- 实现真正的 LLM 流式输出
- 添加用户认证与权限管理
- 增加多文件来源聚合下的排序控制与导出能力
