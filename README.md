# SmartDocs-RAG

## 项目简介
SmartDocs-RAG 是一个本地知识库问答系统，面向文档问答场景，支持文档上传、文本切分、向量化存储、相似检索与基于检索增强生成（RAG）的答案生成。

项目目标是构建一个最小可运行的 RAG 闭环，让用户能够基于本地文档进行问答，并查看回答所依据的参考片段，从而提升结果的可解释性。

---

## 项目功能
- 支持上传本地 TXT / MD 文档
- 支持展示文档原始内容
- 支持按 chunk size 和 overlap 进行文本切分
- 支持将文本块进行向量化并存入 Chroma 向量库
- 支持根据用户问题进行相似度检索
- 支持基于检索结果调用大模型生成回答
- 支持展示参考检索片段及 metadata
- 支持在信息不足时进行拒答
- 支持加载状态提示与流式输出效果

---

## 技术栈
- Python
- Streamlit
- LangChain / langchain-openai
- Chroma
- Embedding 模型（智谱 Embedding-3）
- Chat Model（智谱 GLM 系列）
- python-dotenv

---

## 项目结构
```bash
smartdocs-rag/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── assets/
├── data/
└── src/
    ├── loader.py
    ├── splitter.py
    ├── vectorstore.py
    ├── qa_chain.py
    └── utils.py