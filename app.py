import time
import streamlit as st
from src.loader import load_document
from src.splitter import split_text
from src.vectorstore import build_vectorstore, search_similar_chunks
from src.qa_chain import generate_answer

st.set_page_config(page_title="SmartDocs-RAG", page_icon="📚")

st.title("📚 SmartDocs-RAG")
st.write("一个本地知识库问答系统的初始版本")

uploaded_file = st.file_uploader("请上传一个 TXT 或 MD 文件", type=["txt", "md"])

chunk_size = st.slider("请选择 chunk 大小", min_value=50, max_value=500, value=100, step=10)
overlap = st.slider("请选择 overlap 大小", min_value=0, max_value=100, value=20, step=5)

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.success(f"已上传文件: {file_name}")

    content = load_document(uploaded_file)

    st.subheader("文档内容预览")
    st.write(f"文档字符数：{len(content)}")
    st.text_area("原始内容", content, height=200)

    chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)

    st.subheader("切分结果预览")
    st.write(f"当前共切分出 {len(chunks)} 个 chunks")
    st.write("当前仅展示前 5 个 chunks")

    for i, chunk in enumerate(chunks[:5]):
        st.text_area(f"Chunk {i + 1}", chunk, height=120)

    if st.button("开始向量化"):
        try:
            vectorstore = build_vectorstore(chunks, file_name=file_name)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["current_file_name"] = file_name
            st.success("向量库构建成功！")
            st.info("你现在可以开始提问。")
        except Exception as e:
            st.error(f"向量化失败：{e}")

if "vectorstore" in st.session_state:
    st.subheader("文档问答")
    query = st.text_input("请输入你的问题")

    if query:
        try:
            with st.spinner("正在检索相关片段..."):
                docs = search_similar_chunks(st.session_state["vectorstore"], query, k=3)

            if not docs:
                st.warning("未检索到相关内容，暂时无法回答该问题。请尝试调整提问方式。")
            else:
                with st.spinner("正在生成回答..."):
                    answer = generate_answer(query, docs)

                if not answer or not answer.strip():
                    st.warning("模型未返回有效回答，请稍后重试。")
                else:
                    st.markdown("## 回答结果")
                    answer_placeholder = st.empty()
                    streamed_text = ""

                    for char in answer:
                        streamed_text += char
                        answer_placeholder.markdown(streamed_text)
                        time.sleep(0.01)

                st.markdown("## 检索结果概览")
                st.write(f"共检索到 {len(docs)} 条参考片段。")

                st.markdown("## 参考检索片段")
                for idx, doc in enumerate(docs):
                    source = doc.metadata.get("source", "未知")
                    chunk_id = doc.metadata.get("chunk_id", "未知")

                    st.markdown(f"### 片段 {idx + 1}")
                    st.write(f"来源文件：{source}")
                    st.write(f"Chunk ID：{chunk_id}")
                    st.text_area(
                        f"片段内容 {idx + 1}",
                        doc.page_content,
                        height=150
                    )

        except Exception as e:
            st.error(f"问答失败，请检查模型配置、网络连接或向量库状态。错误信息：{e}")