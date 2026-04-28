import gc

import streamlit as st

from src.chat_history import clear_chat_history, load_chat_history, save_chat_history
from src.loader import load_document
from src.logging_utils import get_logger
from src import qa_chain
from src.splitter import split_document
from src.vectorstore import (
    build_vectorstore,
    delete_vectorstore,
    get_vectorstore_metadata,
    list_vectorstores,
    load_vectorstore,
    normalize_kb_name,
    search_similar_chunks,
    vectorstore_exists,
)

logger = get_logger(__name__)

st.set_page_config(page_title="SmartDocs-RAG", page_icon="📚")


def default_kb_name_for_uploads(uploaded_files) -> str:
    if len(uploaded_files) == 1:
        return normalize_kb_name(uploaded_files[0].name.rsplit(".", 1)[0])
    return "multi_docs_kb"


def group_search_results_by_source(search_results: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for item in search_results:
        source = item["doc"].metadata.get("source", "未知")
        grouped.setdefault(source, []).append(item)
    return grouped


if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "current_kb_name" not in st.session_state:
    st.session_state["current_kb_name"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("📚 SmartDocs-RAG")
st.write("一个支持多文件知识库、持久化存储和检索阈值控制的本地问答原型")

available_vectorstores = list_vectorstores()

with st.sidebar:
    st.header("知识库管理")
    if available_vectorstores:
        selected_kb = st.selectbox("已有知识库", available_vectorstores)
        selected_meta = get_vectorstore_metadata(selected_kb)

        if selected_meta:
            st.caption(f"更新时间：{selected_meta.get('updated_at', '未知')}")
            if selected_meta.get("collection_name"):
                st.caption(f"当前版本：{selected_meta['collection_name']}")
            st.caption(f"文档数：{selected_meta.get('document_count', '?')}｜Chunks：{selected_meta.get('chunk_count', '?')}")
            sources = selected_meta.get("sources", [])
            if sources:
                st.caption("来源文件：" + "、".join(sources))

        load_col, delete_col = st.columns(2)
        with load_col:
            if st.button("加载知识库"):
                try:
                    st.session_state["vectorstore"] = load_vectorstore(selected_kb)
                    st.session_state["current_kb_name"] = selected_kb
                    st.session_state["chat_history"] = load_chat_history(selected_kb)
                    st.success(f"已加载知识库：{selected_kb}")
                except Exception as exc:
                    st.error(f"加载失败：{exc}")
        with delete_col:
            if st.button("删除知识库", type="secondary"):
                try:
                    delete_vectorstore(selected_kb)
                    if st.session_state["current_kb_name"] == selected_kb:
                        st.session_state["vectorstore"] = None
                        st.session_state["current_kb_name"] = None
                        st.session_state["chat_history"] = []
                    st.success(f"已删除知识库：{selected_kb}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"删除失败：{exc}")
    else:
        st.caption("当前还没有已持久化的知识库。")

uploaded_files = st.file_uploader(
    "请上传一个或多个 TXT、MD 或 PDF 文件",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
)
chunk_size = st.slider("请选择 chunk 大小", min_value=100, max_value=1000, value=500, step=50)
overlap = st.slider("请选择 overlap 大小", min_value=0, max_value=200, value=80, step=10)

if uploaded_files:
    try:
        documents = [load_document(uploaded_file) for uploaded_file in uploaded_files]
        logger.info("用户上传文件: %d 个文件", len(documents))
        kb_name = st.text_input("知识库名称", value=default_kb_name_for_uploads(uploaded_files))
        normalized_kb_name = normalize_kb_name(kb_name)
        kb_already_exists = vectorstore_exists(normalized_kb_name)
        existing_meta = get_vectorstore_metadata(normalized_kb_name) if kb_already_exists else None

        total_chars = sum(len(document["text"]) for document in documents)
        total_pages = sum(len(document["pages"]) for document in documents)

        st.success(f"已上传 {len(documents)} 个文件")
        st.caption("文件列表：" + "、".join(document["source"] for document in documents))

        st.subheader("文档概览")
        st.write(f"总字符数：{total_chars}")
        if total_pages:
            st.write(f"累计提取页数：{total_pages}")

        if kb_already_exists:
            if existing_meta:
                st.warning(
                    f"知识库 {normalized_kb_name} 已存在，"
                    f"最近更新时间为 {existing_meta['updated_at']}。"
                )
                st.caption(
                    f"已有内容：{existing_meta['document_count']} 个文档，"
                    f"{existing_meta['chunk_count']} 个 chunks。"
                )
                st.caption(f"当前版本：{existing_meta['collection_name']}")
            else:
                st.warning(
                    f"知识库 {normalized_kb_name} 已存在，但这是旧版本目录或缺少 metadata 的目录。"
                    " 如需继续，请勾选覆盖后重建。"
                )
            overwrite = st.checkbox("确认覆盖已有知识库")
        else:
            overwrite = False

        preview_tab, chunk_tab = st.tabs(["文档预览", "切分预览"])

        with preview_tab:
            for document in documents:
                st.markdown(f"### {document['source']}")
                st.caption(f"文件类型：{document['file_type']}")
                st.text_area(
                    f"原始内容预览 - {document['source']}",
                    document["text"][:3000],
                    height=180,
                )

        split_docs = []
        for document in documents:
            split_docs.extend(
                split_document(document, chunk_size=chunk_size, overlap=overlap)
            )

        with chunk_tab:
            st.write(f"当前共切分出 {len(split_docs)} 个 chunks")
            st.write("当前仅展示前 8 个 chunks")

            for index, chunk_doc in enumerate(split_docs[:8], start=1):
                meta = chunk_doc.metadata
                summary = (
                    f"来源：{meta['source']}｜Chunk ID：{meta['chunk_id']}｜"
                    f"起始位置：{meta['start_index']}"
                )
                if "page" in meta:
                    summary += f"｜页码：{meta['page']}"
                st.caption(summary)
                st.text_area(f"Chunk {index}", chunk_doc.page_content, height=120)

        if st.button("构建并保存知识库"):
            if not kb_name.strip():
                st.error("知识库名称不能为空。")
            elif kb_already_exists and not overwrite:
                st.error("当前知识库已存在，请先勾选“确认覆盖已有知识库”。")
            else:
                try:
                    if overwrite and st.session_state["current_kb_name"] == normalized_kb_name:
                        st.session_state["vectorstore"] = None
                        st.session_state["current_kb_name"] = None
                        gc.collect()

                    vectorstore = build_vectorstore(
                        split_docs,
                        kb_name=kb_name,
                        overwrite=overwrite,
                    )
                    st.session_state["vectorstore"] = vectorstore
                    st.session_state["current_kb_name"] = normalized_kb_name
                    st.session_state["chat_history"] = []
                    clear_chat_history(normalized_kb_name)
                    logger.info("知识库构建成功: %s", normalized_kb_name)
                    st.success(f"知识库构建成功：{normalized_kb_name}")
                    st.info("你现在可以开始提问，或在下次启动时从侧边栏重新加载。")
                    st.rerun()
                except Exception as exc:
                    logger.error("向量化失败: %s", exc, exc_info=True)
                    st.error(f"向量化失败：{exc}")
    except Exception as exc:
        logger.error("文档处理失败: %s", exc, exc_info=True)
        st.error(f"文档处理失败：{exc}")

if st.session_state["vectorstore"] is not None:
    st.subheader("文档问答")
    if st.session_state["current_kb_name"]:
        st.caption(f"当前知识库：{st.session_state['current_kb_name']}")

    current_meta = get_vectorstore_metadata(st.session_state["current_kb_name"]) if st.session_state["current_kb_name"] else None
    available_sources = current_meta.get("sources", []) if current_meta else []

    top_k = st.slider("检索返回条数", min_value=1, max_value=8, value=4, step=1)
    score_threshold = st.slider("检索相关性阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    source_filter = st.multiselect(
        "按来源文件过滤",
        options=available_sources,
        default=[],
        help="留空表示在当前知识库的全部文件中检索。",
    )
    history_col, clear_col = st.columns([4, 1])
    with history_col:
        st.caption(f"当前会话轮数：{len(st.session_state['chat_history'])}")
    with clear_col:
        if st.button("清空会话"):
            st.session_state["chat_history"] = []
            if st.session_state["current_kb_name"]:
                clear_chat_history(st.session_state["current_kb_name"])
            st.rerun()

    if st.session_state["chat_history"]:
        st.markdown("## 会话历史")
        for turn_index, turn in enumerate(st.session_state["chat_history"], start=1):
            with st.container(border=True):
                st.markdown(f"**第 {turn_index} 轮**")
                st.markdown(f"**你：** {turn['question']}")
                st.markdown(f"**助手：** {turn['answer']}")
                if turn.get("citations"):
                    source_summary = "、".join(
                        sorted({citation["source"] for citation in turn["citations"]})
                    )
                    st.caption(f"引用来源：{source_summary}")

    query = st.chat_input("请输入你的问题")

    if query:
        logger.info("用户提问: %r, top_k=%d, threshold=%.2f", query[:50], top_k, score_threshold)
        try:
            with st.spinner("正在检索相关片段..."):
                search_results = search_similar_chunks(
                    st.session_state["vectorstore"],
                    query,
                    k=top_k,
                    score_threshold=score_threshold,
                    allowed_sources=source_filter or None,
                )

            if not search_results:
                st.warning("没有检索到达到阈值的相关内容，当前不建议生成回答。请尝试降低阈值或调整提问方式。")
            else:
                docs = [item["doc"] for item in search_results]
                st.markdown("## 当前问题")
                st.markdown(query)

                st.markdown("## 回答结果")
                with st.spinner("正在生成回答..."):
                    answer = st.write_stream(
                        qa_chain.stream_answer(
                            query,
                            docs,
                            history=st.session_state["chat_history"],
                        )
                    )
                answer_result = qa_chain.build_stream_result(answer, docs)
                answer = answer_result["answer"]

                if not answer or not answer.strip():
                    st.warning("模型未返回有效回答，请稍后重试。")
                else:
                    st.session_state["chat_history"].append(
                        {
                            "question": query,
                            "answer": answer,
                            "citations": answer_result["citations"],
                            "source_filter": source_filter,
                        }
                    )
                    if st.session_state["current_kb_name"]:
                        save_chat_history(
                            st.session_state["current_kb_name"],
                            st.session_state["chat_history"],
                        )
                    if not answer_result["enough_context"]:
                        st.info("模型判断当前上下文信息不足，回答仅基于已有片段做保守输出。")

                    if answer_result["citations"]:
                        st.markdown("## 回答引用")
                        for citation in answer_result["citations"]:
                            label = f"[{citation['index']}] {citation['source']}"
                            if citation["page"] is not None:
                                label += f" | 第 {citation['page']} 页"
                            label += f" | Chunk ID：{citation['chunk_id']}"
                            st.write(label)
                            st.caption(citation["preview"])

                st.markdown("## 检索结果概览")
                st.write(f"共有 {len(search_results)} 条片段通过当前阈值过滤。")
                if source_filter:
                    st.caption("当前来源过滤：" + "、".join(source_filter))

                grouped_results = group_search_results_by_source(search_results)
                st.markdown("## 按文件聚合")
                for source, items in grouped_results.items():
                    st.write(f"{source}：{len(items)} 条相关片段")

                st.markdown("## 参考检索片段")
                item_index = 1
                for source, items in grouped_results.items():
                    st.markdown(f"### 文件：{source}")
                    for item in items:
                        doc = item["doc"]
                        score = item["score"]
                        chunk_id = doc.metadata.get("chunk_id", "未知")
                        page = doc.metadata.get("page")
                        start_index = doc.metadata.get("start_index", "未知")

                        st.write(f"片段 {item_index}")
                        st.write(f"Chunk ID：{chunk_id}")
                        st.write(f"起始位置：{start_index}")
                        if page is not None:
                            st.write(f"页码：{page}")
                        if score is not None:
                            st.write(f"相关性分数：{score:.3f}")
                        st.text_area(
                            f"片段内容 {item_index}",
                            doc.page_content,
                            height=150,
                        )
                        item_index += 1

        except Exception as exc:
            logger.error("问答失败: %s", exc, exc_info=True)
            st.error(f"问答失败，请检查模型配置、网络连接或向量库状态。错误信息：{exc}")
