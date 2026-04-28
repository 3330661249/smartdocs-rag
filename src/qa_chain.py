import json

from langchain_core.prompts import ChatPromptTemplate

from src.config import MAX_QUERY_LENGTH, get_chat_llm
from src.logging_utils import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是一个基于文档内容进行问答的助手。

请严格遵守以下规则：
1. 只能依据给定的上下文回答问题。
2. 不要使用上下文之外的常识或外部知识进行补充。
3. 如果上下文中没有足够信息回答问题，请明确说明信息不足。
4. 回答应简洁、清晰、使用中文。
5. 回答中如果引用了上下文，请在句子末尾标注对应编号，例如 [1]、[2]。
6. 仅输出 JSON，格式如下：
{{
  "answer": "你的回答",
  "enough_context": true,
  "used_citations": [1, 2]
}}"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "【上下文】\n{context}\n\n【用户问题】\n{query}\n\n请开始回答："),
])


def _build_citations(docs: list) -> list[dict]:
    citations = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        citations.append(
            {
                "index": index,
                "source": metadata.get("source", "未知"),
                "chunk_id": metadata.get("chunk_id", "未知"),
                "page": metadata.get("page"),
                "start_index": metadata.get("start_index"),
                "preview": doc.page_content[:120],
            }
        )

    return citations


def _parse_json_response(content: str) -> dict:
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        normalized = "\n".join(lines).strip()

    return json.loads(normalized)


def generate_answer(query: str, docs: list) -> dict:
    if not query or not query.strip():
        raise ValueError("查询内容不能为空。")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"查询长度 {len(query)} 超过最大限制 {MAX_QUERY_LENGTH}。")

    llm = get_chat_llm()

    context_lines = []
    citations = _build_citations(docs)
    for citation, doc in zip(citations, docs):
        label = f"[{citation['index']}] 来源：{citation['source']}"
        if citation["page"] is not None:
            label += f" 第 {citation['page']} 页"
        label += f" | Chunk ID：{citation['chunk_id']}"
        context_lines.append(f"{label}\n{doc.page_content}")

    chain = QA_PROMPT | llm
    response = chain.invoke({"context": "\n".join(context_lines), "query": query})
    content = response.content.strip()

    logger.info("生成回答: query=%r, docs=%d, response_len=%d", query[:50], len(docs), len(content))

    try:
        result = _parse_json_response(content)
    except json.JSONDecodeError:
        result = {
            "answer": content,
            "enough_context": True,
            "used_citations": [citation["index"] for citation in citations],
        }

    used_citation_ids = set(result.get("used_citations", []))
    filtered_citations = [
        citation
        for citation in citations
        if not used_citation_ids or citation["index"] in used_citation_ids
    ]

    return {
        "answer": result.get("answer", "").strip(),
        "enough_context": bool(result.get("enough_context", True)),
        "citations": filtered_citations,
    }
