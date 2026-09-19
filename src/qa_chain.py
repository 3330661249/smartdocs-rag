import json
import re

from langchain_core.prompts import ChatPromptTemplate

from src.config import MAX_QUERY_LENGTH, get_chat_llm
from src.logging_utils import get_logger

logger = get_logger(__name__)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个基于文档内容进行问答的助手。

请严格遵守以下规则：
1. 只能依据给定的上下文回答问题。
2. 不要使用上下文之外的常识或外部知识进行补充。
3. 如果上下文中没有足够信息回答问题，请明确说明信息不足。
4. 回答应简洁、清晰、使用中文。
5. 回答中如果引用了上下文，请在句子末尾标注对应编号，例如 [1]、[2]。
used_citations 必须与回答正文中的编号一致；信息不足时为空数组。
6. 仅输出 JSON，格式如下：
{{
  "answer": "你的回答",
  "enough_context": true,
  "used_citations": [1, 2]
}}""",
        ),
        (
            "human",
            "【历史对话】\n{history}\n\n【上下文】\n{context}\n\n【用户问题】\n{query}\n\n请开始回答：",
        ),
    ]
)

STREAM_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个基于文档内容进行问答的助手。

请严格遵守以下规则：
1. 只能依据给定的上下文回答问题。
2. 不要使用上下文之外的常识或外部知识进行补充。
3. 如果上下文中没有足够信息回答问题，请以“根据当前上下文信息不足”开头说明无法确认。
4. 回答应简洁、清晰、使用中文。
5. 只要使用了某条上下文，请在对应句子末尾标注编号，例如 [1]、[2]。
6. 正常输出回答正文，不要输出 JSON，不要解释你的规则。""",
        ),
        (
            "human",
            "【历史对话】\n{history}\n\n【上下文】\n{context}\n\n【用户问题】\n{query}\n\n请开始回答：",
        ),
    ]
)

INSUFFICIENT_CONTEXT_PREFIX = "根据当前上下文信息不足"


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


def _build_context(citations: list[dict], docs: list) -> str:
    context_lines = []
    for citation, doc in zip(citations, docs):
        label = f"[{citation['index']}] 来源：{citation['source']}"
        if citation["page"] is not None:
            label += f" 第 {citation['page']} 页"
        label += f" | Chunk ID：{citation['chunk_id']}"
        context_lines.append(f"{label}\n{doc.page_content}")
    return "\n".join(context_lines)


def _validate_query(query: str) -> str:
    if not query or not query.strip():
        raise ValueError("用户问题不能为空。")

    normalized_query = query.strip()
    if len(normalized_query) > MAX_QUERY_LENGTH:
        raise ValueError(f"用户问题超过最大限制：{MAX_QUERY_LENGTH} 字符。")

    return normalized_query


def _collect_used_citations(answer: str, citations: list[dict]) -> list[dict]:
    matches = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    return [
        citation
        for citation in citations
        if citation["index"] in matches
    ]


def _build_history(history: list[dict] | None, limit: int = 3) -> str:
    if not history:
        return "无历史对话"

    recent_turns = history[-limit:]
    lines = []
    for index, turn in enumerate(recent_turns, start=1):
        question = str(turn.get("question", "")).strip() or "未记录问题"
        answer = str(turn.get("answer", "")).strip() or "未记录回答"
        lines.append(f"第 {index} 轮用户问题：{question}")
        lines.append(f"第 {index} 轮助手回答：{answer}")
    return "\n".join(lines)


def _failure_result(status: str) -> dict:
    messages = {
        "insufficient_context": "根据当前上下文信息不足，暂时无法确认。",
        "invalid_response": "模型输出格式无效，本次未生成可验证的回答，请重试。",
        "invalid_citations": "模型回答缺少有效引用或引用不一致，本次回答未通过校验，请重试。",
    }
    return {
        "answer": messages[status],
        "enough_context": False,
        "citations": [],
        "status": status,
    }


def _finalize_answer(answer: str, docs: list, enough_context: bool = True, used_citations: list[int] | None = None) -> dict:
    if not answer.strip():
        return _failure_result("invalid_response")
    if not docs or not enough_context or answer.strip().startswith(INSUFFICIENT_CONTEXT_PREFIX):
        return _failure_result("insufficient_context")

    citations = _build_citations(docs)
    inline_ids = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    valid_ids = {citation["index"] for citation in citations}
    if not inline_ids or not inline_ids <= valid_ids:
        return _failure_result("invalid_citations")
    if used_citations is not None and set(used_citations) != inline_ids:
        return _failure_result("invalid_citations")
    return {
        "answer": answer.strip(),
        "enough_context": True,
        "citations": _collect_used_citations(answer, citations),
        "status": "answered",
    }


def build_stream_result(answer: str, docs: list) -> dict:
    """Validate citation identity after streaming; this does not grade semantic support."""
    return _finalize_answer(answer, docs)


def stream_answer(query: str, docs: list, history: list[dict] | None = None):
    normalized_query = _validate_query(query)
    if not docs:
        yield _failure_result("insufficient_context")["answer"]
        return
    citations = _build_citations(docs)
    context = _build_context(citations, docs)
    history_text = _build_history(history)
    logger.info(
        "Stream answer: query_len=%s, docs=%s, citations=%s",
        len(normalized_query),
        len(docs),
        len(citations),
    )

    chain = STREAM_QA_PROMPT | get_chat_llm()
    for chunk in chain.stream(
        {"history": history_text, "context": context, "query": normalized_query}
    ):
        content = getattr(chunk, "content", "")
        if not content:
            continue
        if isinstance(content, list):
            for item in content:
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                if text:
                    yield text
        else:
            yield str(content)


def generate_answer(query: str, docs: list, history: list[dict] | None = None) -> dict:
    query = _validate_query(query)
    if not docs:
        return _failure_result("insufficient_context")

    citations = _build_citations(docs)
    context = _build_context(citations, docs)
    history_text = _build_history(history)
    logger.info(
        "Generate answer: query_len=%s, docs=%s, citations=%s",
        len(query),
        len(docs),
        len(citations),
    )

    chain = QA_PROMPT | get_chat_llm()
    response = chain.invoke({"history": history_text, "context": context, "query": query})
    content = response.content
    if not isinstance(content, str):
        return _failure_result("invalid_response")

    try:
        result = _parse_json_response(content)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON")
        return _failure_result("invalid_response")

    if (
        not isinstance(result, dict)
        or not isinstance(result.get("answer"), str)
        or type(result.get("enough_context")) is not bool
        or not isinstance(result.get("used_citations"), list)
        or any(type(value) is not int for value in result["used_citations"])
    ):
        return _failure_result("invalid_response")
    finalized = _finalize_answer(result["answer"], docs, result["enough_context"], result["used_citations"])
    logger.info("Generate answer complete: status=%s, citations=%s", finalized["status"], len(finalized["citations"]))
    return finalized
