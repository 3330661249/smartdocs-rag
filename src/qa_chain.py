import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def generate_answer(query: str, docs: list) -> str:
    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL")
    model_name = os.getenv("ZHIPU_CHAT_MODEL", "glm-4.7")

    if not api_key:
        raise ValueError("未检测到 ZHIPU_API_KEY，请检查 .env 文件配置。")

    if not base_url:
        raise ValueError("未检测到 ZHIPU_BASE_URL，请检查 .env 文件配置。")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0
    )

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
你是一个基于文档内容进行问答的助手。

请严格遵守以下规则：
1. 只能依据给定的上下文回答问题
2. 不要使用上下文之外的常识或外部知识进行补充
3. 如果上下文中没有足够信息回答问题，请明确回答：
   "根据当前文档内容，无法确定该问题的答案。"
4. 回答应简洁、清晰、使用中文

【上下文】
{context}

【用户问题】
{query}

请开始回答：
"""

    response = llm.invoke(prompt)
    return response.content
