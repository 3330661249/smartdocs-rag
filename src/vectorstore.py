import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def build_vectorstore(chunks, file_name="uploaded_file"):
    if not chunks:
        raise ValueError("chunks 为空，无法构建向量库。")

    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL")
    model_name = "Embedding-3"

    if not api_key:
        raise ValueError("未检测到 ZHIPU_API_KEY，请检查 .env 文件配置。")

    if not base_url:
        raise ValueError("未检测到 ZHIPU_BASE_URL，请检查 .env 文件配置。")

    embeddings = OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url
    )

    metadatas = [
        {"chunk_id": i, "source": file_name}
        for i in range(len(chunks))
    ]

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore


def search_similar_chunks(vectorstore, query, k=3):
    return vectorstore.similarity_search(query, k=k)