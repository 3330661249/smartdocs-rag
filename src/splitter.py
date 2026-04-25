def split_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks