def load_document(uploaded_file) -> str:
    file_name = uploaded_file.name

    if file_name.endswith(".txt") or file_name.endswith(".md"):
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")
        return content
    else:
        return "暂不支持该文件类型。"