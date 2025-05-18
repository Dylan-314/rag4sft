import os
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_community.document_loaders.word_document import Docx2txtLoader
from tempfile import NamedTemporaryFile

EXTENSION_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}

def load_file(file_data: str | bytes, filename: str = "") -> List[Document]:
    """
    支持 str 路径 / bytes + 显式文件名
    """
    # —— 判断是否为本地路径 —— #
    if isinstance(file_data, str) and os.path.exists(file_data):
        ext = os.path.splitext(file_data)[1].lower()
        loader_cls = EXTENSION_LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls(file_data).load()

    # —— 如果是上传的文件内容（bytes），写临时文件 —— #
    if isinstance(file_data, bytes):
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = EXTENSION_LOADER_MAP.get(ext)
        if not loader_cls:
            raise ValueError(f"Unsupported file type: {ext}")

        with NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_data)
            tmp.flush()
            return loader_cls(tmp.name).load()

    raise ValueError("Unsupported input type for load_file")