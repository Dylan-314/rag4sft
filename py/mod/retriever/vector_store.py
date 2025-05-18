from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from .embedder import get_embedder
from .config import PERSIST_ROOT


def get_or_create_chroma(kb_name: str) -> Chroma:
    persist_dir = Path(PERSIST_ROOT) / kb_name
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embedder(),
    )


def add_documents(kb_name: str, docs: List[Document]) -> None:
    vs = get_or_create_chroma(kb_name)
    vs.add_documents(documents=docs)
    vs.persist()
    