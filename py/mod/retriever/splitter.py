from langchain_text_splitters import TokenTextSplitter
from .config import SPLIT_CHUNK_SIZE, SPLIT_CHUNK_OVERLAP


def get_splitter() -> TokenTextSplitter:
    return TokenTextSplitter(
        chunk_size=SPLIT_CHUNK_SIZE,
        chunk_overlap=SPLIT_CHUNK_OVERLAP,
        add_start_index=True,
    )