import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import List, Tuple, Iterable, Union
from langchain.docstore.document import Document
from pathlib import Path
from .search import HybridRetriever

# 指定知识库根目录（不用环境变量）
KB_ROOT_PATH = Path("/Users/yubowang/PycharmProjects/py/mod/retriever/.chroma_db")

from .loader import load_file
from .splitter import get_splitter
from .vector_store import add_documents
from .search import HybridRetriever
# 尝试使用 LlamaParse（可选）
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")  # 环境变量配置
PARSER = None


def _parse_docs(raw_docs: List[Document]) -> List[Document]:
    """使用 LlamaParse 将文档结构化，如果无法使用则自动回退原始文本。"""
    global PARSER
    if PARSER is None:
        try:
            from llama_parse import LlamaParse
            PARSER = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY,
                result_type="markdown",
                verbose=False,
            )
        except Exception:
            PARSER = False  # 显式标记为不可用，避免重复导入

    if PARSER:
        try:
            return PARSER.parse_documents(raw_docs)  # type: ignore
        except Exception:
            pass  # 出错就自动回退

    return raw_docs  # 默认回退：原始文档列表


class KnowledgeBaseManager:
    """知识库管理器，支持入库和混合检索。"""

    def __init__(self, kb_name: str = "default"):
        self.kb_name = kb_name
        self.retriever = HybridRetriever(kb_name)
    # ------------------------------------------------------------
    # Utility: enumerate existing knowledge-bases
    # ------------------------------------------------------------
    @staticmethod
    def list_kbs() -> list[str]:
        """
        返回当前已存在的知识库名称列表。
        枚举 KB_ROOT_DIR 下的子目录（环境变量 KB_ROOT_DIR，否则使用 ~/.chromadb）。
        """
        kb_root = KB_ROOT_PATH
        if not kb_root.exists():
            return []
        return [p.name for p in kb_root.iterdir() if p.is_dir()]

    # —— 文档入库 —— #
    def ingest_files(self, file_objs: Iterable[bytes | str]) -> int:
        """将多个文件加载并写入向量库，返回写入 chunk 数量。"""
        docs: List[Document] = []
        for f in file_objs:
            if hasattr(f, "read") and hasattr(f, "name"):
                docs.extend(load_file(f.read(), filename=f.name))
            else:
                docs.extend(load_file(f))

        parsed_docs = _parse_docs(docs)
        splitter = get_splitter()
        chunks = splitter.split_documents(parsed_docs)
        add_documents(self.kb_name, chunks)
        return len(chunks)

    # —— 混合检索 —— #
    def retrieve(self, query: str, top_k: int = 4, bm25_weight: float = 0.5):
        """执行混合检索（向量 + BM25），返回排序片段。"""
        retriever = HybridRetriever(self.kb_name)
        return retriever.query(query, top_k=top_k, bm25_weight=bm25_weight)

    def search(
        self,
        query: str,
        top_k: int = 6,
        bm25_weight: float = 0.5,
        return_scores: bool = False,
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """
        简化接口：执行混合检索。
        :param bm25_weight: BM25 权重 (0~1)
        :param return_scores: 是否返回 (Document, score) 列表
        """
        hits = self.retrieve(
            query=query,
            top_k=top_k,
            bm25_weight=bm25_weight,
        )
        if return_scores:
            return hits
        return [doc for doc, _ in hits]