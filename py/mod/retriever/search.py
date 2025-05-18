from __future__ import annotations
from typing import List, Tuple

import numpy as np
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from .vector_store import get_or_create_chroma
from .embedder import get_embedder


class HybridRetriever:
    """向量检索 + BM25 全文检索，再简单重排序."""

    def __init__(self, kb_name: str):
        self.vs = get_or_create_chroma(kb_name)
        self.embedder = get_embedder()
        # Build BM25 from all documents in store (could be optimized/cache)
        self._bm25 = None

    def _build_bm25(self):
        if self._bm25:
            return
        all_docs = self.vs.get(include=["documents"])["documents"]
        docs = [Document(page_content=d) for d in all_docs]
        self._bm25 = BM25Retriever.from_documents(docs)

    def query(
            self,
            question: str,
            top_k: int = 4,
            bm25_weight: float = 0.5,
    ) -> List[Tuple[Document, float]]:
        """返回 (Document, score) 列表，score 已归一化为 0~1."""
        # —— Vector search —— #
        vec_hits = self.vs.similarity_search_with_relevance_scores(
            question, k=top_k * 2
        )  # type: ignore
        if vec_hits:
            vec_docs, vec_scores = zip(*vec_hits)
            vec_scores = np.array(vec_scores)
            vec_scores = (vec_scores.max() - vec_scores) / (
                    vec_scores.max() - vec_scores.min() + 1e-6
            )  # 归一化到 0~1
        else:
            vec_docs, vec_scores = [], np.array([])

        # —— BM25 search —— #
        self._build_bm25()
        bm25_hits = self._bm25.get_relevant_documents(question, k=top_k * 2)
        bm25_scores = np.array([
            1.0 - i / max(1, len(bm25_hits) - 1) for i in range(len(bm25_hits))
        ])
        bm25_docs = bm25_hits

        # —— 合并 —— #
        merged: dict[str, Tuple[Document, float]] = {}
        for doc, s in zip(vec_docs, vec_scores):
            merged[doc.page_content[:50]] = (doc, bm25_weight * s)
        for doc, s in zip(bm25_docs, bm25_scores):
            key = doc.page_content[:50]
            if key in merged:
                merged[key] = (doc, merged[key][1] + (1 - bm25_weight) * s)
            else:
                merged[key] = (doc, (1 - bm25_weight) * s)

        # —— 去冗余 —— #
        docs, scores = zip(*merged.values()) if merged else ([], [])
        if docs:
            filter_ = EmbeddingsRedundantFilter(embeddings=self.embedder)
            docs = filter_.transform_documents(list(docs))

        # —— 排序 & 截断 —— #
        sorted_hits = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return sorted_hits