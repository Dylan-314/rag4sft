from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
from mod.retriever.manager import KnowledgeBaseManager

# ✅ 加载环境变量
load_dotenv()

def test_ingest_and_retrieve():
    text = "LangChain makes building with LLMs easier. Retrieval-Augmented Generation (RAG) combines search with generation."
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as tmp:
        tmp.write(text)
        file_path = Path(tmp.name)

    kb = KnowledgeBaseManager("test_kb")
    n_chunks = kb.ingest_files([str(file_path)])
    assert n_chunks > 0

    results = kb.retrieve("What does RAG do?", top_k=2)
    assert results, "Should return at least one result"
    best_doc, score = results[0]
    assert "RAG" in best_doc.page_content
    file_path.unlink()