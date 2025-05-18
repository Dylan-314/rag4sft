"""
集中定义 RAG 主流程 (rag_chain) —— 任何调用 rag_chain.invoke({"question": ...})
都会被 LangSmith 自动追踪，只需在 .env 里设置 LANGSMITH_API_KEY 等变量。

依赖:
- mod.retriever.search.HybridRetriever
- langchain>=0.2.*
- langsmith>=0.1.*
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from mod.retriever.search import HybridRetriever

# ---------------- 渲染 Prompt ---------------- #
SYSTEM_PROMPT = "You are a helpful assistant. Use the context to answer."

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{question}\n\nContext:\n{context}")
])

# ---------------- 检索器 & LLM ---------------- #
# 默认知识库，可在调用方决定 new_retriever = HybridRetriever(kb_name)
retriever = HybridRetriever("default")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# ---------------- LCEL Runnable ---------------- #
rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": lambda q: "\n\n".join(
            [doc.page_content for doc, _ in retriever.query(q, top_k=4)]
        ),
    }
    | prompt
    | llm
)

# 让其它文件 `from mod.rag.chain import rag_chain`
__all__ = ["rag_chain"]