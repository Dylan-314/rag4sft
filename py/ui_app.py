"""Unified Streamlit UI for RAG-SFT System"""

import os
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────
# 0. 全局一次性配置
# ─────────────────────────────
st.set_page_config(page_title="RAG-SFT System", layout="wide")
load_dotenv()
os.environ["OPENAI_API_BASE"] = "https://api.v3.cm/v1"   # 显式指定

# ─────────────────────────────
# 1. 导入子页面
# ─────────────────────────────

from importlib import import_module

chat_ui       = import_module("mod.chat.streamlit_app")
retriever_ui  = import_module("mod.retriever.streamlit_app")
qgen_ui       = import_module("mod.question_generator.streamlit_app")
ans_ui        = import_module("mod.answer_generator.streamlit_app")
# ─────────────────────────────
# 2. 页面映射
# ─────────────────────────────
PAGES = {
    "💬 General Chat": chat_ui.render,
    "📚 Knowledge Retriever": retriever_ui.render,
    "❓ Question Generator": qgen_ui.render,
    "📝 Answer Generator": ans_ui.render,
    "📊 LangSmith Evaluation": "EVAL",   # 占位，特殊渲染
}

# ─────────────────────────────
# 3. Sidebar 导航
# ─────────────────────────────
st.sidebar.title("🔗 Navigation")
page = st.sidebar.radio(
    "Go to",
    list(PAGES.keys()),
    key="nav_page_select",
)

# ─────────────────────────────
# 4. 页面调度
# ─────────────────────────────
if page == "📊 LangSmith Evaluation":
    # -------------- EVAL 页 --------------
    st.header("📊 LangSmith Evaluation Dashboard")

    from langsmith import Client, evaluate
    ls_client = Client()

    dataset_name = st.text_input(
        "Dataset name on LangSmith",
        value="my_dataset",
        key="eval_dataset_name",
    )
    model_name = st.selectbox(
        "LLM model",
        ["gpt-4.1", "deepseek-r1", "gemini-2.5-pro-preview-05-06"],
        key="eval_model_name",
    )

    if st.button("🚀 Run evaluation", key="eval_run"):
        if not dataset_name.strip():
            st.warning("⚠️ 请先填写数据集名称")
            st.stop()
        with st.spinner("Running evaluation …"):
            # 目标函数：简单调用 answer_generator
            from langchain.docstore.document import Document
            from mod.answer_generator.answer_generator import generate_answer

            def target_fn(example):
                doc = Document(page_content=example["input"])
                return generate_answer(
                    example["instruction"],
                    [doc],
                    model=model_name,
                )

            results = evaluate(
                target_fn,
                dataset_name=dataset_name,
                evaluators=["qa"],
                client=ls_client,
            )
        st.success("✅ Finish!")
        st.dataframe(results, use_container_width=True)
else:
    # -------------- 其它子页面 --------------
    PAGES[page]()               # 调用对应模块的 render()