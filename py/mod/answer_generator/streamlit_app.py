import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from mod.retriever.manager import KnowledgeBaseManager
from mod.question_generator.generator import generate_questions
from mod.answer_generator.answer_generator import (
    generate_answer, make_sample, validate_sample, export_jsonl
)

def render():
    st.title("📝 Answer Generator")

    # ---- Sidebar ---- #
    kb_name = st.sidebar.text_input(
        "Knowledge Base", value="default", key="ans_kb_name"
    )
    model = st.sidebar.selectbox(
        "LLM Model",
        ["gpt-4.1", "deepseek-r1", "gemini-2.5-pro-preview-05-06"],
        key="ans_model"
    )
    top_k = st.sidebar.slider("检索 Top K", 1, 10, 4, key="ans_topk")

    # ---- 主界面 ---- #
    question = st.text_input(
        "输入问题（留空则自动生成）",
        key="ans_question"
    )

    if st.button("🚀 生成回答", key="ans_run"):
        kb = KnowledgeBaseManager(kb_name)

        # 1. 若问题为空，自动生成
        if not question.strip():
            st.info("未输入问题，系统将自动生成…")
            chunks = [doc for doc, _ in kb.retrieve("overview", top_k=1)]
            question = generate_questions(chunks, 1, "zh")[0]["question"]
            st.write(f"💡 自动生成问题：**{question}**")

        # 2. 检索上下文
        hits = kb.retrieve(question, top_k=top_k)
        ctx_docs = [doc for doc, _ in hits]
        ctx_text = "\n\n".join(d.page_content for d in ctx_docs)

        # 3. 生成回答
        with st.spinner("LLM 正在回答…"):
            answer = generate_answer(question, ctx_docs, model=model)

        # 4. 展示结果
        st.markdown("### 🤔 Question")
        st.write(question)
        st.markdown("### 📚 Context (Top K)")
        for i, d in enumerate(ctx_docs, 1):
            st.markdown(f"- **Chunk {i}:** {d.page_content[:300]}…")
        st.markdown("### 💡 Answer")
        st.write(answer)

        # 5. 打包样本 + 校验 + 导出
        sample = make_sample(question, ctx_text, answer)
        if validate_sample(sample):
            st.success("✅ 样本格式通过")
            if st.button("💾 导出 JSONL", key="ans_export"):
                path = export_jsonl([sample])
                st.success(f"已保存至 {path}")
        else:
            st.error("❌ 样本格式非法，未导出")