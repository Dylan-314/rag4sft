import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from .generator import generate_questions
from mod.retriever.manager import KnowledgeBaseManager

def render():
    st.title("❓ Question Generator")

    existing_kbs = KnowledgeBaseManager.list_kbs()
    if not existing_kbs:
        st.sidebar.warning("暂无知识库，请先在文档入库中创建。")
        st.stop()
    kb_name = st.sidebar.selectbox("选择知识库", options=existing_kbs, key="qgen_kb_name")
    n_per_chunk = st.sidebar.slider(
        "每个 Chunk 生成问题数",
        1, 5, 2,
        key="qgen_n_per_chunk"
    )
    lang = st.sidebar.radio(
        "问题语言",
        ["zh", "en"],
        horizontal=True,
        key="qgen_lang"
    )
    total_q = st.sidebar.number_input(
        "总问题数",
        min_value=1,
        value=20,
        step=1,
        key="qgen_total_q"
    )

    query = st.text_input(
        "初始检索问题（用于获取相关片段）",
        key="qgen_query"
    )
    if st.button("🎯 生成问题", key="qgen_generate_btn"):
        st.info(
            f"📝 将从“{kb_name}”中基于“{query}”每段生成 {n_per_chunk} 条问题 ({lang})，"
            f"目标总数 {total_q} 条"
        )

        # ---- 主流程：检索 → 生成 → 自动保存 ----
        with st.spinner("🔍 正在检索并生成问题，请稍候..."):
            # 1) 依据知识库名称实例化检索器（路径按你项目实际情况调整）
            kbm = KnowledgeBaseManager(kb_name)
            # 2) 获取与初始查询相关的文本片段（top_k 可按需调整）
            chunks = kbm.search(query, top_k=10)

            # 3) 生成问题；auto_save=True 会把结果写入 exports/ 目录
            results = generate_questions(
                chunks,
                n_per_chunk=n_per_chunk,
                lang=lang,
                total_questions=total_q,
                auto_save=True,
                notify=True,
            )

        st.success(f"✅ 已生成 {len(results)} 条问题，并已保存到 exports/ 目录")