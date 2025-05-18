import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from mod.retriever.manager import KnowledgeBaseManager   # 统一管理知识库

def render() -> None:
    st.title("📚 Knowledge Retriever")

    tab_ingest, tab_search, tab_viz = st.tabs(
        ["📤 文档入库", "🔍 检索", "📊 Embedding Viz"]
    )

    # ========================== 文档入库 ========================== #
    with tab_ingest:
        st.subheader("文档入库")
        kb_name = st.text_input("知识库名称", value="default", key="retr_kb_name")

        uploaded = st.file_uploader(
            "上传文档",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            key="retr_file_upload"
        )

        if st.button("🚀 开始入库", key="retr_ingest_btn"):
            # TODO：将 uploaded 文件写入 kb_name 知识库
            st.success(f"✅ 已将 {len(uploaded)} 个文件入库（{kb_name}）")

    # ========================== 检索 ========================== #
    with tab_search:
        st.subheader("混合检索")

        existing_kbs = KnowledgeBaseManager.list_kbs()
        if not existing_kbs:
            st.warning("暂无知识库，请先在“文档入库”中新建并上传文档。")
            st.stop()

        kb_name = st.selectbox("选择知识库", options=existing_kbs, key="retr_kb_search")

        query = st.text_input("输入检索问题", key="retr_query")
        top_k = st.slider("Top K", 1, 10, 4, key="retr_topk")
        bm25_weight = st.slider(
            "BM25 权重 (0 = 仅向量检索, 1 = 仅 BM25)",
            0.0, 1.0, 0.5, step=0.05,
            key="retr_bm25_weight"
        )

        if st.button("🔍 执行检索", key="retr_search_btn"):
            kbm = KnowledgeBaseManager(kb_name)

            # 直接调用你的 HybridRetriever.query
            results = kbm.search(
                query,
                top_k=top_k,
                bm25_weight=bm25_weight,
                return_scores=True    # 必须返回 (Document, score)
            )

            if not results:
                st.warning("未找到任何相关文档。")
            else:
                for idx, (doc, score) in enumerate(results, 1):
                    st.markdown(f"**{idx}. 相似度: {score:.4f}**")
                    st.write(doc.page_content)
                    st.divider()

    # ========================== 嵌入可视化 ========================== #
    with tab_viz:
        st.subheader("嵌入可视化 (UMAP)")

        existing_kbs = KnowledgeBaseManager.list_kbs()
        if not existing_kbs:
            st.warning("暂无知识库，请先在“文档入库”中新建并上传文档。")
            st.stop()

        kb_name = st.selectbox("选择知识库", options=existing_kbs, key="retr_kb_viz")

        if st.button("🔄 加载并绘图", key="retr_viz_btn"):
            import umap
            import pandas as pd
            import plotly.express as px

            st.info("正在加载嵌入数据并执行 UMAP...")

            kbm = KnowledgeBaseManager(kb_name)
            # —— 关键修改 —— #
            # 直接从底层向量库获取 embeddings 与元数据
            raw = kbm.retriever.vs.get(include=["embeddings", "metadatas"])
            embeddings = raw["embeddings"]            # List[List[float]]
            labels     = [md.get("source", "unknown") for md in raw["metadatas"]]

            # UMAP 3D 降维
            reducer      = umap.UMAP(n_components=3, random_state=42)
            embedding_3d = reducer.fit_transform(embeddings)

            df = pd.DataFrame({
                "x": embedding_3d[:, 0],
                "y": embedding_3d[:, 1],
                "z": embedding_3d[:, 2],
                "label": labels
            })

            fig = px.scatter_3d(
                df,
                x="x", y="y", z="z",
                color=df["label"].astype(str),
                title="3D UMAP Embedding Visualization"
            )
            st.plotly_chart(fig)