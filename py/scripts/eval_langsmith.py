"""
运行方法:
  $ python scripts/eval_langsmith.py

要求:
  - .env 已设置 LANGSMITH_API_KEY / LANGCHAIN_TRACING_V2=true / LANGCHAIN_ENDPOINT
  - 数据集 "RAG_QA_Test" 已通过 UI 或 upload_dataset.py 创建
"""


from langsmith import evaluate, RunEvalConfig
from mod.rag.chain import rag_chain   # 主链
import os

DATASET_NAME = "RAG_QA_Test"
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "rag_baseline")

eval_cfg = RunEvalConfig(
    evaluators=[
        "retrieval_precision",
        "faithfulness",
        "answer_relevance",
    ]
)

if __name__ == "__main__":
    results = evaluate(
        dataset_name=DATASET_NAME,
        func=rag_chain,
        evaluation=eval_cfg,
        project_name=PROJECT_NAME,
    )
    agg = results["aggregated"]
    print(f"✅ 评估完成 | faithfulness={agg['faithfulness']:.3f} "
          f"| retrieval_precision={agg['retrieval_precision']:.3f}")