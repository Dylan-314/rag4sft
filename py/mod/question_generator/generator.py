from __future__ import annotations
from typing import List, Dict, Optional
import json
from openai import OpenAI
from langchain.docstore.document import Document
from .config import OPENAI_API_KEY, OPENAI_API_BASE, DEFAULT_MODEL
from .prompt import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH
import sys, os
from pathlib import Path
from datetime import datetime
from .config import EXPORT_DIR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def _build_messages(chunk: Document, n: int, lang: str) -> list[dict]:
    sys_prompt = SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN
    sys_prompt = sys_prompt.format(n=n)
    user_prompt = f"文本片段：\n\"\"\"\n{chunk.page_content}\n\"\"\""
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

def _save_results(results: List[Dict], export_dir: Path = EXPORT_DIR) -> Path:
    """Save generated QA pairs to a timestamped JSONL file and return the file path."""
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = export_dir / f"qa_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return out_path

def _notify_success(path: Path) -> None:
    """Show a toast in Streamlit if available; otherwise print to stdout."""
    try:
        import streamlit as st
        st.success(f"✅ 生成结果已保存到 {path}")
    except ModuleNotFoundError:
        print(f"[INFO] 生成结果已保存到 {path}")

def generate_questions(
    chunks: List[Document],
    n_per_chunk: int = 2,
    lang: str = "zh",
    model: str = DEFAULT_MODEL,
    total_questions: Optional[int] = None,
    auto_save: bool = True,
    notify: bool = True,
) -> List[Dict]:
    """
    对每个 chunk 调用 LLM 生成问题。
    参数
    ----
    total_questions : int | None
        希望最终返回的总问题数。达到该数量后立即停止并返回。
    返回 JSON-able 列表：{"chunk_id": X, "question": "...", "type": "..."}
    """
    results: List[Dict] = []
    for idx, chunk in enumerate(chunks, 1):
        messages = _build_messages(chunk, n_per_chunk, lang)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        try:
            questions = json.loads(resp.choices[0].message.content)
            for q in questions:
                q["chunk_id"] = idx
                results.append(q)
                if total_questions is not None and len(results) >= total_questions:
                    break
        except Exception:
            # 若模型输出格式不合法，回退为单一问题
            results.append(
                {"chunk_id": idx, "question": resp.choices[0].message.content, "type": "unknown"}
            )
            if total_questions is not None and len(results) >= total_questions:
                break
    # ---- 自动保存 & 成功提示 ----
    if auto_save:
        if total_questions is not None:
            results = results[:total_questions]
        path = _save_results(results)
        if notify:
            _notify_success(path)
    return results