from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from openai import OpenAI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

load_dotenv()   # 读取 .env
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.v3.cm/v1",
)

EXPORT_DIR = Path(__file__).parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer ONLY with the given context; if unsure, say 'I don't know'."
)

def generate_answer(question: str, ctx_docs: list[Document], model: str) -> str:
    """调用指定模型，返回回答字符串"""
    context = "\n\n".join(d.page_content for d in ctx_docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ---------- SFT 样本工具 ---------- #
def make_sample(question: str, context: str, answer: str) -> dict:
    return {"instruction": question, "input": context, "output": answer}

def validate_sample(s: dict) -> bool:
    return all(k in s and isinstance(s[k], str) and s[k].strip()
               for k in ("instruction", "input", "output"))

def export_jsonl(samples: list[dict], filename: str | None = None) -> Path:
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sft_{ts}.jsonl"
    path = EXPORT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return path