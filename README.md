# rag4sft
# **rag4sft**

*A Retrieval‑Augmented Generation pipeline for automatically creating high‑quality Supervised Fine‑Tuning (SFT) datasets from your domain documents.*

---

## ✨ Key Features

| Module              | Folder          | Description                                                                                                       |
| ------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| General Chat        | `mod/chat`      | Multi‑model chat (GPT‑4.1, Deepseek‑r1, Gemini‑2.5) with or without retrieval; Streamlit UI & file upload support |
| Knowledge Retriever | `mod/retriever` | Document ingestion → chunking → embedding (OpenAI) → vector DB (Chroma) → similarity search                       |
| Question Generator  | `mod/qgen`      | Generates domain‑specific Q‑A pairs from retrieved context                                                        |
| Answer Synthesizer  | `mod/ans`       | Normalises & diversifies answers; adds rationales/metadata if required                                            |
| Data Export         | `mod/export`    | Writes SFT records to JSON/JSONL ready for model fine‑tuning                                                      |

*Fully modular:* each component can be invoked from the UI **or** imported as a Python package.

---

## 📐 Architecture

```mermaid
graph LR
    A[Domain Docs] --> B[Loader (LlamaHub)]
    B --> C[Structured Parse (llama‑parse)]
    C --> D[Chunking (TokenTextSplitter)]
    D --> E[Embedding (OpenAI text‑embedding‑3‑small)]
    E --> F[Chroma Vector DB]
    F -->|Top‑k| G[Retriever]
    G --> H[LLM Generator]
    H --> I[Answer Synthesiser]
    I --> J[Export SFT JSON]
```

---

## 🗂️ Folder Structure (simplified)

```text
rag4sft/
├── mod/
│   ├── chat/
│   ├── retriever/
│   ├── qgen/
│   ├── ans/
│   └── export/
├── data/            # sample docs & demo outputs
├── requirements.txt
└── README.md        # you are here
```

> **Note**  All former `modules/` paths are now **`mod/`** (see commit `rename‑modules‑to‑mod`). Sidebar widgets use unique keys with prefixes `chat_`, `retr_`, `qgen_`.

---

## 🚀 Quickstart

### 1. Prerequisites

* Python ≥ 3.10 (conda or venv recommended)
* An **OpenAI‑compatible** endpoint (e.g. `https://api.v3.cm/v1`)

### 2. Installation

```bash
# clone repo
git clone https://github.com/Dylan-314/rag4sft.git
cd rag4sft

# create env
conda create -n rag4sft python=3.10 -y
conda activate rag4sft

# install deps
pip install -r requirements.txt
```

### 3. Configuration

Add your keys to a `.env` file **or** export them:

```bash
export OPENAI_API_KEY="sk‑..."
export OPENAI_API_BASE="https://api.v3.cm/v1"  # must be explicit per project spec
```

### 4. Run Streamlit UI

```bash
streamlit run mod/chat/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), choose a model, upload docs, and start generating!

### 5. CLI Example

```bash
python -m mod.retriever.cli \
  --input docs/*.pdf \
  --top_k 5 \
  --out data/sft_samples.jsonl
```

---

## 🛠️ Advanced Configuration

| Env Var      | Purpose                  | Default    |
| ------------ | ------------------------ | ---------- |
| `RAG_TOP_K`  | Similarity search depth  | `3`        |
| `RAG_TEMP`   | LLM sampling temperature | `0.7`      |
| `CHROMA_DIR` | Vector store location    | `.chroma/` |

All variables can also be passed as CLI flags (see `--help`).

---

## 📊 Evaluation

We recommend **LangSmith** + **RAGAS** for automated assessment (accuracy, faithfulness, latency). Sample notebooks are under `notebooks/`.

---

## 🤝 Contributing

1. Fork the repo & create a branch, e.g. `feat/my-awesome-idea`
2. Commit with conventional messages
3. Open a PR and describe your changes

All code must pass `ruff` + `mypy` + unit tests (`pytest`).

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

Thanks to the LangChain, OpenAI, and Chroma communities for the amazing tooling that powers **rag4sft**.
