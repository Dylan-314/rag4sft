from pathlib import Path
import os

# —— 基础环境 —— #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.v3.cm/v1"
LLAMA_CLOUD_API_KEY = "llx-hmWJsFzFgR2q0avW9FfqH27w7qZhuqtsUZPfjpWlAdea7ETU"
# —— 向量配置 —— #
EMBED_MODEL = "text-embedding-3-large"
PERSIST_ROOT = Path(__file__).resolve().parent / ".chroma_db"
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)

# —— 分块参数 —— #
SPLIT_CHUNK_SIZE = 1024
SPLIT_CHUNK_OVERLAP = 128