from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# ✅ 从 .env 中读取变量
load_dotenv()

def get_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://api.v3.cm/v1",
    )