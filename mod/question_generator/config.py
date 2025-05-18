"""模块三配置：统一读取 .env，并显式指定 API URL"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # 从项目根目录读取 .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://api.v3.cm/v1"

# 默认模型（可在 UI 里修改）
DEFAULT_MODEL = "gpt-4.1"

# 每个 chunk 默认生成的问题数
DEFAULT_QUESTIONS_PER_CHUNK = 2

# 数据导出目录
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)