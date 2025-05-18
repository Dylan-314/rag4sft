
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import logging, os
from dotenv import load_dotenv
from openai import OpenAI
from mod.chat.models import ChatModel, DEFAULT_MODEL

load_dotenv()  # 读取 .env

class ChatService:
    """调用统一 OpenAI 兼容端点的聊天服务"""

    def __init__(
        self,
        model: str | ChatModel = DEFAULT_MODEL,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> None:
        self.model: ChatModel = (
            ChatModel(model) if not isinstance(model, ChatModel) else model
        )
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self._logger = logging.getLogger(__name__)

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.v3.cm/v1",
        )

    def chat(self, history: list[dict[str, str]]) -> str:
        messages = [{"role": "system", "content": self.system_prompt}] + history
        resp = self.client.chat.completions.create(
            model=self.model.value,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()