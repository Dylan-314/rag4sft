from enum import Enum

class ChatModel(str, Enum):
    GPT_41 = "gpt-4.1"
    DEEPSEEK_R1 = "deepseek-r1"
    GEMINI_25 = "gemini-2.5-pro-preview-05-06"

DEFAULT_MODEL = ChatModel.GPT_41