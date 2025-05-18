"""Few-shot Prompt 模版"""

SYSTEM_PROMPT_ZH = """\
你是一位专业的提问生成器，请根据给定的文本片段，为下游知识检索和问答任务生成 {n} 个**覆盖关键细节**、**多样性高**的中文问题。\
输出 JSON 数组，每个元素形如：{{"question": "xxx", "type": "background"}}。不要添加额外字段或解释。
"""

SYSTEM_PROMPT_EN = SYSTEM_PROMPT_ZH.replace("中文", "英文").replace("Chinese", "English")