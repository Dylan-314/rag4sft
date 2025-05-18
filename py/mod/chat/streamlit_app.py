# mod/chat/streamlit_app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from mod.chat.chat_service import ChatService
from mod.chat.models import ChatModel
def render():
    st.title("💬 General Chat")

    # ---- Sidebar ---- #
    model_choice = st.sidebar.selectbox(
        "选择模型",
        [m.value for m in ChatModel],
        key="chat_model"
    )
    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.7, step=0.05,
        key="chat_temperature"
    )

    # ---- Session State ---- #
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---- 显示历史 ---- #
    for msg in st.session_state.chat_history:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        st.markdown(f"{avatar} **{msg['role'].title()}:** {msg['content']}")

    # ---- 输入 ---- #
    user_input = st.text_input(
        "请输入问题",
        key="chat_user_input",
        placeholder="Hello, how are you?"
    )
    if st.button("发送", key="chat_send_btn") and user_input.strip():
        # 1. 记录用户
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        # 2. 调用 ChatService
        chat_svc = ChatService(model=model_choice, temperature=temperature)
        with st.spinner("LLM thinking…"):
            answer = chat_svc.chat(st.session_state.chat_history)
        # 3. 记录助手
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
        st.rerun()  # 刷新显示

if __name__ == "__main__":
     render()