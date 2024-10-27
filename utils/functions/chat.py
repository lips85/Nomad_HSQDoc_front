import streamlit as st
import requests
from langchain_core.callbacks import BaseCallbackHandler


class ChatMemory:
    @staticmethod
    def save_message_db(message, role, token=0):
        response = requests.post(
            st.session_state["messages_url"],
            headers={"jwt": st.session_state.jwt},
            json={
                "message_role": role,
                "message_content": message,
                "model": st.session_state["openai_model"],
                "token": token,
            },
        )
        if response.status_code != 200:
            st.error("Failed to Save Message")

    @staticmethod
    def save_message(message, role):
        st.session_state["messages"][st.session_state["messages_url"]].append(
            {"message": message, "role": role}
        )

    @staticmethod
    def send_message(message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            ChatMemory.save_message(message, role)
            ChatMemory.save_message_db(message, role)

    @staticmethod
    def paint_history():
        if st.session_state["messages_url"] in st.session_state["messages"].keys():
            for message in st.session_state["messages"][
                st.session_state["messages_url"]
            ]:
                ChatMemory.send_message(message["message"], message["role"], save=False)


# 콜백 핸들러 클래스 정의
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None
        self.total_token = 0

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        ChatMemory.save_message(self.message, "ai")
        ChatMemory.save_message_db(self.message, "ai", self.total_token)

    def on_llm_new_token(self, token, *args, **kwargs):
        print(token)
        self.message += token
        self.message_box.markdown(self.message)
        self.total_token += 1
