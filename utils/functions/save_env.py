import re
import requests
import streamlit as st
from utils.constant.constant import API_KEY_PATTERN
import os
from dotenv import load_dotenv

load_dotenv()

# backend urls
BACKEND_URL = os.getenv("BACKEND_URL")
USER_PROFILE_URL = BACKEND_URL + "api/v1/users/profile/"


class SaveEnv:
    @staticmethod
    def save_openai_api_key():

        st.session_state["openai_api_key_check"] = bool(
            re.match(API_KEY_PATTERN, st.session_state["openai_api_key"])
        )
        if st.session_state["openai_api_key_check"]:
            response = requests.put(
                USER_PROFILE_URL,
                headers={"jwt": st.session_state.jwt},
                json={
                    "openai_api_key": st.session_state["openai_api_key"],
                },
            )
            if response.status_code != 200:
                st.error("Failed to save API key")

    @staticmethod
    def save_claude_api_key():

        st.session_state["claude_api_key_check"] = bool(
            re.match(API_KEY_PATTERN, st.session_state["claude_api_key"])
        )
        if st.session_state["claude_api_key_check"]:

            response = requests.put(
                USER_PROFILE_URL,
                headers={"jwt": st.session_state.jwt},
                json={
                    "claude_api_key": st.session_state["claude_api_key"],
                },
            )
            if response.status_code != 200:
                st.error("Failed to save API key")

    @staticmethod
    def save_file():
        st.session_state["file_check"] = st.session_state.file is not None

    @staticmethod
    def save_openai_model():
        st.session_state["openai_model_check"] = (
            st.session_state["openai_model"] != "선택해주세요"
        )

    @staticmethod
    def save_url():
        if st.session_state["url"]:
            st.session_state["url_check"] = True
            st.session_state["url_name"] = (
                st.session_state["url"].split("://")[1].replace("/", "_")
            )
        else:
            st.session_state["url_check"] = False
            st.session_state["url_name"] = None
