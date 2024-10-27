import os
import requests
import streamlit as st

from dotenv import load_dotenv


from utils.constant.constant import AI_MODEL
from utils.functions.save_env import SaveEnv
from utils.functions.debug import Debug


load_dotenv()

st.title("Settings(API KEY, Model, file)")

BACKEND_URL = os.getenv("BACKEND_URL")
FILE_UPLOAD_URL = BACKEND_URL + "uploads/"
USERS_URL = BACKEND_URL + "api/v1/users/"


# front 손보기
for key, default in [
    # jwt token을 담기 위한 session state
    ("jwt", None),
    # 로그인 했는지를 나타내는 session state
    ("is_login", False),
    # 현재 유저가 보고 있는 대화방의 대화 기록 url을 나타내는 session state
    ("messages_url", None),
    # 현재 유저가 보고 있는 대화방의 url을 나타내는 session state
    ("conversation_url", None),
    # 업로드한 파일 이름을 나타내는 session state
    ("file_name", None),
    # 업로드한 파일 경로을 나타내는 session state
    ("file_path", None),
    # langchain
    ("messages", {}),
    ("openai_api_key", None),
    ("claude_api_key", None),
    ("openai_api_key_check", False),
    ("claude_api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("file_check", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def clear_session_keys():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)


if "jwt" not in st.session_state or st.session_state["jwt"] is None:
    st.error("Please Log In")
else:
    with st.form("upload_file"):
        uploaded_file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
            key="file",
        )
        upload_request = st.form_submit_button(
            "Upload File",
            on_click=SaveEnv.save_file,
        )
        if upload_request:
            # 파일을 장에 저장

            os.makedirs("./.cache/files", exist_ok=True)
            st.session_state["file_path"] = f"./.cache/files/{uploaded_file.name}"
            with open(st.session_state["file_path"], "wb") as f:
                f.write(uploaded_file.read())

            st.session_state["file_name"] = uploaded_file.name
            uploaded_file_path_for_django = FILE_UPLOAD_URL + uploaded_file.name
            response = requests.put(
                st.session_state["conversation_url"],
                headers={"jwt": st.session_state.jwt},
                data={
                    "file_name": st.session_state["file_name"],
                    "file_url": uploaded_file_path_for_django,
                },
            )
            if st.session_state["file_check"]:
                st.success("😄문서가 업로드되었습니다.😄")
            else:
                st.warning("문서를 업로드해주세요.")
    st.divider()

    st.text_input(
        "OpenAI API_KEY 입력",
        placeholder="sk-...",
        on_change=SaveEnv.save_openai_api_key,
        key="openai_api_key",
    )

    if st.session_state["openai_api_key_check"]:
        st.success("😄OpenAI API_KEY가 저장되었습니다.😄")
    else:
        st.warning("OpenAI API_KEY를 넣어주세요.")

    st.button(
        "hary의 OpenAI API_KEY (디버그용)",
        on_click=Debug.my_openai_api_key,
        key="my_openai_key_button",
    )

    st.text_input(
        "Anthropic API_KEY 입력",
        placeholder="sk-...",
        on_change=SaveEnv.save_claude_api_key,
        key="claude_api_key",
    )

    if st.session_state["claude_api_key_check"]:
        st.success("😄Anthropic API_KEY가 저장되었습니다.😄")
    else:
        st.warning("Anthropic API_KEY를 넣어주세요.")

    st.button(
        "hary의 Anthropic API_KEY (디버그용)",
        on_click=Debug.my_anthropic_api_key,
        key="my_anthropic_key_button",
    )
    st.divider()
    st.selectbox(
        "Model을 골라주세요.",
        options=AI_MODEL,
        on_change=SaveEnv.save_openai_model,
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄모델이 선택되었니다.😄")
    else:
        st.warning("모델을 선택해주세요.")
    st.divider()

    st.write(
        """
            Made by hary, seedjin298.
            
            Github
            https://github.com/lips85/Nomad_HSQDoc_backend
            https://github.com/lips85/Nomad_HSQDoc_frontend
            """
    )
    st.divider()
    st.write("Click to LogOut")
    logout_request = st.button(
        "LogOut",
        disabled=not st.session_state.is_login,
    )
    if logout_request:
        response = requests.post(
            USERS_URL + "logout/",
            headers={"jwt": st.session_state.jwt},
        )
        if response.status_code == 200:
            clear_session_keys()
            # 로그아웃 후 rerun -> 바로 로그인 form이 나타남
            # st.success("LogOut Success!")
            st.rerun()
        else:
            st.error("Failed to LogOut")
