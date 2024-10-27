import re
import os
import requests
import streamlit as st

# from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate


from langchain_core.runnables import RunnableLambda

from langchain_core.runnables import RunnablePassthrough


# 파일 분리 (상수들)
from utils.constant.constant import AI_MODEL, API_KEY_PATTERN

# 파일 분리 (함수들)
from utils.functions.save_env import SaveEnv
from utils.functions.debug import Debug
from utils.functions.chat import ChatMemory, ChatCallbackHandler

# 디버그용
from dotenv import load_dotenv

load_dotenv()

# 디버그용 2
# set_debug(True)

# 해야 할 것
# 파일 모듈화
# 보여주는 로직 변경
# 사이드 바에 conversation 제목들을 보여주고, 클릭하면 메인 페이지에 반영되게
# 통계화면 보여주는거


# backend urls

FILE_UPLOAD_URL = "https://nomad-hsqdoc.onrender.com/uploads/"
USERS_URL = "https://nomad-hsqdoc.onrender.com/api/v1/users/"
CONVERSATIONS_URL = "https://nomad-hsqdoc.onrender.com/api/v1/conversations/"
MESSAGES_URL = "https://nomad-hsqdoc.onrender.com/api/v1/messages/"


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


st.set_page_config(
    page_title="HSQDoc",
    page_icon="📃",
)

st.title("Welcome to HSQDoc!")

if st.session_state["is_login"]:
    st.subheader("Create a Conversation or Choose a Existing Conversation to Start")


class FileController:
    # 파일 임베딩 함수
    @staticmethod
    @st.cache_resource(show_spinner="Embedding file...")
    def embed_file(file_name, file_path):
        # os.makedirs("./.cache/files", exist_ok=True)
        # file_path = f"./.cache/files/{file.name}"
        # with open(file_path, "wb") as f:
        #     f.write(file.read())

        cache_dir = LocalFileStore(f"./.cache/embeddings/open_ai/{file_name}")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n", ".", "?", "!"],
            chunk_size=1000,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["openai_api_key"])
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings, cache_dir
        )
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        return vectorstore.as_retriever()

    # 문서 포맷팅 함수
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(document.page_content for document in docs)


# 사이드바 설정


if st.session_state["jwt"] is None:
    with st.form(key="login_or_register"):
        tab1, tab2 = st.tabs(["login", "Register"])
        with tab1:

            username = st.text_input(
                "Username",
            )
            password = st.text_input(
                "Password",
                type="password",
            )

            login_request = st.form_submit_button(
                "Login",
                disabled=st.session_state.is_login,
            )

            if login_request:
                login_data = {
                    "username": username,
                    "password": password,
                }
                response = requests.post(USERS_URL + "login/", json=login_data)
                if response.status_code == 200:
                    st.session_state.is_login = True
                    token = response.json()["token"]
                    st.session_state.jwt = token
                    st.session_state["username"] = username
                    # 로인 후 rerun 하는걸로 form 안 보이게 하기
                    # 그대신 rerun하면 st.success가 안 보이게 된다: 생기자마자 rerun으로 사라지기 때문
                    # st.success("Welcome! You are logged in!")
                    st.rerun()
                elif response.status_code == 400:
                    error_message = response.json()["error"]
                    st.error(error_message)

        with tab2:
            first_name = st.text_input(
                "Enter Your First Name",
            )
            last_name = st.text_input(
                "Enter Your Last Name",
            )
            email = st.text_input(
                "Enter Your Email",
            )
            gender = st.selectbox(
                "Choose Your Gender",
                ["male", "female"],
            )

            username = st.text_input(
                "Enter Your Username, This is Required",
            )
            password = st.text_input(
                "Enter Your Password, This is Required",
                type="password",
            )
            check_password = st.text_input(
                "Reenter Your Password",
                type="password",
            )
            register_request = st.form_submit_button(
                "Register",
            )
            validate_password = password == check_password
            if not validate_password:
                st.error("Password is Different")

            if register_request and validate_password:
                register_data = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "gender": gender,
                    "username": username,
                    "password": password,
                }
                response = requests.post(USERS_URL, json=register_data)
                if response.status_code == 200:
                    st.success("Register Success! Now You can LogIn")
                else:
                    if response.json()["error"]:
                        st.error(response.json()["error"])
                    else:
                        st.error("Register Fail")
else:

    # 유저의 api key 가져오기
    response = requests.get(
        USERS_URL + "profile/",
        headers={"jwt": st.session_state.jwt},
    )
    if response.status_code == 200:
        openai_api_key = response.json()["openai_api_key"]
        claude_api_key = response.json()["claude_api_key"]
        if openai_api_key != "":
            st.session_state["openai_api_key"] = openai_api_key
            st.session_state["openai_api_key_check"] = True
        if claude_api_key != "":
            st.session_state["claude_api_key"] = claude_api_key
            st.session_state["claude_api_key_check"] = True

    with st.sidebar:
        conversations_data = requests.get(
            CONVERSATIONS_URL,
            headers={"jwt": st.session_state.jwt},
        )
        if conversations_data.status_code == 200:
            conversations = conversations_data.json()
            conversations_options = ["Create Conversation"]
            for conversation in conversations:
                conversations_options.append(conversation["title"])
            chosen_option = st.selectbox(
                "Create or Choose a Conversation", conversations_options
            )
            if chosen_option == "Create Conversation":
                with st.form(key="create_conversation", clear_on_submit=True):
                    new_title = st.text_input(
                        "Write a Title for Your Conversation With AI",
                    )
                    create_conversation_request = st.form_submit_button(
                        "Create Conversation",
                    )

                    if create_conversation_request:
                        response = requests.post(
                            CONVERSATIONS_URL,
                            headers={"jwt": st.session_state.jwt},
                            json={
                                "title": new_title,
                            },
                        )
                        if response.status_code != 200:
                            error_message = response.json()["error"]
                            st.error(error_message)
                        else:
                            st.rerun()

            else:
                chosen_conversation_index = (
                    conversations_options.index(chosen_option) - 1
                )
                chosen_conversation_id = conversations[int(chosen_conversation_index)][
                    "id"
                ]
                st.session_state["messages_url"] = (
                    MESSAGES_URL + str(chosen_conversation_id) + "/"
                )
                st.session_state["conversation_url"] = (
                    CONVERSATIONS_URL + str(chosen_conversation_id) + "/"
                )

                # 과거 대화 기록 가져오기
                if (
                    st.session_state["messages_url"]
                    not in st.session_state["messages"].keys()
                ):
                    st.session_state["messages"][st.session_state["messages_url"]] = []
                    messages_data = requests.get(
                        st.session_state["messages_url"],
                        headers={"jwt": st.session_state.jwt},
                    )
                    if messages_data.status_code == 200:
                        messages = messages_data.json()
                        for message in messages:
                            ChatMemory.save_message(
                                message["message_content"],
                                message["message_role"],
                            )
                    else:
                        st.error("Please Choose Proper Conversation")

                # 과거 업로드한 파일 가져오기
                response = requests.get(
                    st.session_state["conversation_url"],
                    headers={"jwt": st.session_state.jwt},
                )
                if response.status_code == 200:
                    file_name = response.json()["file_name"]
                    st.session_state["file_name"] = file_name
                    st.session_state["file_path"] = f"./.cache/files/{file_name}"
                    if st.session_state["file_name"] != "":
                        st.session_state["file_check"] = True

        else:
            st.error("Please log in")

    # 메인 로직
if st.session_state["is_login"]:
    if (
        st.session_state["openai_api_key_check"]
        and st.session_state["file_check"]
        and st.session_state["openai_model_check"]
    ):
        if chosen_option != "Create Conversation":
            if st.session_state["openai_model"] == AI_MODEL[1]:
                llm = ChatOpenAI(
                    temperature=0.1,
                    streaming=True,
                    callbacks=[ChatCallbackHandler()],
                    model=st.session_state["openai_model"],
                    openai_api_key=st.session_state["openai_api_key"],
                )
                print("you chose openai")
            elif st.session_state["openai_model"] == AI_MODEL[2]:
                llm = ChatAnthropic(
                    temperature=0.1,
                    streaming=True,
                    callbacks=[ChatCallbackHandler()],
                    model=st.session_state["openai_model"],
                    anthropic_api_key=st.session_state["claude_api_key"],
                )
                print("you chose claude")

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are an AI that reads documents for me. Please answer based on the document given below. 
                        Make sure to check every detail inside the document.
                        If the information is not in the document, answer the question with "The required information is not in the document." Never make up answers.
                        Please answer in the questioner's language 
                        
                        Context : {context}
                        """,
                    ),
                    ("human", "{question}"),
                ]
            )
            retriever = (
                FileController.embed_file(
                    st.session_state["file_name"], st.session_state["file_path"]
                )
                if (
                    st.session_state["file_check"]
                    and (st.session_state["file_name"] != "")
                )
                else None
            )
            if retriever:
                ChatMemory.send_message("I'm ready! Ask away!", "ai", save=False)
                ChatMemory.paint_history()
                message = st.chat_input("Ask anything about your file...")

                if message:
                    if (
                        re.match(API_KEY_PATTERN, st.session_state["openai_api_key"])
                        or re.match(API_KEY_PATTERN, st.session_state["claude_api_key"])
                    ) and (st.session_state["openai_model"] in AI_MODEL):
                        ChatMemory.send_message(message, "human")
                        chain = (
                            {
                                "context": retriever
                                | RunnableLambda(FileController.format_docs),
                                "question": RunnablePassthrough(),
                            }
                            | prompt
                            | llm
                        )

                        try:
                            with st.chat_message("ai"):
                                ai_answer = chain.invoke(message)

                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            st.warning(
                                "OPENAI_API_KEY or 모델 선택을 다시 진행해주세요."
                            )
                    else:
                        ChatMemory.send_message(
                            "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요.",
                            "ai",
                        )
            else:
                st.session_state["messages"] = {}
    with st.sidebar:
        if chosen_option != "Create Conversation":
            with st.form(key="update_or_delete_conversation", clear_on_submit=True):
                st.write("Update or Delete Conversation")
                updated_title = st.text_input("Write a New Conversation Title")
                update_request = st.form_submit_button("Update Title")
                delete_request = st.form_submit_button("Delete Conversation")

                if update_request:
                    response = requests.put(
                        st.session_state["conversation_url"],
                        headers={"jwt": st.session_state.jwt},
                        json={
                            "title": updated_title,
                        },
                    )
                    if response.status_code != 200:
                        st.error("Update Failed. Try Again")
                    else:
                        st.rerun()

                if delete_request:
                    response = requests.delete(
                        st.session_state["conversation_url"],
                        headers={"jwt": st.session_state.jwt},
                    )
                    if response.status_code != 204:
                        st.error("Delete Failed. Try Again")
                    else:
                        st.rerun()

            st.divider()
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
                https://github.com/lips85/Nomad_HSQDoc
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
