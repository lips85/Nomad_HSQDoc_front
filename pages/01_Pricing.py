import streamlit as st
import requests
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import random

load_dotenv()

# backend urls
BACKEND_URL = os.getenv("BACKEND_URL")
USER_STATS_URL = BACKEND_URL + "api/v1/users/stats"

st.set_page_config(layout="wide", page_title="사용량: 비용")

st.title("사용량: 비용")


def get_cost(cost_per_1M_tokens):
    cost = cost_per_1M_tokens / 1000000
    return f"${cost:.4f}"  # 소수점 4자리까지 표시


# 파스텔 톤 색상 정의
pastel_colors = [
    "#FFB3BA",
    "#BAFFC9",
    "#BAE1FF",
    "#FFFFBA",
    "#FFDFBA",
    "#E0BBE4",
    "#B5EAD7",
    "#C7CEEA",
    "#FFDAC1",
    "#FF9AA2",
    "#B5B9FF",
    "#AFF8DB",
]

if "jwt" not in st.session_state or st.session_state["jwt"] is None:
    st.error("로그인해 주세요")
else:
    response = requests.get(
        USER_STATS_URL,
        headers={"jwt": st.session_state.jwt},
    )
    if response.status_code != 200:
        st.error("로그인해 주세요")
    else:
        user_stats = response.json()
        user_total_conversations = user_stats["user_total_conversations"]
        user_total_tokens = user_stats["user_total_tokens"]
        user_total_cost = user_stats["user_total_cost"]

        col1, col2 = st.columns([2, 1])

        with col1:
            with st.form("platform_tokens"):
                st.subheader("1. 플랫폼별 토큰 사용량")
                platforms = ["OpenAI", "Claude"]
                token_counts = [
                    user_total_tokens["total_tokens_openai"],
                    user_total_tokens["total_tokens_claude"],
                ]

                platform_tokens = go.Figure(
                    data=[
                        go.Bar(
                            x=platforms,
                            y=token_counts,
                            text=token_counts,
                            textposition="auto",
                            width=0.25,
                            marker_color=pastel_colors[
                                :2
                            ],  # 첫 두 개의 파스텔 색상 사용
                        )
                    ]
                )
                platform_tokens.update_layout(
                    xaxis_title="플랫폼",
                    yaxis_title="토큰 수",
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=50),
                    bargap=0.5,
                )
                st.plotly_chart(platform_tokens, use_container_width=True)
                st.form_submit_button("새로고침")

            with st.form("conversation_usage"):
                st.subheader("2. 대화별 토큰 사용량")
                conversation_titles = [
                    f"{i+1}. {conv['title']}"
                    for i, conv in enumerate(user_total_conversations)
                ]
                conversation_tokens = [
                    conv["total_tokens"] for conv in user_total_conversations
                ]

                # 대화 수에 맞춰 색상 선택 (색상이 부족하면 반복 사용)
                colors = (
                    pastel_colors * (len(conversation_titles) // len(pastel_colors) + 1)
                )[: len(conversation_titles)]
                random.shuffle(colors)  # 색상을 무작위로 섞음

                conversation_usage = go.Figure(
                    data=[
                        go.Bar(
                            x=conversation_titles,
                            y=conversation_tokens,
                            text=conversation_tokens,
                            textposition="auto",
                            width=0.25,
                            marker_color=colors,
                        )
                    ]
                )
                conversation_usage.update_layout(
                    xaxis_title="대화 제목",
                    yaxis_title="토큰 수",
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=50),
                    xaxis_tickangle=-45,
                    bargap=0.2,
                )
                st.plotly_chart(conversation_usage, use_container_width=True)
                st.form_submit_button("새로고침")

        with col2:
            with st.form("platform_tokens_summary"):
                st.subheader("3. 플랫폼별 토큰 사용량")
                st.metric(
                    "OpenAI 토큰 수", f"{user_total_tokens['total_tokens_openai']:,}"
                )
                st.metric(
                    "Claude 토큰 수", f"{user_total_tokens['total_tokens_claude']:,}"
                )
                st.form_submit_button("새로고침")

            with st.form("platform_costs"):
                st.subheader("4. 플랫폼별 비용")
                st.metric(
                    "OpenAI 비용",
                    get_cost(user_total_cost["total_cost_openai_per_1M_tokens"]),
                )
                st.metric(
                    "Claude 비용",
                    get_cost(user_total_cost["total_cost_claude_per_1M_tokens"]),
                )
                st.form_submit_button("새로고침")

            with st.form("total_summary"):
                st.subheader("5. 총계")
                st.metric("총 토큰 수", f"{user_total_tokens['total_tokens']:,}")
                st.metric(
                    "총 비용", get_cost(user_total_cost["total_cost_per_1M_tokens"])
                )
                st.form_submit_button("새로고침")

        # 사이드바에 대화 선택 옵션 추가
        with st.sidebar:
            with st.form("conversation_details"):
                st.subheader("6. 각 대화의 사용량 확인")
                conversation_title_list = [
                    f"{i+1}. {conv['title']}"
                    for i, conv in enumerate(user_total_conversations)
                ]
                selected_conversation_title = st.selectbox(
                    "대화 선택",
                    conversation_title_list,
                    index=None,
                    placeholder="대화 선택...",
                )
                if st.form_submit_button("확인"):
                    if selected_conversation_title:
                        selected_conversation = user_total_conversations[
                            int(selected_conversation_title.split(".")[0]) - 1
                        ]

                        st.header(
                            f"총 메시지 수: {selected_conversation['total_messages']}"
                        )
                        st.header(
                            f"총 토큰 수: {selected_conversation['total_tokens']:,}"
                        )
                        st.header(
                            f"총 비용: {get_cost(selected_conversation['total_cost_per_1M_tokens'])}"
                        )
                        st.header(f"파일 이름: {selected_conversation['file_name']}")

                        st.header("OpenAI 사용량")
                        st.subheader(
                            f"총 토큰: {selected_conversation['models']['openai']['tokens']:,}"
                        )
                        st.subheader(
                            f"총 비용: {get_cost(selected_conversation['models']['openai']['cost_per_1M_tokens'])}"
                        )

                        st.header("Claude 사용량")
                        st.subheader(
                            f"총 토큰: {selected_conversation['models']['claude']['tokens']:,}"
                        )
                        st.subheader(
                            f"총 비용: {get_cost(selected_conversation['models']['claude']['cost_per_1M_tokens'])}"
                        )
