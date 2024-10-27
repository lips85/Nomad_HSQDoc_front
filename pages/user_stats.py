import streamlit as st
import requests

USER_STATS_URL = "http://127.0.0.1:8000/api/v1/users/stats"

st.title("User Stats")


def get_cost(cost_per_1M_tokens):
    cost = round(cost_per_1M_tokens / 1000000, 2)
    if cost < 0.01:
        return "Below $0.01"
    return f"${cost}"


if "jwt" not in st.session_state or st.session_state["jwt"] is None:
    st.error("Please Log In")
else:
    response = requests.get(
        USER_STATS_URL,
        headers={"jwt": st.session_state.jwt},
    )
    if response.status_code != 200:
        st.error("Please Log In")
    else:
        user_stats = response.json()
        user_total_conversations = user_stats["user_total_conversations"]
        user_total_tokens = user_stats["user_total_tokens"]
        user_total_cost = user_stats["user_total_cost"]
        user_total_messages = user_stats["user_total_messages"]
        user_total_files = user_stats["total_files"]
        st.header(f"Total Conversation Number: {len(user_total_conversations)}")
        st.header(f"Total Messages Number: {user_total_messages}")
        st.header(f"Total File Number: {len(user_total_files)}")
        st.header(f"Total Token Number: {user_total_tokens['total_tokens']}")
        st.subheader(
            f"Total OpenAI Token Number: {user_total_tokens['total_tokens_openai']}"
        )
        st.subheader(
            f"Total Claude Token Number: {user_total_tokens['total_tokens_claude']}"
        )
        st.header(
            f"Total Cost: {get_cost(user_total_cost['total_cost_per_1M_tokens'])}"
        )
        st.subheader(
            f"Total OpenAI Cost: {get_cost(user_total_cost['total_cost_openai_per_1M_tokens'])}"
        )
        st.subheader(
            f"Total Claude Cost: {get_cost(user_total_cost['total_cost_claude_per_1M_tokens'])}"
        )

        with st.sidebar:
            conversation_title_list = []
            for conversation in user_total_conversations:
                conversation_title_list.append(conversation["title"])
            selected_conversation_title = st.selectbox(
                "Check Usage of Each Conversation",
                conversation_title_list,
                index=None,
                placeholder="Select Conversation...",
            )
            if selected_conversation_title:
                selected_conversation_index = conversation_title_list.index(
                    selected_conversation_title
                )
                selected_conversation = user_total_conversations[
                    selected_conversation_index
                ]

                st.header(
                    f"Total Messages Number: {selected_conversation['total_messages']}",
                )
                st.header(
                    f"Total Token Number: {selected_conversation['total_tokens']}",
                )
                st.header(
                    f"Total Cost Number: {get_cost(selected_conversation['total_cost_per_1M_tokens'])}",
                )
                st.header(
                    f"File Name: {selected_conversation['file_name']}",
                )

                st.header("OpenAI Usage")
                st.subheader(
                    f"Total Token: {selected_conversation['models']['openai']['tokens']}",
                )
                st.subheader(
                    f"Total Cost: {get_cost(selected_conversation['models']['openai']['cost_per_1M_tokens'])}",
                )

                st.header("Claude Usage")
                st.subheader(
                    f"Total Token: {selected_conversation['models']['claude']['tokens']}",
                )
                st.subheader(
                    f"Total Cost: {get_cost(selected_conversation['models']['claude']['cost_per_1M_tokens'])}",
                )
