# from dotenv import load_dotenv
# load_dotenv()

import streamlit as st
# from utils import print_messages, StreamHandler
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler

import uuid


st.set_page_config(page_title="🌈 챗GPT 3.5")
st.title("🌈 챗GPT 3.5")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = dict()

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())


# with st.sidebar:
    # session_id = st.text_input("Session ID", value="abc123")

    # clear_btn = st.button("대화기록 초기화")
    # if clear_btn:
        # st.session_state["messages"] = []
        # st.experimental_rerun()




class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def print_messages():
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)


print_messages()


def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in st.session_state["store"] :
        st.session_state["store"] [session_ids] = ChatMessageHistory()
    return st.session_state["store"] [session_ids]





if user_input:= st.chat_input("질문을 입력하세요."):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))





    # AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())



        # 모델 생성
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])

        # 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "질문에 답변해 주세요.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | llm
        # response = chain.invoke({"question": user_input})

        chain_with_memory = (
            RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
        )

        response = chain_with_memory.invoke(
            {"question": user_input},
            # 세션 관리
            config = {"configurable": {"session_id": st.session_state["session_id"]}},
        )




        st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))
