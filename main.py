from langchain_openai import ChatOpenAI
import os
import streamlit as st
import uuid
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler


st.title("건강보험 AI 동호회 챗봇")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message[0]).write(chat_message[1])


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "store" not in st.session_state:
    st.session_state["store"] = dict()

def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    print(session_ids)
    if session_ids not in st.session_state["store"] :
        st.session_state["store"] [session_ids] = ChatMessageHistory()
    return st.session_state["store"] [session_ids]





class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

        


question = st.chat_input("질문을 입력하세요.")
if question:
    st.chat_message("user").write(question)


    with st.chat_message("assistant"):

        stream_handler = StreamHandler(st.empty())

    
        key = os.getenv('OPENAI_KEY')
        llm = ChatOpenAI(openai_api_key=key, streaming=True, callbacks=[stream_handler])


        prompt = ChatPromptTemplate.from_messages( [ ("system", "질문에 답변해 주세요."),
                                                      MessagesPlaceholder(variable_name="history"),
                                                    ("human", "{question}") ]  )

        chain = prompt | llm

        chain_with_memory = ( RunnableWithMessageHistory(chain,
                                                         get_session_history,
                                                         input_messages_key="question",
                                                         history_messages_key="history",))


        answer = chain_with_memory.invoke({"question": question},
                                          config = {"configurable": {"session_id": st.session_state["session_id"]}})

        answer = answer.content

        st.session_state["messages"].append(("user", question))
        st.session_state["messages"].append(("assistant", answer))


