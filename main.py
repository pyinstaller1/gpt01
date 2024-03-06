
from langchain.chat_models import ChatOpenAI
import streamlit as st


chat_model = ChatOpenAI(openai_api_key="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")


st.title('건강보험 GPT')

content = st.text_input('오픈AI의 GPT4.0 LLM 기반 챗봇')

if st.button('질문하기'):
    with st.spinner('답변 작성 중...'):
        result = chat_model.predict(content)
        st.write(result)



















