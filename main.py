
from langchain.chat_models import ChatOpenAI
import streamlit as st


chat_model = ChatOpenAI(openai_api_key="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")


st.title('GPT 3.5 연동 챗봇')

content = st.text_input('GPT 3.5 LLM 기반 챗봇... 질문하기 버튼을 누르세요.')

if st.button('질문하기'):
    with st.spinner('답변 작성 중...'):
        result = chat_model.predict(content)
        st.write(result)



















