
from langchain.chat_models import ChatOpenAI
import streamlit as st


chat_model = ChatOpenAI(openai_api_key="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")


st.title('인공지능 시인')

content = st.text_input('시의 주제를 제시해주세요.')

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        result = chat_model.predict(content + "에 대해 시를 써줘")
        st.write(result)



















