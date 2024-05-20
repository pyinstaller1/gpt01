from langchain_openai import ChatOpenAI
import os
import streamlit as st



import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(page_title="챗봇", page_icon=":robot_face:")
    st.title(":red[NHIS]&nbsp;_챗봇_  &nbsp;&nbsp;&nbsp; :robot_face:")
    st.text("PDF 파일을 등록하시고 질문하세요.")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    with st.sidebar:
        uploaded_files = st.file_uploader("PDF 자료를 여기에 첨부하세요.", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("첨부된 파일 등록")

    if  len(uploaded_files) != 0:

        if process:   # 첨부파일 등록 버튼을 누른 경우
            files_text = get_text(uploaded_files)            # PDF 파일 받기
            message_info = st.info("파일을 벡터 DB에 저장하고 있습니다.")
            text_chunks = get_text_chunks(files_text)      # 청크 문장 단위로 나누기
            vetorestore = get_vectorstore(text_chunks)   # 벡터 DB 저장
            key = os.getenv('OPENAI_KEY')
            st.session_state.conversation = get_conversation_chain(vetorestore, key)   # 벡터DB 내용과 GPT LLM을 연결
            st.session_state.processComplete = True
            message_info.empty()

        history = StreamlitChatMessageHistory(key="chat_messages")

    elif len(uploaded_files) == 0 and "messages" in st.session_state:
        if process:
            st.error("파일을 첨부하세요.")




    if "messages" not in st.session_state:
        st.session_state["messages"] = [("assistant", "질문하시면 친절하게 답변 드리겠습니다.")]

    for chat_message in st.session_state.messages:
        st.chat_message(chat_message[0]).write(chat_message[1])

    question = st.chat_input("질문을 입력하세요.")
    if question:

        if  len(uploaded_files) != 0:                # 첨부파일이 있으면, RAG
            st.chat_message("user").write(question)
            with st.chat_message("assistant"):
                chain = st.session_state.conversation
                with st.spinner("RAG 답변 생성 중..."):
                    result = chain({"question": question})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    answer = result['answer']
                    st.write(answer)

                    source_documents = result['source_documents']
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)


        if  len(uploaded_files) == 0:            # 첨부 파일이 없으면, 챗GPT 3.5
            st.chat_message("user").write(question)


            with st.spinner("챗GPT가 답변 생성 중..."):
                key = os.getenv('OPENAI_KEY')
                llm = ChatOpenAI(openai_api_key = key)
                answer = llm.invoke(question)
                answer = answer.content
                st.chat_message("assistant").write(answer)
    


        st.session_state["messages"].append(["user", question])
        st.session_state["messages"].append(["assistant", answer])










def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore, key):
    llm = ChatOpenAI(openai_api_key=key, model_name='gpt-3.5-turbo', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

def process_local_file(file_path):
    # 파일 확장자에 따라 적절한 로더 선택
    if '.pdf' in file_path:
        loader = PyPDFLoader(file_path)
    elif '.docx' in file_path:
        loader = Docx2txtLoader(file_path)
    # 여기에 필요한 다른 파일 타입에 대한 처리를 추가할 수 있습니다.
    else:
        raise ValueError(f"Unsupported file type for {file_path}")

    # 파일 로드 및 분할
    documents = loader.load_and_split()
    return documents

if __name__ == '__main__':
    main()
