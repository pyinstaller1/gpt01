from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

import gradio as gr

from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = """Use the following pieces of context to answer the users question shortly.
Given the following summaries of a long document and a question, create a final answer with retrieval.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
---------------
{summaries}

You MUST answer in Korean and in markdown format:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]

prompt = ChatPromptTemplate.from_messages(messages)



loader = PyPDFLoader("건강보험자료.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(api_key="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")
vector_store = Chroma.from_documents(texts, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":2})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="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")

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever = retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

query = "건강보험은?"
result = chain(query)

for doc in result['source_documents']:
  print('내용 : ' + doc.page_content[0:100].replace('\n', ' '))
  print('파일 : ' + doc.metadata['source'])
  print('페이지 : ' + str(doc.metadata['page']))



def respond(message, chat_history):

    result = chain(message)
    bot_message = result['answer']

    for i, doc in enumerate(result['source_documents']):
        bot_message += '[' + str(i+1) + '] ' + doc.metadata['source'] + '(' + str(doc.metadata['page']) + ') '

    chat_history.append((message, bot_message))

    return "", chat_history

with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
    chatbot = gr.Chatbot(label="채팅창")
    msg = gr.Textbox(label="입력")
    clear = gr.Button("초기화")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True)


print(7)


























"""


import gradio as gr

def greet(name):
  return "안녕! " + name

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()















import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat_model = ChatOpenAI(openai_api_key="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")
# result = chat_model([HumanMessage(content="안녕? 넌 누구니?")])

def respond(message, chat_history):

  result = chat_model([HumanMessage(content=message)])
  
  bot_message = result.content
  # bot_message = chat_model.predict(message)

  chat_history.append((message, bot_message))

  return "", chat_history

with gr.Blocks() as demo:
  chatbot = gr.Chatbot(label="채팅창")
  msg = gr.Textbox(label="입력")
  clear = gr.Button("초기화")

  msg.submit(respond, [msg, chatbot], [msg, chatbot])
  clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True)

"""
