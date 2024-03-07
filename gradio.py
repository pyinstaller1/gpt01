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

