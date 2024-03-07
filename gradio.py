# import openai
import gradio as gr

from langchain.chat_models import ChatOpenAI



chat_model = ChatOpenAI(openai_api_key="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")

result = chat_model.predict("hi")
print(result)

print(8)







# client = openai.OpenAI(api_key="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")

def respond(message, chat_history):

  # client = openai.OpenAI(api_key="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")

  """
  response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages = [{"role": "user", "content": message}],
      max_tokens=256,
      temperature=0
  )

  print(response.choices)
  print(response.choices[0].message.content)
  """

  # bot_message = response.choices[0].message.content
  bot_message = chat_model.predict(message)
  chat_history.append((message, bot_message))

  return "", chat_history

with gr.Blocks() as demo:
  chatbot = gr.Chatbot(label="채팅창")
  msg = gr.Textbox(label="입력")
  clear = gr.Button("초기화")

  msg.submit(respond, [msg, chatbot], [msg, chatbot])
  clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True)

print(8)

