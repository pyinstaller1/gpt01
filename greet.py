import gradio as gr

def greet(name):
  return "안녕! " + name

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()


