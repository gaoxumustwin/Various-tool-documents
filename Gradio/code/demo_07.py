# 下面才是完整代码
import time
import gradio as gr

def slow_echa(message, history):
    for i in range(len(message)):
        time.sleep(0.1)
        yield "You typed:"+ message[: i + 1]

demo = gr.ChatInterface(slow_echa, type="messages")

demo.launch()