import gradio as gr
def echo(message, history):
    # message 为用户当次输入，return的返回结果为当次的bot输出
    return message
demo = gr.ChatInterface(
    fn=echo, 
    type="messages", 
    examples=["hello", "hola", "merhaba"],
    title="Echo Bot")
demo.launch()