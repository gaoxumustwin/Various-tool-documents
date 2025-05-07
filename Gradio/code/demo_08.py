import gradio as gr
# 这是修改后的函数
def reverse_and_count(text):
    reversed_text = text[::-1]
    length = len(text)
    return reversed_text, length

with gr.Blocks()as demo:
    gr.Markdown("<h1><center>文本处理工具<center/></h1>") # 标题
    gr.Markdown("输入一段文字，查看其倒序形式及字符数。") # 介绍

    with gr.Row():#水平排列
        with gr.Column(): # 第一列
            input_text = gr.Textbox(label="请输入一些文字")
        with gr.Column(): # 第二列
            output_reversed = gr.Textbox(label="倒序结果")
    btn = gr.Button("提交") # 定义提交按钮
    output_length = gr.Number(label="字符总数") 
    gr.Examples([["你好,世界"],["Hello World"]],inputs=[input_text])
    #组件支持的事件，这里代表着页面上的按钮被点击时会触发reverse_and_count 这个函数
    btn.click(fn=reverse_and_count,
              inputs=[input_text],
              outputs=[output_reversed, output_length])

demo.launch()