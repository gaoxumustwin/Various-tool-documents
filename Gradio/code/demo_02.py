import gradio as gr

# 这是修改后的函数
def reverse_and_count(text):
    reversed_text = text[::-1]
    length = len(text)
    return reversed_text, length

demo = gr.Interface(fn=reverse_and_count,
                    inputs="text", # 输入框
                    # flagging_mode="never",
                    outputs=["text", "number"], # 第一个输出是文本，第二个输出是一个数字
                    title="文本处理工具", # 设置页面标题
                    description="输入一段文字，查看其倒序形式及字符数。", # 添加简短说明
                    examples=[["你好，世界"], ["Hello World"]] # 在页面添加示例
                )
demo.launch()