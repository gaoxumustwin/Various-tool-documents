import gradio as gr

def reverse_and_count(text):
    reversed_text = text[::-1]
    return reversed_text

demo = gr.Interface(fn=reverse_and_count, inputs="text", outputs="text")
demo.launch()   