import gradio as gr
# 下面的示例中，我们可以通过输入文字，将其输出到 output 侧，并最终通过 Slider 选择的值为后面添加不同个数的!
def greet(name, intensity):
	return "Hello,"+ name +"!"* intensity

demo = gr.Interface(fn = greet,
                    inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1)],
                    outputs=[gr.Text(label="greeting",lines=3)]
                    )
demo.launch()