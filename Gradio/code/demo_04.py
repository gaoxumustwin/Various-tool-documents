import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
def plot_function(x_min, x_max, n_points):
    x=np.linspace(x_min, x_max, n_points)
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y)
    plt.title("sine Wave")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    return plt

demo = gr.Interface(
    fn=plot_function,
    inputs=[
        gr.Number(label="X Min" ),
        gr.Number(label="X Max"),
        gr.Number(label="Number of Points")
    ],
    outputs=[gr.Plot()],
    title="Function Plotter",
    description="plot a sine wave function based on the given parameters."
)
demo.launch()