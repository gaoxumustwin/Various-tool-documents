import gradio as gr
import numpy as np
import cv2
def image_to_sketch(image):
    gray_image = image.convert('L')
    inverted_image = 255-np.array(gray_image)
    blurred = cv2.GaussianBlur(inverted_image, (21,21), 0)
    inverted_blurred=255-blurred
    pencil_sketch = cv2.divide(np.array(gray_image), inverted_blurred, scale=256.0)
    return pencil_sketch

demo = gr.Interface(fn=image_to_sketch,
                    inputs=[gr.Image(label="Input Image", type="pil")],
                    outputs=[gr.Image(label="Pencil Sketch")],
                    title="Image to Pencil Sketch",
                    description="Convért an input image to a pencil sketch."
)
demo.launch()