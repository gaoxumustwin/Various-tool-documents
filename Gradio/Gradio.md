# Gradio

## 简介

官方网站：https://www.gradio.app/

Gradio 是一个用于快速创建和部署机器学习模型界面的 Python 库，允许用户通过Python 代码来构建交互式的 Web 界面。



## 安装Gradio

Gradio的安装可以使用pip

```sh
pip install Gradio
```



## 创建第一个Gradio应用

应用功能：一个接收文本输入，将这个文本倒装返回

首先定义 Gradio 界面的核心逻辑函数

```python
def reverse_and_count(text):
    reversed_text = text[::-1]
    return reversed_text
```

接下来使用Gradio构建用户界面，这里我们需要一个文本输入框和一个用于显示结果的文本输出区域

```python
import gradio as gr

demo = gr.Interface(fn=reverse_and_count, inputs="text", outputs="text") 
demo.launch()   
```

上面的代码中:

- fn参数指定了处理输入输出数据的函数
- inputs 指定输入类型，在这里是文本，  如果显示的使用 `inputs=[gr.Textbox(label="text")]`  其中这里的label后的text就必须和reverse_and_count的参数强绑定
- outputs 设置了期望的输出格式，同样为文本。

最后一步是启动 Gradio 提供的服务，我们可以通过浏览器访问到我们刚才创建的界面

```python
demo.launch()   
# launch 里面可以指定ip地址 和 端口号
# 例如： demo.launch(server_name="ip地址"，server_poer=端口号)   
```

执行上述代码后，你可以在控制台看到一条在7860端口打开的 URL（http://127.0.0.1:7860/），打开指定的 URL 即可查看效果。

![](img\1.png)



## Gradio的Interface

Gradio 的 Interface是 Gradio 中高度封装的一种布局方式，比较适合面向小白用户，允许用户通过几行代码就可以生成 WebUl.

Interface 的核心是在第一个应用中提到的三个参数:

- h:指定了处理输入数据的函数，并将结果进行输出。
- inputs:指定输入类型。
- outputs:设置了期望的输出格式，

除了核心的参数之外，其余参数大多和整体的界面布局有关。

### 更新 Gradio 界面

我们更新前面的 Gradio 界面，假设我们想要构建一个应用，该应用不仅能够反转文本,还能计算给定字符串的长 度

```python
import gradio as gr

# 这是修改后的函数
def reverse_and_count(text):
    reversed_text = text[::-1]
    length = len(text)
    return reversed_text, length

demo = gr.Interface(fn=reverse_and_count,
                    inputs="text",
                    # flagging_mode="never",
                    outputs=["text", "number"], # 第一个输出是文本，第二个输出是一个数字 （接收核心函数中的俩个返回值）
                    title="文本处理工具", # 设置页面标题
                    description="输入一段文字，查看其倒序形式及字符数。", # 添加简短说明
                    examples=[["你好，世界"], ["Hello World"]]
                )
demo.launch()
```

效果：

![](img\2.png)

Interface支持接收一些参数来改变页面的显示

### 更改组件初始值

Interface 的输入输出除了可以接收字符串外，还可以接收由 Gradio 导出的任意其他组件实例，我们可以为这些实例添加初始值和配置项，来实现更定制化的需求:

```python
import gradio as gr
# 下面的示例中，我们可以通过输入文字，将其输出到 output 侧，并最终通过 Slider 选择的值为后面添加不同个数的 “!”
def greet(name, intensity):
	return "Hello,"+ name +"!"* intensity

demo = gr.Interface(fn = greet,
                    inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1)],
                    outputs=[gr.Text(label="greeting",lines=3)] # lines 设置行数为3   greeting和text都是Gradio的组件实例
                    )
demo.launch()
```

效果如下：

![](img\3.png)

输入输出的展示模块也可以进行单独修改，Gradio的Interface的输入和输出除了可以接收字符串外还可以接收由Gradio导出的任意其他组件实例，可以为这些实例添加初始值和配置项来实现定制化的需求

### 其他组件示例

#### 绘制函数

使用 Number 组件输入函数的参数范围和点数，使用PIot 组件绘制正弦波图

```python
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
```

效果如下：

![](img\4.png)

#### 图像处理

使用 lmage 组件输入图像，并将其转换为铅笔素描。

```python
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
```

效果如下：

![](img\5.png)

还有**音频处理和视频处理**等等，这些都可以在官方的文档中找到



## 使用 Chatlnterface 生成对话界面

Gradio 的 ChatInterface 同样是 Gradio 中高度封装的一种布局方式，但主要用于创建聊天机器人的 UI界面。与 Interface 不同，Chatinterface 只有一个核心参数:

- fn: 根据用户输入和聊天历史记录来控制聊天机器人的响应值

```python
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
```

效果：

![](img\6.png)

模仿大模型的流式输出  使用time进行间隔  yield将单词大模型的输出结果返回

```python
import time
import gradio as gr

def slow_echa(message, history):
    for i in range(len(message)):
        time.sleep(0.1)
        yield "You typed:"+ message[: i + 1]

demo = gr.ChatInterface(slow_echa, type="messages")

demo.launch()
```

结合质谱AI进行流式输出

```python
# pip install zhipuai 请先在终端进行安装
import gradio as gr
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="0f56bcd3ce36d22b5b6564de4faeebfe.nvc4AlZb8rw1WhFG")

def get_zhipu_response(prompt):
    response = client.chat.completions.create(
        model="glm-4-flashx",
        messages=[
            {
                "role": "system",
                "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。" 
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        top_p= 0.7,
        temperature= 0.95,
        max_tokens=1024,
        tools = [{"type":"web_search","web_search":{"search_result":True}}],
        stream=True # 开启流式输出
    )
    for trunk in response:
        yield trunk.choices[0].delta.content # 质谱AI流式输出的固定写法 
        # yield 如果一个函数带了yield，那他就不是一个函数了，而是一个生成器对象  所以返回的是一个生成器对象

def main(query):
    prompt = f"请你回答下面的问题：{query}"
    zhipu_gen = get_zhipu_response(prompt) # zhipu_gen 是一个生成器对象

    acumulated_result = ""
    for char in zhipu_gen: # 一个字一个字吐
        acumulated_result += char
        yield acumulated_result  # 如果直接吐char会被刷新

if __name__ == "__main__":

    demo = gr.Interface(
        fn = main,
        inputs=gr.Textbox(),
        outputs = gr.Markdown()
    )

    demo.launch(
        server_name = "127.0.0.1",
        server_port = 55555
    )   
```





## 自定义界面与复杂布局

​	Gradio除了提供 Interface与 ChatInterface 等高级封装方式来快速实现界面布局外，还支持我们自定义地控制页面布局。
​	BlocKs 是 Gradio 的低阶 AP，使用它我们可以像搭建积木一样装饰页面，并对一些函数的触发时机与数据流的输入输出提供更强大的灵活性。Interface与ChatInterface 本质上也是基于它来二次封装的，

### 更改布局方式

现在让我们更新一下前面的使用 Interface创建的反转文本应用。

```python
import gradio as gr
# 这是修改后的函数
# Functions
def reverse_and_count(text):
    reversed_text = text[::-1]
    length = len(text)
    return reversed_text, length

# Components
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
    
    # events
    #组件支持的事件，这里代表着页面上的按钮被点击时会触发reverse_and_count 这个函数
    btn.click(fn=reverse_and_count,
              inputs=[input_text],
              outputs=[output_reversed, output_length])

demo.launch()
```

结果显示：

![](img\7.png)

代码中没有了Interface取而代之的是Blocks，在Blocks的代码中有些是界面布局有些是事件绑定

使用Blocks的方式就可以更加自由的改变界面布局的方式

Blocks作为Gradio最基础的页面构成方式还支持与其他的方式混用

## Gradio 的组件与运行机制

​	Gradio 的组件是 Gradio 中的基本单元，它不仅可以映射到前端作为页面显示元素，还可以在 Python 端绑定事件运行用户的实际推理函数。

Gradio 整体的运行机制如下:

通常在一个应用中，我们会遇到下面几种概念:

- Components:组件本身，此概念下它只是作为构成界面的基本元素，如文本框、按钮等。
- Events:用户与界面交互时触发的事件，如点击按钮、输入文本等行为。
- Functions:用户输入并生成输出的函数。

本质上，Gradio 是一个基于事件监听机制的应用框架，它通常由前端发送事件信号，然后在 Python 端由对应绑定了事件的组件进行接收，传递给用户的输入输出函数，最后又由 Grado 将用户的返回结果传递给前端界面。

- 前端:用户在界面上进行操作(例如输入文本或点击按钮)时，这些操作会由前端组件发送到function函数。
- Python 端:Python 端接收到用户的请求后，会调用相应的函数来处理输入数据，并将结果返回给前端。
- 更新界面:前端接收到后端返回的结果后，会更新相应的组件(如文本框的内容)。

## 自定义组件与三方组件

之前小节中提到的一些组件示例和很多没讲到的 Gradio 组件都可以在官方文档中找到。

除此之外，Gradio 支持自定义组件，自定义组件底层与 Gradio 原始组件的实现机制一致，在使用时一般不会有其他特殊限制(除了某些自定义组件本身有特殊要求)，你可以查看这里来获取更多如何制作自定义组件的信息。

### 三方组件

Gradio 原始组件之外的组件都是三方组件，它们通常用于支持一些 Gradio 原始组件无法实现的功能

#### modelscope-studio

modelscope-studio（https://modelscope.cn/studios/modelscope/madelscope-studio）是一个基于 Gradio 的三方组件库，它可以为开发者提供更定制化的界面搭建能力和更丰富的组件使用形式。

在 https://www.gradio.app/custom-components/gallery 可以查看所有的第三方组件

在原有 Gradio 组件之上， modelscope-studio 提供了多种基础组件来辅助开发者优化界面布局，如 divspan、text 等前端的基本元素，并且还集成了Ant Design等著名的前端组件库来帮助开发者快速构建更加美观的界面。

安装：

```sh
pip install modelscope_studio
```

