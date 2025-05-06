# DeepSeek R1本地部署

## ollama

‌**[Ollama](https://www.baidu.com/s?rsv_idx=1&wd=Ollama&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=6138pVi3KtJShDZwX%2FTvjuBS3CBXG70guTwmL%2FDoWm8tTQywcl9XmJbMZJ8&rsv_dl=re_dqa_generate&sa=re_dqa_generate)是一个开源的大语言模型服务工具，专为在本地机器上便捷部署和运行大型语言模型而设计**‌。它通过提供便捷的模型管理、丰富的预建模型库、跨平台支持以及灵活的自定义选项，使得开发者和研究人员能够在本地环境中高效利用大型语言模型进行各种自然语言处理任务，而无需依赖云服务或复杂的基础设施设置。‌

Ollama基于[Go语言](https://www.baidu.com/s?rsv_idx=1&wd=Go语言&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=0c7bV2qe5eICNls6w2u44UEFNu1qEGWUvC0F%2FrieIslC8uW564AronG9%2B%2F4&rsv_dl=re_dqa_generate&sa=re_dqa_generate)开发，具有类似[Docker](https://www.baidu.com/s?rsv_idx=1&wd=Docker&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=f05eVaDirji%2FBiuo2hlUF3sDPSfXeH7x5zgKMAXdkFGE7%2BQtxN6iAA0G6No&rsv_dl=re_dqa_generate&sa=re_dqa_generate)的命令行工具功能，支持通过命令行进行模型的管理和交互。用户可以通过简单的命令行操作来列出、拉取、推送、拷贝和删除模型，极大地简化了大模型的本地部署和使用过程。

Ollama支持多个开源大语言模型，包括[Llama 3](https://www.baidu.com/s?rsv_idx=1&wd=Llama 3&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=1b32x7S5%2FT0D%2B4xBGdnZc3NP1VPzWX3Ai94Q0HMqFL5dXSz3hEaoQBn5%2BFE&rsv_dl=re_dqa_generate&sa=re_dqa_generate)、[Gemma 2](https://www.baidu.com/s?rsv_idx=1&wd=Gemma 2&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=1b32x7S5%2FT0D%2B4xBGdnZc3NP1VPzWX3Ai94Q0HMqFL5dXSz3hEaoQBn5%2BFE&rsv_dl=re_dqa_generate&sa=re_dqa_generate)、[Code Llama](https://www.baidu.com/s?rsv_idx=1&wd=Code Llama&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=bb16EwsN%2FmblkRdVaeFOuVMmtR5edRXpjgmGNokgxzz7Ry83qgO8AvxuMaw&rsv_dl=re_dqa_generate&sa=re_dqa_generate)等，用户可以根据需求选择合适的模型进行部署和使用。此外，Ollama还提供了web框架[gin](https://www.baidu.com/s?rsv_idx=1&wd=gin&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=0bc52QP93%2FmuUgkb1Qmve3dbFuzQYvozNqjQOSQz3Cnv6pjfkEbVoT%2FJbHg&rsv_dl=re_dqa_generate&sa=re_dqa_generate)，使得用户可以通过API接口与模型进行交互，类似于[OpenAI](https://www.baidu.com/s?rsv_idx=1&wd=OpenAI&fenlei=256&usm=2&ie=utf-8&rsv_pq=e93de165000167b1&oq=ollma是什么&rsv_t=0bc52QP93%2FmuUgkb1Qmve3dbFuzQYvozNqjQOSQz3Cnv6pjfkEbVoT%2FJbHg&rsv_dl=re_dqa_generate&sa=re_dqa_generate)提供的接口。

ollama的仓库地址：https://github.com/ollama/ollama.git

ollama类似于python的pip，可以用指令来安装和操作大模型

ollama可以将你在huggingface 下载并微调、量化后的模型接入到ollama



## DeepSeek R1热点的原因

1. 开源
2. 训练成本时openai o1模型的二十分之1，推理成本也是很低，API价格更低
3. 使用成本低，因为模型天然加入了COT



## ollama下载安装

ollama的网站地址：https://ollama.com/  根据系统选择安装程序

ollama的默认安装地址为：C:\Users\XueLi_G\AppData\Local\Programs\Ollama



## ollama的使用

安装完成后，打开cmd命令行窗口，输入ollama，会显示如下内容

```sh
C:\Users\XueLi_G>ollama
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```

出现上面的指令提示就证明ollama已经安装完成

查看ollama的版本

```sh
C:\Users\XueLi_G>ollama -v
ollama version is 0.5.7
```

 查看ollama安装了哪些大模型

```sh
C:\Users\XueLi_G>ollama list
NAME                ID              SIZE      MODIFIED
deepseek-r1:1.5b    a42b25d8c10a    1.1 GB    41 hours ago
```



## ollama模型部署

ollama可部署大模型链接：https://ollama.com/search，选择好要部署的模型和参数

例如：deepsseek R1 7b  连接为：https://ollama.com/library/deepseek-r1:7b

会出现一条指令：ollama run deepseek-r1:7b，将这个指令复制到cmd终端

这条指令会下载和运行deepsseek R1 7b ，如果已经下载好了就可以直接进行交互，没有下载好就先去下载模型

默认的端口号：http://localhost:11434/v1/

ctrl + d 退出终端交互



## ollama将deepseek 以api的方式部署

在cmd终端输入OllamaServer，启动ollama的服务，正常启动会出一大堆提示，非正常启动会显示端口占用

运行起来之后就可以使用api的方式，来与deepseek R1 7b的模型进行交互

在ollama的使用文档 ollama/docs/openai.md中 有交代如何使用代码的形式与ollama调用

```python
from openai import OpenAI  # openai已经写好了底层大模型调用的方式  很多大模型都是直接使用openai写好的这个调用方式

client = OpenAI(
    base_url='http://localhost:11434/v1/', # 部署ollama的电脑ip地址：端口号
 
    # required but ignored
    api_key='ollama', # 本地部署随便写
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='llama3.2',
)

response = client.chat.completions.create(
    model="llava",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG0AAABmCAYAAADBPx+VAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjlug8ZtTt4kVF0kLUYYmCCtD/DrQ5YhMGbA9L3ucdjh0y8kOHW5gU/VEEmJTcL4Pz/f7mgoAbYkAAAAAElFTkSuQmCC",
                },
            ],
        }
    ],
    max_tokens=300,
)

completion = client.completions.create(
    model="llama3.2",
    prompt="Say this is a test",
)

list_completion = client.models.list()

model = client.models.retrieve("llama3.2")

embeddings = client.embeddings.create(
    model="all-minilm",
    input=["why is the sky blue?", "why is the grass green?"],
)
```







