# LLama Factory

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是一款开源低代码大模型微调框架，集成了业界最广泛使用的微调技术，支持通过 Web UI 界面零代码微调大模型，目前已经成为开源社区内最受欢迎的微调框架之一。

官方仓库地址：https://github.com/hiyouga/LLaMA-Factory

官方指导书：https://llamafactory.readthedocs.io/zh-cn/latest/

写给自己的一句话：**训练并不复杂，关键是如何根据训练的结果判断，训练是否充分，是否过拟合，精度效果怎么样，没训练好的原因是什么，如何去调整，是换数据还是加参数，还是换模型**

**到公司，如果拿到新的服务器，那么LLama Factory的环境构建过程都需要走一遍**



## LLama Factory的环境

参考LLama Factory的README文件，目前版本其推荐的环境为：

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 1.13.1  | 2.6.0     |
| transformers | 4.41.2  | 4.50.0    |
|              |         |           |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |



## 安装 LLaMA Factory

**python环境安装**

```
pip3 install deepspeed modelscope transformers
```

**此步骤为必需**

```sh
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[metrics]" # -e是额外选择安装依赖项 即pip安装基础包以及安装torch和metrics的依赖项
```

可选的额外依赖项：torch、torch-npu、metrics、deepspeed、liger-kernel、bitsandbytes、hqq、eetq、gptq、awq、aqlm、vllm、galore、badam、adam-mini、qwen、modelscope、quality

注意：遇到包冲突时，可使用 `pip install --no-deps -e .` 解决。



## LLama Factory工程

setup.py是一个用于定义python软件包安装的脚本（pip install会自动执行这个脚本），通过setup()函数来配置包的各种属性，包括名称，版本和各种依赖项等

setup.py代码：

```python
import os
import re
from typing import List

from setuptools import find_packages, setup

'''
这个函数会打开src\llamafactory\extras\env.py文件，读取其内容，并使用正则表达式提取名字为VERSION的变量值，即版本号
'''
def get_version() -> str:
    with open(os.path.join("src", "llamafactory", "extras", "env.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version

'''
这个函数会打开requirements.txt文件，读取其内容，并过滤掉以#为开头的注释行，然后返回一个包含每行依赖项的列表
'''
def get_requires() -> List[str]:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines

def get_console_scripts() -> List[str]:
    console_scripts = ["llamafactory-cli = llamafactory.cli:main"]
    if os.environ.get("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "1"]:
        console_scripts.append("lmf = llamafactory.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=1.13.1"],
    "torch-npu": ["torch==2.1.0", "torch-npu==2.1.0.post3", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.14.4"],
    "liger-kernel": ["liger-kernel"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.17.0", "auto-gptq>=0.5.0"],
    "awq": ["autoawq"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3,<=0.6.0"],
    "galore": ["galore-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "qwen": ["transformers_stream_generator"],
    "modelscope": ["modelscope"],
    "dev": ["ruff", "pytest"],
}


def main():
    setup(
        name="llamafactory",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga" "@" "buaa.edu.cn",
        description="Easy-to-use LLM fine-tuning framework",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["LLaMA", "BLOOM", "Falcon", "LLM", "ChatGPT", "transformer", "pytorch", "deep learning"],
        license="Apache 2.0 License",
        url="https://github.com/hiyouga/LLaMA-Factory",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()}, # 最关键的 定义了一个命令行工具， 这里llamafactory-cli命令会调用llamafactory.cli模块中的main函数
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
```

正是如此安装完成后才会可以在命令行终端使用：llamafactory-cli



## llamafactory-cli 命令

例如：在终端输入llamafactory-cli，会显示如下信息

![28](img\28.png)

```sh
llamafactory-cli api -h：启动OpenAI类型的api服务器
llamafactory-cli chat -h：在cli中启动聊天界面
llamafactory-cli eval -h：评估模型
llamafactory-cli export -h：合并LoRA适配器和导出模型
llamafactory-cli train -h：训练模型
llamafactory-cli webchat -h：在Web UI中启动聊天界面
llamafactory-cli webui：推出LlamaBoard
llamafactory-cli version：显示版本信息
```

例如：使用llamafactory-cli 分别对 Llama3-8B-Instruct 模型进行 LoRA **微调（训练）**、**推理**和**合并**。

```
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml 
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

例如训练：

修改好examples/train_lora/llama3_lora_sft.yaml 

![5](img\35.png)

训练完成后如下：

![5](img\36.png)

高级用法请参考 [examples/README_zh.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README_zh.md)（包括多 GPU 微调）。

使用 `llamafactory-cli help` 显示帮助信息。

在终端执行which命令找到llamafactory-cli的位置，

![30](img\30.png)

其实llamafactory-cli的代码逻辑来自于 src\llamafactory\cli.pyz中的main函数，即执行llamafactory-cli做一些任务的时候，实际上都是在调用src\llamafactory\cli.pyz中的main函数

llamafactory-cli的debug在后面

工程和算法上常用llamafactory-cli 

### llamafactory-cli 的一些指令

参考：https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/README_zh.md





## 模型下载

可以从魔搭社区下载模型，魔塔社区的连接为：https://modelscope.cn/home，点击模型库搜索模型进行下载，例如：搜索0.5B(大小约为1G 0.5*2)的通义千问2

![5](img\5.png)

进入到搜索到的模型页面，点击模型文件，选择下载模型

![5](img\6.png)

会提供SDK下载 Git下载以及命令行下载，例如：

- SDK下载 

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-0.5B-Instruct')
```

**snapshot_download** 里面的名称不要去修改

- Git下载  （稍微稳定）

请确保 lfs 已经被正确安装

```sh
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2-0.5B-Instruct.git
```

- 命令行下载

请先通过如下命令安装ModelScope

```sh
pip install modelscope
```

下载完整模型repo

```sh
modelscope download --model Qwen/Qwen2-0.5B-Instruct
```

下载单个文件（以README.md为例）

```sh
modelscope download --model Qwen/Qwen2-0.5B-Instruct README.md
```

![5](img\7.png)

下载完成后模型的存储位置为：![5](img\8.png)

将Qwen2-0___5B-Instruct移动到与LLaMA-Factory同级的models目录下面

```sh
mv /root/.cache/modelscope/hub/Qwen/Qwen2-0___5B-Instruct /root/models
```

![5](img\9.png)



## 数据

数据形式有alpaca和share gpt两种数据形式

### alpaca数据形式

```json
{
    "instruction": "【OCNLI】请判断以下两句话的逻辑关系，选项：蕴含、中性、不相关",
    "input": "第一句话：一月份跟二月份肯定有一个月份有。第二句话：肯定有一个月份有。",
    "output": "蕴含"
},
```

- instruction：指令
- input：内容
- output：结果

在训练的时候实际的输入是 instruction + input

### share gpt格式

```json
"conversations":{
 	{
    "from": "human",
    "input": "【OCNLI】请判断以下两句话的逻辑关系，选项：蕴含、中性、不相关;用户输入：第一句话：一月份跟二月份肯定有一个月份有。第二句话：肯定有一个月份有。",
	},
	{
    "from": "gpt",
    "input": "蕴含",
	}
}
```







## LLaMA Board 可视化微调（由 [Gradio](https://github.com/gradio-app/gradio) 驱动）

```
llamafactory-cli webui
```

llamafactory-cli是前面环境安装的时候生成

会自行跳转：http://localhost:7860/   使用VsCode连接远程服务器的时候可以直接跳转，因为有端口映射

- 设置页面的语言格式，例如：设置中文显示

![4](img\4.png)

- 模型名称和模型路径，模型名称要与模型路径使用的模型对应

![4](img\10.png)

一般填写模型路径，模型下载看下面

- 微调方法，它是一种模型适配的方式，个人显卡建议lora，使用其他显卡资源不够，省显存

![4](img\23.png)

- 检查点路径 

类似于yolov5的断点训练 如果你设置了某个batch保存一次结果，那么你下一次训练模型可以使用从这个结果继续训练，第一次训练的时候没有检查点所以不用设置

- 训练阶段，即训练方式的调节

下面以Supervised Fine-Tuning有监督微调为例开始训练和推理

![4](img\17.png)

- 数据路径和数据集

![4](img\19.png)

它对应data目录下面的内容

![4](img\20.png)

像Supervised Fine-Tuning有监督微调使用的训练数据多为问答对，例如以identity.json为例：

![21](img\21.png)

- 预览命令

当配置好各种参数后选择预览命令，会生成训练命令

![4](img\11.png)

训练的时候可以在终端使用: watch -n 1 nvidia-smi 每间隔一秒打印一下显卡的使用情况 

效果如下所示，模型训练的结果会保存到saves目录下面

![5](img\13.png)

训练完成后并没有结束，这是因为这个终端控制的是浏览器页面

可以按ctrl+c退出，此时浏览器页面也不会响应

![5](img\15.png)



## 训练权重加载和推理

参考examples下面的README_zh.md

### 推理 LoRA 模型 

执行下面代码的时候一定要在LLaMA-Factory目录下面执行，执行前要关闭训练的各种窗口

#### 使用命令行接口

```sh
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### 使用浏览器界面

```sh
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

- llamafactory-cli是LLaMA-Factory的启动命令
- webchat是使用浏览器界面去启动
- examples/inference/llama3_lora_sft.yaml是配置文件，自己训练出来的模型要修改这个配置文件里面的内容,其文件内容如下所示： **在使用的时候建议复制出来一份在自己的文件上修改**

```yaml
model_name_or_path: /root/models/Qwen2-0___5B-Instruct # 模型文件的路径
adapter_name_or_path: /root/LLaMA-Factory/saves/Qwen2-0.5B-Instruct/lora/train_2024-09-25-15-42-07 # 使用LoRA训练出来的模型权重
template: qwen # 模板 默认是llama3 前面设置模型名称 就是为了在这个地方确认模型的模板
finetuning_type: lora
```

执行后会打印模型的合并过程，最后跳转到浏览器页面

![5](img\16.png)

对话回复这一过程是使用GPU的，每一次提交GPU都会出现峰值

#### 启动 OpenAI 风格 API

```sh
llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```



## 预训练操作 - Pre-Training

### 调整训练阶段

![5](img\18.png)



### 数据设置

Pre-Training使用的数据集是纯文本的，例如：c4_demo.json

![5](img\22.png)

### 整体设置

在浏览器设置的情况如下所示：

Pre-Training训练比Supervised Fine-Tuning训练在界面上的区别只有下面两个红色框的地方

![5](img\24.png)

Pre-Training训练比Supervised Fine-Tuning训练消耗更多的资源，在后台看GPU利用率可以明显得到

### 推理

与前面一样！！



## DPO训练

DPO理解为大模型训练的强化学习的简化版本

大模型训练分为三个阶段：

- 预训练阶段---让大模型学习海量的文本数据
- Supervised Fine-Tuning有监督微调---让模型学会问答的形式激发其对第一个阶段学到知识的使用
- DPO训练---让模型回答的更好，与Supervised Fine-Tuning的区别是训练的数据的问题会接两个答案，一个好，一个不好，让其回答靠近好的答案，远离不好的答案，让模型一直学一直学，逐渐越来越好，越来越靠近人类的偏好，所以DPO也叫人类偏好对齐

DPO训练的数据如下：

chosen是好的答案

rejected是不好的答案

![5](img\25.png)

DPO的训练页面：

![5](img\26.png)



## 数据集修改和配置

​	在网页上可以选择的数据集均存放在data下面，关于数据集文件的格式，请参考 data/README_zh.md的内容。你可以使用 HuggingFace / ModelScope 上的数据集或加载本地数据集。

**使用自定义数据集时，在将自己的数据放到这个目录下面的同时，还需要更新 `data/dataset_info.json` 文件**

dataset_info.json上的数据集会在Web上进行显示

例如：以前面的SFT训练的identity数据集为例

![5](img\27.png)

dataset_info.json各种属性的介绍在data/README_zh.md都提及到了



## llamafactory-cli train debug

服务器VsCode debug配置，安装好插件后，重启一下窗口

![5](img\32.png)

以llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml 为例配置debug文件

![5](img\33.png)

其中llama3_lora_sft.yaml 为训练时候的配置文件，需要准备或者修改文件内部的模型model_name_or_path

![5](img\35.png)

点击开始调试

![5](img\34.png)

一般会存在着路径问题：找不到数据，此时可以通过设置cwd来避免这个问题

```json
// 工作目录路径问题设置
"cwd": "${fileDirname}", // 设置工作目录为当前打开文件的所在目录
或者
"cwd": "${workspaceFolder}/src", //  "cwd": "${workspaceFolder}"设置工作目录为当前工作空间的根目录  后面加/src 是相对于工作空间根目录
```

![5](img\37.png)

debug某些中间过程的函数执行

![5](img\38.png)



## LLama Factory SFT 思路

有资源 有时间的情况下 使用 SFT全量训练，其效果是最好的

业务流程：

- 接到一个任务：某个行业的问答机器人（非常好落地）、某个行业的分类
- 模型选型（7B的模型，float16） 7B指的是参数量，不是大小
  - 训练：涉及 参数存储、梯度、优化器的状态、其它
    - **倍数关系：20**， 7B 模型大约需要 140G 显存   0.5B需要10G显存
  - 推理：涉及 参数量
    - **倍数关系：2**，7B 模型需要 14G 显存做推理  0.5B需要1G显存
  - 下载模型
    - 复制下载链接，如果碰到模型有问题，复制下载链接使用 wget 下载
- 处理数据 （占用大量的时间）
  - 多进程、多线程
  - 数据的增广处理  （策略有很多，例如：有很多大模型都是让GPT生成数据他们训练）
- 安装 LLama Facotry
  - 安装 deepspeed
- 训练模型
  - 修改配置文件：数据集、训练参数...



## lora训练

它是原模型不动，lora的秩一般喜欢用8-64之间



## LLama Factory实战 - Qwen1.5-0.5B SFT 微调

### 模型

#### 下载

- git下载

```sh
git lfs install  # 确保 lfs 已经被正确安装
git clone https://www.modelscope.cn/Qwen/Qwen1.5-0.5B-Chat.git
```

- SDK下载

```python
pip install modelscope

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen1.5-0.5B-Chat')
```

#### 测试

测试代码参考 modelscope的Quickstart的代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-0.5B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

如果运行时候出现HeaderTooLarge的问题，则说明model.safetensors文件发生了损坏，需要重新下载 可以使用wget



### GPU服务器租借

一般建议小任务的情况下：GPU 使用3090



### 数据

#### 数据集准备

identity.json数据集是大模型的自我认知数据集，主要是让大模型知道自己是谁，可以做什么

将LLaMA-Factory\data下面的identity.json数据复制一份例如叫myidentity，修改里面的{{name}} 和 {{author}} ，并且把所有的英文数据删掉

#### 数据配置准备

需要修改LLaMA-Factory\data下面所有数据集的文件 dataset_info.json

```json
"myidentity": { # myidentity的名称和下面的配置文件中的dataset名称对应
    "file_name": "myidentity.json" # file_name要和LLaMA-Factory\data下面下面的数据集名称对应
               # 如果数据集在LLaMA-Factory\data文件夹下面的时候就不需要指定绝对路径，相对路径即可
  },
# 仿照下面的identity制作的
"identity": {
    "file_name": "identity.json"
  },
  "alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json"
  },
```

#### 数据增广

一般同一个问题使用十几二十种方法去问，再加上大模型的泛化，效果会很好

但是我们前面的数据集问句长，答案长，大模型还是0.5B，再怎么泛化大模型也不会有什么好的结果，所以我们需要做EMR数据增广，将问答对弄的多一点

从其他数据集中摘取一些数据添加到我们的数据集中 

例如：使用大模型 根据一个对话对 来进行增广生成一些数据对话对，其prompt设计如下所示：

```python
处理这个数据：
'''json
	*****
'''
要求如下：
1. 根据给出的数据内容，另外写出10条对话
2. 对话的格式跟原来的数据一致
3. 生成的内容不能脱离原来的对话内容
```

根据这个prompt就会生成很多的数据，这样的话我们原本只有200条数据，经过这个数据增广后会生成与之有关的2000条数据

**使用大模型生成数据，可以使用大模型api来完成结合多线程**

下面是使用智谱轻言大模型来进行数据增广的多线程代码

~~~python
# pip install zhipuai 请先在终端进行安装

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from zhipuai import ZhipuAI

def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def get_zhipu_response(prompt):
    client = ZhipuAI(api_key="0f56bcd3ce36d22b5b6564de4faeebfe.nvc4AlZb8rw1WhFG")
    response = client.chat.completions.create(
        model="glm-4-plus",
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
        top_p=0.7,
        temperature=0.95,
        max_tokens=1024
    )
    text = response.choices[0].message.content
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)

    if match:
        json_content = match.group(1)
        return json_content


def process_chunk(chunk, chunk_id):
    res = []
    exceptions = []
    for chat in tqdm(chunk, desc=f"Processing chunk: {chunk_id}"):
        prompt = f'''处理这个数据：
```json
{chat}
```
要求如下：
1. 根据给出的数据内容，另外写出20条对话
2. 对话的格式跟原来的数据一致
3. 生成的内容不能脱离原来的对话内容'''
        try:
            response = get_zhipu_response(prompt)
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            continue

        if response:
            try:
                res.extend(eval(response))
            except:
                exceptions.append(chat)
                continue

    return res


if __name__ == "__main__":
    data_root = "/root/workspace/data/output"
    os.makedirs(data_root, exist_ok=True)

    file = "/root/workspace/data/one_round_chats_100.json"
    all_chats = read_json(file)

    num_worker = 100
    chunk_size = len(all_chats) // num_worker

    chunks = [all_chats[i:i + chunk_size] for i in range(0, len(all_chats), chunk_size)]


    res = []
    with ThreadPoolExecutor(max_workers=num_worker) as executor:
        results = list(tqdm(executor.map(lambda chunk: process_chunk(chunk[1], chunk[0]), enumerate(chunks)), total=len(chunks), desc="Total Progress"))

    for result in results:
        res.extend(result)

    with open(f"{data_root}/one_round_chats-aug.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
~~~





### 配置文件修改

将LLaMA-Factory\examples\train_full\qwen2vl_full_sft.yaml复制一份名为：qwen1.5_0.5B_full_sft.yaml，我们在这个复制后的文件进行修改

其修改后的文件内容如下所示：

```sh
### model
model_name_or_path: path/to/Qwen1.5_0.5B***  # 更换模型路径为自己的  最好使用绝对路径  也可以放之前自己训练好的 他就会从自己之前训练出来的继续训练 

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json # deepspeed多机多卡配置文件

### dataset
dataset: myidentity # 修改为自己的数据集   这里放多个数据集 就是一起训练  必须要要和dataset_info.json中的数据名一致
template: qwen # 由原本的qwen2_vl 修改为qwen  这个在你下载的模型文件中可以找到
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16 # 数据处理线程
dataloader_num_workers: 4 # 数据分发线程

### output
output_dir: path/to/saves/qwen2_vl-7b/full/sft # 训练生成的模型权重等等  最好使用绝对路径  里面的每一个checkpoint文件夹都是一个模型
logging_steps: 10
save_steps: 500 # 每500轮就会保存一下   
plot_loss: true
overwrite_output_dir: true
save_only_model: true # 设置为True就不会保存中间的梯度，只保存模型本身 设置为False的目的是断点训练，可以随时在任何地方重启训练

### train
per_device_train_batch_size: 1 # 每个设备训练批次的size
gradient_accumulation_steps: 2 # 梯度累积
learning_rate: 1.0e-5 # 学习率
num_train_epochs: 3.0 # 训练多少轮  微调一般训练2-5的epoch
lr_scheduler_type: cosine # 学习率的策略-余弦退火学习率
warmup_ratio: 0.1 # 预热
bf16: true 
ddp_timeout: 180000000

### eval   下面没有注释掉便是打开状态
val_size: 0.1 # 把10%的数据作为测试集
per_device_eval_batch_size: 1
eval_strategy: steps # 以steps为标准去检验
eval_steps: 500 # 跑多少步去测试以下   一般先不开	 
```

注意：template模板如何选择

| **qwen**     | 通用文本生成/分类任务   | 基础模板，适合纯文本输入（如分类、生成）                     |
| ------------ | ----------------------- | ------------------------------------------------------------ |
| **qwen2_vl** | 多模态任务（图像+文本） | 扩展了视觉标记处理，适合Qwen2.5的多模态版本（如VL-*系列模型） |



#### 训练时的调整测率

将eval中的eval_steps的步数调小一些，比如调成5，然后以当前的学习率为中心调节学习率，向上或向下观察效果，此时就能得出学习率的探索区间



### 训练

在单机上进行指令监督微调

```
cd  path/to/workspace/LLaMA-Factory   # 必须要进入到 LLaMA-Factory项目里才可以执行
FORCE_TORCHRUN=1 llamafactory-cli train \
/root/workspace/LLaMA-Factory/examples/train_full/qwen1.5_0.5B_full_sft.yaml
```

![](img\39.png)

- Total optimization = (Num example * Num Epochs) / (Instantaneous batch size per device * 设备数 * Gradient Accumulation steps)   例如：75 *2 / 1 * 1 * 2 = 75 (从0开始就是74)
- Total train batch size = Instantaneous batch size per device * 设备数 * Gradient Accumulation steps





### 验证模型

可以使用前面模型测试的代码，只不过将路径更换为训练出来的模型权重路径

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/root/workspace/save/Qwen1.5-0.5B-Chat/full/sft",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/root/workspace/save/Qwen1.5-0.5B-Chat/full/sft")

while True:
    prompt = input("【用户】：")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"【Qwen】:{response}")
```

测试：

```sh
root@autodl-container-a574409304-5cb971d9:~/workspace# python model_infer.py 
[2025-04-15 16:24:57,391] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
【用户】：你是 谁？
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
【Qwen】:您好，我是由 gaoxu 发明的 mofei。我可以为您提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。
```





## 大模型精度提升

对于大模型回答问题的准确性，我们可以从两个方面去提升：1. 数据增加并且质量提升 2. 选择大一点的模型



## 辅助工具推荐

### tmux

tmux 是一个后台管理工具，可以将一些需要长时间运行的程序丢到里面去

安装

```
sudo apt-get update
sudo apt-get install tmux
```

常用命令

```sh
# 创建一个后台窗口
tmux new -s [后台名称] # 后台名称是自己命名的

# 退出后台窗口
ctrl+b  等一会... 再按 d

# 查看所有后台运行的session
tmux list-session

# kill 掉一个 session
tmux kill-session -t [session的名称]

# 进入一个 session
tmux attach -t [session的名称]

# 设置能滚动鼠标
ctrl+b  等一会... 再按 :
set -g mouse on
```

最明显的好处是：如果不使用tmux 你不小心关掉了命令行终端 或 再打开找不到命令行终端了，使用tmux就不会出现这样的问题