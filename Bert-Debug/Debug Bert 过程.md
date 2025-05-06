# VsCode Debug

## 安装扩展

无论是服务器还是个人电脑都需要去安装python扩展，python扩展会自动安装python Debugger的扩展

## Debug各个按钮介绍

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/10.png)

第一个按钮为：直接跳到下一个断点运行

第二个按钮为：逐行运行

第三个按钮为：进入函数体

第四个按钮为：跳出函数体

第五个按钮为：重新运行

第六个按钮为：终止调试

## 设置配置文件解决输入参数和工作目录路径问题

创建生成配置文件

进入Debug程序界面，点击create a launch.json file,选择python Debugger	 

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/2.png)

选择python Debugger后会出现很多选项，

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/3.png)

最上面的选项为：Debug 当前这个文件

第二个选项为：带参数Debug 当前这个文件

我们选择第二个 ，选择完成后，生成如下内容文件：

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/4.png)

生成的launch.json内容如下所示：

1. name：当前DEBUG配置的名称
2. type:语言
3. request是最重要的参数，它能选择两种类型，一个是launch模式，一个是attach模式：launch模式:由VS Code来启动一个独立的具有debug功能的程序(默认选择)attach模式：监听一个已启动的程序（其必须已经开启debug模式）
4. program: 文件的绝对路径，一般不需要改动
5. console: 终端的类型， integratedTerminal 指使用vscode终端
6. args:参数输入

生成的launch.json文件在你的项目中的.vscode文件夹下面

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/9.png)

- 解决工作目录路径问题

可以通过设置cwd来避免这个问题

```json
// 工作目录路径问题设置
"cwd": "${fileDirname}", // 设置工作目录为当前打开文件的所在目录
或者
"cwd": "${workspaceFolder}/src", //  "cwd": "${workspaceFolder}"设置工作目录为当前工作空间的根目录  后面加/src 是相对于工作空间根目录
```

- 解决输入参数的问题

可以通过直接在args上传入指定的参数来解决

```json
// 输入参数问题解决
// "args": "${command:pickArgs}", 让用户在启动调试会话的时候，通过一个交互式输入框手动选择或输入参数
"args": ["--weights", "best.pt",
         "--imgsize", "640",
         "--cuda", "0"
        ]   
```

使用了launch.json配置文件去Debug的时候，选择python Debugger:Debug using launch.json这个选项

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/5.png)

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/6.png)

如果使用的是"args": "${command:pickArgs}"，则在Debug的时候在下面的命令行窗口输入参数

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/7.png)

可以一次写多个配置，每个配置取一个名字，就像下面一样

![](E:/%E9%AB%98%E6%A0%A1%E5%9F%B9%E8%AE%AD/%E5%8D%97%E8%88%AA%E5%AE%9E%E8%AE%AD/md/img/VscodeDebug/8.png)

debuger配置`.vscode`下`launch.json`添加

```json
{
    "name": "Python Debugger: Current File with Arguments",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",  // 指定文件
    // 工作目录路径问题设置
    // "cwd": "${fileDirname}", // 设置工作目录为当前打开文件的所在目录
    // "cwd": "${workspaceFolder}/src", //  "cwd": "${workspaceFolder}"设置工作目录为当前工作空间的根目录  后面加/src 是相对于工作空间根目录
    "console": "integratedTerminal",
    // "justMyCode": true // false表示可以进入第三方库（如Pytorch）里进行调试

    // 输入参数问题解决
    "args": "${command:pickArgs}", //让用户在启动调试会话的时候，通过一个交互式输入框手动选择或输入参数
    // "args": ["--weights", "best.pt",
    //         "--imgsize", "640",
    //         "--cuda", "0"
    //     ]       
}
```

## Vs Code json文件中的变量解释

```json
以：D://Users//XueLi_G//Desktop//NUAA//.vscode//tasks.json 为例
${workspaceFolder} :表示当前workspace文件夹路径，也即D://Users//XueLi_G//Desktop//NUAA
${workspaceRootFolderName}:表示workspace的文件夹名，也即NUAA
${file}:文件自身的绝对路径，也即D://Users//XueLi_G//Desktop//NUAA//.vscode//tasks.json
${relativeFile}:文件在workspace中的路径，也即.vscode//tasks.json
${fileBasenameNoExtension}:当前文件的文件名，不带后缀，也即tasks
${fileBasename}:当前文件的文件名，tasks.json
${fileDirname}:文件所在的文件夹路径，也即D://Users//XueLi_G//Desktop//NUAA//.vscode
${fileExtname}:当前文件的后缀，也即.json
${lineNumber}:当前文件光标所在的行号
${env:PATH}:系统中的环境变量
```

# Debug BertModel的过程

## 入口代码

自己或找大模型编写一个使用Bert_Model的代码

```python
from transformers import AutoModel, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 导入分词器
Bert_Tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/tiansz/bert-base-chinese")

# 导入模型
Bert_Model = AutoModel.from_pretrained("/root/.cache/modelscope/hub/models/tiansz/bert-base-chinese").to(device)

# BertEncoder = Bert_Model.encoder.layer #  ModuleList 有12个 BertLayer
# BertEmbeddings = Bert_Model.embeddings 
'''
(
    (word_embeddings): Embedding(21128, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
'''

input_text = "高"
# 使用分词器对输入文本进行编码
# add_special_tokens=True 会自动添加 [CLS] 和 [SEP] 标记  表示任务开始 和 任务结束
# return_tensors="pt"用于指定返回的数据类型。它告诉分词器（如AutoTokenizer）将编码后的数据返回为 PyTorch 张量（Tensor）。这使得数据可以直接用于 PyTorch 模型的输入。
inputs = Bert_Tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
# {'input_ids': tensor([[ 101, 7770,  102]]), 'token_type_ids': tensor([[0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1]])}

# 获取输入张量
input_ids = inputs["input_ids"].to(device) # tensor([[ 101, 7770,  102]])
attention_mask = inputs["attention_mask"].to(device) # tensor([[1, 1, 1]]

# 将输入传递给模型
outputs = Bert_Model(input_ids, attention_mask=attention_mask)

# 打印输出
print("Last hidden state shape:", outputs.last_hidden_state.shape)
print("Pooler output shape:", outputs.pooler_output.shape)

# Last hidden state shape: torch.Size([1, 3, 768])
# Pooler output shape: torch.Size([1, 768])
```

## 设置

VsoCode使用Launch.json文件进行debug调试，并且需要在json文件中加入如下命令，

```sh
"justMyCode": false
```

加入如下的命令可以让遇到非自己的代码时进行跳转：

## 过程

将断点打在 Bert_Model = AutoModel.from_pretrained(model_path) 和 outputs = Bert_Model(input_ids, attention_mask=attention_mask) 两处，使用Launch.json文件开始debug，直接跳转到第二处断点，点击进入函数体

![](img\1.png)

此时会跳转进_ wrapped_ call_impl函数中，逐行运行到return self._ call_impl(*args, **kwargs) ，再次点击进入函数体

![](img\2.png)

此时跳转进 _ call_ impl函数中给，不断逐行运行，直到return forward_call(*args, **kwargs)，此时再次点击进入函数体

![](img\3.png)

到此我们便进入到了BertModel的源码位置，其在我服务器上的地址为：/root/miniconda3/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py

给forward的第一行和 init的第一行有效代码打上断点，下次在给前面的入口代码打断点的时候，debug可以直接进入到这里；



## BertConfig

在debug的过程中会经常看到BertConfig的相关信息：

Bert的BertConfig的一些参数如下，可以当作字典查询

```python
BertConfig {
  "_attn_implementation_autoset": true,
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.50.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
```

