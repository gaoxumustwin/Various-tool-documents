# OpenCompass

## 介绍

git:   https://github.com/open-compass/opencompass

doc: https://opencompass.org.cn/doc



## 大模型测评

1. 在测试集上的准确度
2. 并发性能
3. 速度上的性能  （1个人访问和多个人的访问大模型吐字的速度是不一样的）



## 任务

1. 通用能力评测：配置opencompass环境，得出qwen0.5B 和1.5B对话模型在C-EVAL、cmmlu和GSM8K （均是前1000题）的成绩并且观察分数在不同大小模型上的体现？

2. 测试0.5B和1.5B的并发性能：1并发每秒多少token、10并发每秒多少token



## 准备

### 服务器环境

Autodl

PyTorch  2.1.2

Python  3.10   -  (ubuntu22.04)

CUDA  11.8

GPU    RTX 3090 (24GB)

### 环境准备

1. 需要注意的是，OpenCompass 要求 `pytorch>=1.13`  python>=3.10。

2. 安装 OpenCompass：

   可以使用pip安装，但是如果希望使用 OpenCompass 的最新功能，需要从源代码构建它：

```sh
source /etc/network_turbo #autodl的科学上网

git clone https://github.com/open-compass/opencompass 
cd opencompass
pip install -e .
```



## 工作流

### 数据准备

#### 机制介绍

**在OpenCompass中，评估数据只需解压到项目目录下即可被自动找到，而不需要在评估指令中显式指定路径，这得益于其智能路径解析机制和预定义的目录结构规范**。

OpenCompass 采用约定优于配置的设计理念，**强制要求数据必须存放在特定目录下（如 `data/` 或 `opencompass/data/`）**

```sh
opencompass/
├── data/          # 默认数据目录
│   ├── math/ 		# 示例：math 数据集
│	├── ceval		# 示例：ceval 数据集
│	├── cmmul/		# 示例：cmmul 数据集
│	├── gsm8k/		# 示例：gsm8k 数据集
│   └── ...        # 其他数据集
├── tools/
└── ...
```

OpenCompass 的代码中**硬编码了数据查找路径**，默认会扫描 `data/` 或配置文件指定的目录。

#### 配置数据集

​	配置数据集实际上指的是OpenCompass为每个数据集都实现了数据配置代码文件，这些代码文件在OpenCompass项目中的 `opencompass/configs/datasets` 目录下的结构，如下所示：

```
configs/datasets/
├── agieval
├── apps
├── ARC_c
├── ...
├── CLUE_afqmc  # 数据集
│   ├── CLUE_afqmc_gen_901306.py  # 不同版本数据集配置文件
├── cmmlu  # 数据集
│   ├──cmmlu_0shot_cot_gen_305931.py  # 不同版本数据集配置文件
│   ├──cmmlu_0shot_nocot_llmjudge_gen_e1cd9a.py
│   ├──cmmlu_gen.py
│   ├──cmmlu_gen_c13365.py
│   ├──cmmlu_llm_judge_gen.py
│   ├──cmmlu_llmjudge_gen_e1cd9a.py
│   ├──cmmlu_ppl.py
│   ├──cmmlu_ppl_041cbf.py
│   ├──cmmlu_ppl_8b9c76.py
│   ├──cmmlu_stem_0shot_nocot_gen_3653db.py
│   ├──cmmlu_stem_0shot_nocot_llmjudge_gen_3653db.py
│   └──cmmlu_stem_0shot_nocot_xml_gen_3653db.py
├── ...
├── XLSum
├── Xsum
└── z_bench
```

在  `opencompass/configs/datasets` 目录结构下，存储了`opencompass`默认支持的所有数据集，**在各个数据集对应的文件夹下存在多个数据集配置文件代码，在评估的时候指定的数据集的名称实际上指定的是数据配置文件代码的名称，这是因为：**

数据集配置文件代码名由以下命名方式构成 `{数据集名称}_{评测方式}_{prompt版本号}.py`，以 `cmmlu/cmmlu_gen_c13365.py` 为例，该配置文件则为**cmmlu**数据集，对应的评测方式为 `gen`，即生成式评测，对应的prompt版本号为 `c13365`；同样的， `cmmlu_ppl_041cbf.py` 指评测方式为`ppl`即判别式评测，prompt版本号为 `041cbf`。

数据集配置通常有两种类型：`ppl` 和 `gen`，分别指示使用的评估方法。其中 `ppl` 表示辨别性评估，`gen` 表示生成性评估。对话模型仅使用 `gen` 生成式评估。

除此之外，不带版本号的文件，例如：` cmmlu_gen.py` 则指向该评测方式最新的prompt配置文件，通常来说会是精度最高的prompt。

**不同的数据集 评估速度是不一样的，数据集指定的名称查看如下所示：**

例如：

```
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace/opencompass# python tools/list_configs.py cmmlu
+--------------------------------------------+----------------------------------------------------------------------------------+
| Dataset                                    | Config Path                                                                      |
|--------------------------------------------+----------------------------------------------------------------------------------|
| cmmlu_0shot_cot_gen_305931                 | opencompass/configs/datasets/cmmlu/cmmlu_0shot_cot_gen_305931.py                 |
| cmmlu_0shot_nocot_llmjudge_gen_e1cd9a      | opencompass/configs/datasets/cmmlu/cmmlu_0shot_nocot_llmjudge_gen_e1cd9a.py      |
| cmmlu_gen                                  | opencompass/configs/datasets/cmmlu/cmmlu_gen.py                                  |
| cmmlu_gen_c13365                           | opencompass/configs/datasets/cmmlu/cmmlu_gen_c13365.py                           |
| cmmlu_llm_judge_gen                        | opencompass/configs/datasets/cmmlu/cmmlu_llm_judge_gen.py                        |
| cmmlu_llmjudge_gen_e1cd9a                  | opencompass/configs/datasets/cmmlu/cmmlu_llmjudge_gen_e1cd9a.py                  |
| cmmlu_ppl                                  | opencompass/configs/datasets/cmmlu/cmmlu_ppl.py                                  |
| cmmlu_ppl_041cbf                           | opencompass/configs/datasets/cmmlu/cmmlu_ppl_041cbf.py                           |
| cmmlu_ppl_8b9c76                           | opencompass/configs/datasets/cmmlu/cmmlu_ppl_8b9c76.py                           |
| cmmlu_stem_0shot_nocot_gen_3653db          | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_gen_3653db.py          |
| cmmlu_stem_0shot_nocot_llmjudge_gen_3653db | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_llmjudge_gen_3653db.py |
| cmmlu_stem_0shot_nocot_xml_gen_3653db      | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_xml_gen_3653db.py      |
| demo_cmmlu_base_ppl                        | opencompass/configs/datasets/demo/demo_cmmlu_base_ppl.py                         |
| demo_cmmlu_chat_gen                        | opencompass/configs/datasets/demo/demo_cmmlu_chat_gen.py                         |
+--------------------------------------------+----------------------------------------------------------------------------------+
```

#### 数据配置代码文件的内容

在各个数据集配置的代码文件中，数据集将会被定义在 `{}_datasets` 变量当中，例如下面 `cmmlu/cmmlu_0shot_cot_gen_305931.py`（cmmlu_gen的数据配置代码文件中是使用 `mmengine` 配置中直接import的机制来构建数据集，导入的cmmlu_0shot_cot_gen_305931.py的cmmlu_datasets） 中的 `cmmlu_datasets`

```python
cmmlu_datasets.append(
    dict(
        type=CMMLUDataset,
        path='opencompass/cmmlu',
        name=_name,
        abbr=f'cmmlu-{_name}',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer',
            train_split='dev',
            test_split='test'),
        infer_cfg=cmmlu_infer_cfg,
        eval_cfg=cmmlu_eval_cfg,
    )
)
```

以及 `ceval/ceval_gen_5f30c7.py` 中的 `ceval_datasets`。（ceval_gen的数据配置代码文件中是使用 `mmengine` 配置中直接import的机制来构建数据集，导入的是ceval_gen_5f30c7.py的ceval_datasets）

```python
ceval_datasets.append(
    dict(
        type=CEvalDataset,
        path='opencompass/ceval-exam',
        name=_name,
        abbr='ceval-' + _name if _split == 'val' else 'ceval-test-' +
        _name,
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer',
            train_split='dev',
            test_split=_split),
        infer_cfg=ceval_infer_cfg,
        eval_cfg=ceval_eval_cfg,
    ))
```

还有 `gsm8k/gsm8k_gen_1d7fe4.py` 中的 gsm8k_datasets。（gsm8k_gen的数据配置代码文件中是使用 `mmengine` 配置中直接import的机制来构建数据集，导入的gsm8k_gen_1d7fe4.py的gsm8k_datasets）  

```python
gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
```

**以上面的三个数据集为例， 如果想同时评测这三个数据集，可以在 `configs` 目录下新建一个配置文件，我们使用 `mmengine` 配置中直接import的机制来构建数据集部分的参数，如下所示：**

```python
from mmengine.config import read_base

with read_base():
    from .cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403
    from .ceval_gen_5f30c7 import ceval_datasets  # noqa: F401, F403
    from .gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, F403

datasets = []
datasets += cmmlu_datasets
datasets += ceval_datasets
datasets += gsm8k_datasets
```

由此我们可以根据需要，选择不同能力不同数据集以及不同评测方式的配置文件来构建评测脚本中数据集的部分。

#### 数据配置代码文件的其它策略

##### 数据集多次评测

在数据集配置中可以通过设置参数`n`来对同一数据集进行多次评测，最终返回平均指标，例如：

```python
cmmlu_datasets.append(
    dict(
        type=CMMLUDataset,
        path='opencompass/cmmlu',
        name=_name,
        n=10, # 进行10次评测
        abbr=f'cmmlu-{_name}',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer',
            train_split='dev',
            test_split='test'),
        infer_cfg=cmmlu_infer_cfg,
        eval_cfg=cmmlu_eval_cfg,
    )
)
```

##### 设置评测指标

对于二值评测指标（例如accuracy，pass-rate等），还可以通过设置参数`k`配合`n`进行[G-Pass@k](http://arxiv.org/abs/2412.13147)评测。

其中 n 为评测次数, c 为 n 次运行中通过或正确的次数。配置例子如下：

```
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='opencompass/aime2024',
        k=[2, 4], # 返回 G-Pass@2和G-Pass@4的结果
        n=12, # 12次评测
        ...
    )
]
```

#### 数据集下载

`OpenCompassData-core-***.zip` 是 OpenCompass 项目发布的一个核心数据集集合，包含了用于模型评测的各种基准数据集。这些数据集是 OpenCompass 进行模型评估和对比的基础。

数据下载到opencompass目录下面，直接解压即可，不需要改名字，也不要移路径，解压后就是一个data目录。

```sh
 # 下载数据集到 opencompass/ 处
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

解压打印可以看到有非常多的数据集

```
(base) root@autodl-container-509c44b497-bb5040a3:~/autodl-tmp/.autodl/workspace/opencompass/data# ls
AGIEval  CLUE          LCSTS      Xsum   commonsenseqa    gsm8k      lambada  mmlu        piqa  strategyqa  tydiqa
ARC      FewCLUE       SuperGLUE  ceval  drop             hellaswag  math     nq          race  summedits   winogrande
BBH      GAOKAO-BENCH  TheoremQA  cmmlu  flores_first100  humaneval  mbpp     openbookqa  siqa  triviaqa    xstory_cloze
```



### 模型准备

使用魔塔社区的SDK下载

```sh
pip install modelscope
```

- 0.5B Qwen chat 

魔塔社区的地址：https://www.modelscope.cn/models/Qwen/Qwen1.5-0.5B-Chat

```
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen1.5-0.5B-Chat')
```

- 1.5B Qwen2.5 instruct

```sh
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
```





### 启动评估

#### 接入vLLM

加速评估

#### 命令行评估

在 OpenCompass 中，每个评估任务由待评估的模型和数据集组成。评估的入口点是 `run.py`。用户可以通过命令行或配置文件选择要测试的模型和数据集。

用户可以通过命令行直接设置模型参数，无需额外的配置文件。

例如，对于Qwen1.5-0.5B-Chat 模型，您可以使用以下命令进行评估：

由于 OpenCompass 默认并行启动评估过程，我们可以在第一次运行时以 `--debug` 模式启动评估，并检查是否存在问题，**第一次如果没发生问题则去掉--debug模型**

```sh
python run.py \
    --hf-type chat \
    --datasets demo_cmmlu_chat_gen \
    --hf-path /root/.cache/modelscope/hub/models/Qwen/Qwen1___5-0___5B-Chat \
    --tokenizer-path /root/.cache/modelscope/hub/models/Qwen/Qwen1___5-0___5B-Chat \
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
    --model-kwargs trust_remote_code=True device_map='auto' \
    --max-seq-len 2048 \
    --max-out-len 16 \
    --batch-size 4 \
    --hf-num-gpus 1 \
    --debug	
```

对于上面的命令进行详解：

```sh
python run.py \
	--hf-type chat \  # HuggingFace 模型类型，可选值为 chat 或 base  chat是对话模型  base是基类模型
	--datasets demo_cmmlu_chat_gen \ # 数据集  可使用tools/list_configs.py查看
	--hf-path /path/to/Qwen1.5-0.5B-Chat/ \ # HuggingFace 模型路径
	--tokenizer-path /path/to/Qwen1.5-0.5B-Chat/ \ # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
	--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \ # 构建 tokenizer 的参数   后面的是他的三个参数
	--model-kwargs trust_remote_code=True device_map='auto' \ # 构建模型的参数
	--max-seq-len 2048 \ # 模型可以接受的最大序列长度
	--max-out-len 16 \ # 生成的最大 token 数
	--batch-size 4 \ # 批量大小
	--num-gpus 1 \ # 运行一个模型实例所需的 GPU 数量
	--debug # 在 `--debug` 模式下，任务将按顺序执行，并实时打印输出。
```

模型和数据集的配置文件预存于 `configs/models` 和 `configs/datasets` 中。用户可以使用 `tools/list_configs.py` 查看或过滤当前可用的模型和数据集配置。

```sh
# 以下命令要在opencompass中执行

# 列出所有配置
python tools/list_configs.py
# 列出与 Qwen 和mmlu相关的所有配置
python tools/list_configs.py Qwen cmmlu

# 列出与cmmlu相关的所有配置
python tools/list_configs.py cmmlu
```

例如：运行 将产生如下输出：

```sh
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace/opencompass# python tools/list_configs.py cmmlu
+--------------------------------------------+----------------------------------------------------------------------------------+
| Dataset                                    | Config Path                                                                      |
|--------------------------------------------+----------------------------------------------------------------------------------|
| cmmlu_0shot_cot_gen_305931                 | opencompass/configs/datasets/cmmlu/cmmlu_0shot_cot_gen_305931.py                 |
| cmmlu_0shot_nocot_llmjudge_gen_e1cd9a      | opencompass/configs/datasets/cmmlu/cmmlu_0shot_nocot_llmjudge_gen_e1cd9a.py      |
| cmmlu_gen                                  | opencompass/configs/datasets/cmmlu/cmmlu_gen.py                                  |
| cmmlu_gen_c13365                           | opencompass/configs/datasets/cmmlu/cmmlu_gen_c13365.py                           |
| cmmlu_llm_judge_gen                        | opencompass/configs/datasets/cmmlu/cmmlu_llm_judge_gen.py                        |
| cmmlu_llmjudge_gen_e1cd9a                  | opencompass/configs/datasets/cmmlu/cmmlu_llmjudge_gen_e1cd9a.py                  |
| cmmlu_ppl                                  | opencompass/configs/datasets/cmmlu/cmmlu_ppl.py                                  |
| cmmlu_ppl_041cbf                           | opencompass/configs/datasets/cmmlu/cmmlu_ppl_041cbf.py                           |
| cmmlu_ppl_8b9c76                           | opencompass/configs/datasets/cmmlu/cmmlu_ppl_8b9c76.py                           |
| cmmlu_stem_0shot_nocot_gen_3653db          | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_gen_3653db.py          |
| cmmlu_stem_0shot_nocot_llmjudge_gen_3653db | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_llmjudge_gen_3653db.py |
| cmmlu_stem_0shot_nocot_xml_gen_3653db      | opencompass/configs/datasets/cmmlu/cmmlu_stem_0shot_nocot_xml_gen_3653db.py      |
| demo_cmmlu_base_ppl                        | opencompass/configs/datasets/demo/demo_cmmlu_base_ppl.py                         |
| demo_cmmlu_chat_gen                        | opencompass/configs/datasets/demo/demo_cmmlu_chat_gen.py                         |
+--------------------------------------------+----------------------------------------------------------------------------------+
```

用户可以使用第一列中的名称作为 `python run.py` 中 `--models` 和 `--datasets` 的输入参数。对于数据集，同一名称的不同后缀通常表示其提示或评估方法不同。

**执行评估指令后** 如果一切正常，您应该看到屏幕上显示 “Starting inference process”，且进度条开始前进：

```
[2023-07-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

#### 配置文件评估

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

这里举一个例子： 以对话模型Qwen-0.5B-chat 和 Qwen2-1.5B-Instruct 在 gsm8k、 ceval、cmmlu 下采样数据集上的评估。它们的配置文件可以在opencompass/examples中找到。

此配置通过 [继承机制](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/config.html#id3) 引入所需的数据集和模型配置，并以所需格式组合 `datasets` 和 `models` 字段。

代码的内容如下：

```python
from mmengine.config import read_base

with read_base():
    # 数据读入
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, F403

    # 模型读入
    from opencompass.configs.models.qwen.hf_qwen1_5_0_5b_chat import \
        models as hf_qwen1_5_0_5b_chat_1_5b_models
    from opencompass.configs.models.qwen.hf_qwen2_1_5b_instruct import \
        models as hf_qwen2_1_5b_chat_1_5b_models

datasets = gsm8k_datasets + cmmlu_datasets + ceval_datasets
models = hf_qwen1_5_0_5b_chat_1_5b_models + hf_qwen2_1_5b_chat_1_5b_models
```

运行任务时，我们只需将配置文件的路径传递给 `run.py`即可：

```
python run.py examples/eval_chat_demo_qwen.py --debug
```

我们可以在第一次运行时以 `--debug` 模式启动评估，并检查是否存在问题，**第一次如果没发生问题则去掉--debug模型**

这里要强调两个东西：1.models 2. datasets

- models

OpenCompass 提供了一系列预定义的模型配置，位于 `configs/models` 下。以下是与 [InternLM2-Chat-1.8B](https://github.com/open-compass/opencompass/blob/main/configs/models/hf_internlm/hf_internlm2_chat_1_8b.py)（`configs/models/hf_internlm/hf_internlm2_chat_1_8b.py`）相关的配置片段：

```python
# 使用 `HuggingFacewithChatTemplate` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的对话模型
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='internlm2-chat-1.8b-hf',         # 模型的缩写
        path='internlm/internlm2-chat-1_8b',   # 模型的 HuggingFace 路径
        max_out_len=1024,                      # 生成的最大 token 数
        batch_size=8,                          # 批量大小
        run_cfg=dict(num_gpus=1),              # 该模型所需的 GPU 数量
    )
]
```

使用配置时，我们可以通过命令行参数 `--models` 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 `models` 列表中。

注意：

使用配置文件进行评估的时候，需要去找到对应的模型配置文件修改模型的路径为本地路径，例如：修改/root/autodl-tmp/.autodl/workspace/opencompass/examples/eval_chat_demo_qwen0.5B.py中的path

- datasets

与模型类似，数据集的配置文件也提供在 `configs/datasets` 下。用户可以在命令行中使用 `--datasets`，或通过继承在配置文件中导入相关配置

下面是来自 `configs/eval_chat_demo.py` 的与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets  # 读取 GSM8K 配置，使用 4-shot，基于生成式进行评估
    from .datasets.demo.demo_math_chat_gen import math_datasets    # 读取 MATH 配置，使用 0-shot，基于生成式进行评估

datasets = gsm8k_datasets + math_datasets       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：`ppl` 和 `gen`，分别指示使用的评估方法。其中 `ppl` 表示辨别性评估，`gen` 表示生成性评估。对话模型仅使用 `gen` 生成式评估。

此外，[configs/datasets/collections](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections) 收录了各种数据集集合，方便进行综合评估。OpenCompass 通常使用 [`chat_OC15.py`](https://github.com/open-compass/opencompass/blob/main/configs/dataset_collections/chat_OC15.py) 进行全面的模型测试。要复制结果，只需导入该文件，例如：

```sh
python run.py --models hf_internlm2_chat_1_8b --datasets chat_OC15 --debug
```

我们可以去掉--debug模式，以正常模式运行以下命令：

```
python run.py examples/eval_chat_demo_qwen.py -w outputs/demo
```

在正常模式下，评估任务将在后台并行执行，其输出将被重定向到输出目录 `outputs/demo/{TIMESTAMP}`。前端的进度条只指示已完成任务的数量，而不考虑其成功或失败。**任何后端任务失败都只会在终端触发警告消息。**

#### 评估日志

```sh
(base) root@autodl-container-509c44b497-bb5040a3:~/autodl-tmp/.autodl/workspace/opencompass# python run.py examples/eval_chat_demo_qwen.py 
04/28 22:20:03 - OpenCompass - INFO - Current exp folder: outputs/default/20250428_222003
04/28 22:20:04 - OpenCompass - WARNING - SlurmRunner is not used, so the partition argument is ignored.
# 两个模型两个任务
04/28 22:20:04 - OpenCompass - INFO - Partitioned into 2 tasks.
# 每个模型对应的数据集进行评估
launch OpenICLInfer[qwen1.5-0.5b-chat-hf/demo_gsm8k,qwen1.5-0.5b-chat-hf/cmmlu-agronomy,qwen1.5-0.5b-chat-hf/cmmlu-anatomy,qwen1.5-0.5b-chat-hf/cmmlu-ancient_chinese,qwen1.5-0.5b-chat-hf/cmmlu-arts,qwen1.5-0.5b-chat-hf/cmmlu-astronomy,qwen1.5-0.5b-chat-hf/cmmlu-business_ethics,qwen1.5-0.5b-chat-hf/cmmlu-chinese_civil_service_exam,qwen1.5-0.5b-chat-hf/cmmlu-chinese_driving_rule,qwen1.5-0.5b-chat-hf/cmmlu-chinese_food_culture,qwen1.5-0.5b-chat-hf/cmmlu-chinese_foreign_policy,qwen1.5-0.5b-chat-hf/cmmlu-chinese_history,qwen1.5-0.5b-chat-hf/cmmlu-chinese_literature,qwen1.5-0.5b-chat-hf/cmmlu-chinese_teacher_qualification,qwen1.5-0.5b-chat-hf/cmmlu-clinical_knowledge,qwen1.5-0.5b-chat-hf/cmmlu-college_actuarial_science,qwen1.5-0.5b-chat-hf/cmmlu-college_education,qwen1.5-0.5b-chat-hf/cmmlu-college_engineering_hydrology,qwen1.5-0.5b-chat-hf/cmmlu-college_law,qwen1.5-0.5b-chat-hf/cmmlu-college_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-college_medical_statistics,qwen1.5-0.5b-chat-hf/cmmlu-college_medicine,qwen1.5-0.5b-chat-hf/cmmlu-computer_science,qwen1.5-0.5b-chat-hf/cmmlu-computer_security,qwen1.5-0.5b-chat-hf/cmmlu-conceptual_physics,qwen1.5-0.5b-chat-hf/cmmlu-construction_project_management,qwen1.5-0.5b-chat-hf/cmmlu-economics,qwen1.5-0.5b-chat-hf/cmmlu-education,qwen1.5-0.5b-chat-hf/cmmlu-electrical_engineering,qwen1.5-0.5b-chat-hf/cmmlu-elementary_chinese,qwen1.5-0.5b-chat-hf/cmmlu-elementary_commonsense,qwen1.5-0.5b-chat-hf/cmmlu-elementary_information_and_technology,qwen1.5-0.5b-chat-hf/cmmlu-elementary_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-ethnology,qwen1.5-0.5b-chat-hf/cmmlu-food_science,qwen1.5-0.5b-chat-hf/cmmlu-genetics,qwen1.5-0.5b-chat-hf/cmmlu-global_facts,qwen1.5-0.5b-chat-hf/cmmlu-high_school_biology,qwen1.5-0.5b-chat-hf/cmmlu-high_school_chemistry,qwen1.5-0.5b-chat-hf/cmmlu-high_school_geography,qwen1.5-0.5b-chat-hf/cmmlu-high_school_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-high_school_physics,qwen1.5-0.5b-chat-hf/cmmlu-high_school_politics,qwen1.5-0.5b-chat-hf/cmmlu-human_sexuality,qwen1.5-0.5b-chat-hf/cmmlu-international_law,qwen1.5-0.5b-chat-hf/cmmlu-journalism,qwen1.5-0.5b-chat-hf/cmmlu-jurisprudence,qwen1.5-0.5b-chat-hf/cmmlu-legal_and_moral_basis,qwen1.5-0.5b-chat-hf/cmmlu-logical,qwen1.5-0.5b-chat-hf/cmmlu-machine_learning,qwen1.5-0.5b-chat-hf/cmmlu-management,qwen1.5-0.5b-chat-hf/cmmlu-marketing,qwen1.5-0.5b-chat-hf/cmmlu-marxist_theory,qwen1.5-0.5b-chat-hf/cmmlu-modern_chinese,qwen1.5-0.5b-chat-hf/cmmlu-nutrition,qwen1.5-0.5b-chat-hf/cmmlu-philosophy,qwen1.5-0.5b-chat-hf/cmmlu-professional_accounting,qwen1.5-0.5b-chat-hf/cmmlu-professional_law,qwen1.5-0.5b-chat-hf/cmmlu-professional_medicine,qwen1.5-0.5b-chat-hf/cmmlu-professional_psychology,qwen1.5-0.5b-chat-hf/cmmlu-public_relations,qwen1.5-0.5b-chat-hf/cmmlu-security_study,qwen1.5-0.5b-chat-hf/cmmlu-sociology,qwen1.5-0.5b-chat-hf/cmmlu-sports_science,qwen1.5-0.5b-chat-hf/cmmlu-traditional_chinese_medicine,qwen1.5-0.5b-chat-hf/cmmlu-virology,qwen1.5-0.5b-chat-hf/cmmlu-world_history,qwen1.5-0.5b-chat-hf/cmmlu-world_religions,qwen1.5-0.5b-chat-hf/ceval-computer_network,qwen1.5-0.5b-chat-hf/ceval-operating_system,qwen1.5-0.5b-chat-hf/ceval-computer_architecture,qwen1.5-0.5b-chat-hf/ceval-college_programming,qwen1.5-0.5b-chat-hf/ceval-college_physics,qwen1.5-0.5b-chat-hf/ceval-college_chemistry,qwen1.5-0.5b-chat-hf/ceval-advanced_mathematics,qwen1.5-0.5b-chat-hf/ceval-probability_and_statistics,qwen1.5-0.5b-chat-hf/ceval-discrete_mathematics,qwen1.5-0.5b-chat-hf/ceval-electrical_engineer,qwen1.5-0.5b-chat-hf/ceval-metrology_engineer,qwen1.5-0.5b-chat-hf/ceval-high_school_mathematics,qwen1.5-0.5b-chat-hf/ceval-high_school_physics,qwen1.5-0.5b-chat-hf/ceval-high_school_chemistry,qwen1.5-0.5b-chat-hf/ceval-high_school_biology,qwen1.5-0.5b-chat-hf/ceval-middle_school_mathematics,qwen1.5-0.5b-chat-hf/ceval-middle_school_biology,qwen1.5-0.5b-chat-hf/ceval-middle_school_physics,qwen1.5-0.5b-chat-hf/ceval-middle_school_chemistry,qwen1.5-0.5b-chat-hf/ceval-veterinary_medicine,qwen1.5-0.5b-chat-hf/ceval-college_economics,qwen1.5-0.5b-chat-hf/ceval-business_administration,qwen1.5-0.5b-chat-hf/ceval-marxism,qwen1.5-0.5b-chat-hf/ceval-mao_zedong_thought,qwen1.5-0.5b-chat-hf/ceval-education_science,qwen1.5-0.5b-chat-hf/ceval-teacher_qualification,qwen1.5-0.5b-chat-hf/ceval-high_school_politics,qwen1.5-0.5b-chat-hf/ceval-high_school_geography,qwen1.5-0.5b-chat-hf/ceval-middle_school_politics,qwen1.5-0.5b-chat-hf/ceval-middle_school_geography,qwen1.5-0.5b-chat-hf/ceval-modern_chinese_history,qwen1.5-0.5b-chat-hf/ceval-ideological_and_moral_cultivation,qwen1.5-0.5b-chat-hf/ceval-logic,qwen1.5-0.5b-chat-hf/ceval-law,qwen1.5-0.5b-chat-hf/ceval-chinese_language_and_literature,qwen1.5-0.5b-chat-hf/ceval-art_studies,qwen1.5-0.5b-chat-hf/ceval-professional_tour_guide,qwen1.5-0.5b-chat-hf/ceval-legal_professional,qwen1.5-0.5b-chat-hf/ceval-high_school_chinese,qwen1.5-0.5b-chat-hf/ceval-high_school_history,qwen1.5-0.5b-chat-hf/ceval-middle_school_history,qwen1.5-0.5b-chat-hf/ceval-civil_servant,qwen1.5-0.5b-chat-hf/ceval-sports_science,qwen1.5-0.5b-chat-hf/ceval-plant_protection,qwen1.5-0.5b-chat-hf/ceval-basic_medicine,qwen1.5-0.5b-chat-hf/ceval-clinical_medicine,qwen1.5-0.5b-chat-hf/ceval-urban_and_rural_planner,qwen1.5-0.5b-chat-hf/ceval-accountant,qwen1.5-0.5b-chat-hf/ceval-fire_engineer,qwen1.5-0.5b-chat-hf/ceval-environmental_impact_assessment_engineer,qwen1.5-0.5b-chat-hf/ceval-tax_accountant,qwen1.5-0.5b-chat-hf/ceval-physician] on GPU 0  # 使用GPU0 进行评估
```





### 可视化评估结果

评估完成后，评估结果表格将打印如下：

```
dataset     version    metric    mode      qwen2-1.5b-instruct-hf    internlm2-chat-1.8b-hf
----------  ---------  --------  ------  ------------------------  ------------------------
demo_gsm8k  1d7fe4     accuracy  gen                        56.25                     32.81
demo_math   393424     accuracy  gen                        18.75                     14.06
```

所有运行输出将定向到 `outputs/demo/` 目录，结构如下：

```
outputs/default/
├── 20200220_120000
├── 20230220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...
```

打印评测结果的过程可被进一步定制化，用于输出一些数据集的平均分 (例如 MMLU, C-Eval 等)。

另外，所有指定-r 但是没有指定对应时间戳将会按照排序选择最新的文件夹作为输出目录。

运行日志：

```
04/28 22:20:13 - OpenCompass - INFO - Task [qwen1.5-0.5b-chat-hf/demo_gsm8k,qwen1.5-0.5b-chat-hf/cmmlu-agronomy,qwen1.5-0.5b-chat-hf/cmmlu-anatomy,qwen1.5-0.5b-chat-hf/cmmlu-ancient_chinese,qwen1.5-0.5b-chat-hf/cmmlu-arts,qwen1.5-0.5b-chat-hf/cmmlu-astronomy,qwen1.5-0.5b-chat-hf/cmmlu-business_ethics,qwen1.5-0.5b-chat-hf/cmmlu-chinese_civil_service_exam,qwen1.5-0.5b-chat-hf/cmmlu-chinese_driving_rule,qwen1.5-0.5b-chat-hf/cmmlu-chinese_food_culture,qwen1.5-0.5b-chat-hf/cmmlu-chinese_foreign_policy,qwen1.5-0.5b-chat-hf/cmmlu-chinese_history,qwen1.5-0.5b-chat-hf/cmmlu-chinese_literature,qwen1.5-0.5b-chat-hf/cmmlu-chinese_teacher_qualification,qwen1.5-0.5b-chat-hf/cmmlu-clinical_knowledge,qwen1.5-0.5b-chat-hf/cmmlu-college_actuarial_science,qwen1.5-0.5b-chat-hf/cmmlu-college_education,qwen1.5-0.5b-chat-hf/cmmlu-college_engineering_hydrology,qwen1.5-0.5b-chat-hf/cmmlu-college_law,qwen1.5-0.5b-chat-hf/cmmlu-college_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-college_medical_statistics,qwen1.5-0.5b-chat-hf/cmmlu-college_medicine,qwen1.5-0.5b-chat-hf/cmmlu-computer_science,qwen1.5-0.5b-chat-hf/cmmlu-computer_security,qwen1.5-0.5b-chat-hf/cmmlu-conceptual_physics,qwen1.5-0.5b-chat-hf/cmmlu-construction_project_management,qwen1.5-0.5b-chat-hf/cmmlu-economics,qwen1.5-0.5b-chat-hf/cmmlu-education,qwen1.5-0.5b-chat-hf/cmmlu-electrical_engineering,qwen1.5-0.5b-chat-hf/cmmlu-elementary_chinese,qwen1.5-0.5b-chat-hf/cmmlu-elementary_commonsense,qwen1.5-0.5b-chat-hf/cmmlu-elementary_information_and_technology,qwen1.5-0.5b-chat-hf/cmmlu-elementary_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-ethnology,qwen1.5-0.5b-chat-hf/cmmlu-food_science,qwen1.5-0.5b-chat-hf/cmmlu-genetics,qwen1.5-0.5b-chat-hf/cmmlu-global_facts,qwen1.5-0.5b-chat-hf/cmmlu-high_school_biology,qwen1.5-0.5b-chat-hf/cmmlu-high_school_chemistry,qwen1.5-0.5b-chat-hf/cmmlu-high_school_geography,qwen1.5-0.5b-chat-hf/cmmlu-high_school_mathematics,qwen1.5-0.5b-chat-hf/cmmlu-high_school_physics,qwen1.5-0.5b-chat-hf/cmmlu-high_school_politics,qwen1.5-0.5b-chat-hf/cmmlu-human_sexuality,qwen1.5-0.5b-chat-hf/cmmlu-international_law,qwen1.5-0.5b-chat-hf/cmmlu-journalism,qwen1.5-0.5b-chat-hf/cmmlu-jurisprudence,qwen1.5-0.5b-chat-hf/cmmlu-legal_and_moral_basis,qwen1.5-0.5b-chat-hf/cmmlu-logical,qwen1.5-0.5b-chat-hf/cmmlu-machine_learning,qwen1.5-0.5b-chat-hf/cmmlu-management,qwen1.5-0.5b-chat-hf/cmmlu-marketing,qwen1.5-0.5b-chat-hf/cmmlu-marxist_theory,qwen1.5-0.5b-chat-hf/cmmlu-modern_chinese,qwen1.5-0.5b-chat-hf/cmmlu-nutrition,qwen1.5-0.5b-chat-hf/cmmlu-philosophy,qwen1.5-0.5b-chat-hf/cmmlu-professional_accounting,qwen1.5-0.5b-chat-hf/cmmlu-professional_law,qwen1.5-0.5b-chat-hf/cmmlu-professional_medicine,qwen1.5-0.5b-chat-hf/cmmlu-professional_psychology,qwen1.5-0.5b-chat-hf/cmmlu-public_relations,qwen1.5-0.5b-chat-hf/cmmlu-security_study,qwen1.5-0.5b-chat-hf/cmmlu-sociology,qwen1.5-0.5b-chat-hf/cmmlu-sports_science,qwen1.5-0.5b-chat-hf/cmmlu-traditional_chinese_medicine,qwen1.5-0.5b-chat-hf/cmmlu-virology,qwen1.5-0.5b-chat-hf/cmmlu-world_history,qwen1.5-0.5b-chat-hf/cmmlu-world_religions,qwen1.5-0.5b-chat-hf/ceval-computer_network,qwen1.5-0.5b-chat-hf/ceval-operating_system,qwen1.5-0.5b-chat-hf/ceval-computer_architecture,qwen1.5-0.5b-chat-hf/ceval-college_programming,qwen1.5-0.5b-chat-hf/ceval-college_physics,qwen1.5-0.5b-chat-hf/ceval-college_chemistry,qwen1.5-0.5b-chat-hf/ceval-advanced_mathematics,qwen1.5-0.5b-chat-hf/ceval-probability_and_statistics,qwen1.5-0.5b-chat-hf/ceval-discrete_mathematics,qwen1.5-0.5b-chat-hf/ceval-electrical_engineer,qwen1.5-0.5b-chat-hf/ceval-metrology_engineer,qwen1.5-0.5b-chat-hf/ceval-high_school_mathematics,qwen1.5-0.5b-chat-hf/ceval-high_school_physics,qwen1.5-0.5b-chat-hf/ceval-high_school_chemistry,qwen1.5-0.5b-chat-hf/ceval-high_school_biology,qwen1.5-0.5b-chat-hf/ceval-middle_school_mathematics,qwen1.5-0.5b-chat-hf/ceval-middle_school_biology,qwen1.5-0.5b-chat-hf/ceval-middle_school_physics,qwen1.5-0.5b-chat-hf/ceval-middle_school_chemistry,qwen1.5-0.5b-chat-hf/ceval-veterinary_medicine,qwen1.5-0.5b-chat-hf/ceval-college_economics,qwen1.5-0.5b-chat-hf/ceval-business_administration,qwen1.5-0.5b-chat-hf/ceval-marxism,qwen1.5-0.5b-chat-hf/ceval-mao_zedong_thought,qwen1.5-0.5b-chat-hf/ceval-education_science,qwen1.5-0.5b-chat-hf/ceval-teacher_qualification,qwen1.5-0.5b-chat-hf/ceval-high_school_politics,qwen1.5-0.5b-chat-hf/ceval-high_school_geography,qwen1.5-0.5b-chat-hf/ceval-middle_school_politics,qwen1.5-0.5b-chat-hf/ceval-middle_school_geography,qwen1.5-0.5b-chat-hf/ceval-modern_chinese_history,qwen1.5-0.5b-chat-hf/ceval-ideological_and_moral_cultivation,qwen1.5-0.5b-chat-hf/ceval-logic,qwen1.5-0.5b-chat-hf/ceval-law,qwen1.5-0.5b-chat-hf/ceval-chinese_language_and_literature,qwen1.5-0.5b-chat-hf/ceval-art_studies,qwen1.5-0.5b-chat-hf/ceval-professional_tour_guide,qwen1.5-0.5b-chat-hf/ceval-legal_professional,qwen1.5-0.5b-chat-hf/ceval-high_school_chinese,qwen1.5-0.5b-chat-hf/ceval-high_school_history,qwen1.5-0.5b-chat-hf/ceval-middle_school_history,qwen1.5-0.5b-chat-hf/ceval-civil_servant,qwen1.5-0.5b-chat-hf/ceval-sports_science,qwen1.5-0.5b-chat-hf/ceval-plant_protection,qwen1.5-0.5b-chat-hf/ceval-basic_medicine,qwen1.5-0.5b-chat-hf/ceval-clinical_medicine,qwen1.5-0.5b-chat-hf/ceval-urban_and_rural_planner,qwen1.5-0.5b-chat-hf/ceval-accountant,qwen1.5-0.5b-chat-hf/ceval-fire_engineer,qwen1.5-0.5b-chat-hf/ceval-environmental_impact_assessment_engineer,qwen1.5-0.5b-chat-hf/ceval-tax_accountant,qwen1.5-0.5b-chat-hf/ceval-physician]
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
04/28 22:20:16 - OpenCompass - INFO - using stop words: ['<|endoftext|>', '<|im_start|>', '<|im_end|>']

Map:   0%|          | 0/7473 [00:00<?, ? examples/s]
Map:  24%|██▍       | 1808/7473 [00:00<00:00, 17957.74 examples/s]
Map:  48%|████▊     | 3616/7473 [00:00<00:00, 17946.21 examples/s]
Map:  81%|████████  | 6057/7473 [00:00<00:00, 16998.12 examples/s]
Map: 100%|██████████| 7473/7473 [00:00<00:00, 16663.67 examples/s]

Map:   0%|          | 0/1319 [00:00<?, ? examples/s]
Map: 100%|██████████| 1319/1319 [00:00<00:00, 14062.65 examples/s]
04/28 22:20:19 - OpenCompass - INFO - Start inferencing [qwen1.5-0.5b-chat-hf/demo_gsm8k]
[2025-04-28 22:20:19,103] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting build dataloader
[2025-04-28 22:20:19,104] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...

  0%|          | 0/2 [00:00<?, ?it/s]04/28 22:20:19 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:20:19 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d0b5240>], 'max_new_tokens': 512, 'pad_token_id': 151643}
/root/miniconda3/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:636: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.8` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(

 50%|█████     | 1/2 [00:15<00:15, 15.43s/it]04/28 22:20:34 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:20:34 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d736aa0>], 'max_new_tokens': 512, 'pad_token_id': 151643}

100%|██████████| 2/2 [00:35<00:00, 18.12s/it]
100%|██████████| 2/2 [00:35<00:00, 17.71s/it]

Map:   0%|          | 0/5 [00:00<?, ? examples/s]
Map: 100%|██████████| 5/5 [00:00<00:00, 1283.37 examples/s]

Map:   0%|          | 0/169 [00:00<?, ? examples/s]
Map: 100%|██████████| 169/169 [00:00<00:00, 9774.51 examples/s]
04/28 22:20:54 - OpenCompass - INFO - Start inferencing [qwen1.5-0.5b-chat-hf/cmmlu-agronomy]
[2025-04-28 22:20:54,614] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting build dataloader
[2025-04-28 22:20:54,614] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...

  0%|          | 0/6 [00:00<?, ?it/s]04/28 22:20:54 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:20:54 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d8ea6b0>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

 17%|█▋        | 1/6 [00:05<00:26,  5.37s/it]04/28 22:20:59 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:20:59 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d8eb340>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

 33%|███▎      | 2/6 [00:13<00:28,  7.07s/it]04/28 22:21:08 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:21:08 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d8e80d0>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

 50%|█████     | 3/6 [00:19<00:19,  6.48s/it]04/28 22:21:14 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:21:14 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d8a7730>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

 67%|██████▋   | 4/6 [00:25<00:12,  6.29s/it]04/28 22:21:20 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:21:20 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d734c40>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

 83%|████████▎ | 5/6 [00:31<00:06,  6.39s/it]04/28 22:21:26 - OpenCompass - INFO - Generation Args of Huggingface: 
04/28 22:21:26 - OpenCompass - INFO - {'stopping_criteria': [<opencompass.models.huggingface_above_v4_33._get_stopping_criteria.<locals>.MultiTokenEOSCriteria object at 0x7f837d7348b0>], 'max_new_tokens': 1024, 'pad_token_id': 151643}

100%|██████████| 6/6 [00:34<00:00,  4.96s/it]
100%|██████████| 6/6 [00:34<00:00,  5.69s/it]
```



### 评测结果

- qwen2.5-1.5b-instruct

qwen2.5-1.5b-instruct在三个数据上的结果：

| dataset                                        | version | metric   | mode | qwen2.5-1.5b-instruct-hf |
| ---------------------------------------------- | ------- | -------- | ---- | ------------------------ |
| demo_gsm8k                                     | 17d0dc  | accuracy | gen  | 62.50                    |
| cmmlu-agronomy                                 | 1b5abe  | accuracy | gen  | 53.25                    |
| cmmlu-anatomy                                  | f3f8bb  | accuracy | gen  | 55.41                    |
| cmmlu-ancient_chinese                          | 43111c  | accuracy | gen  | 29.88                    |
| cmmlu-arts                                     | b6e1d6  | accuracy | gen  | 82.50                    |
| cmmlu-astronomy                                | 3bd739  | accuracy | gen  | 39.39                    |
| cmmlu-business_ethics                          | 4a2346  | accuracy | gen  | 51.20                    |
| cmmlu-chinese_civil_service_exam               | 6e22c2  | accuracy | gen  | 50.00                    |
| cmmlu-chinese_driving_rule                     | 5c8e68  | accuracy | gen  | 82.44                    |
| cmmlu-chinese_food_culture                     | aa203f  | accuracy | gen  | 50.74                    |
| cmmlu-chinese_foreign_policy                   | 1b2a69  | accuracy | gen  | 57.01                    |
| cmmlu-chinese_history                          | 7f3da0  | accuracy | gen  | 56.35                    |
| cmmlu-chinese_literature                       | 16f7f9  | accuracy | gen  | 41.67                    |
| cmmlu-chinese_teacher_qualification            | cae559  | accuracy | gen  | 71.51                    |
| cmmlu-clinical_knowledge                       | e1ff3c  | accuracy | gen  | 52.32                    |
| cmmlu-college_actuarial_science                | dd69d9  | accuracy | gen  | 25.47                    |
| cmmlu-college_education                        | cf6884  | accuracy | gen  | 71.96                    |
| cmmlu-college_engineering_hydrology            | b3296d  | accuracy | gen  | 50.94                    |
| cmmlu-college_law                              | 1f7f65  | accuracy | gen  | 51.85                    |
| cmmlu-college_mathematics                      | dfeadf  | accuracy | gen  | 28.57                    |
| cmmlu-college_medical_statistics               | 3c9fc0  | accuracy | gen  | 60.38                    |
| cmmlu-college_medicine                         | 20ea93  | accuracy | gen  | 53.48                    |
| cmmlu-computer_science                         | 3e570b  | accuracy | gen  | 61.27                    |
| cmmlu-computer_security                        | 25ada2  | accuracy | gen  | 68.42                    |
| cmmlu-conceptual_physics                       | 85fa17  | accuracy | gen  | 53.06                    |
| cmmlu-construction_project_management          | 9c916e  | accuracy | gen  | 39.57                    |
| cmmlu-economics                                | fa5173  | accuracy | gen  | 56.60                    |
| cmmlu-education                                | 1b5cdc  | accuracy | gen  | 59.51                    |
| cmmlu-electrical_engineering                   | 1214ff  | accuracy | gen  | 58.14                    |
| cmmlu-elementary_chinese                       | 9c88f7  | accuracy | gen  | 53.57                    |
| cmmlu-elementary_commonsense                   | bcaca6  | accuracy | gen  | 60.10                    |
| cmmlu-elementary_information_and_technology    | b028a0  | accuracy | gen  | 78.99                    |
| cmmlu-elementary_mathematics                   | 874577  | accuracy | gen  | 51.74                    |
| cmmlu-ethnology                                | 5b63f2  | accuracy | gen  | 52.59                    |
| cmmlu-food_science                             | a580ec  | accuracy | gen  | 51.75                    |
| cmmlu-genetics                                 | 94fe78  | accuracy | gen  | 43.75                    |
| cmmlu-global_facts                             | 7cc427  | accuracy | gen  | 56.38                    |
| cmmlu-high_school_biology                      | 7f868f  | accuracy | gen  | 39.05                    |
| cmmlu-high_school_chemistry                    | dc5fee  | accuracy | gen  | 35.61                    |
| cmmlu-high_school_geography                    | b92326  | accuracy | gen  | 56.78                    |
| cmmlu-high_school_mathematics                  | 64e6b9  | accuracy | gen  | 45.73                    |
| cmmlu-high_school_physics                      | 361088  | accuracy | gen  | 43.64                    |
| cmmlu-high_school_politics                     | 343b99  | accuracy | gen  | 49.65                    |
| cmmlu-human_sexuality                          | ffde7a  | accuracy | gen  | 54.76                    |
| cmmlu-international_law                        | 87dfd3  | accuracy | gen  | 43.24                    |
| cmmlu-journalism                               | 1e8127  | accuracy | gen  | 52.91                    |
| cmmlu-jurisprudence                            | 3782cd  | accuracy | gen  | 56.45                    |
| cmmlu-legal_and_moral_basis                    | 5f37c3  | accuracy | gen  | 86.92                    |
| cmmlu-logical                                  | c85511  | accuracy | gen  | 48.78                    |
| cmmlu-machine_learning                         | 0bdf84  | accuracy | gen  | 55.74                    |
| cmmlu-management                               | 869b6e  | accuracy | gen  | 68.57                    |
| cmmlu-marketing                                | 1cfb4c  | accuracy | gen  | 60.56                    |
| cmmlu-marxist_theory                           | fc22e6  | accuracy | gen  | 72.49                    |
| cmmlu-modern_chinese                           | dd73b3  | accuracy | gen  | 40.52                    |
| cmmlu-nutrition                                | 0da8ab  | accuracy | gen  | 48.97                    |
| cmmlu-philosophy                               | 5bf8d5  | accuracy | gen  | 58.10                    |
| cmmlu-professional_accounting                  | 36b39a  | accuracy | gen  | 69.14                    |
| cmmlu-professional_law                         | 0af151  | accuracy | gen  | 40.28                    |
| cmmlu-professional_medicine                    | 6a1d3e  | accuracy | gen  | 46.54                    |
| cmmlu-professional_psychology                  | c15514  | accuracy | gen  | 64.66                    |
| cmmlu-public_relations                         | d5be35  | accuracy | gen  | 56.32                    |
| cmmlu-security_study                           | 84c059  | accuracy | gen  | 64.44                    |
| cmmlu-sociology                                | 9645be  | accuracy | gen  | 59.29                    |
| cmmlu-sports_science                           | 3249c4  | accuracy | gen  | 54.55                    |
| cmmlu-traditional_chinese_medicine             | 9a6a77  | accuracy | gen  | 52.43                    |
| cmmlu-virology                                 | 02753f  | accuracy | gen  | 67.46                    |
| cmmlu-world_history                            | 9a94e5  | accuracy | gen  | 56.52                    |
| cmmlu-world_religions                          | 5c2ff5  | accuracy | gen  | 56.88                    |
| ceval-computer_network                         | db9ce2  | accuracy | gen  | 68.42                    |
| ceval-operating_system                         | 1c2571  | accuracy | gen  | 52.63                    |
| ceval-computer_architecture                    | a74dad  | accuracy | gen  | 76.19                    |
| ceval-college_programming                      | 4ca32a  | accuracy | gen  | 70.27                    |
| ceval-college_physics                          | 963fa8  | accuracy | gen  | 42.11                    |
| ceval-college_chemistry                        | e78857  | accuracy | gen  | 41.67                    |
| ceval-advanced_mathematics                     | ce03e2  | accuracy | gen  | 21.05                    |
| ceval-probability_and_statistics               | 65e812  | accuracy | gen  | 27.78                    |
| ceval-discrete_mathematics                     | e894ae  | accuracy | gen  | 43.75                    |
| ceval-electrical_engineer                      | ae42b9  | accuracy | gen  | 56.76                    |
| ceval-metrology_engineer                       | ee34ea  | accuracy | gen  | 87.50                    |
| ceval-high_school_mathematics                  | 1dc5bf  | accuracy | gen  | 22.22                    |
| ceval-high_school_physics                      | adf25f  | accuracy | gen  | 84.21                    |
| ceval-high_school_chemistry                    | 2ed27f  | accuracy | gen  | 52.63                    |
| ceval-high_school_biology                      | 8e2b9a  | accuracy | gen  | 68.42                    |
| ceval-middle_school_mathematics                | bee8d5  | accuracy | gen  | 57.89                    |
| ceval-middle_school_biology                    | 86817c  | accuracy | gen  | 90.48                    |
| ceval-middle_school_physics                    | 8accf6  | accuracy | gen  | 89.47                    |
| ceval-middle_school_chemistry                  | 167a15  | accuracy | gen  | 95.00                    |
| ceval-veterinary_medicine                      | b4e08d  | accuracy | gen  | 73.91                    |
| ceval-college_economics                        | f3f4e6  | accuracy | gen  | 52.73                    |
| ceval-business_administration                  | c1614e  | accuracy | gen  | 54.55                    |
| ceval-marxism                                  | cf874c  | accuracy | gen  | 78.95                    |
| ceval-mao_zedong_thought                       | 51c7a4  | accuracy | gen  | 87.50                    |
| ceval-education_science                        | 591fee  | accuracy | gen  | 79.31                    |
| ceval-teacher_qualification                    | 4e4ced  | accuracy | gen  | 88.64                    |
| ceval-high_school_politics                     | 5c0de2  | accuracy | gen  | 78.95                    |
| ceval-high_school_geography                    | 865461  | accuracy | gen  | 73.68                    |
| ceval-middle_school_politics                   | 5be3e7  | accuracy | gen  | 85.71                    |
| ceval-middle_school_geography                  | 8a63be  | accuracy | gen  | 91.67                    |
| ceval-modern_chinese_history                   | fc01af  | accuracy | gen  | 86.96                    |
| ceval-ideological_and_moral_cultivation        | a2aa4a  | accuracy | gen  | 100.00                   |
| ceval-logic                                    | f5b022  | accuracy | gen  | 59.09                    |
| ceval-law                                      | a110a1  | accuracy | gen  | 45.83                    |
| ceval-chinese_language_and_literature          | 0f8b68  | accuracy | gen  | 47.83                    |
| ceval-art_studies                              | 2a1300  | accuracy | gen  | 63.64                    |
| ceval-professional_tour_guide                  | 4e673e  | accuracy | gen  | 79.31                    |
| ceval-legal_professional                       | ce8787  | accuracy | gen  | 56.52                    |
| ceval-high_school_chinese                      | 315705  | accuracy | gen  | 36.84                    |
| ceval-high_school_history                      | 7eb30a  | accuracy | gen  | 70.00                    |
| ceval-middle_school_history                    | 48ab4a  | accuracy | gen  | 90.91                    |
| ceval-civil_servant                            | 87d061  | accuracy | gen  | 57.45                    |
| ceval-sports_science                           | 70f27b  | accuracy | gen  | 68.42                    |
| ceval-plant_protection                         | 8941f9  | accuracy | gen  | 59.09                    |
| ceval-basic_medicine                           | c409d6  | accuracy | gen  | 73.68                    |
| ceval-clinical_medicine                        | 49e82d  | accuracy | gen  | 59.09                    |
| ceval-urban_and_rural_planner                  | 95b885  | accuracy | gen  | 65.22                    |
| ceval-accountant                               | 002837  | accuracy | gen  | 65.31                    |
| ceval-fire_engineer                            | bc23f5  | accuracy | gen  | 67.74                    |
| ceval-environmental_impact_assessment_engineer | c64e2d  | accuracy | gen  | 64.52                    |
| ceval-tax_accountant                           | 3a5e3c  | accuracy | gen  | 61.22                    |
| ceval-physician                                | 6e277d  | accuracy | gen  | 71.43                    |

qwen2.5-1.5b-instruct在三个数据上的评测日志：

```
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# python model_down.py 
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct
2025-04-29 10:43:46,844 - modelscope - INFO - Got 10 files, start to download ...
Downloading [generation_config.json]: 100%|█████████████████████████| 242/242 [00:00<00:00, 601B/s]
Downloading [config.json]: 100%|██████████████████████████████████| 660/660 [00:00<00:00, 1.51kB/s]
Downloading [configuration.json]: 100%|██████████████████████████| 2.00/2.00 [00:00<00:00, 4.56B/s]
Downloading [LICENSE]: 100%|██████████████████████████████████| 11.1k/11.1k [00:00<00:00, 24.6kB/s]
Downloading [README.md]: 100%|████████████████████████████████| 4.80k/4.80k [00:00<00:00, 9.92kB/s]
Downloading [tokenizer_config.json]: 100%|████████████████████| 7.13k/7.13k [00:00<00:00, 16.6kB/s]
Downloading [merges.txt]: 100%|███████████████████████████████| 1.59M/1.59M [00:01<00:00, 1.67MB/s]
Downloading [vocab.json]: 100%|███████████████████████████████| 2.65M/2.65M [00:01<00:00, 2.00MB/s]
Downloading [tokenizer.json]: 100%|███████████████████████████| 6.71M/6.71M [00:01<00:00, 3.83MB/s]
Downloading [model.safetensors]: 100%|████████████████████████| 2.88G/2.88G [04:45<00:00, 10.8MB/s]
Processing 10 items: 100%|██████████████████████████████████████| 10.0/10.0 [04:45<00:00, 28.5s/it]
2025-04-29 10:48:32,309 - modelscope - INFO - Download model 'Qwen/Qwen2.5-1.5B-Instruct' successfully.
2025-04-29 10:48:32,309 - modelscope - INFO - Creating symbolic link [/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-1.5B-Instruct].|████████                   | 2.00M/6.71M [00:00<00:01, 2.53MB/s]
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# | 2.88G/2.88G [04:45<00:00, 11.7MB/s]
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# | 6.71M/6.71M [00:01<00:00, 5.19MB/s]
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# 
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# 
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# 
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace# cd opencompass/
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace/opencompass# ls
LICENSE                            data                  opencompass           run.py
MANIFEST.in                        dataset-index.yml     opencompass.egg-info  setup.py
OpenCompassData-core-20240207.zip  docs                  outputs               tests
README.md                          examples              requirements          tmp
README_zh-CN.md                    icl_inference_output  requirements.txt      tools
(base) root@autodl-container-484d49b8aa-17f91b5c:~/workspace/opencompass# python run.py examples/eval_chat_demo_qwen.py
04/29 11:07:16 - OpenCompass - INFO - Current exp folder: outputs/default/20250429_110716
04/29 11:07:17 - OpenCompass - WARNING - SlurmRunner is not used, so the partition argument is ignored.
04/29 11:07:17 - OpenCompass - INFO - Partitioned into 1 tasks.
launch OpenICLInfer[qwen2.5-1.5b-instruct-hf/demo_gsm8k,qwen2.5-1.5b-instruct-hf/cmmlu-agronomy,qwen2.5-1.5b-instruct-hf/cmmlu-anatomy,qwen2.5-1.5b-instruct-hf/cmmlu-ancient_chinese,qwen2.5-1.5b-instruct-hf/cmmlu-arts,qwen2.5-1.5b-instruct-hf/cmmlu-astronomy,qwen2.5-1.5b-instruct-hf/cmmlu-business_ethics,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_civil_service_exam,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_driving_rule,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_food_culture,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_foreign_policy,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_history,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_literature,qwen2.5-1.5b-instruct-hf/cmmlu-chinese_teacher_qualification,qwen2.5-1.5b-instruct-hf/cmmlu-clinical_knowledge,qwen2.5-1.5b-instruct-hf/cmmlu-college_actuarial_science,qwen2.5-1.5b-instruct-hf/cmmlu-college_education,qwen2.5-1.5b-instruct-hf/cmmlu-college_engineering_hydrology,qwen2.5-1.5b-instruct-hf/cmmlu-college_law,qwen2.5-1.5b-instruct-hf/cmmlu-college_mathematics,qwen2.5-1.5b-instruct-hf/cmmlu-college_medical_statistics,qwen2.5-1.5b-instruct-hf/cmmlu-college_medicine,qwen2.5-1.5b-instruct-hf/cmmlu-computer_science,qwen2.5-1.5b-instruct-hf/cmmlu-computer_security,qwen2.5-1.5b-instruct-hf/cmmlu-conceptual_physics,qwen2.5-1.5b-instruct-hf/cmmlu-construction_project_management,qwen2.5-1.5b-instruct-hf/cmmlu-economics,qwen2.5-1.5b-instruct-hf/cmmlu-education,qwen2.5-1.5b-instruct-hf/cmmlu-electrical_engineering,qwen2.5-1.5b-instruct-hf/cmmlu-elementary_chinese,qwen2.5-1.5b-instruct-hf/cmmlu-elementary_commonsense,qwen2.5-1.5b-instruct-hf/cmmlu-elementary_information_and_technology,qwen2.5-1.5b-instruct-hf/cmmlu-elementary_mathematics,qwen2.5-1.5b-instruct-hf/cmmlu-ethnology,qwen2.5-1.5b-instruct-hf/cmmlu-food_science,qwen2.5-1.5b-instruct-hf/cmmlu-genetics,qwen2.5-1.5b-instruct-hf/cmmlu-global_facts,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_biology,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_chemistry,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_geography,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_mathematics,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_physics,qwen2.5-1.5b-instruct-hf/cmmlu-high_school_politics,qwen2.5-1.5b-instruct-hf/cmmlu-human_sexuality,qwen2.5-1.5b-instruct-hf/cmmlu-international_law,qwen2.5-1.5b-instruct-hf/cmmlu-journalism,qwen2.5-1.5b-instruct-hf/cmmlu-jurisprudence,qwen2.5-1.5b-instruct-hf/cmmlu-legal_and_moral_basis,qwen2.5-1.5b-instruct-hf/cmmlu-logical,qwen2.5-1.5b-instruct-hf/cmmlu-machine_learning,qwen2.5-1.5b-instruct-hf/cmmlu-management,qwen2.5-1.5b-instruct-hf/cmmlu-marketing,qwen2.5-1.5b-instruct-hf/cmmlu-marxist_theory,qwen2.5-1.5b-instruct-hf/cmmlu-modern_chinese,qwen2.5-1.5b-instruct-hf/cmmlu-nutrition,qwen2.5-1.5b-instruct-hf/cmmlu-philosophy,qwen2.5-1.5b-instruct-hf/cmmlu-professional_accounting,qwen2.5-1.5b-instruct-hf/cmmlu-professional_law,qwen2.5-1.5b-instruct-hf/cmmlu-professional_medicine,qwen2.5-1.5b-instruct-hf/cmmlu-professional_psychology,qwen2.5-1.5b-instruct-hf/cmmlu-public_relations,qwen2.5-1.5b-instruct-hf/cmmlu-security_study,qwen2.5-1.5b-instruct-hf/cmmlu-sociology,qwen2.5-1.5b-instruct-hf/cmmlu-sports_science,qwen2.5-1.5b-instruct-hf/cmmlu-traditional_chinese_medicine,qwen2.5-1.5b-instruct-hf/cmmlu-virology,qwen2.5-1.5b-instruct-hf/cmmlu-world_history,qwen2.5-1.5b-instruct-hf/cmmlu-world_religions,qwen2.5-1.5b-instruct-hf/ceval-computer_network,qwen2.5-1.5b-instruct-hf/ceval-operating_system,qwen2.5-1.5b-instruct-hf/ceval-computer_architecture,qwen2.5-1.5b-instruct-hf/ceval-college_programming,qwen2.5-1.5b-instruct-hf/ceval-college_physics,qwen2.5-1.5b-instruct-hf/ceval-college_chemistry,qwen2.5-1.5b-instruct-hf/ceval-advanced_mathematics,qwen2.5-1.5b-instruct-hf/ceval-probability_and_statistics,qwen2.5-1.5b-instruct-hf/ceval-discrete_mathematics,qwen2.5-1.5b-instruct-hf/ceval-electrical_engineer,qwen2.5-1.5b-instruct-hf/ceval-metrology_engineer,qwen2.5-1.5b-instruct-hf/ceval-high_school_mathematics,qwen2.5-1.5b-instruct-hf/ceval-high_school_physics,qwen2.5-1.5b-instruct-hf/ceval-high_school_chemistry,qwen2.5-1.5b-instruct-hf/ceval-high_school_biology,qwen2.5-1.5b-instruct-hf/ceval-middle_school_mathematics,qwen2.5-1.5b-instruct-hf/ceval-middle_school_biology,qwen2.5-1.5b-instruct-hf/ceval-middle_school_physics,qwen2.5-1.5b-instruct-hf/ceval-middle_school_chemistry,qwen2.5-1.5b-instruct-hf/ceval-veterinary_medicine,qwen2.5-1.5b-instruct-hf/ceval-college_economics,qwen2.5-1.5b-instruct-hf/ceval-business_administration,qwen2.5-1.5b-instruct-hf/ceval-marxism,qwen2.5-1.5b-instruct-hf/ceval-mao_zedong_thought,qwen2.5-1.5b-instruct-hf/ceval-education_science,qwen2.5-1.5b-instruct-hf/ceval-teacher_qualification,qwen2.5-1.5b-instruct-hf/ceval-high_school_politics,qwen2.5-1.5b-instruct-hf/ceval-high_school_geography,qwen2.5-1.5b-instruct-hf/ceval-middle_school_politics,qwen2.5-1.5b-instruct-hf/ceval-middle_school_geography,qwen2.5-1.5b-instruct-hf/ceval-modern_chinese_history,qwen2.5-1.5b-instruct-hf/ceval-ideological_and_moral_cultivation,qwen2.5-1.5b-instruct-hf/ceval-logic,qwen2.5-1.5b-instruct-hf/ceval-law,qwen2.5-1.5b-instruct-hf/ceval-chinese_language_and_literature,qwen2.5-1.5b-instruct-hf/ceval-art_studies,qwen2.5-1.5b-instruct-hf/ceval-professional_tour_guide,qwen2.5-1.5b-instruct-hf/ceval-legal_professional,qwen2.5-1.5b-instruct-hf/ceval-high_school_chinese,qwen2.5-1.5b-instruct-hf/ceval-high_school_history,qwen2.5-1.5b-instruct-hf/ceval-middle_school_history,qwen2.5-1.5b-instruct-hf/ceval-civil_servant,qwen2.5-1.5b-instruct-hf/ceval-sports_science,qwen2.5-1.5b-instruct-hf/ceval-plant_protection,qwen2.5-1.5b-instruct-hf/ceval-basic_medicine,qwen2.5-1.5b-instruct-hf/ceval-clinical_medicine,qwen2.5-1.5b-instruct-hf/ceval-urban_and_rural_planner,qwen2.5-1.5b-instruct-hf/ceval-accountant,qwen2.5-1.5b-instruct-hf/ceval-fire_engineer,qwen2.5-1.5b-instruct-hf/ceval-environmental_impact_assessment_engineer,qwen2.5-1.5b-instruct-hf/ceval-tax_accountant,qwen2.5-1.5b-instruct-hf/ceval-physician] on GPU 0
100%|███████████████████████████████████████████████████████████| 1/1 [3:48:54<00:00, 13734.97s/it]
04/29 14:56:12 - OpenCompass - WARNING - Default to dump eval details, it might take extraspace to save all the evaluation details. Set --dump-eval-details False to skip the details dump
04/29 14:56:12 - OpenCompass - INFO - Partitioned into 120 tasks.
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/demo_gsm8k] on CPU                                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-agronomy] on CPU                                 
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-anatomy] on CPU                                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-ancient_chinese] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-arts] on CPU                                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-astronomy] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-business_ethics] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_civil_service_exam] on CPU               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_driving_rule] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_food_culture] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_foreign_policy] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_history] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_literature] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-chinese_teacher_qualification] on CPU            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-clinical_knowledge] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_actuarial_science] on CPU                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_education] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_engineering_hydrology] on CPU            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_law] on CPU                              
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_mathematics] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_medical_statistics] on CPU               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-college_medicine] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-computer_science] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-computer_security] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-conceptual_physics] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-construction_project_management] on CPU          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-economics] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-education] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-electrical_engineering] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-elementary_chinese] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-elementary_commonsense] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-elementary_information_and_technology] on CPU    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-elementary_mathematics] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-ethnology] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-food_science] on CPU                             
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-genetics] on CPU                                 
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-global_facts] on CPU                             
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_biology] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_chemistry] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_geography] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_mathematics] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_physics] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-high_school_politics] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-human_sexuality] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-international_law] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-journalism] on CPU                               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-jurisprudence] on CPU                            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-legal_and_moral_basis] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-logical] on CPU                                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-machine_learning] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-management] on CPU                               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-marketing] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-marxist_theory] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-modern_chinese] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-nutrition] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-philosophy] on CPU                               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-professional_accounting] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-professional_law] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-professional_medicine] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-professional_psychology] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-public_relations] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-security_study] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-sociology] on CPU                                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-sports_science] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-traditional_chinese_medicine] on CPU             
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-virology] on CPU                                 
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-world_history] on CPU                            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-world_religions] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-computer_network] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-operating_system] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-computer_architecture] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-college_programming] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-college_physics] on CPU                          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-college_chemistry] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-advanced_mathematics] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-probability_and_statistics] on CPU               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-discrete_mathematics] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-electrical_engineer] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-metrology_engineer] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_mathematics] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_physics] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_chemistry] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_biology] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_mathematics] on CPU                
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_biology] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_physics] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_chemistry] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-veterinary_medicine] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-college_economics] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-business_administration] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-marxism] on CPU                                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-mao_zedong_thought] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-education_science] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-teacher_qualification] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_politics] on CPU                     
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_geography] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_politics] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_geography] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-modern_chinese_history] on CPU                   
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-ideological_and_moral_cultivation] on CPU        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-logic] on CPU                                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-law] on CPU                                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-chinese_language_and_literature] on CPU          
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-art_studies] on CPU                              
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-professional_tour_guide] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-legal_professional] on CPU                       
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_chinese] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-high_school_history] on CPU                      
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-middle_school_history] on CPU                    
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-civil_servant] on CPU                            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-sports_science] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-plant_protection] on CPU                         
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-basic_medicine] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-clinical_medicine] on CPU                        
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-urban_and_rural_planner] on CPU                  
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-accountant] on CPU                               
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-fire_engineer] on CPU                            
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-environmental_impact_assessment_engineer] on CPU 
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-tax_accountant] on CPU                           
launch OpenICLEval[qwen2.5-1.5b-instruct-hf/ceval-physician] on CPU                                
100%|████████████████████████████████████████████████████████████| 120/120 [16:26<00:00,  8.22s/it]
dataset                                         version    metric    mode      qwen2.5-1.5b-instruct-hf
----------------------------------------------  ---------  --------  ------  --------------------------
demo_gsm8k                                      17d0dc     accuracy  gen                          62.50
cmmlu-agronomy                                  1b5abe     accuracy  gen                          53.25
cmmlu-anatomy                                   f3f8bb     accuracy  gen                          55.41
cmmlu-ancient_chinese                           43111c     accuracy  gen                          29.88
cmmlu-arts                                      b6e1d6     accuracy  gen                          82.50
cmmlu-astronomy                                 3bd739     accuracy  gen                          39.39
cmmlu-business_ethics                           4a2346     accuracy  gen                          51.20
cmmlu-chinese_civil_service_exam                6e22c2     accuracy  gen                          50.00
cmmlu-chinese_driving_rule                      5c8e68     accuracy  gen                          82.44
cmmlu-chinese_food_culture                      aa203f     accuracy  gen                          50.74
cmmlu-chinese_foreign_policy                    1b2a69     accuracy  gen                          57.01
cmmlu-chinese_history                           7f3da0     accuracy  gen                          56.35
cmmlu-chinese_literature                        16f7f9     accuracy  gen                          41.67
cmmlu-chinese_teacher_qualification             cae559     accuracy  gen                          71.51
cmmlu-clinical_knowledge                        e1ff3c     accuracy  gen                          52.32
cmmlu-college_actuarial_science                 dd69d9     accuracy  gen                          25.47
cmmlu-college_education                         cf6884     accuracy  gen                          71.96
cmmlu-college_engineering_hydrology             b3296d     accuracy  gen                          50.94
cmmlu-college_law                               1f7f65     accuracy  gen                          51.85
cmmlu-college_mathematics                       dfeadf     accuracy  gen                          28.57
cmmlu-college_medical_statistics                3c9fc0     accuracy  gen                          60.38
cmmlu-college_medicine                          20ea93     accuracy  gen                          53.48
cmmlu-computer_science                          3e570b     accuracy  gen                          61.27
cmmlu-computer_security                         25ada2     accuracy  gen                          68.42
cmmlu-conceptual_physics                        85fa17     accuracy  gen                          53.06
cmmlu-construction_project_management           9c916e     accuracy  gen                          39.57
cmmlu-economics                                 fa5173     accuracy  gen                          56.60
cmmlu-education                                 1b5cdc     accuracy  gen                          59.51
cmmlu-electrical_engineering                    1214ff     accuracy  gen                          58.14
cmmlu-elementary_chinese                        9c88f7     accuracy  gen                          53.57
cmmlu-elementary_commonsense                    bcaca6     accuracy  gen                          60.10
cmmlu-elementary_information_and_technology     b028a0     accuracy  gen                          78.99
cmmlu-elementary_mathematics                    874577     accuracy  gen                          51.74
cmmlu-ethnology                                 5b63f2     accuracy  gen                          52.59
cmmlu-food_science                              a580ec     accuracy  gen                          51.75
cmmlu-genetics                                  94fe78     accuracy  gen                          43.75
cmmlu-global_facts                              7cc427     accuracy  gen                          56.38
cmmlu-high_school_biology                       7f868f     accuracy  gen                          39.05
cmmlu-high_school_chemistry                     dc5fee     accuracy  gen                          35.61
cmmlu-high_school_geography                     b92326     accuracy  gen                          56.78
cmmlu-high_school_mathematics                   64e6b9     accuracy  gen                          45.73
cmmlu-high_school_physics                       361088     accuracy  gen                          43.64
cmmlu-high_school_politics                      343b99     accuracy  gen                          49.65
cmmlu-human_sexuality                           ffde7a     accuracy  gen                          54.76
cmmlu-international_law                         87dfd3     accuracy  gen                          43.24
cmmlu-journalism                                1e8127     accuracy  gen                          52.91
cmmlu-jurisprudence                             3782cd     accuracy  gen                          56.45
cmmlu-legal_and_moral_basis                     5f37c3     accuracy  gen                          86.92
cmmlu-logical                                   c85511     accuracy  gen                          48.78
cmmlu-machine_learning                          0bdf84     accuracy  gen                          55.74
cmmlu-management                                869b6e     accuracy  gen                          68.57
cmmlu-marketing                                 1cfb4c     accuracy  gen                          60.56
cmmlu-marxist_theory                            fc22e6     accuracy  gen                          72.49
cmmlu-modern_chinese                            dd73b3     accuracy  gen                          40.52
cmmlu-nutrition                                 0da8ab     accuracy  gen                          48.97
cmmlu-philosophy                                5bf8d5     accuracy  gen                          58.10
cmmlu-professional_accounting                   36b39a     accuracy  gen                          69.14
cmmlu-professional_law                          0af151     accuracy  gen                          40.28
cmmlu-professional_medicine                     6a1d3e     accuracy  gen                          46.54
cmmlu-professional_psychology                   c15514     accuracy  gen                          64.66
cmmlu-public_relations                          d5be35     accuracy  gen                          56.32
cmmlu-security_study                            84c059     accuracy  gen                          64.44
cmmlu-sociology                                 9645be     accuracy  gen                          59.29
cmmlu-sports_science                            3249c4     accuracy  gen                          54.55
cmmlu-traditional_chinese_medicine              9a6a77     accuracy  gen                          52.43
cmmlu-virology                                  02753f     accuracy  gen                          67.46
cmmlu-world_history                             9a94e5     accuracy  gen                          56.52
cmmlu-world_religions                           5c2ff5     accuracy  gen                          56.88
ceval-computer_network                          db9ce2     accuracy  gen                          68.42
ceval-operating_system                          1c2571     accuracy  gen                          52.63
ceval-computer_architecture                     a74dad     accuracy  gen                          76.19
ceval-college_programming                       4ca32a     accuracy  gen                          70.27
ceval-college_physics                           963fa8     accuracy  gen                          42.11
ceval-college_chemistry                         e78857     accuracy  gen                          41.67
ceval-advanced_mathematics                      ce03e2     accuracy  gen                          21.05
ceval-probability_and_statistics                65e812     accuracy  gen                          27.78
ceval-discrete_mathematics                      e894ae     accuracy  gen                          43.75
ceval-electrical_engineer                       ae42b9     accuracy  gen                          56.76
ceval-metrology_engineer                        ee34ea     accuracy  gen                          87.50
ceval-high_school_mathematics                   1dc5bf     accuracy  gen                          22.22
ceval-high_school_physics                       adf25f     accuracy  gen                          84.21
ceval-high_school_chemistry                     2ed27f     accuracy  gen                          52.63
ceval-high_school_biology                       8e2b9a     accuracy  gen                          68.42
ceval-middle_school_mathematics                 bee8d5     accuracy  gen                          57.89
ceval-middle_school_biology                     86817c     accuracy  gen                          90.48
ceval-middle_school_physics                     8accf6     accuracy  gen                          89.47
ceval-middle_school_chemistry                   167a15     accuracy  gen                          95.00
ceval-veterinary_medicine                       b4e08d     accuracy  gen                          73.91
ceval-college_economics                         f3f4e6     accuracy  gen                          52.73
ceval-business_administration                   c1614e     accuracy  gen                          54.55
ceval-marxism                                   cf874c     accuracy  gen                          78.95
ceval-mao_zedong_thought                        51c7a4     accuracy  gen                          87.50
ceval-education_science                         591fee     accuracy  gen                          79.31
ceval-teacher_qualification                     4e4ced     accuracy  gen                          88.64
ceval-high_school_politics                      5c0de2     accuracy  gen                          78.95
ceval-high_school_geography                     865461     accuracy  gen                          73.68
ceval-middle_school_politics                    5be3e7     accuracy  gen                          85.71
ceval-middle_school_geography                   8a63be     accuracy  gen                          91.67
ceval-modern_chinese_history                    fc01af     accuracy  gen                          86.96
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy  gen                         100.00
ceval-logic                                     f5b022     accuracy  gen                          59.09
ceval-law                                       a110a1     accuracy  gen                          45.83
ceval-chinese_language_and_literature           0f8b68     accuracy  gen                          47.83
ceval-art_studies                               2a1300     accuracy  gen                          63.64
ceval-professional_tour_guide                   4e673e     accuracy  gen                          79.31
ceval-legal_professional                        ce8787     accuracy  gen                          56.52
ceval-high_school_chinese                       315705     accuracy  gen                          36.84
ceval-high_school_history                       7eb30a     accuracy  gen                          70.00
ceval-middle_school_history                     48ab4a     accuracy  gen                          90.91
ceval-civil_servant                             87d061     accuracy  gen                          57.45
ceval-sports_science                            70f27b     accuracy  gen                          68.42
ceval-plant_protection                          8941f9     accuracy  gen                          59.09
ceval-basic_medicine                            c409d6     accuracy  gen                          73.68
ceval-clinical_medicine                         49e82d     accuracy  gen                          59.09
ceval-urban_and_rural_planner                   95b885     accuracy  gen                          65.22
ceval-accountant                                002837     accuracy  gen                          65.31
ceval-fire_engineer                             bc23f5     accuracy  gen                          67.74
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy  gen                          64.52
ceval-tax_accountant                            3a5e3c     accuracy  gen                          61.22
ceval-physician                                 6e277d     accuracy  gen                          71.43
04/29 15:12:38 - OpenCompass - INFO - write summary to /root/workspace/opencompass/outputs/default/20250429_110716/summary/summary_20250429_110716.txt
04/29 15:12:38 - OpenCompass - INFO - write csv to /root/workspace/opencompass/outputs/default/20250429_110716/summary/summary_20250429_110716.csv


The markdown format results is as below:

| dataset | version | metric | mode | qwen2.5-1.5b-instruct-hf |
|----- | ----- | ----- | ----- | -----|
| demo_gsm8k | 17d0dc | accuracy | gen | 62.50 |
| cmmlu-agronomy | 1b5abe | accuracy | gen | 53.25 |
| cmmlu-anatomy | f3f8bb | accuracy | gen | 55.41 |
| cmmlu-ancient_chinese | 43111c | accuracy | gen | 29.88 |
| cmmlu-arts | b6e1d6 | accuracy | gen | 82.50 |
| cmmlu-astronomy | 3bd739 | accuracy | gen | 39.39 |
| cmmlu-business_ethics | 4a2346 | accuracy | gen | 51.20 |
| cmmlu-chinese_civil_service_exam | 6e22c2 | accuracy | gen | 50.00 |
| cmmlu-chinese_driving_rule | 5c8e68 | accuracy | gen | 82.44 |
| cmmlu-chinese_food_culture | aa203f | accuracy | gen | 50.74 |
| cmmlu-chinese_foreign_policy | 1b2a69 | accuracy | gen | 57.01 |
| cmmlu-chinese_history | 7f3da0 | accuracy | gen | 56.35 |
| cmmlu-chinese_literature | 16f7f9 | accuracy | gen | 41.67 |
| cmmlu-chinese_teacher_qualification | cae559 | accuracy | gen | 71.51 |
| cmmlu-clinical_knowledge | e1ff3c | accuracy | gen | 52.32 |
| cmmlu-college_actuarial_science | dd69d9 | accuracy | gen | 25.47 |
| cmmlu-college_education | cf6884 | accuracy | gen | 71.96 |
| cmmlu-college_engineering_hydrology | b3296d | accuracy | gen | 50.94 |
| cmmlu-college_law | 1f7f65 | accuracy | gen | 51.85 |
| cmmlu-college_mathematics | dfeadf | accuracy | gen | 28.57 |
| cmmlu-college_medical_statistics | 3c9fc0 | accuracy | gen | 60.38 |
| cmmlu-college_medicine | 20ea93 | accuracy | gen | 53.48 |
| cmmlu-computer_science | 3e570b | accuracy | gen | 61.27 |
| cmmlu-computer_security | 25ada2 | accuracy | gen | 68.42 |
| cmmlu-conceptual_physics | 85fa17 | accuracy | gen | 53.06 |
| cmmlu-construction_project_management | 9c916e | accuracy | gen | 39.57 |
| cmmlu-economics | fa5173 | accuracy | gen | 56.60 |
| cmmlu-education | 1b5cdc | accuracy | gen | 59.51 |
| cmmlu-electrical_engineering | 1214ff | accuracy | gen | 58.14 |
| cmmlu-elementary_chinese | 9c88f7 | accuracy | gen | 53.57 |
| cmmlu-elementary_commonsense | bcaca6 | accuracy | gen | 60.10 |
| cmmlu-elementary_information_and_technology | b028a0 | accuracy | gen | 78.99 |
| cmmlu-elementary_mathematics | 874577 | accuracy | gen | 51.74 |
| cmmlu-ethnology | 5b63f2 | accuracy | gen | 52.59 |
| cmmlu-food_science | a580ec | accuracy | gen | 51.75 |
| cmmlu-genetics | 94fe78 | accuracy | gen | 43.75 |
| cmmlu-global_facts | 7cc427 | accuracy | gen | 56.38 |
| cmmlu-high_school_biology | 7f868f | accuracy | gen | 39.05 |
| cmmlu-high_school_chemistry | dc5fee | accuracy | gen | 35.61 |
| cmmlu-high_school_geography | b92326 | accuracy | gen | 56.78 |
| cmmlu-high_school_mathematics | 64e6b9 | accuracy | gen | 45.73 |
| cmmlu-high_school_physics | 361088 | accuracy | gen | 43.64 |
| cmmlu-high_school_politics | 343b99 | accuracy | gen | 49.65 |
| cmmlu-human_sexuality | ffde7a | accuracy | gen | 54.76 |
| cmmlu-international_law | 87dfd3 | accuracy | gen | 43.24 |
| cmmlu-journalism | 1e8127 | accuracy | gen | 52.91 |
| cmmlu-jurisprudence | 3782cd | accuracy | gen | 56.45 |
| cmmlu-legal_and_moral_basis | 5f37c3 | accuracy | gen | 86.92 |
| cmmlu-logical | c85511 | accuracy | gen | 48.78 |
| cmmlu-machine_learning | 0bdf84 | accuracy | gen | 55.74 |
| cmmlu-management | 869b6e | accuracy | gen | 68.57 |
| cmmlu-marketing | 1cfb4c | accuracy | gen | 60.56 |
| cmmlu-marxist_theory | fc22e6 | accuracy | gen | 72.49 |
| cmmlu-modern_chinese | dd73b3 | accuracy | gen | 40.52 |
| cmmlu-nutrition | 0da8ab | accuracy | gen | 48.97 |
| cmmlu-philosophy | 5bf8d5 | accuracy | gen | 58.10 |
| cmmlu-professional_accounting | 36b39a | accuracy | gen | 69.14 |
| cmmlu-professional_law | 0af151 | accuracy | gen | 40.28 |
| cmmlu-professional_medicine | 6a1d3e | accuracy | gen | 46.54 |
| cmmlu-professional_psychology | c15514 | accuracy | gen | 64.66 |
| cmmlu-public_relations | d5be35 | accuracy | gen | 56.32 |
| cmmlu-security_study | 84c059 | accuracy | gen | 64.44 |
| cmmlu-sociology | 9645be | accuracy | gen | 59.29 |
| cmmlu-sports_science | 3249c4 | accuracy | gen | 54.55 |
| cmmlu-traditional_chinese_medicine | 9a6a77 | accuracy | gen | 52.43 |
| cmmlu-virology | 02753f | accuracy | gen | 67.46 |
| cmmlu-world_history | 9a94e5 | accuracy | gen | 56.52 |
| cmmlu-world_religions | 5c2ff5 | accuracy | gen | 56.88 |
| ceval-computer_network | db9ce2 | accuracy | gen | 68.42 |
| ceval-operating_system | 1c2571 | accuracy | gen | 52.63 |
| ceval-computer_architecture | a74dad | accuracy | gen | 76.19 |
| ceval-college_programming | 4ca32a | accuracy | gen | 70.27 |
| ceval-college_physics | 963fa8 | accuracy | gen | 42.11 |
| ceval-college_chemistry | e78857 | accuracy | gen | 41.67 |
| ceval-advanced_mathematics | ce03e2 | accuracy | gen | 21.05 |
| ceval-probability_and_statistics | 65e812 | accuracy | gen | 27.78 |
| ceval-discrete_mathematics | e894ae | accuracy | gen | 43.75 |
| ceval-electrical_engineer | ae42b9 | accuracy | gen | 56.76 |
| ceval-metrology_engineer | ee34ea | accuracy | gen | 87.50 |
| ceval-high_school_mathematics | 1dc5bf | accuracy | gen | 22.22 |
| ceval-high_school_physics | adf25f | accuracy | gen | 84.21 |
| ceval-high_school_chemistry | 2ed27f | accuracy | gen | 52.63 |
| ceval-high_school_biology | 8e2b9a | accuracy | gen | 68.42 |
| ceval-middle_school_mathematics | bee8d5 | accuracy | gen | 57.89 |
| ceval-middle_school_biology | 86817c | accuracy | gen | 90.48 |
| ceval-middle_school_physics | 8accf6 | accuracy | gen | 89.47 |
| ceval-middle_school_chemistry | 167a15 | accuracy | gen | 95.00 |
| ceval-veterinary_medicine | b4e08d | accuracy | gen | 73.91 |
| ceval-college_economics | f3f4e6 | accuracy | gen | 52.73 |
| ceval-business_administration | c1614e | accuracy | gen | 54.55 |
| ceval-marxism | cf874c | accuracy | gen | 78.95 |
| ceval-mao_zedong_thought | 51c7a4 | accuracy | gen | 87.50 |
| ceval-education_science | 591fee | accuracy | gen | 79.31 |
| ceval-teacher_qualification | 4e4ced | accuracy | gen | 88.64 |
| ceval-high_school_politics | 5c0de2 | accuracy | gen | 78.95 |
| ceval-high_school_geography | 865461 | accuracy | gen | 73.68 |
| ceval-middle_school_politics | 5be3e7 | accuracy | gen | 85.71 |
| ceval-middle_school_geography | 8a63be | accuracy | gen | 91.67 |
| ceval-modern_chinese_history | fc01af | accuracy | gen | 86.96 |
| ceval-ideological_and_moral_cultivation | a2aa4a | accuracy | gen | 100.00 |
| ceval-logic | f5b022 | accuracy | gen | 59.09 |
| ceval-law | a110a1 | accuracy | gen | 45.83 |
| ceval-chinese_language_and_literature | 0f8b68 | accuracy | gen | 47.83 |
| ceval-art_studies | 2a1300 | accuracy | gen | 63.64 |
| ceval-professional_tour_guide | 4e673e | accuracy | gen | 79.31 |
| ceval-legal_professional | ce8787 | accuracy | gen | 56.52 |
| ceval-high_school_chinese | 315705 | accuracy | gen | 36.84 |
| ceval-high_school_history | 7eb30a | accuracy | gen | 70.00 |
| ceval-middle_school_history | 48ab4a | accuracy | gen | 90.91 |
| ceval-civil_servant | 87d061 | accuracy | gen | 57.45 |
| ceval-sports_science | 70f27b | accuracy | gen | 68.42 |
| ceval-plant_protection | 8941f9 | accuracy | gen | 59.09 |
| ceval-basic_medicine | c409d6 | accuracy | gen | 73.68 |
| ceval-clinical_medicine | 49e82d | accuracy | gen | 59.09 |
| ceval-urban_and_rural_planner | 95b885 | accuracy | gen | 65.22 |
| ceval-accountant | 002837 | accuracy | gen | 65.31 |
| ceval-fire_engineer | bc23f5 | accuracy | gen | 67.74 |
| ceval-environmental_impact_assessment_engineer | c64e2d | accuracy | gen | 64.52 |
| ceval-tax_accountant | 3a5e3c | accuracy | gen | 61.22 |
| ceval-physician | 6e277d | accuracy | gen | 71.43 |

04/29 15:12:38 - OpenCompass - INFO - write markdown summary to /root/workspace/opencompass/outputs/default/20250429_110716/summary/summary_20250429_110716.md
```

- Qwen 0.5B的结果

| dataset                                        | version | metric   | mode | qwen1.5-0.5b-chat-hf |
| ---------------------------------------------- | ------- | -------- | ---- | -------------------- |
| demo_gsm8k                                     | 17d0dc  | accuracy | gen  | 3.12                 |
| cmmlu-agronomy                                 | 1b5abe  | accuracy | gen  | 0.00                 |
| cmmlu-anatomy                                  | f3f8bb  | accuracy | gen  | 0.68                 |
| cmmlu-ancient_chinese                          | 43111c  | accuracy | gen  | 1.83                 |
| cmmlu-arts                                     | b6e1d6  | accuracy | gen  | 3.75                 |
| cmmlu-astronomy                                | 3bd739  | accuracy | gen  | 0.61                 |
| cmmlu-business_ethics                          | 4a2346  | accuracy | gen  | 11.48                |
| cmmlu-chinese_civil_service_exam               | 6e22c2  | accuracy | gen  | 0.00                 |
| cmmlu-chinese_driving_rule                     | 5c8e68  | accuracy | gen  | 6.11                 |
| cmmlu-chinese_food_culture                     | aa203f  | accuracy | gen  | 8.09                 |
| cmmlu-chinese_foreign_policy                   | 1b2a69  | accuracy | gen  | 6.54                 |
| cmmlu-chinese_history                          | 7f3da0  | accuracy | gen  | 7.43                 |
| cmmlu-chinese_literature                       | 16f7f9  | accuracy | gen  | 0.98                 |
| cmmlu-chinese_teacher_qualification            | cae559  | accuracy | gen  | 0.00                 |
| cmmlu-clinical_knowledge                       | e1ff3c  | accuracy | gen  | 0.00                 |
| cmmlu-college_actuarial_science                | dd69d9  | accuracy | gen  | 0.00                 |
| cmmlu-college_education                        | cf6884  | accuracy | gen  | 0.93                 |
| cmmlu-college_engineering_hydrology            | b3296d  | accuracy | gen  | 1.89                 |
| cmmlu-college_law                              | 1f7f65  | accuracy | gen  | 0.93                 |
| cmmlu-college_mathematics                      | dfeadf  | accuracy | gen  | 0.00                 |
| cmmlu-college_medical_statistics               | 3c9fc0  | accuracy | gen  | 0.00                 |
| cmmlu-college_medicine                         | 20ea93  | accuracy | gen  | 0.00                 |
| cmmlu-computer_science                         | 3e570b  | accuracy | gen  | 0.49                 |
| cmmlu-computer_security                        | 25ada2  | accuracy | gen  | 2.34                 |
| cmmlu-conceptual_physics                       | 85fa17  | accuracy | gen  | 12.24                |
| cmmlu-construction_project_management          | 9c916e  | accuracy | gen  | 3.60                 |
| cmmlu-economics                                | fa5173  | accuracy | gen  | 3.77                 |
| cmmlu-education                                | 1b5cdc  | accuracy | gen  | 0.61                 |
| cmmlu-electrical_engineering                   | 1214ff  | accuracy | gen  | 1.16                 |
| cmmlu-elementary_chinese                       | 9c88f7  | accuracy | gen  | 1.98                 |
| cmmlu-elementary_commonsense                   | bcaca6  | accuracy | gen  | 4.04                 |
| cmmlu-elementary_information_and_technology    | b028a0  | accuracy | gen  | 0.42                 |
| cmmlu-elementary_mathematics                   | 874577  | accuracy | gen  | 0.87                 |
| cmmlu-ethnology                                | 5b63f2  | accuracy | gen  | 2.96                 |
| cmmlu-food_science                             | a580ec  | accuracy | gen  | 3.50                 |
| cmmlu-genetics                                 | 94fe78  | accuracy | gen  | 1.70                 |
| cmmlu-global_facts                             | 7cc427  | accuracy | gen  | 20.13                |
| cmmlu-high_school_biology                      | 7f868f  | accuracy | gen  | 1.78                 |
| cmmlu-high_school_chemistry                    | dc5fee  | accuracy | gen  | 0.76                 |
| cmmlu-high_school_geography                    | b92326  | accuracy | gen  | 3.39                 |
| cmmlu-high_school_mathematics                  | 64e6b9  | accuracy | gen  | 0.61                 |
| cmmlu-high_school_physics                      | 361088  | accuracy | gen  | 1.82                 |
| cmmlu-high_school_politics                     | 343b99  | accuracy | gen  | 1.40                 |
| cmmlu-human_sexuality                          | ffde7a  | accuracy | gen  | 13.49                |
| cmmlu-international_law                        | 87dfd3  | accuracy | gen  | 7.57                 |
| cmmlu-journalism                               | 1e8127  | accuracy | gen  | 0.00                 |
| cmmlu-jurisprudence                            | 3782cd  | accuracy | gen  | 2.43                 |
| cmmlu-legal_and_moral_basis                    | 5f37c3  | accuracy | gen  | 17.29                |
| cmmlu-logical                                  | c85511  | accuracy | gen  | 1.63                 |
| cmmlu-machine_learning                         | 0bdf84  | accuracy | gen  | 6.56                 |
| cmmlu-management                               | 869b6e  | accuracy | gen  | 0.00                 |
| cmmlu-marketing                                | 1cfb4c  | accuracy | gen  | 6.67                 |
| cmmlu-marxist_theory                           | fc22e6  | accuracy | gen  | 8.47                 |
| cmmlu-modern_chinese                           | dd73b3  | accuracy | gen  | 2.59                 |
| cmmlu-nutrition                                | 0da8ab  | accuracy | gen  | 1.38                 |
| cmmlu-philosophy                               | 5bf8d5  | accuracy | gen  | 4.76                 |
| cmmlu-professional_accounting                  | 36b39a  | accuracy | gen  | 1.14                 |
| cmmlu-professional_law                         | 0af151  | accuracy | gen  | 1.42                 |
| cmmlu-professional_medicine                    | 6a1d3e  | accuracy | gen  | 0.27                 |
| cmmlu-professional_psychology                  | c15514  | accuracy | gen  | 0.00                 |
| cmmlu-public_relations                         | d5be35  | accuracy | gen  | 3.45                 |
| cmmlu-security_study                           | 84c059  | accuracy | gen  | 6.67                 |
| cmmlu-sociology                                | 9645be  | accuracy | gen  | 1.33                 |
| cmmlu-sports_science                           | 3249c4  | accuracy | gen  | 1.82                 |
| cmmlu-traditional_chinese_medicine             | 9a6a77  | accuracy | gen  | 1.62                 |
| cmmlu-virology                                 | 02753f  | accuracy | gen  | 1.18                 |
| cmmlu-world_history                            | 9a94e5  | accuracy | gen  | 3.11                 |
| cmmlu-world_religions                          | 5c2ff5  | accuracy | gen  | 11.25                |
| ceval-computer_network                         | db9ce2  | accuracy | gen  | 36.84                |
| ceval-operating_system                         | 1c2571  | accuracy | gen  | 26.32                |
| ceval-computer_architecture                    | a74dad  | accuracy | gen  | 14.29                |
| ceval-college_programming                      | 4ca32a  | accuracy | gen  | 32.43                |
| ceval-college_physics                          | 963fa8  | accuracy | gen  | 31.58                |
| ceval-college_chemistry                        | e78857  | accuracy | gen  | 37.50                |
| ceval-advanced_mathematics                     | ce03e2  | accuracy | gen  | 15.79                |
| ceval-probability_and_statistics               | 65e812  | accuracy | gen  | 38.89                |
| ceval-discrete_mathematics                     | e894ae  | accuracy | gen  | 18.75                |
| ceval-electrical_engineer                      | ae42b9  | accuracy | gen  | 37.84                |
| ceval-metrology_engineer                       | ee34ea  | accuracy | gen  | 45.83                |
| ceval-high_school_mathematics                  | 1dc5bf  | accuracy | gen  | 44.44                |
| ceval-high_school_physics                      | adf25f  | accuracy | gen  | 42.11                |
| ceval-high_school_chemistry                    | 2ed27f  | accuracy | gen  | 42.11                |
| ceval-high_school_biology                      | 8e2b9a  | accuracy | gen  | 21.05                |
| ceval-middle_school_mathematics                | bee8d5  | accuracy | gen  | 21.05                |
| ceval-middle_school_biology                    | 86817c  | accuracy | gen  | 47.62                |
| ceval-middle_school_physics                    | 8accf6  | accuracy | gen  | 42.11                |
| ceval-middle_school_chemistry                  | 167a15  | accuracy | gen  | 50.00                |
| ceval-veterinary_medicine                      | b4e08d  | accuracy | gen  | 26.09                |
| ceval-college_economics                        | f3f4e6  | accuracy | gen  | 30.91                |
| ceval-business_administration                  | c1614e  | accuracy | gen  | 27.27                |
| ceval-marxism                                  | cf874c  | accuracy | gen  | 47.37                |
| ceval-mao_zedong_thought                       | 51c7a4  | accuracy | gen  | 37.50                |
| ceval-education_science                        | 591fee  | accuracy | gen  | 44.83                |
| ceval-teacher_qualification                    | 4e4ced  | accuracy | gen  | 50.00                |
| ceval-high_school_politics                     | 5c0de2  | accuracy | gen  | 42.11                |
| ceval-high_school_geography                    | 865461  | accuracy | gen  | 36.84                |
| ceval-middle_school_politics                   | 5be3e7  | accuracy | gen  | 52.38                |
| ceval-middle_school_geography                  | 8a63be  | accuracy | gen  | 66.67                |
| ceval-modern_chinese_history                   | fc01af  | accuracy | gen  | 39.13                |
| ceval-ideological_and_moral_cultivation        | a2aa4a  | accuracy | gen  | 47.37                |
| ceval-logic                                    | f5b022  | accuracy | gen  | 36.36                |
| ceval-law                                      | a110a1  | accuracy | gen  | 37.50                |
| ceval-chinese_language_and_literature          | 0f8b68  | accuracy | gen  | 30.43                |
| ceval-art_studies                              | 2a1300  | accuracy | gen  | 33.33                |
| ceval-professional_tour_guide                  | 4e673e  | accuracy | gen  | 44.83                |
| ceval-legal_professional                       | ce8787  | accuracy | gen  | 39.13                |
| ceval-high_school_chinese                      | 315705  | accuracy | gen  | 31.58                |
| ceval-high_school_history                      | 7eb30a  | accuracy | gen  | 70.00                |
| ceval-middle_school_history                    | 48ab4a  | accuracy | gen  | 50.00                |
| ceval-civil_servant                            | 87d061  | accuracy | gen  | 42.55                |
| ceval-sports_science                           | 70f27b  | accuracy | gen  | 31.58                |
| ceval-plant_protection                         | 8941f9  | accuracy | gen  | 54.55                |
| ceval-basic_medicine                           | c409d6  | accuracy | gen  | 47.37                |
| ceval-clinical_medicine                        | 49e82d  | accuracy | gen  | 40.91                |
| ceval-urban_and_rural_planner                  | 95b885  | accuracy | gen  | 54.35                |
| ceval-accountant                               | 002837  | accuracy | gen  | 38.78                |
| ceval-fire_engineer                            | bc23f5  | accuracy | gen  | 35.48                |
| ceval-environmental_impact_assessment_engineer | c64e2d  | accuracy | gen  | 22.58                |
| ceval-tax_accountant                           | 3a5e3c  | accuracy | gen  | 28.57                |
| ceval-physician                                | 6e277d  | accuracy | gen  | 38.78                |

运行日志：

```
aunch OpenICLEval[qwen2.5-1.5b-instruct-hf/cmmlu-logical] on CPU                       
 70%|██████████████████████████████████████████                  | 168/240 [22:10<08:51,  7.38s/it]
outputs/default/20250429_100836/logs/eval/qwen2.5-1.5b-instruct-hf/ceval-physician.out
100%|████████████████████████████████████████████████████████████| 240/240 [32:45<00:00,  8.19s/it]

dataset                                         version    metric    mode      qwen1.5-0.5b-chat-hf  qwen2.5-1.5b-instruct-hf
----------------------------------------------  ---------  --------  ------  ----------------------  --------------------------
demo_gsm8k                                      17d0dc     accuracy  gen                       3.12  -
cmmlu-agronomy                                  1b5abe     accuracy  gen                       0.00  -
cmmlu-anatomy                                   f3f8bb     accuracy  gen                       0.68  -
cmmlu-ancient_chinese                           43111c     accuracy  gen                       1.83  -
cmmlu-arts                                      b6e1d6     accuracy  gen                       3.75  -
cmmlu-astronomy                                 3bd739     accuracy  gen                       0.61  -
cmmlu-business_ethics                           4a2346     accuracy  gen                      11.48  -
cmmlu-chinese_civil_service_exam                6e22c2     accuracy  gen                       0.00  -
cmmlu-chinese_driving_rule                      5c8e68     accuracy  gen                       6.11  -
cmmlu-chinese_food_culture                      aa203f     accuracy  gen                       8.09  -
cmmlu-chinese_foreign_policy                    1b2a69     accuracy  gen                       6.54  -
cmmlu-chinese_history                           7f3da0     accuracy  gen                       7.43  -
cmmlu-chinese_literature                        16f7f9     accuracy  gen                       0.98  -
cmmlu-chinese_teacher_qualification             cae559     accuracy  gen                       0.00  -
cmmlu-clinical_knowledge                        e1ff3c     accuracy  gen                       0.00  -
cmmlu-college_actuarial_science                 dd69d9     accuracy  gen                       0.00  -
cmmlu-college_education                         cf6884     accuracy  gen                       0.93  -
cmmlu-college_engineering_hydrology             b3296d     accuracy  gen                       1.89  -
cmmlu-college_law                               1f7f65     accuracy  gen                       0.93  -
cmmlu-college_mathematics                       dfeadf     accuracy  gen                       0.00  -
cmmlu-college_medical_statistics                3c9fc0     accuracy  gen                       0.00  -
cmmlu-college_medicine                          20ea93     accuracy  gen                       0.00  -
cmmlu-computer_science                          3e570b     accuracy  gen                       0.49  -
cmmlu-computer_security                         25ada2     accuracy  gen                       2.34  -
cmmlu-conceptual_physics                        85fa17     accuracy  gen                      12.24  -
cmmlu-construction_project_management           9c916e     accuracy  gen                       3.60  -
cmmlu-economics                                 fa5173     accuracy  gen                       3.77  -
cmmlu-education                                 1b5cdc     accuracy  gen                       0.61  -
cmmlu-electrical_engineering                    1214ff     accuracy  gen                       1.16  -
cmmlu-elementary_chinese                        9c88f7     accuracy  gen                       1.98  -
cmmlu-elementary_commonsense                    bcaca6     accuracy  gen                       4.04  -
cmmlu-elementary_information_and_technology     b028a0     accuracy  gen                       0.42  -
cmmlu-elementary_mathematics                    874577     accuracy  gen                       0.87  -
cmmlu-ethnology                                 5b63f2     accuracy  gen                       2.96  -
cmmlu-food_science                              a580ec     accuracy  gen                       3.50  -
cmmlu-genetics                                  94fe78     accuracy  gen                       1.70  -
cmmlu-global_facts                              7cc427     accuracy  gen                      20.13  -
cmmlu-high_school_biology                       7f868f     accuracy  gen                       1.78  -
cmmlu-high_school_chemistry                     dc5fee     accuracy  gen                       0.76  -
cmmlu-high_school_geography                     b92326     accuracy  gen                       3.39  -
cmmlu-high_school_mathematics                   64e6b9     accuracy  gen                       0.61  -
cmmlu-high_school_physics                       361088     accuracy  gen                       1.82  -
cmmlu-high_school_politics                      343b99     accuracy  gen                       1.40  -
cmmlu-human_sexuality                           ffde7a     accuracy  gen                      13.49  -
cmmlu-international_law                         87dfd3     accuracy  gen                       7.57  -
cmmlu-journalism                                1e8127     accuracy  gen                       0.00  -
cmmlu-jurisprudence                             3782cd     accuracy  gen                       2.43  -
cmmlu-legal_and_moral_basis                     5f37c3     accuracy  gen                      17.29  -
cmmlu-logical                                   c85511     accuracy  gen                       1.63  -
cmmlu-machine_learning                          0bdf84     accuracy  gen                       6.56  -
cmmlu-management                                869b6e     accuracy  gen                       0.00  -
cmmlu-marketing                                 1cfb4c     accuracy  gen                       6.67  -
cmmlu-marxist_theory                            fc22e6     accuracy  gen                       8.47  -
cmmlu-modern_chinese                            dd73b3     accuracy  gen                       2.59  -
cmmlu-nutrition                                 0da8ab     accuracy  gen                       1.38  -
cmmlu-philosophy                                5bf8d5     accuracy  gen                       4.76  -
cmmlu-professional_accounting                   36b39a     accuracy  gen                       1.14  -
cmmlu-professional_law                          0af151     accuracy  gen                       1.42  -
cmmlu-professional_medicine                     6a1d3e     accuracy  gen                       0.27  -
cmmlu-professional_psychology                   c15514     accuracy  gen                       0.00  -
cmmlu-public_relations                          d5be35     accuracy  gen                       3.45  -
cmmlu-security_study                            84c059     accuracy  gen                       6.67  -
cmmlu-sociology                                 9645be     accuracy  gen                       1.33  -
cmmlu-sports_science                            3249c4     accuracy  gen                       1.82  -
cmmlu-traditional_chinese_medicine              9a6a77     accuracy  gen                       1.62  -
cmmlu-virology                                  02753f     accuracy  gen                       1.18  -
cmmlu-world_history                             9a94e5     accuracy  gen                       3.11  -
cmmlu-world_religions                           5c2ff5     accuracy  gen                      11.25  -
ceval-computer_network                          db9ce2     accuracy  gen                      36.84  -
ceval-operating_system                          1c2571     accuracy  gen                      26.32  -
ceval-computer_architecture                     a74dad     accuracy  gen                      14.29  -
ceval-college_programming                       4ca32a     accuracy  gen                      32.43  -
ceval-college_physics                           963fa8     accuracy  gen                      31.58  -
ceval-college_chemistry                         e78857     accuracy  gen                      37.50  -
ceval-advanced_mathematics                      ce03e2     accuracy  gen                      15.79  -
ceval-probability_and_statistics                65e812     accuracy  gen                      38.89  -
ceval-discrete_mathematics                      e894ae     accuracy  gen                      18.75  -
ceval-electrical_engineer                       ae42b9     accuracy  gen                      37.84  -
ceval-metrology_engineer                        ee34ea     accuracy  gen                      45.83  -
ceval-high_school_mathematics                   1dc5bf     accuracy  gen                      44.44  -
ceval-high_school_physics                       adf25f     accuracy  gen                      42.11  -
ceval-high_school_chemistry                     2ed27f     accuracy  gen                      42.11  -
ceval-high_school_biology                       8e2b9a     accuracy  gen                      21.05  -
ceval-middle_school_mathematics                 bee8d5     accuracy  gen                      21.05  -
ceval-middle_school_biology                     86817c     accuracy  gen                      47.62  -
ceval-middle_school_physics                     8accf6     accuracy  gen                      42.11  -
ceval-middle_school_chemistry                   167a15     accuracy  gen                      50.00  -
ceval-veterinary_medicine                       b4e08d     accuracy  gen                      26.09  -
ceval-college_economics                         f3f4e6     accuracy  gen                      30.91  -
ceval-business_administration                   c1614e     accuracy  gen                      27.27  -
ceval-marxism                                   cf874c     accuracy  gen                      47.37  -
ceval-mao_zedong_thought                        51c7a4     accuracy  gen                      37.50  -
ceval-education_science                         591fee     accuracy  gen                      44.83  -
ceval-teacher_qualification                     4e4ced     accuracy  gen                      50.00  -
ceval-high_school_politics                      5c0de2     accuracy  gen                      42.11  -
ceval-high_school_geography                     865461     accuracy  gen                      36.84  -
ceval-middle_school_politics                    5be3e7     accuracy  gen                      52.38  -
ceval-middle_school_geography                   8a63be     accuracy  gen                      66.67  -
ceval-modern_chinese_history                    fc01af     accuracy  gen                      39.13  -
ceval-ideological_and_moral_cultivation         a2aa4a     accuracy  gen                      47.37  -
ceval-logic                                     f5b022     accuracy  gen                      36.36  -
ceval-law                                       a110a1     accuracy  gen                      37.50  -
ceval-chinese_language_and_literature           0f8b68     accuracy  gen                      30.43  -
ceval-art_studies                               2a1300     accuracy  gen                      33.33  -
ceval-professional_tour_guide                   4e673e     accuracy  gen                      44.83  -
ceval-legal_professional                        ce8787     accuracy  gen                      39.13  -
ceval-high_school_chinese                       315705     accuracy  gen                      31.58  -
ceval-high_school_history                       7eb30a     accuracy  gen                      70.00  -
ceval-middle_school_history                     48ab4a     accuracy  gen                      50.00  -
ceval-civil_servant                             87d061     accuracy  gen                      42.55  -
ceval-sports_science                            70f27b     accuracy  gen                      31.58  -
ceval-plant_protection                          8941f9     accuracy  gen                      54.55  -
ceval-basic_medicine                            c409d6     accuracy  gen                      47.37  -
ceval-clinical_medicine                         49e82d     accuracy  gen                      40.91  -
ceval-urban_and_rural_planner                   95b885     accuracy  gen                      54.35  -
ceval-accountant                                002837     accuracy  gen                      38.78  -
ceval-fire_engineer                             bc23f5     accuracy  gen                      35.48  -
ceval-environmental_impact_assessment_engineer  c64e2d     accuracy  gen                      22.58  -
ceval-tax_accountant                            3a5e3c     accuracy  gen                      28.57  -
ceval-physician                                 6e277d     accuracy  gen                      38.78  -
04/29 11:13:55 - OpenCompass - INFO - write summary to /root/workspace/opencompass/outputs/default/20250429_100836/summary/summary_20250429_100836.txt
04/29 11:13:55 - OpenCompass - INFO - write csv to /root/workspace/opencompass/outputs/default/20250429_100836/summary/summary_20250429_100836.csv


The markdown format results is as below:

| dataset | version | metric | mode | qwen1.5-0.5b-chat-hf | qwen2.5-1.5b-instruct-hf |
|----- | ----- | ----- | ----- | ----- | -----|
| demo_gsm8k | 17d0dc | accuracy | gen | 3.12 | - |
| cmmlu-agronomy | 1b5abe | accuracy | gen | 0.00 | - |
| cmmlu-anatomy | f3f8bb | accuracy | gen | 0.68 | - |
| cmmlu-ancient_chinese | 43111c | accuracy | gen | 1.83 | - |
| cmmlu-arts | b6e1d6 | accuracy | gen | 3.75 | - |
| cmmlu-astronomy | 3bd739 | accuracy | gen | 0.61 | - |
| cmmlu-business_ethics | 4a2346 | accuracy | gen | 11.48 | - |
| cmmlu-chinese_civil_service_exam | 6e22c2 | accuracy | gen | 0.00 | - |
| cmmlu-chinese_driving_rule | 5c8e68 | accuracy | gen | 6.11 | - |
| cmmlu-chinese_food_culture | aa203f | accuracy | gen | 8.09 | - |
| cmmlu-chinese_foreign_policy | 1b2a69 | accuracy | gen | 6.54 | - |
| cmmlu-chinese_history | 7f3da0 | accuracy | gen | 7.43 | - |
| cmmlu-chinese_literature | 16f7f9 | accuracy | gen | 0.98 | - |
| cmmlu-chinese_teacher_qualification | cae559 | accuracy | gen | 0.00 | - |
| cmmlu-clinical_knowledge | e1ff3c | accuracy | gen | 0.00 | - |
| cmmlu-college_actuarial_science | dd69d9 | accuracy | gen | 0.00 | - |
| cmmlu-college_education | cf6884 | accuracy | gen | 0.93 | - |
| cmmlu-college_engineering_hydrology | b3296d | accuracy | gen | 1.89 | - |
| cmmlu-college_law | 1f7f65 | accuracy | gen | 0.93 | - |
| cmmlu-college_mathematics | dfeadf | accuracy | gen | 0.00 | - |
| cmmlu-college_medical_statistics | 3c9fc0 | accuracy | gen | 0.00 | - |
| cmmlu-college_medicine | 20ea93 | accuracy | gen | 0.00 | - |
| cmmlu-computer_science | 3e570b | accuracy | gen | 0.49 | - |
| cmmlu-computer_security | 25ada2 | accuracy | gen | 2.34 | - |
| cmmlu-conceptual_physics | 85fa17 | accuracy | gen | 12.24 | - |
| cmmlu-construction_project_management | 9c916e | accuracy | gen | 3.60 | - |
| cmmlu-economics | fa5173 | accuracy | gen | 3.77 | - |
| cmmlu-education | 1b5cdc | accuracy | gen | 0.61 | - |
| cmmlu-electrical_engineering | 1214ff | accuracy | gen | 1.16 | - |
| cmmlu-elementary_chinese | 9c88f7 | accuracy | gen | 1.98 | - |
| cmmlu-elementary_commonsense | bcaca6 | accuracy | gen | 4.04 | - |
| cmmlu-elementary_information_and_technology | b028a0 | accuracy | gen | 0.42 | - |
| cmmlu-elementary_mathematics | 874577 | accuracy | gen | 0.87 | - |
| cmmlu-ethnology | 5b63f2 | accuracy | gen | 2.96 | - |
| cmmlu-food_science | a580ec | accuracy | gen | 3.50 | - |
| cmmlu-genetics | 94fe78 | accuracy | gen | 1.70 | - |
| cmmlu-global_facts | 7cc427 | accuracy | gen | 20.13 | - |
| cmmlu-high_school_biology | 7f868f | accuracy | gen | 1.78 | - |
| cmmlu-high_school_chemistry | dc5fee | accuracy | gen | 0.76 | - |
| cmmlu-high_school_geography | b92326 | accuracy | gen | 3.39 | - |
| cmmlu-high_school_mathematics | 64e6b9 | accuracy | gen | 0.61 | - |
| cmmlu-high_school_physics | 361088 | accuracy | gen | 1.82 | - |
| cmmlu-high_school_politics | 343b99 | accuracy | gen | 1.40 | - |
| cmmlu-human_sexuality | ffde7a | accuracy | gen | 13.49 | - |
| cmmlu-international_law | 87dfd3 | accuracy | gen | 7.57 | - |
| cmmlu-journalism | 1e8127 | accuracy | gen | 0.00 | - |
| cmmlu-jurisprudence | 3782cd | accuracy | gen | 2.43 | - |
| cmmlu-legal_and_moral_basis | 5f37c3 | accuracy | gen | 17.29 | - |
| cmmlu-logical | c85511 | accuracy | gen | 1.63 | - |
| cmmlu-machine_learning | 0bdf84 | accuracy | gen | 6.56 | - |
| cmmlu-management | 869b6e | accuracy | gen | 0.00 | - |
| cmmlu-marketing | 1cfb4c | accuracy | gen | 6.67 | - |
| cmmlu-marxist_theory | fc22e6 | accuracy | gen | 8.47 | - |
| cmmlu-modern_chinese | dd73b3 | accuracy | gen | 2.59 | - |
| cmmlu-nutrition | 0da8ab | accuracy | gen | 1.38 | - |
| cmmlu-philosophy | 5bf8d5 | accuracy | gen | 4.76 | - |
| cmmlu-professional_accounting | 36b39a | accuracy | gen | 1.14 | - |
| cmmlu-professional_law | 0af151 | accuracy | gen | 1.42 | - |
| cmmlu-professional_medicine | 6a1d3e | accuracy | gen | 0.27 | - |
| cmmlu-professional_psychology | c15514 | accuracy | gen | 0.00 | - |
| cmmlu-public_relations | d5be35 | accuracy | gen | 3.45 | - |
| cmmlu-security_study | 84c059 | accuracy | gen | 6.67 | - |
| cmmlu-sociology | 9645be | accuracy | gen | 1.33 | - |
| cmmlu-sports_science | 3249c4 | accuracy | gen | 1.82 | - |
| cmmlu-traditional_chinese_medicine | 9a6a77 | accuracy | gen | 1.62 | - |
| cmmlu-virology | 02753f | accuracy | gen | 1.18 | - |
| cmmlu-world_history | 9a94e5 | accuracy | gen | 3.11 | - |
| cmmlu-world_religions | 5c2ff5 | accuracy | gen | 11.25 | - |
| ceval-computer_network | db9ce2 | accuracy | gen | 36.84 | - |
| ceval-operating_system | 1c2571 | accuracy | gen | 26.32 | - |
| ceval-computer_architecture | a74dad | accuracy | gen | 14.29 | - |
| ceval-college_programming | 4ca32a | accuracy | gen | 32.43 | - |
| ceval-college_physics | 963fa8 | accuracy | gen | 31.58 | - |
| ceval-college_chemistry | e78857 | accuracy | gen | 37.50 | - |
| ceval-advanced_mathematics | ce03e2 | accuracy | gen | 15.79 | - |
| ceval-probability_and_statistics | 65e812 | accuracy | gen | 38.89 | - |
| ceval-discrete_mathematics | e894ae | accuracy | gen | 18.75 | - |
| ceval-electrical_engineer | ae42b9 | accuracy | gen | 37.84 | - |
| ceval-metrology_engineer | ee34ea | accuracy | gen | 45.83 | - |
| ceval-high_school_mathematics | 1dc5bf | accuracy | gen | 44.44 | - |
| ceval-high_school_physics | adf25f | accuracy | gen | 42.11 | - |
| ceval-high_school_chemistry | 2ed27f | accuracy | gen | 42.11 | - |
| ceval-high_school_biology | 8e2b9a | accuracy | gen | 21.05 | - |
| ceval-middle_school_mathematics | bee8d5 | accuracy | gen | 21.05 | - |
| ceval-middle_school_biology | 86817c | accuracy | gen | 47.62 | - |
| ceval-middle_school_physics | 8accf6 | accuracy | gen | 42.11 | - |
| ceval-middle_school_chemistry | 167a15 | accuracy | gen | 50.00 | - |
| ceval-veterinary_medicine | b4e08d | accuracy | gen | 26.09 | - |
| ceval-college_economics | f3f4e6 | accuracy | gen | 30.91 | - |
| ceval-business_administration | c1614e | accuracy | gen | 27.27 | - |
| ceval-marxism | cf874c | accuracy | gen | 47.37 | - |
| ceval-mao_zedong_thought | 51c7a4 | accuracy | gen | 37.50 | - |
| ceval-education_science | 591fee | accuracy | gen | 44.83 | - |
| ceval-teacher_qualification | 4e4ced | accuracy | gen | 50.00 | - |
| ceval-high_school_politics | 5c0de2 | accuracy | gen | 42.11 | - |
| ceval-high_school_geography | 865461 | accuracy | gen | 36.84 | - |
| ceval-middle_school_politics | 5be3e7 | accuracy | gen | 52.38 | - |
| ceval-middle_school_geography | 8a63be | accuracy | gen | 66.67 | - |
| ceval-modern_chinese_history | fc01af | accuracy | gen | 39.13 | - |
| ceval-ideological_and_moral_cultivation | a2aa4a | accuracy | gen | 47.37 | - |
| ceval-logic | f5b022 | accuracy | gen | 36.36 | - |
| ceval-law | a110a1 | accuracy | gen | 37.50 | - |
| ceval-chinese_language_and_literature | 0f8b68 | accuracy | gen | 30.43 | - |
| ceval-art_studies | 2a1300 | accuracy | gen | 33.33 | - |
| ceval-professional_tour_guide | 4e673e | accuracy | gen | 44.83 | - |
| ceval-legal_professional | ce8787 | accuracy | gen | 39.13 | - |
| ceval-high_school_chinese | 315705 | accuracy | gen | 31.58 | - |
| ceval-high_school_history | 7eb30a | accuracy | gen | 70.00 | - |
| ceval-middle_school_history | 48ab4a | accuracy | gen | 50.00 | - |
| ceval-civil_servant | 87d061 | accuracy | gen | 42.55 | - |
| ceval-sports_science | 70f27b | accuracy | gen | 31.58 | - |
| ceval-plant_protection | 8941f9 | accuracy | gen | 54.55 | - |
| ceval-basic_medicine | c409d6 | accuracy | gen | 47.37 | - |
| ceval-clinical_medicine | 49e82d | accuracy | gen | 40.91 | - |
| ceval-urban_and_rural_planner | 95b885 | accuracy | gen | 54.35 | - |
| ceval-accountant | 002837 | accuracy | gen | 38.78 | - |
| ceval-fire_engineer | bc23f5 | accuracy | gen | 35.48 | - |
| ceval-environmental_impact_assessment_engineer | c64e2d | accuracy | gen | 22.58 | - |
| ceval-tax_accountant | 3a5e3c | accuracy | gen | 28.57 | - |
| ceval-physician | 6e277d | accuracy | gen | 38.78 | - |

04/29 11:13:55 - OpenCompass - INFO - write markdown summary to /root/workspace/opencompass/outputs/default/20250429_100836/summary/summary_20250429_100836.md
```



## 最终总结

以下是对 **Qwen-0.5B** 和 **Qwen-1.5B** 两个模型在三个数据集（CEval、CMMLU、GSM8K）上的性能表现总结和对比分析：

---

### **1. CEval 数据集**

CEval 是一个面向中文语言理解的多任务基准测试，涵盖 STEM（科学、技术、工程、数学）、社会科学、人文学科等多个领域。以下是 Qwen-0.5B 和 Qwen-1.5B 在该数据集上的表现：

| 模型      | CEval 总体准确率 | STEM 领域准确率 | 社会科学准确率 | 人文学科准确率 |
| --------- | ---------------- | --------------- | -------------- | -------------- |
| Qwen-0.5B | 54.28%           | 47.90%          | 68.98%         | 55.75%         |
| Qwen-1.5B | **66.28%**       | **60.86%**      | **77.17%**     | **67.87%**     |

#### **关键观察：**

- **总体表现：** Qwen-1.5B 的总体准确率为 **66.28%**，显著高于 Qwen-0.5B 的 **54.28%**。
- **STEM 领域：** Qwen-1.5B 在 STEM 领域提升明显，从 **47.90% 提升到 60.86%**。
- **社会科学：** Qwen-1.5B 在社会科学领域的准确率也有所提升，从 **68.98% 提升到 77.17%**。
- **人文学科：** Qwen-1.5B 在人文领域的表现也有改善，从 **55.75% 提升到 67.87%**。

#### **亮点科目（以 Qwen-1.5B 为主）：**

- **高分科目：**
  - 中学化学：95%
  - 思想道德修养与法律基础（Ideological_and_moral_cultivation）：100%
  - 中学生物学：90.48%
- **低分科目：**
  - 高等数学：15.79%
  - 概率与统计：27.78%

这表明 Qwen-1.5B 在中学阶段的知识掌握较强，但在高等数学等复杂推理任务上仍存在挑战。

---

### **2. CMMLU 数据集**

CMMLU 是一个针对中文语言理解的大规模多学科评估数据集，涵盖多个知识领域和专业考试内容。

| 模型      | CMMLU 总体准确率 | STEM 领域准确率 | 社会科学准确率 | 人文学科准确率 |
| --------- | ---------------- | --------------- | -------------- | -------------- |
| Qwen-0.5B | 25.78%           | 16.41%          | 26.73%         | 27.69%         |
| Qwen-1.5B | **2.68%**        | **1.23%**       | **2.54%**      | **2.97%**      |

#### **关键观察：**

- **总体表现：** 与 Qwen-0.5B 相比，Qwen-1.5B 在 CMMLU 上的表现大幅下降，从 **25.78% 下降到 2.68%**。
- **STEM 领域：** Qwen-1.5B 在 STEM 领域的准确率下降明显，从 **16.41% 下降到 1.23%**。
- **社会科学：** Qwen-1.5B 在社会科学领域的准确率也大幅降低，从 **26.73% 下降到 2.54%**。
- **人文学科：** Qwen-1.5B 在人文学科领域的表现也显著下降，从 **27.69% 下降到 2.97%**。

#### **可能原因分析：**

- Qwen-1.5B 可能未针对 CMMLU 数据集进行充分优化或微调。
- 可能由于参数优化方向不同，导致其在某些特定类型的问题上表现不佳。

#### **个别高分情况：**

- Qwen-0.5B 在“Legal_and_moral_basis（法律与道德基础）”这一子任务中表现较好，准确率为 **66.82%**。

---

### **3. GSM8K 数据集**

GSM8K 是一个专门用于测评数学推理能力的数据集，包含 8,500 个小学水平的数学问题。

| 模型      | GSM8K 准确率 |
| --------- | ------------ |
| Qwen-0.5B | 38.97%       |
| Qwen-1.5B | **63.00%**   |

#### **关键观察：**

- Qwen-1.5B 在 GSM8K 数学推理任务中的表现显著优于 Qwen-0.5B，准确率从 **38.97% 提升到 63.00%**。
- 这表明 Qwen-1.5B 的数学推理能力和解题技巧有明显改进。

---

### **综合对比总结**

| 指标           | Qwen-0.5B                | Qwen-1.5B          | 对比结论                      |
| -------------- | ------------------------ | ------------------ | ----------------------------- |
| **CEval 总体** | 54.28%                   | **66.28%**         | Qwen-1.5B 明显更优            |
| **CMMLU 总体** | 25.78%                   | **2.68%**          | Qwen-0.5B 更优                |
| **GSM8K 数学** | 38.97%                   | **63.00%**         | Qwen-1.5B 明显更优            |
| **优势领域**   | 中文理解、法律与道德基础 | 中文理解、数学推理 | 各有侧重                      |
| **劣势领域**   | 高等数学、概率与统计     | 所有 CMMLU 子任务  | Qwen-1.5B 在 CMMLU 上表现较差 |

---

### **总结建议**

- 如果您的应用场景主要集中在 **中文理解、数学推理** 等领域，Qwen-1.5B 是更好的选择。
- 如果您的任务涉及 **复杂知识类问题（如 STEM、社会科学）**，且需要稳定的多领域表现，Qwen-0.5B 在 CMMLU 数据集上的表现更稳定。
- 建议进一步对 Qwen-1.5B 进行针对性的训练或优化，尤其是在 **CMMLU 数据集相关任务** 上，以弥补目前的不足。



## 附录

### 数据介绍

#### cmmlu 数据

###### 介绍

CMMLU是一个综合性的中文评估基准，专门用于评估中文背景下语言模型的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。它包括需要自然科学计算和推理的任务，以及涉及人文、社会科学知识和中国驾驶规则等实际方面的任务。此外，CMMLU中的许多任务都有针对中国的答案，这些答案可能不适用于其他地区或语言。因此，CMMLU是一个完全本地化的中文评估基准。

###### 数据样例

数据集具有

- 问题：问题的主体
- A、 B、C、D：模型应该选择的选项
- 答案：问题的正确答案

```
Question：同一物种的两类细胞各产生一种分泌蛋白，组成这两种蛋白质的各种氨基酸含量相同，但排列顺序不同。其原因是参与这两种蛋白质合成的：
A. tRNA种类不同
B. 同一密码子所决定的氨基酸不同
C. mRNA碱基序列不同
D. 核糖体成分不同
Answer：C
```

###### 数据加载

huggingface的地址：https://huggingface.co/datasets/haonan-li/cmmlu

魔塔社区上的的数据地址： https://modelscope.cn/datasets/opencompass/cmmlu/summary

- 下载前的准备

```sh
source /etc/network_turbo # Autodl上的科学上网
pip install datasets  # 安装 Hugging Face 数据集库
```

- 加载某一个种类的数据

```
from datasets import load_dataset
cmmlu=load_dataset(r"haonan-li/cmmlu", 'agronomy')
print(cmmlu['test'][0])

# 数据被下载到：/root/.cache/huggingface/datasets/haonan-li___cmmlu/agronomy
```

- 一次性加载任意种类的数据

```sh
task_list = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history', 'chinese_literature', 
'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science', 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics', 'college_medical_statistics', 'college_medicine', 'computer_science',
'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education', 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense', 'elementary_information_and_technology', 'elementary_mathematics', 
'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics', 'high_school_politics', 'human_sexuality',
'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical', 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting', 'professional_law', 
'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology', 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

from datasets import load_dataset
cmmlu = {k: load_dataset(r"haonan-li/cmmlu", k) for k in task_list}

# 数据被下载到：/root/.cache/huggingface/datasets/haonan-li___cmmlu
```

下载过程：

```
root@autodl-container-c38f4899d3-e6629854:~/workspace# python data_down.py 
cmmlu_v1_0_1.zip: 100%|████████████████████████████████████████████████████████████████████| 1.08M/1.08M [00:03<00:00, 270kB/s]
Generating test split: 169 examples [00:00, 2562.80 examples/s]
Generating dev split: 5 examples [00:00, 997.65 examples/s]
.....
Generating test split: 160 examples [00:00, 1563.53 examples/s]
Generating dev split: 5 examples [00:00, 445.41 examples/s]
```

我们需要将下载好的数据，复制到opencompass下面的data目录下，如下所示：

```sh
opencompass/
├── data/          # 默认数据目录
│   ├── math/     # 示例：math 数据集
│	├── cmmul/    # 示例：cmmul 数据集
│   └── ...        # 其他数据集
├── tools/
└── ...
```

#### ceval数据

###### 介绍

C-Eval是一个全面的中国基础模型评估套件。它由13948道多项选择题组成，涵盖52个不同学科和4个难度级别。请访问我们的网站和GitHub或查看我们的论文以了解更多详细信息。

每个主题由三个部分组成：dev、val和test。每个受试者的开发集由五个示例组成，并对少数镜头评估进行了解释。val集旨在用于超参数调整。测试集用于模型评估。测试拆分上的标签不会发布，用户需要提交结果以自动获得测试准确性

###### 数据样例

```
id: 1
question: 25 °C时，将pH=2的强酸溶液与pH=13的强碱溶液混合，所得混合液的pH=11，则强酸溶液与强碱溶液 的体积比是(忽略混合后溶液的体积变化)____
A: 11:1
B: 9:1
C: 1:11
D: 1:9
answer: B
explanation: 
1. pH=13的强碱溶液中c(OH-)=0.1mol/L, pH=2的强酸溶液中c(H+)=0.01mol/L，酸碱混合后pH=11，即c(OH-)=0.001mol/L。
2. 设强酸和强碱溶液的体积分别为x和y，则：c(OH-)=(0.1y-0.01x)/(x+y)=0.001，解得x:y=9:1。
```

###### 数据加载

gitHub地址：https://github.com/hkust-nlp/ceval/tree/main

hugginface地址：https://huggingface.co/datasets/ceval/ceval-exam

魔塔社区的地址：https://modelscope.cn/datasets/opencompass/ceval-exam

```python
from datasets import load_dataset
dataset=load_dataset(r"ceval/ceval-exam",name="computer_network", cache_dir="/root/autodl-tmp/workspace/datasets")

print(dataset['val'][0])
# {'id': 0, 'question': '使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____', 'A': '1', 'B': '2', 'C': '3', 'D': '4', 'answer': 'C', 'explanation': ''}

```

#### gsm8k数据

###### 介绍

GSM8K是由人类问题编写者创建的8500个高质量、语言多样的小学数学单词问题的数据集。数据集分为7500个训练问题和1000个测试问题。这些问题需要2到8个步骤来解决，解决方案主要涉及使用基本算术运算（+−×÷）执行一系列基本计算，以获得最终答案。一个聪明的中学生应该能够解决每一个问题。它可用于多步骤数学推理。

###### 数据样例

数据集具有

- 问题：问题的主体
- 答案：计算过程和答案

```
{
    "question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?",
    "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3"
}
```

###### 数据加载

hugginface地址：https://huggingface.co/datasets/openai/gsm8k

魔塔社区的地址：https://modelscope.cn/datasets/modelscope/gsm8k

```python
source /etc/network_turbo

# hugginface下载地址
from datasets import load_dataset
dataset=load_dataset(r"openai/gsm8k",name="main", cache_dir="/root/autodl-tmp/workspace/datasets")


# 魔塔下载地址
from datasets import load_dataset
dataset=load_dataset(r"openai/gsm8k",name="main", split='train', cache_dir="/root/autodl-tmp/workspace/datasets")
```



### 基座模型和对话模型

基座模型：

- 基座模型是**通用预训练模型**，通过海量无标注数据（如网页文本、书籍、代码等）进行自监督学习，掌握语言的基础规律和世界知识。 
- **典型代表**：GPT-3、LLaMA、PaLM、BERT。
- **应用场景** 文本摘要/翻译  代码生成  知识问答（需额外检索增强）  下游任务的微调基座

对话模型：

- 对话模型是**基于基座模型微调**的专用模型，通过指令微调（Instruction Tuning）和人类反馈强化学习（RLHF）优化对话交互能力。
- **典型代表**：ChatGPT、Claude、Gemini、LLaMA-2-Chat。
- **应用场景**：智能客服  虚拟助手  教育陪练  娱乐社交

举例说明:

- Meta LLaMA 系列
  - `LLaMA-2`（基座）：适合研究者微调
  - `LLaMA-2-Chat`（对话模型）：直接用于聊天应用
- OpenAI 系列
  - `GPT-3`（基座） → `ChatGPT`（对话模型）


