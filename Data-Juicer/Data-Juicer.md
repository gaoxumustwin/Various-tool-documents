# Data-Juicer

## 简介

Data-Juicer 是一个一站式系统，由阿里开发，面向大模型的文本及多模态数据处理。我们提供了一个基于 JupyterLab 的 [Playground](http://8.138.149.181/)，您可以从浏览器中在线试用 Data-Juicer。 

地址：https://github.com/modelscope/data-juicer



## 安装

### 前置条件

- 推荐 Python>=3.9,<=3.10
- gcc >= 5 (at least C++14 support)

### 从源码安装 (指定使用场景)

- 运行以下命令以安装 `data_juicer` 可编辑模式的最新基础版本

```
source /etc/network_turbo
git clone https://github.com/modelscope/data-juicer.git

cd <path_to_data_juicer>
pip install -v -e .  # 安装最小依赖，支持基础功能
pip install -v -e .[tools] # 安装部分工具库的依赖
```

部分算子功能依赖于较大的或者平台兼容性不是很好的第三方库，因此用户可按需额外安装可选的依赖项:

依赖选项如下表所示:

标签描述`.` 或 `.[mini]`为基本 Data-Juicer 安装最小依赖项。`.[all]`为除沙盒之外的所有 OP 安装依赖项。`.[sci]`为与科学用途相关的 OP 安装依赖项。`.[dist]`安装用于分布式数据处理的额外依赖项。`.[dev]`安装作为贡献者开发软件包的依赖项。`.[tools]`安装专用工具（例如质量分类器）的依赖项。`.[sandbox]`安装沙盒的所有依赖项。

### 从源码安装 (指定部分算子)

- 只安装部分算子依赖

随着OP数量的增长，全OP环境的依赖安装会变得越来越重。为此，我们提供了两个替代的、更轻量的选项，作为使用命令`pip install -v -e .[sci]`安装所有依赖的替代：

- 自动最小依赖安装：在执行Data-Juicer的过程中，将自动安装最小依赖。也就是说你可以安装mini后直接执行，但这种方式可能会导致一些(滞后的)依赖冲突。

- 手动最小依赖安装：可以通过如下指令手动安装适合特定执行配置的最小依赖，可以提前确定依赖冲突、使其更易解决:

  ```
  # 从源码安装
  python tools/dj_install.py --config path_to_your_data-juicer_config_file
  
  # 使用命令行工具
  dj-install --config path_to_your_data-juicer_config_file
  ```

### 使用 pip 安装

- 运行以下命令用 `pip` 安装 `data_juicer` 的最新发布版本：

```
pip install py-data-juicer
```

- 注意
  - 使用这种方法安装时，只有`data_juicer`中的基础的 API 和2个基础工具 （数据[处理](https://github.com/modelscope/data-juicer/blob/main/README_ZH.md#%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86)与[分析](https://github.com/modelscope/data-juicer/blob/main/README_ZH.md#%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90)）可以使用。如需更定制化地使用完整功能，建议[从源码进行安装](https://github.com/modelscope/data-juicer/blob/main/README_ZH.md#%E4%BB%8E%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)。
  - pypi 的发布版本较源码的最新版本有一定的滞后性，如需要随时跟进 `data_juicer` 的最新功能支持，建议[从源码进行安装](https://github.com/modelscope/data-juicer/blob/main/README_ZH.md#%E4%BB%8E%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)。

### 使用 Docker 安装

- 您可以选

  - 从DockerHub直接拉取我们的预置镜像:

    ```
    docker pull datajuicer/data-juicer:<version_tag>
    ```

  - 或者运行如下命令用我们提供的 [Dockerfile](https://github.com/modelscope/data-juicer/blob/main/Dockerfile) 来构建包括最新版本的 `data-juicer` 的 docker 镜像：

    ```
    docker build -t datajuicer/data-juicer:<version_tag> .
    ```

  - `<version_tag>`的格式类似于`v0.2.0`，与发布（Release）的版本号相同。

### 安装校验

```python
import data_juicer as dj
print(dj.__version__)
```



## 数据配置

DJ 支持多种数据集输入类型，包括本地文件、远程数据集（如 huggingface）；还支持数据验证和数据混合。

### 配置数据输入文件的两种方法

#### 简单场景

本地/HF（远程） 文件的单一路径

```
dataset_path: '/path/to/your/dataset' # 数据集目录或文件的路径
```

#### 高级方法

支持子配置项和更多功能

```
dataset:
configs:
- type: 'local'
path: 'path/to/your/dataset' # 数据集目录或文件的路径
```

### 支持的数据集格式

**支持的数据集格式参考项目的源码： load_strategy.py(项目的/data_juicer/core/data/load_strategy.py)，如下所示：**

```python
@DataLoadStrategyRegistry.register('ray', 'local', '*')
class RayLocalJsonDataLoadStrategy(RayDataLoadStrategy):
                               f'Error: {str(e)}')


@DataLoadStrategyRegistry.register('ray', 'remote', 'huggingface')
class RayHuggingfaceDataLoadStrategy(RayDataLoadStrategy):
            'Huggingface data load strategy for Ray is not implemented')


@DataLoadStrategyRegistry.register('default', 'local', '*')
class DefaultLocalDataLoadStrategy(DefaultDataLoadStrategy):
        return formatter.load_dataset(load_data_np, self.cfg)


@DataLoadStrategyRegistry.register('default', 'remote', 'huggingface')
class DefaultHuggingfaceDataLoadStrategy(DefaultDataLoadStrategy):
                            global_cfg=self.cfg)


@DataLoadStrategyRegistry.register('default', 'remote', 'modelscope')
class DefaultModelScopeDataLoadStrategy(DefaultDataLoadStrategy):
            'ModelScope data load strategy is not implemented')


@DataLoadStrategyRegistry.register('default', 'remote', 'arxiv')
class DefaultArxivDataLoadStrategy(DefaultDataLoadStrategy):
            'Arxiv data load strategy is not implemented')


@DataLoadStrategyRegistry.register('default', 'remote', 'wiki')
class DefaultWikiDataLoadStrategy(DefaultDataLoadStrategy):
        raise NotImplementedError('Wiki data load strategy is not implemented')


@DataLoadStrategyRegistry.register('default', 'remote', 'commoncrawl')
class DefaultCommonCrawlDataLoadStrategy(DefaultDataLoadStrategy):
            'CommonCrawl data load strategy is not implemented')
```

**各种数据配置文件位于：项目的/data-juicer/configs/datasets/*.yaml**

下面以本地数据集、Remote Huggingface 数据集、Remote  modelscope 数据集和远程 Arxiv 数据集为例

#### 本地数据集配置

`local_json.yaml`（configs/datasets/local_json.yam） 配置文件用于指定以 JSON 格式本地存储的数据集。*path* 是必需的，用于指定本地数据集路径，可以是单个文件或目录。*format* 是可选的，用于指定数据集格式。 对于本地文件，DJ 将自动检测文件格式并相应地加载数据集。支持 parquet、jsonl、json、csv、tsv、txt 和 jsonl.gz 等格式 

```
dataset:
configs:
- type: local
path: path/to/your/local/dataset.json
format: json
```

```
dataset:
configs:
- type: local
path: path/to/your/local/dataset.parquet
format: parquet
```

#### Remote Huggingface 数据集配置

`remote_huggingface.yaml` （configs/datasets/remote_huggingface.yaml）配置文件用于指定 huggingface 数据集。*type* 和 *source* 固定为 'remote' 和 'huggingface'，以定位 huggingface 加载逻辑。*path* 是必需的，用于标识 huggingface 数据集。*name*、*split* 和 *limit* 是可选的，用于指定数据集名称/拆分并限制要加载的样本数量。

```yaml
dataset:
configs:
- type: 'remote'
    source: 'huggingface'
    path: "HuggingFaceFW/fineweb"
    name: "CC-MAIN-2024-10"
    split: "train"
    limit: 1000
```

#### Remote modelscope 数据集配置

`remote_modelscope.yaml`（configs/datasets/remote_modelscope.yaml） 配置文件用于指定 modelscope数据集。*type* 和 *source* 固定为 'remote' 和 'modelscope'，以定位 modelscope加载逻辑。*path* 是必需的，用于标识 modelscope数据集。*name*、*split* 和 *limit* 是可选的，用于指定数据集名称/拆分并限制要加载的样本数量。 

```yaml
# global parameters
project_name: 'dataset-remote-modelscope'
dataset:
  configs:
    - type: 'remote'
      source: 'modelscope'
      path: 'modelscope/clue'
      subset_name: 'afqmc'
      split: 'train'
      limit: 1000
```

#### 远程 Arxiv 数据集配置

`remote_arxiv.yaml` 配置文件用于指定以 JSON 格式远程存储的数据集。*type* 和 *source* 固定为 'remote' 和 'arxiv'，以定位 arxiv 加载逻辑。 *lang*、*dump_date*、*force_download* 和 *url_limit* 是可选的，用于指定数据集语言、转储日期、强制下载和 URL 限制。

```
dataset:
configs:
- type: 'remote'
source: 'arxiv'
lang: 'en'
dump_date: 'latest'
force_download: false
url_limit: 2
```

**远程 ArXiv 数据集**通常指通过 API 或爬虫从 ArXiv 获取的论文元数据或全文数据，用于学术研究、自然语言处理（NLP）或机器学习任务。

### 配置数据混合

`mixture.yaml`（项目/configs/datasets/mixture.yaml） 配置文件演示了如何指定数据混合规则。DJ 将通过对数据集的一部分进行采样并应用适当的权重来混合数据集。 

```
dataset:
max_sample_num: 10000
configs:
- type: 'local'
weight: 1.0
path: 'path/to/json/file'
- type: 'local'
weight: 1.0
path: 'path/to/csv/file'
```

### 配置数据验证

`validator.yaml`（data-juicer/configs/datasets/validation_swift_messages.yaml 或 data-juicer/configs/datasets/validation_required_fields.yaml） 配置文件演示了如何指定数据验证规则。DJ 将通过对数据集的一部分进行采样并应用验证规则来验证数据集。

其对应的python功能代码为： data_validator.py（data-juicer/data-juicer/blob/main/data_juicer/core/data/data_validator.py)

```
dataset:
configs:
- type: local
path: path/to/data.json

validators:
- type: swift_messages
min_turns: 2
max_turns: 20
sample_size: 1000
- type: required_fields
required_fields:
- "text"
- "metadata"
- "language"
field_types:
text: "str"
metadata: "dict"
language: "str"
```





## 数据处理

### 全局配置文件

`data-juicer/configs/config_all.yaml` 是 **Data-Juicer** 工具中的一个**全局配置文件**，用于定义数据清洗和预处理的所有可用操作符（Operators）及其默认参数。它的核心作用是：

1. **集中管理所有数据处理操作符**，避免在单个任务配置中重复定义。
2. **提供默认参数**，简化用户配置（用户只需覆盖需要修改的参数）。
3. **支持模块化组合**，用户可以通过 `process.yaml` 选择性调用这些操作符。

![](D:\gx\Desktop\cutting-edge technology\Data-Juicer\img\1.jpg)

**与 process.yaml 的关系**

- `process.yaml` 是**任务级配置文件**，通过引用 `config_all.yaml` 中的操作符组合出具体流程。
- 用户在 `process.yaml` 中只需指定操作符名称和需要修改的参数，其余参数自动继承自 `config_all.yaml`
- **优先级规则**：
  - `process.yaml` 中显式指定的参数 > `config_all.yaml` 的默认值。
  - 未在 `process.yaml` 中定义的参数将自动继承全局配置。



### 数据集的流程配置文件

位于：data-juicer/configs/demo/process.yaml，`process.yaml`用于定义数据处理的流程和参数

```
# Process config example for dataset

# 全局参数
project_name: 'demo-process' # 项目名称（仅标识用途）
dataset_path: './demos/data/demo-dataset.jsonl'  # 输入数据路径
np: 4  # 处理数据集的子进程数

export_path: './outputs/demo-process/demo-processed.jsonl' # 处理后的数据路径 （加速处理）

# 具体清洗步骤
# 可叠加多个操作符（如去重、长度过滤、敏感词过滤等）
process:
  - language_id_score_filter: # 操作符：语言识别与过滤
      lang: 'zh'  # 目标语言（中文）
      min_score: 0.8 # 语言置信度阈值（≥0.8才保留）
```

**language_id_score_filter就是一个op算子（所有的算子在data-juicer/configs/config_all.yaml）**

这里language_id_score_filter 的作用：

- 使用语言检测模型（如fasttext）计算每条文本的语言类别和置信度。
- 仅保留语言为中文（`zh`）且置信度≥0.8的文本，过滤其他语言或低质量中文文本。

去除数据集中的非中文内容（如混入的英文、乱码等），确保数据符合大模型训练的语言要求（例如专练中文模型时需纯中文语料）。

### 数据处理命令

以配置文件路径作为参数来运行 `process_data.py` 或者 `dj-process` 命令行工具来处理数据集。

```
# 适用于从源码安装
python tools/process_data.py --config configs/demo/process.yaml

# 使用命令行工具
dj-process --config configs/demo/process.yaml
```

- **注意**：使用未保存在本地的第三方模型或资源的算子第一次运行可能会很慢，因为这些算子需要将相应的资源下载到缓存目录中。默认的下载缓存目录为`~/.cache/data_juicer`。您可通过设置 shell 环境变量 `DATA_JUICER_CACHE_HOME` 更改缓存目录位置，您也可以通过同样的方式更改 `DATA_JUICER_MODELS_CACHE` 或 `DATA_JUICER_ASSETS_CACHE` 来分别修改模型缓存或资源缓存目录:
- **注意**：对于使用了第三方模型的算子，在填写config文件时需要去声明其对应的`mem_required`（可以参考`config_all.yaml`文件中的设置）。Data-Juicer在运行过程中会根据内存情况和算子模型所需的memory大小来控制对应的进程数，以达成更好的数据处理的性能效率。而在使用CUDA环境运行时，如果不正确的声明算子的`mem_required`情况，则有可能导致CUDA Out of Memory。

```
# 缓存主目录
export DATA_JUICER_CACHE_HOME="/path/to/another/directory"
# 模型缓存目录
export DATA_JUICER_MODELS_CACHE="/path/to/another/directory/models"
# 资源缓存目录
export DATA_JUICER_ASSETS_CACHE="/path/to/another/directory/assets"
```

### 数据处理工具

下面是一些预处理脚本，用于在使用Data Juicer之前对数据集进行额外处理。

#### 按语言将数据集拆分为子数据集

dataset_split_by_language.py脚本将按语言信息将原始数据集拆分为不同的子数据集。

```sh
python tools/preprocess/dataset_split_by_language.py        \
    --src_dir             <src_dir>          \
    --target_dir          <target_dir>       \
    --suffixes            <suffixes>         \
    --text_key            <text_key>         \
    --num_proc            <num_proc>

# get help
python tools/preprocess/dataset_split_by_language.py --help
```

- src_dir：您只需将此参数设置为存储数据集的路径。
- target_dir：存储转换后的jsonl文件的结果目录。
- text_key：存储示例文本的字段的键名。默认值：文本
- suffixes：将要读取的文件的后缀。默认值：无
- num_proc（可选）：流程工人的数量。默认值为1。

#### 将原始的Alpaca-CoT数据转化为jsonl

使用 `raw_alpaca_cot_merge_add_meta.py` 工具转换原始 Alpaca-CoT 数据。

该工具用于将从 HuggingFace 下载的原始 Alpaca-CoT 数据集转换为 jsonl 格式文件。

```
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py       \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --num_proc          <num_proc>

# get help
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py --help
```

- `src_dir`：只需将该参数设置为存放 Alpaca-CoT 数据的路径
- `target_dir`：用于存储转换后 jsonl 文件的结果目录
- `num_proc`（可选）：进程工作线程数，默认为 1

#### 重新格式化csv或tsv文件

reformat_csv_nan_value.py用于将可能包含空值（Nan）的CSV或TSV文件重新格式化为多个JSONL文件。

```
python tools/preprocess/reformat_csv_nan_value.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --suffixes          <suffixes>        \
    --is_tsv            <is_tsv>          \
    --keep_default_na   <keep_default_na> \
    --num_proc          <num_proc>

# get help
python tools/preprocess/reformat_csv_nan_value.py --help
```

- `src_dir`：指定待处理文件的存储路径（支持*.csv/*.tsv格式文件）
- `target_dir`：转换后JSONL文件的输出目录
- `suffixes`：需处理的文件后缀列表（多参数示例：--suffixes '.tsv' '.csv'）
- `is_tsv`：布尔值，为True时使用'\t'作为分隔符，默认为','（CSV格式）
- `keep_default_na`：为False时解析所有字符串为NaN值，否则仅解析默认NaN标识
- `num_proc`（可选）：处理进程数（默认值：1）

#### JSONL文件格式化

reformat_jsonl_nan_value.py工具用于重新格式化可能包含空值(Nan)的JSONL文件。

```
python tools/preprocess/reformat_jsonl_nan_value.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --num_proc          <num_proc>

# get help
python tools/preprocess/reformat_jsonl_nan_value.py --help
```

- `src_dir`：指定待处理的JSONL文件存储路径（支持*.jsonl格式文件）
- `target_dir`：格式化后JSONL文件的输出目录
- `num_proc`（可选参数）：设置处理进程数（默认值：1个进程）

#### JSONL文件元字段序列化工具说明

serialize_meta.py工具用于对JSONL文件中的元字段进行序列化处理，解决因元字段结构不一致导致的HuggingFace Dataset读取失败问题。当不同样本的元字段存在差异（包括字段类型不一致）时，可将除指定字段外的所有元字段序列化为字符串格式，便于后续Data-Juicer处理流程。处理完成后通常需要使用deserialize_meta.py工具进行反序列化。

```sh
python tools/preprocess/serialize_meta.py           \
    --src_dir           <src_dir>         \
    --target_dir        <target_dir>      \
    --text_key          <text_key>        \
    --serialized_key    <serialized_key>  \
    --num_proc          <num_proc>

# get help
python tools/preprocess/serialize_meta.py --help
```

- `src_dir`：待处理的JSONL文件存储路径
- `target_dir`：转换后的JSONL文件保存路径
- `text_key`：指定不进行序列化的字段键名（默认值：'text'）
- `serialized_key`：序列化后的元信息存储字段键名（默认值：'source_info'）
- `num_proc`（可选）：处理进程数（默认值：1）



## 分布式数据处理

分布式处理文档：https://github.com/modelscope/data-juicer/blob/main/docs/Distributed_ZH.md

Data-Juicer 现在基于[RAY](https://www.ray.io/)实现了多机分布式数据处理。 对应Demo可以通过如下命令运行：

```
# 运行文字数据处理
python tools/process_data.py --config ./demos/process_on_ray/configs/demo.yaml

# 运行视频数据处理
python tools/process_data.py --config ./demos/process_video_on_ray/configs/demo.yaml
```

- 如果需要在多机上使用RAY执行数据处理，需要确保所有节点都可以访问对应的数据路径，即将对应的数据路径挂载在共享文件系统（如NAS）中。
- RAY 模式下的去重算子与单机版本不同，所有 RAY 模式下的去重算子名称都以 `ray` 作为前缀，例如 `ray_video_deduplicator` 和 `ray_document_deduplicator`。



## 数据分析

以配置文件路径为参数运行 `analyze_data.py` 或者 `dj-analyze` 命令行工具来分析数据集。

```sh
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/analyzer.yaml

# 使用命令行工具
dj-analyze --config configs/demo/analyzer.yaml

# 你也可以使用"自动"模式来避免写一个新的数据菜谱。它会使用全部可产出统计信息的 Filter 来分析
# 你的数据集的一小部分（如1000条样本，可通过 `auto_num` 参数指定）
dj-analyze --auto --dataset_path xx.jsonl [--auto_num 1000]
```

- 注意：Analyzer 只用于能在 stats 字段里产出统计信息的 Filter 算子和能在 meta 字段里产出 tags 或类别标签的其他算子。除此之外的其他的算子会在分析过程中被忽略。我们使用以下两种注册器来装饰相关的算子：
  - `NON_STATS_FILTERS`：装饰那些**不能**产出任何统计信息的 Filter 算子。
  - `TAGGING_OPS`：装饰那些能在 meta 字段中产出 tags 或类别标签的算子。



## 数据可视化

- 运行 `app.py` 来在浏览器中可视化您的数据集。
- **注意**：只可用于从源码安装的方法。

```
streamlit run app.py
```





## 使用 data-juicer 完成对数据的清洗

参考例子：https://zhuanlan.zhihu.com/p/756151574

我们以data-juicer对OCEMOTION_train1128.csv文件清洗为案例；

### csv数据清洗

首先使用reformat_csv_nan_value.py对csv将可能包含空值（Nan）重新格式化为多个JSONL文件。

```sh
python tools/preprocess/reformat_jsonl_nan_value.py           \
    --src_dir  dataset/OCEMOTION_train1128.csv
    --target_dir output/OCEMOTION_train1128.jsonl
    --num_proc  4
```

### 数据分析

```
# 适用于从源码安装
python tools/analyze_data.py --config configs/demo/config_OCEMOTION.yaml

# 使用命令行工具
dj-analyze --config configs/demo/config_OCEMOTION.yaml

# 你也可以使用"自动"模式来避免写一个新的数据菜谱。它会使用全部可产出统计信息的 Filter 来分析
# 你的数据集的一小部分（如1000条样本，可通过 `auto_num` 参数指定）
dj-analyze --auto --dataset_path xx.jsonl [--auto_num 1000]
```

analyzer.yaml是自己创建的，其实就是process.yaml文件

```
# Process config example including:
#   - all global arguments
#   - all ops and their arguments

# global parameters
project_name: 'OCEMOTION-process'        
dataset_path: 'output/OCEMOTION_train1128.jsonl'  
                                                           
export_path: 'output/OCEMOTION_train1128_refine.jsonl'              

np: 4                                                       # number of subprocess to process your dataset
text_keys: 'response'                                         

# process schedule: a list of several process operators with their arguments
process:
  - clean_email_mapper:           # remove emails from text.
  - clean_html_mapper:           # remove html formats form text.
......
```

### 数据筛选和清洗

```sh
dj-process  --config config_OCEMOTION.yaml
```

