# Megatron-LM

web:https://github.com/NVIDIA/Megatron-LM.git

衍生：

https://github.com/alibaba/Pai-Megatron-Patch



## 涉及到的名词

**NCCL**： NVIDIA 的分布式通信库，用于高效的 GPU 间通信。Megatron-LM 还可以使用 **MPI**（消息传递接口）进行多节点通信。

**APEX**（A PyTorch Extension）是 NVIDIA 开发的一个高性能 PyTorch 扩展库，主要用于加速混合精度训练和分布式训练。Megatron-LM 依赖 APEX 的 `amp` 模块实现 FP16/FP32 混合精度训练，减少显存占用并提升计算速度。

**Megatron-LM** 专注于大规模 Transformer 语言模型（如 GPT、BERT）的分布式训练

**Megatron-Core** 作为 **模块化底层库**，为 Megatron-LM 和其他框架提供基础组件





## Megatron-LM 环境配置

### Doker安装  借助彗星云和矩池云

这种设置环境的方法是使用NGC提供的NVIDIA PyTorch容器，它包含所有必需的安装。(亲测慢)  

教程：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html

#### 云平台地址

彗星云：https://portal.huixingyun.com/

矩池云：https://matpool.com/

#### 镜像准备、上传并加载

最好将 nvidia_24.01py3的镜像在本机docker pull下来，然后save，使用FTP工具无损上传到服务器，我出现过U盘传输导致镜像损坏的问题

使用 docker load -i 加载镜像

```sh
(base) root@gpu-109b9874432b6964ac398-1-wfeqmlinxugb:~# ls
data  my-app.tar  nvidia_24.01py3.tar  public  snap
(base) root@gpu-109b9874432b6964ac398-1-wfeqmlinxugb:~# docker load -i nvidia_24.01py3.tar 
a1360aae5271: Loading layer [==================================================>]  29.55MB/29.55MB
aad8fbafa1e9: Loading layer [==================================================>]  132.6MB/132.6MB
5f70bf18a086: Loading layer [==================================================>]      32B/32B
a1d5f3691cf2: Loading layer [==================================================>]  155.2MB/155.2MB
408dbdcce3b1: Loading layer [==================================================>]   15.1kB/15.1kB
d922555f193d: Loading layer [==================================================>]  2.963GB/2.963GB
7fa243a59b17: Loading layer [==================================================>]  11.53kB/11.53kB
967fc1edf1f6: Loading layer [==================================================>]     180B/180B
61f4c0fdd178: Loading layer [==================================================>]  5.719kB/5.719kB
5f70bf18a086: Loading layer [==================================================>]      32B/32B
c272ef64f0f5: Loading layer [==================================================>]  31.37MB/31.37MB
5ca5b67aba1f: Loading layer [==================================================>]   86.9MB/86.9MB
94fb17ce577d: Loading layer [==================================================>]     502B/502B
7b64623cadb7: Loading layer [==================================================>]  476.4MB/476.4MB
3fc4574df740: Loading layer [==================================================>]  64.56MB/64.56MB
bd7fb5f6570e: Loading layer [==================================================>]  8.538MB/8.538MB
0277b7e58fa6: Loading layer [==================================================>]  27.26MB/27.26MB
c3f04d280695: Loading layer [==================================================>]  12.75MB/12.75MB
c630ef1b7b61: Loading layer [==================================================>]     115B/115B
565ddb635a79: Loading layer [==================================================>]  499.3MB/499.3MB
7d6bea944b14: Loading layer [==================================================>]  642.8MB/642.8MB
10880933e715: Loading layer [==================================================>]  54.88MB/54.88MB
2646186d8524: Loading layer [==================================================>]   8.43kB/8.43kB
acd428170ab7: Loading layer [==================================================>]  69.24MB/69.24MB
37dc326733c1: Loading layer [==================================================>]     475B/475B
213a49c78600: Loading layer [==================================================>]  15.75MB/15.75MB
818822d7503f: Loading layer [==================================================>]  2.287GB/2.287GB
7b43594788a2: Loading layer [==================================================>]  320.4kB/320.4kB
bc1d39032f97: Loading layer [==================================================>]    112MB/112MB
8ae576ba2e94: Loading layer [==================================================>]  302.9MB/302.9MB
1c09125232fc: Loading layer [==================================================>]  28.74MB/28.74MB
5c8f5b34bde3: Loading layer [==================================================>]  1.117kB/1.117kB
09e62598d36f: Loading layer [==================================================>]  468.8MB/468.8MB
1f5a19781014: Loading layer [==================================================>]  1.344MB/1.344MB
0ea8f48013ca: Loading layer [==================================================>]   61.7kB/61.7kB
2cc93dd0720d: Loading layer [==================================================>]  1.832GB/1.832GB
ccb64ab424a8: Loading layer [==================================================>]  1.124kB/1.124kB
cee9721d2e5d: Loading layer [==================================================>]  1.115kB/1.115kB
85f4236315a0: Loading layer [==================================================>]  6.765MB/6.765MB
7e02e3f124ab: Loading layer [==================================================>]  30.13MB/30.13MB
a867e76178d1: Loading layer [==================================================>]  37.03MB/37.03MB
fa2a013a2a9a: Loading layer [==================================================>]  1.567MB/1.567MB
47585a11ae4f: Loading layer [==================================================>]  6.428MB/6.428MB
98a833959cd6: Loading layer [==================================================>]  60.49MB/60.49MB
ee8be49aec6b: Loading layer [==================================================>]  14.56MB/14.56MB
6ab1e25b0f53: Loading layer [==================================================>]  121.8MB/121.8MB
023d19c1b22b: Loading layer [==================================================>]  109.4MB/109.4MB
c9daaf1b703d: Loading layer [==================================================>]   13.9kB/13.9kB
6cbef91399c5: Loading layer [==================================================>]     503B/503B
Loaded image: nvcr.io/nvidia/pytorch:24.01-py3
(base) root@gpu-109b9874432b6964ac398-1-wfeqmlinxugb:~# ls
```

#### 运行镜像

```sh
(base) root@gpu-109b9874432b6964ac398-1-wfeqmlinxugb:~# docker images
REPOSITORY               TAG         IMAGE ID       CREATED         SIZE
nvcr.io/nvidia/pytorch   24.01-py3   8470a68886ff   14 months ago   22GB
# 启动容器（必须加 --gpus all） 才能使用GPU
(base) root@gpu-109b9874432b6964ac398-1-wfeqmlinxugb:~# docker run -itd \
  --gpus all \
  --name megatron \
  -v /your/data:/workspace \
  nvcr.io/nvidia/pytorch:24.01-py3 \
  bash

root@09f1ed6b7124:/workspace# 
root@09f1ed6b7124:/workspace# ls
NVIDIA_Deep_Learning_Container_License.pdf  README.md  docker-examples  examples  tutorials
```

安装一些python包

```sh
pip install einops datasets nltk sentencepiece pybind11 ninja transformers  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

克隆项目

```sh
# /root/workspace
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM
pip install -r requirements_mlm.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
# 这个文件是为实际训练任务准备的，特别是 Masked Language Model（MLM）和其他语言模型的预训练任务。
# 包含了完整的依赖项，包括深度学习框架（如 PyTorch）、分布式训练工具（如 NCCL）、以及其他必要的库。
```

然后参考GPT的例子

##### 数据操作

- 下载数据

```sh
创建好 data tokenizers
进入到data的上一级

mkdir data
# 外网上的数据需要用本地上传
docker cp /root/data_json.zip f855f4c0edaf:/root/data
```

- 上传数据配置文件

指定注意的是如果使用的是Docker需要进行文件挂载才可以让宿主机和主机使用

例如：**挂载多个文件   别用docker run 用一次会创建不同的机器**

```
# 语法：docker cp <宿主机文件路径> <容器ID>:<容器内目标路径>
docker cp /宿主机/文件.txt 5faeb112b5ba:/容器内/目标路径/

mkdir -p ./tokenizers/GPT2_tokenizer/
docker cp /root/GPT/vocab.json f855f4c0edaf:/root/tokenizers/GPT2_tokenizer
docker cp /root/GPT/merges.txt f855f4c0edaf:/root/tokenizers/GPT2_tokenizer
```

- 数据清洗

```
# GPT模型
export BASE_PATH=/root
python ${BASE_PATH}/Megatron-LM/tools/preprocess_data.py \
       --input ${BASE_PATH}/data/merged_json.json \
       --output-prefix ${BASE_PATH}/data/merged_json \
       --tokenizer-type GPT2BPETokenizer \
       --vocab-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json \
       --merge-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt \
       --workers 16 \
       --append-eod
```

##### 模型操作

在/worspca/下面创建pretrain_gpt.sh，其内容为：

```sh
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1 # 每节点GPU数量
MASTER_ADDR=localhost # 主节点地址（单机默认为localhost）
MASTER_PORT=6000 # 主节点端口
NUM_NODES=1 # 节点总数（单机训练）
NODE_RANK=0 # 当前节点rank 
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # 全局GPU数量

# Paths
BASE_PATH=/root
SRC_PATH=/root/Megatron-LM/pretrain_gpt.py

# Log
LOG_NAME=GPT2_pretrain_WS${WORLD_SIZE}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

# Data 
DATA_PATH=${BASE_PATH}/data/merged_json_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# Save Model
CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
mkdir -p ${CHECKPOINT_PATH}

# Tokenizer files
VOCAB_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json
MERGE_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每节点进程数
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点IP
    --master_port $MASTER_PORT  # 主节点端口
)

GPT_MODEL_ARGS=(
    --num-layers 12   # 减少层数
    --hidden-size 768   # 减少隐藏层维度
    --num-attention-heads 12  # 减少注意力头数量
    --seq-length 512   # 减少序列长度
    --max-position-embeddings 512    
    --attention-backend auto # 自动选择最优注意力实现
)

TRAINING_ARGS=(
    --micro-batch-size 1  # 每个GPU的batch大小
    --global-batch-size 16  # 全局batch大小（需梯度累积）
    --train-iters 1000  # 总训练步数
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 # 混合精度训练
    --lr 0.0001  # 学习率
    --lr-decay-style cosine  # 余弦退火
    --min-lr 0.00001
    --lr-warmup-fraction .01 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1  # 张量并行度1
    --pipeline-model-parallel-size 1  # 流水线并行度1
)

DATA_ARGS=(
    --data-path $DATA_PATH   # 预处理数据路径
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1  # 训练/验证/测试集比例
    # --mock-data       # 启用模拟数据模式（可选）
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 100 
    --eval-interval 50 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 5
)

# ENGINE_ARGS=(
#     --transformer_impl transformer_engine 
# )

# 通过torchrun启动分布式训练
torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    # ${ENGINE_ARGS[@]}
```



### 手动安装 AutoDL GPU服务器各种环境配置

#### 科学上网

```sh
source /etc/network_turbo
```

#### python环境准备

```sh
pip install einops datasets nltk sentencepiece pybind11 ninja transformers 
```

#### 其它库安装

手动安装最新版本的PyTorch、CUDA、NCCL以及NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start)发行版，以及nltk库。 

- cuda home设置

```sh
CUDA_HOME=/usr/local/cuda-12.4  
PATH=$CUDA_HOME/bin:$PATH
LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

- 安装 Apex（支持 FP16）

```sh
git clone https://github.com/NVIDIA/apex.git
cd apex
#pip install -v --disable-pip-version-check --no-cache-dir \
#    . --global-option="--cpp_ext" --global-option="--cuda_ext"
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

添加 `--no-build-isolation` 可避免因环境隔离导致的依赖问题。

- 克隆项目

```sh
# /root/workspace
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM
pip install -r requirements_mlm.txt
# 这个文件是为实际训练任务准备的，特别是 Masked Language Model（MLM）和其他语言模型的预训练任务。
# 包含了完整的依赖项，包括深度学习框架（如 PyTorch）、分布式训练工具（如 NCCL）、以及其他必要的库。
```

- 安装TransformerEngine

```sh
# 重新克隆并初始化子模块
git clone --recurse-submodules https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
pip install -v .
```

#### 查看本机各种版本

- NCCL

```
>>> import torch
>>> print(torch.cuda.nccl.version())  # 输出 NCCL 版本号
(2, 21, 5)
```

- CUDA

```
root@autodl-container-b027488635-f5d4ceeb:~# nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

- PyTorch

```
>>> import torch
>>> torch.__version__
'2.5.1+cu124'
```



## 训练

无论是哪种训练方式 都需要在workspace的目录下创建：pretrain_gpt_*.sh

### 单机单卡

`pretrain_gpt.sh` 在 /examples 下有提供，`pretrain_llama.py` 可以直接 copy `pretrain_gpt.py`
实例脚本如下：

```sh
#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2 # 每节点GPU数量
# Change for multinode config
MASTER_ADDR=localhost # 主节点地址（单机默认为localhost）
MASTER_PORT=6000 # 主节点端口
NUM_NODES=1 # 节点总数（单机训练）
NODE_RANK=0 # 当前节点rank 
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # 全局GPU数量

CHECKPOINT_PATH=$1 #<Specify path>  # 检查点保存路径（如：/checkpoints）
TENSORBOARD_LOGS_PATH=$2 #<Specify path>  # TensorBoard日志路径（如：/logs）
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json # 词汇表文件
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt # BPE合并文件
DATA_PATH=$5 #<Specify path and file prefix>_text_document # 预处理后的数据路径前缀

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每节点进程数
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点IP
    --master_port $MASTER_PORT  # 主节点端口
)

GPT_MODEL_ARGS=(
    --num-layers 96   # 96层Transformer
    --hidden-size 12288   # 12K隐藏层维度
    --num-attention-heads 96  # 96个注意力头
    --seq-length 2048   #  序列长度2048 
    --max-position-embeddings 2048    
    --attention-backend auto # Can use (flash/fused/unfused/local) # 自动选择最优注意力实现
)

TRAINING_ARGS=(
    --micro-batch-size 1  #  每个GPU的batch大小
    --global-batch-size 1536  # 全局batch大小（需梯度累积）
    --rampup-batch-size 16 16 5859375 
    --train-iters 500000  # 总训练步数
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 # 混合精度训练
    --lr 6.0e-5  # 学习率
    --lr-decay-style cosine  # 余弦退火
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8  # 张量并行度8
	--pipeline-model-parallel-size 16  # 流水线并行度16
)

# 张量并行：单个层在8个GPU上拆分
# 流水线并行：不同层组在16个GPU上拆分
# 总并行度：8×16=128，至少需要128个GPU才能运行

DATA_ARGS=(
    --data-path $DATA_PATH   # 预处理数据路径
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1  # 训练/验证/测试集比例
)
# 需预先通过preprocess_data.py处理成二进制格式


EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# 通过torchrun启动分布式训练
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

```

### 单机多卡训练

```sh
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2 # 每节点GPU数量
MASTER_ADDR=localhost # 主节点地址（单机默认为localhost）
MASTER_PORT=6000 # 主节点端口
NUM_NODES=1 # 节点总数（单机训练）
NODE_RANK=0 # 当前节点rank 
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # 全局GPU数量

# Paths
BASE_PATH=/root/workspace
SRC_PATH=/root/workspace/Megatron-LM/pretrain_gpt.py

# Log
LOG_NAME=GPT2_pretrain_WS${WORLD_SIZE}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

# Data 
DATA_PATH=${BASE_PATH}/data/oscar-en-10k-meg-gpt_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# Save Model
CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
mkdir -p ${CHECKPOINT_PATH}

# Tokenizer files
VOCAB_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json
MERGE_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每节点进程数
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点IP
    --master_port $MASTER_PORT  # 主节点端口
)

GPT_MODEL_ARGS=(
    --num-layers 12   # 减少层数
    --hidden-size 768   # 减少隐藏层维度
    --num-attention-heads 12  # 减少注意力头数量
    --seq-length 512   # 减少序列长度
    --max-position-embeddings 512    
    --attention-backend auto # 自动选择最优注意力实现
)

TRAINING_ARGS=(
    --micro-batch-size 2  # 每个GPU的batch大小
    --global-batch-size 16  # 全局batch大小（需梯度累积）
    --train-iters 1000  # 总训练步数
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 # 混合精度训练
    --lr 0.0001  # 学习率
    --lr-decay-style cosine  # 余弦退火
    --min-lr 0.00001
    --lr-warmup-fraction .01 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2  # 张量并行度1
    --pipeline-model-parallel-size 1  # 流水线并行度1
)

DATA_ARGS=(
    --data-path $DATA_PATH   # 预处理数据路径
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1  # 训练/验证/测试集比例
    # --mock-data       # 启用模拟数据模式（可选）
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 100 
    --eval-interval 50 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 5
)

# ENGINE_ARGS=(
#     --transformer_impl transformer_engine 
# )

# 通过torchrun启动分布式训练
torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    # ${ENGINE_ARGS[@]}
```

使用下面的指令查看：

```sh
root@autodl-container-7bda11a2fa-7c0b7ce3:~# nvidia-smi
Mon Apr 14 16:00:06 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100S-PCIE-32GB          On  |   00000000:00:07.0 Off |                  Off |
| N/A   39C    P0             77W /  250W |    3090MiB /  32768MiB |     99%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100S-PCIE-32GB          On  |   00000000:00:0B.0 Off |                  Off |
| N/A   39C    P0             98W /  250W |    3302MiB /  32768MiB |     39%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```



```bash
[rank1]:[W414 16:54:27.214824770 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
[rank0]:[W414 16:54:27.221573952 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```



### 多机多卡

代码再GPT训练中



## GPT2案例复现

### 准备工作

```sh
mkdir -p /root/workspace/data

# GPT
cd /root/workspace && mkdir -p ./tokenizers/GPT2_tokenizer && cd tokenizers/GPT2_tokenizer
```

### 数据准备

很小的数据集

```sh
cd /root/workspace

python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'
```

### 配置文件准备

最好下载到本地在上传 wget下载有问题

#### 词汇文件 vocab.json

https://huggingface.co/openai-community/gpt2/blob/main/vocab.json

这个json文件是乱码，需要自己去转换一下

#### 合并表 merges.txt

 https://huggingface.co/openai-community/gpt2/blob/main/merges.txt

### 预处理数据

### 程序

tools/preprocess_data.py  使用以下命令将数据 tokenize、shuffle 进行训练:

```sh
# GPT模型
export BASE_PATH=/root/workspace
python ${BASE_PATH}/Megatron-LM/tools/preprocess_data.py \
       --input ${BASE_PATH}/data/oscar-en-10k.jsonl \
       --output-prefix ${BASE_PATH}/data/oscar-en-10k-meg-gpt\
       --tokenizer-type GPT2BPETokenizer \
       --vocab-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json \
       --merge-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt \
       --workers 16 \
       --append-eod
```

workers 选项指的是预处理中使用的线程数量。

/workspace/data/oscar-en-10k-meg-gpt_text_document.bin 和 /workspace/data/oscar-en-10k-meg-gpt_text_document.idx 两个文件用于训练。

执行记录：

```sh
Opening /root/workspace/data/oscar-en-10k.jsonl
Processed 1000 documents (771.7006209955063 docs/s, 9.578794600474247 MB/s).
Processed 2000 documents (979.0478177907919 docs/s, 12.25653908805528 MB/s).
Processed 3000 documents (798.2964312416949 docs/s, 10.340221307869045 MB/s).
Processed 4000 documents (884.3645392571202 docs/s, 11.393466466163384 MB/s).
Processed 5000 documents (911.7690637658194 docs/s, 11.99618223833907 MB/s).
Processed 6000 documents (966.5748133792005 docs/s, 12.674512009005962 MB/s).
Processed 7000 documents (950.0273244917109 docs/s, 12.384330325390382 MB/s).
Processed 8000 documents (998.9877169917394 docs/s, 12.930725645938885 MB/s).
Processed 9000 documents (1014.2027529850262 docs/s, 13.086762671403317 MB/s).
Processed 10000 documents (1043.1646875000622 docs/s, 13.515629523410261 MB/s).



预处理成功完成后会在/root/workspace/data下面生成：
-rw-r--r-- 1 root root 259M Apr 13 17:01 /root/workspace/data/oscar-en-10k-meg-gpt_text_document.bin
-rw-r--r-- 1 root root 196K Apr 13 17:01 /root/workspace/data/oscar-en-10k-meg-gpt_text_document.idx


root@autodl-container-178642aab5-f4be30c1:~/workspace# cd data
root@autodl-container-178642aab5-f4be30c1:~/workspace/data# ls
oscar-en-10k-meg-gpt_text_document.bin  oscar-en-10k-meg-gpt_text_document.idx  oscar-en-10k.jsonl
```

### 训练

在/root/worspca/下面创建pretrain_gpt.sh，其内容为：

```sh
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1 # 每节点GPU数量
MASTER_ADDR=localhost # 主节点地址（单机默认为localhost）
MASTER_PORT=6000 # 主节点端口
NUM_NODES=1 # 节点总数（单机训练）
NODE_RANK=0 # 当前节点rank 
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # 全局GPU数量

# Paths
BASE_PATH=/root/workspace
SRC_PATH=/root/workspace/Megatron-LM/pretrain_gpt.py

# Log
LOG_NAME=GPT2_pretrain_WS${WORLD_SIZE}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

# Data 
DATA_PATH=${BASE_PATH}/data/oscar-en-10k-meg-gpt_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# Save Model
CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
mkdir -p ${CHECKPOINT_PATH}

# Tokenizer files
VOCAB_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json
MERGE_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每节点进程数
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点IP
    --master_port $MASTER_PORT  # 主节点端口
)

GPT_MODEL_ARGS=(
    --num-layers 12   # 减少层数
    --hidden-size 768   # 减少隐藏层维度
    --num-attention-heads 12  # 减少注意力头数量
    --seq-length 512   # 减少序列长度
    --max-position-embeddings 512    
    --attention-backend auto # 自动选择最优注意力实现
)

TRAINING_ARGS=(
    --micro-batch-size 1  # 每个GPU的batch大小
    --global-batch-size 16  # 全局batch大小（需梯度累积）
    --train-iters 1000  # 总训练步数
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 # 混合精度训练
    --lr 0.0001  # 学习率
    --lr-decay-style cosine  # 余弦退火
    --min-lr 0.00001
    --lr-warmup-fraction .01 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1  # 张量并行度1
    --pipeline-model-parallel-size 1  # 流水线并行度1
)

DATA_ARGS=(
    --data-path $DATA_PATH   # 预处理数据路径
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1  # 训练/验证/测试集比例
    # --mock-data       # 启用模拟数据模式（可选）
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 100 
    --eval-interval 50 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 5
)

# ENGINE_ARGS=(
#     --transformer_impl transformer_engine 
# )

# 通过torchrun启动分布式训练
torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    # ${ENGINE_ARGS[@]}
```

控制台打印：

```
root@autodl-container-800b4fb0ba-b2cf04d8:~/workspace# bash pretrain_gpt.sh 
using world size: 1, data-parallel size: 1, context-parallel size: 1, hierarchical context-parallel sizes: Nonetensor-model-parallel size: 1, encoder-tensor-model-parallel size: 0, pipeline-model-parallel size: 1, encoder-pipeline-model-parallel size: 0
Number of virtual stages per pipeline stage: None
WARNING: Setting args.check_for_nan_in_loss_and_grad to False since dynamic loss scaling is being used
using torch.float16 for parameters ...
------------------------ arguments ------------------------
  account_for_embedding_in_pipeline_split ......... False
  account_for_loss_in_pipeline_split .............. False
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.95
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. True
  add_position_embedding .......................... True
  add_qkv_bias .................................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  align_grad_reduce ............................... True
  align_param_gather .............................. False
  app_tag_run_name ................................ None
  app_tag_run_version ............................. 0.0.0
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  apply_rope_fusion ............................... False
  async_save ...................................... None
  async_tensor_model_parallel_allreduce ........... True
  attention_backend ............................... AttnBackend.auto
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  auto_detect_ckpt_format ......................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  bias_swiglu_fusion .............................. True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  calc_ft_timeouts ................................ False
  calculate_per_token_loss ........................ False
  check_for_large_grads ........................... False
  check_for_nan_in_loss_and_grad .................. False
  check_for_spiky_loss ............................ False
  check_weight_hash_across_dp_replicas_interval ... None
  ckpt_assume_constant_structure .................. False
  ckpt_convert_format ............................. None
  ckpt_convert_save ............................... None
  ckpt_convert_update_legacy_dist_opt_format ...... False
  ckpt_format ..................................... torch_dist
  ckpt_fully_parallel_load ........................ False
  ckpt_fully_parallel_save ........................ True
  ckpt_fully_parallel_save_deprecated ............. False
  ckpt_step ....................................... None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  clone_scatter_output_in_embedding ............... True
  config_logger_dir ............................... 
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  context_parallel_size ........................... 1
  cp_comm_type .................................... ['p2p']
  create_attention_mask_in_dataloader ............. True
  cross_entropy_fusion_impl ....................... native
  cross_entropy_loss_fusion ....................... False
  cuda_graph_scope ................................ full
  cuda_graph_warmup_steps ......................... 3
  data_args_path .................................. None
  data_cache_path ................................. None
  data_parallel_random_init ....................... False
  data_parallel_sharding_strategy ................. no_shard
  data_parallel_size .............................. 1
  data_path ....................................... ['/root/workspace/data/oscar-en-10k-meg-gpt_text_document']
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. single
  ddp_average_in_collective ....................... False
  ddp_bucket_size ................................. None
  ddp_num_buckets ................................. None
  ddp_pad_buckets_for_high_nccl_busbw ............. False
  decoder_first_pipeline_num_layers ............... None
  decoder_last_pipeline_num_layers ................ None
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  decoupled_lr .................................... None
  decoupled_min_lr ................................ None
  decrease_batch_size_if_needed ................... False
  defer_embedding_wgrad_compute ................... False
  deprecated_use_mcore_models ..................... False
  deterministic_mode .............................. False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  disable_bf16_reduced_precision_matmul ........... False
  disable_straggler_on_startup .................... False
  dist_ckpt_format_deprecated ..................... None
  dist_ckpt_strictness ............................ assume_ok_unexpected
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  enable_cuda_graph ............................... False
  enable_ft_package ............................... False
  enable_gloo_process_groups ...................... True
  enable_msc ...................................... True
  enable_one_logger ............................... True
  encoder_num_layers .............................. 12
  encoder_pipeline_model_parallel_size ............ 0
  encoder_seq_length .............................. 512
  encoder_tensor_model_parallel_size .............. 0
  end_weight_decay ................................ 0.1
  eod_mask_loss ................................... False
  error_injection_rate ............................ 0
  error_injection_type ............................ transient_error
  eval_interval ................................... 50
  eval_iters ...................................... 5
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  exp_avg_dtype ................................... torch.float32
  exp_avg_sq_dtype ................................ torch.float32
  expert_model_parallel_size ...................... 1
  expert_tensor_parallel_size ..................... 1
  external_cuda_graph ............................. False
  ffn_hidden_size ................................. 3072
  finetune ........................................ False
  first_last_layers_bf16 .......................... False
  flash_decode .................................... False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8 ............................................. None
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_param_gather ................................ False
  fp8_recipe ...................................... delayed
  fp8_wgrad ....................................... True
  global_batch_size ............................... 16
  grad_reduce_in_bf16 ............................. False
  gradient_accumulation_fusion .................... True
  gradient_reduce_div_fusion ...................... True
  group_query_attention ........................... False
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 768
  hierarchical_context_parallel_sizes ............. None
  hybrid_attention_ratio .......................... 0.0
  hybrid_mlp_ratio ................................ 0.0
  hybrid_override_pattern ......................... None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... -1
  inference_dynamic_batching ...................... False
  inference_dynamic_batching_buffer_guaranteed_fraction  0.2
  inference_dynamic_batching_buffer_overflow_factor  None
  inference_dynamic_batching_buffer_size_gb ....... 40.0
  inference_dynamic_batching_max_requests_override  None
  inference_dynamic_batching_max_tokens_override .. None
  inference_max_batch_size ........................ 8
  inference_max_seq_length ........................ 2560
  inference_rng_tracker ........................... False
  init_method_std ................................. 0.006
  init_method_xavier_uniform ...................... False
  init_model_with_meta_device ..................... False
  initial_loss_scale .............................. 4294967296
  is_hybrid_model ................................. False
  iter_per_epoch .................................. 1250
  iterations_to_skip .............................. []
  keep_fp8_transpose_cache_when_using_custom_fsdp . False
  kv_channels ..................................... 64
  kv_lora_rank .................................... 32
  lazy_mpu_init ................................... None
  load ............................................ /root/workspace/checkpoint/GPT2_pretrain_WS1
  local_rank ...................................... 0
  log_interval .................................... 10
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_progress .................................... False
  log_straggler ................................... False
  log_throughput .................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  logging_level ................................... None
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. 0.01
  lr_warmup_init .................................. 0.0
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  lr_wsd_decay_iters .............................. None
  lr_wsd_decay_samples ............................ None
  lr_wsd_decay_style .............................. exponential
  main_grads_dtype ................................ torch.float32
  main_params_dtype ............................... torch.float32
  make_vocab_size_divisible_by .................... 128
  mamba_head_dim .................................. 64
  mamba_num_groups ................................ 8
  mamba_state_dim ................................. 128
  manual_gc ....................................... False
  manual_gc_eval .................................. True
  manual_gc_interval .............................. 0
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  max_tokens_to_oom ............................... 12000
  memory_snapshot_path ............................ snapshot.pickle
  merge_file ...................................... /root/workspace/tokenizers/GPT2_tokenizer/merges.txt
  micro_batch_size ................................ 1
  microbatch_group_size_per_vp_stage .............. None
  mid_level_dataset_surplus ....................... 0.005
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mlp_chunks_for_prefill .......................... 1
  mmap_bin_files .................................. True
  mock_data ....................................... False
  moe_aux_loss_coeff .............................. 0.0
  moe_enable_deepep ............................... False
  moe_expert_capacity_factor ...................... None
  moe_extended_tp ................................. False
  moe_ffn_hidden_size ............................. None
  moe_grouped_gemm ................................ False
  moe_input_jitter_eps ............................ None
  moe_layer_freq .................................. 1
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ False
  moe_per_layer_logging ........................... False
  moe_permute_fusion .............................. False
  moe_router_bias_update_rate ..................... 0.001
  moe_router_dtype ................................ None
  moe_router_enable_expert_bias ................... False
  moe_router_group_topk ........................... None
  moe_router_load_balancing_type .................. aux_loss
  moe_router_num_groups ........................... None
  moe_router_pre_softmax .......................... False
  moe_router_score_function ....................... softmax
  moe_router_topk ................................. 2
  moe_router_topk_scaling_factor .................. None
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_token_dispatcher_type ....................... allgather
  moe_token_drop_policy ........................... probs
  moe_use_legacy_grouped_gemm ..................... False
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None
  mrope_section ................................... None
  mscale .......................................... 1.0
  mscale_all_dim .................................. 1.0
  mtp_loss_scaling_factor ......................... 0.1
  mtp_num_layers .................................. None
  multi_latent_attention .......................... False
  nccl_communicator_config_path ................... None
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  non_persistent_ckpt_type ........................ None
  non_persistent_global_ckpt_dir .................. None
  non_persistent_local_ckpt_algo .................. fully_parallel
  non_persistent_local_ckpt_dir ................... None
  non_persistent_save_interval .................... None
  norm_epsilon .................................... 1e-05
  normalization ................................... LayerNorm
  num_attention_heads ............................. 12
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_dataset_builder_threads ..................... 1
  num_distributed_optimizer_instances ............. 1
  num_experts ..................................... None
  num_layers ...................................... 12
  num_layers_at_end_in_bf16 ....................... 1
  num_layers_at_start_in_bf16 ..................... 1
  num_layers_per_virtual_pipeline_stage ........... None
  num_query_groups ................................ 1
  num_virtual_stages_per_pipeline_rank ............ None
  num_workers ..................................... 2
  object_storage_cache_path ....................... None
  one_logger_async ................................ False
  one_logger_project .............................. megatron-lm
  one_logger_run_name ............................. None
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  optimizer_cpu_offload ........................... False
  optimizer_offload_fraction ...................... 1.0
  output_bert_embeddings .......................... False
  overlap_cpu_optimizer_d2h_h2d ................... False
  overlap_grad_reduce ............................. False
  overlap_p2p_comm ................................ False
  overlap_p2p_comm_warmup_flush ................... False
  overlap_param_gather ............................ False
  overlap_param_gather_with_optimizer_step ........ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  per_split_data_args_path ........................ None
  perform_initialization .......................... True
  pin_cpu_grads ................................... True
  pin_cpu_params .................................. True
  pipeline_model_parallel_comm_backend ............ None
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  position_embedding_type ......................... learned_absolute
  pretrained_checkpoint ........................... None
  profile ......................................... False
  profile_ranks ................................... [0]
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  q_lora_rank ..................................... None
  qk_head_dim ..................................... 128
  qk_layernorm .................................... False
  qk_pos_emb_head_dim ............................. 64
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... None
  recompute_method ................................ None
  recompute_modules ............................... None
  recompute_num_layers ............................ None
  record_memory_history ........................... False
  relative_attention_max_distance ................. 128
  relative_attention_num_buckets .................. 32
  replication ..................................... False
  replication_factor .............................. 2
  replication_jump ................................ None
  rerun_mode ...................................... disabled
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  result_rejected_tracker_filename ................ None
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_attention_gate ............................ 1
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_project_dir ............................... None
  retro_verify_neighbor_count ..................... True
  rope_scaling_factor ............................. 8.0
  rotary_base ..................................... 10000
  rotary_interleaved .............................. False
  rotary_percent .................................. 1.0
  rotary_scaling_factor ........................... 1.0
  rotary_seq_len_interpolation_factor ............. None
  run_workload_inspector_server ................... False
  sample_rate ..................................... 1.0
  save ............................................ /root/workspace/checkpoint/GPT2_pretrain_WS1
  save_interval ................................... 100
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  skipped_train_samples ........................... 0
  spec ............................................ None
  split ........................................... 949,50,1
  squared_relu .................................... False
  start_weight_decay .............................. 0.1
  straggler_ctrlr_port ............................ 65535
  straggler_minmax_count .......................... 1
  suggested_communication_unit_size ............... None
  swiglu .......................................... False
  swin_backbone_type .............................. tiny
  te_rng_tracker .................................. False
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  test_mode ....................................... False
  tiktoken_num_special_tokens ..................... 1000
  tiktoken_pattern ................................ None
  tiktoken_special_tokens ......................... None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. GPT2BPETokenizer
  tp_comm_bootstrap_backend ....................... nccl
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_ag .............................. True
  tp_comm_overlap_cfg ............................. None
  tp_comm_overlap_rs .............................. True
  tp_comm_overlap_rs_dgrad ........................ False
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True
  train_data_path ................................. None
  train_iters ..................................... 1000
  train_samples ................................... None
  train_sync_interval ............................. None
  transformer_impl ................................ transformer_engine
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_cpu_initialization .......................... None
  use_custom_fsdp ................................. False
  use_dist_ckpt ................................... True
  use_dist_ckpt_deprecated ........................ False
  use_distributed_optimizer ....................... False
  use_flash_attn .................................. False
  use_legacy_models ............................... False
  use_mp_args_from_checkpoint_args ................ False
  use_one_sent_docs ............................... False
  use_persistent_ckpt_worker ...................... False
  use_precision_aware_optimizer ................... False
  use_pytorch_profiler ............................ False
  use_ring_exchange_p2p ........................... False
  use_rope_scaling ................................ False
  use_rotary_position_embeddings .................. False
  use_tokenizer_model_from_checkpoint_args ........ True
  use_torch_fsdp2 ................................. False
  use_torch_optimizer_for_cpu_offload ............. False
  use_tp_pp_dp_mapping ............................ False
  v_head_dim ...................................... 128
  valid_data_path ................................. None
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /root/workspace/tokenizers/GPT2_tokenizer/vocab.json
  vocab_size ...................................... None
  wandb_exp_name .................................. 
  wandb_project ................................... 
  wandb_save_dir .................................. 
  weight_decay .................................... 0.1
  weight_decay_incr_style ......................... constant
  wgrad_deferral_limit ............................ 0
  world_size ...................................... 1
  yaml_cfg ........................................ None
-------------------- end of arguments ---------------------
INFO:megatron.core.num_microbatches_calculator:setting number of microbatches to constant 16
> building GPT2BPETokenizer tokenizer ...
 > padded vocab (size: 50257) with 47 dummy tokens (new size: 50304)
WARNING: one_logger package is required to enable e2e metrics tracking. please go to https://confluence.nvidia.com/display/MLWFO/Package+Repositories for details to install it
WARNING:megatron.core.rerun_state_machine:RerunStateMachine initialized in mode RerunMode.DISABLED
> initializing torch distributed ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: Entering directory '/root/workspace/Megatron-LM/megatron/core/datasets'
make: Nothing to be done for 'default'.
make: Leaving directory '/root/workspace/Megatron-LM/megatron/core/datasets'
>>> done with dataset index builder. Compilation time: 0.051 seconds
> compiling and loading fused kernels ...
[rank0]:[W413 23:40:58.027177023 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
>>> done with compiling and loading fused kernels. Compilation time: 0.264 seconds
time to initialize megatron (seconds): 1.961
[after megatron is initialized] datetime: 2025-04-13 23:41:00 
building GPT model ...
/root/workspace/Megatron-LM/megatron/core/models/gpt/gpt_layer_specs.py:91: UserWarning: The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated and will be removed soon. Please update your code accordingly.
  warnings.warn(
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 124082688
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=False, overlap_grad_reduce=False, overlap_param_gather=False, align_param_gather=False, use_distributed_optimizer=False, num_distributed_optimizer_instances=1, check_for_nan_in_grad=False, check_for_large_grads=False, bucket_size=None, pad_buckets_for_high_nccl_busbw=False, average_in_collective=False, fp8_param_gather=False, use_custom_fsdp=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, suggested_communication_unit_size=None, preserve_fp32_weights=True, keep_fp8_transpose_cache_when_using_custom_fsdp=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
Params for bucket 1 (124082688 elements, 124082688 padded size):
        module.decoder.final_layernorm.bias
        module.decoder.layers.9.self_attention.linear_qkv.bias
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.8.mlp.linear_fc1.weight
        module.decoder.layers.7.self_attention.linear_qkv.weight
        module.decoder.layers.6.self_attention.linear_qkv.weight
        module.decoder.layers.11.mlp.linear_fc2.weight
        module.decoder.layers.5.mlp.linear_fc1.weight
        module.decoder.layers.4.mlp.linear_fc1.weight
        module.decoder.layers.3.mlp.linear_fc1.weight
        module.decoder.layers.2.mlp.linear_fc1.weight
        module.decoder.layers.10.mlp.linear_fc1.weight
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.self_attention.linear_proj.bias
        module.decoder.layers.6.self_attention.linear_proj.bias
        module.decoder.layers.11.mlp.linear_fc1.bias
        module.decoder.layers.8.self_attention.linear_qkv.weight
        module.decoder.layers.0.mlp.linear_fc1.weight
        module.decoder.layers.9.self_attention.linear_proj.bias
        module.decoder.layers.5.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_qkv.weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_proj.bias
        module.decoder.layers.10.self_attention.linear_qkv.bias
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.10.self_attention.linear_proj.weight
        module.decoder.layers.8.self_attention.linear_proj.bias
        module.decoder.layers.7.mlp.linear_fc2.weight
        module.decoder.layers.6.mlp.linear_fc2.bias
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.0.self_attention.linear_qkv.bias
        module.decoder.layers.5.self_attention.linear_proj.bias
        module.decoder.layers.4.self_attention.linear_proj.bias
        module.decoder.layers.3.self_attention.linear_proj.bias
        module.decoder.layers.1.self_attention.linear_qkv.bias
        module.decoder.layers.0.mlp.linear_fc2.bias
        module.decoder.layers.10.self_attention.linear_proj.bias
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.8.mlp.linear_fc2.bias
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.10.mlp.linear_fc2.weight
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.5.mlp.linear_fc2.bias
        module.decoder.layers.4.mlp.linear_fc2.bias
        module.decoder.layers.3.mlp.linear_fc2.bias
        module.decoder.layers.2.mlp.linear_fc2.bias
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.1.mlp.linear_fc1.weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.10.self_attention.linear_qkv.weight
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.1.mlp.linear_fc2.bias
        module.decoder.layers.10.mlp.linear_fc2.bias
        module.decoder.layers.7.mlp.linear_fc1.bias
        module.decoder.layers.6.mlp.linear_fc1.bias
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.2.self_attention.linear_proj.bias
        module.decoder.layers.11.mlp.linear_fc2.bias
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.9.mlp.linear_fc2.bias
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.9.mlp.linear_fc1.bias
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.8.mlp.linear_fc1.bias
        module.decoder.layers.7.self_attention.linear_qkv.bias
        module.decoder.layers.6.self_attention.linear_qkv.bias
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.11.self_attention.linear_proj.bias
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.mlp.linear_fc2.bias
        module.decoder.layers.6.mlp.linear_fc2.weight
        module.decoder.layers.5.mlp.linear_fc1.bias
        module.decoder.layers.4.mlp.linear_fc1.bias
        module.decoder.layers.3.mlp.linear_fc1.bias
        module.decoder.layers.2.mlp.linear_fc1.bias
        module.decoder.layers.7.self_attention.linear_proj.weight
        module.decoder.layers.6.self_attention.linear_proj.weight
        module.decoder.layers.1.mlp.linear_fc1.bias
        module.embedding.position_embeddings.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_bias
        module.decoder.layers.11.mlp.linear_fc1.weight
        module.decoder.layers.8.self_attention.linear_qkv.bias
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.0.mlp.linear_fc1.bias
        module.decoder.layers.9.mlp.linear_fc2.weight
        module.decoder.layers.9.mlp.linear_fc1.weight
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc2.weight
        module.decoder.layers.5.self_attention.linear_qkv.bias
        module.decoder.layers.4.self_attention.linear_qkv.bias
        module.decoder.layers.3.self_attention.linear_qkv.bias
        module.decoder.layers.2.self_attention.linear_qkv.bias
        module.decoder.layers.5.mlp.linear_fc2.weight
        module.decoder.layers.7.mlp.linear_fc1.weight
        module.decoder.layers.11.self_attention.linear_qkv.weight
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.9.self_attention.linear_qkv.weight
        module.decoder.layers.8.self_attention.linear_proj.weight
        module.decoder.layers.6.mlp.linear_fc1.weight
        module.decoder.layers.4.mlp.linear_fc2.weight
        module.decoder.layers.3.mlp.linear_fc2.weight
        module.decoder.layers.2.mlp.linear_fc2.weight
        module.decoder.layers.10.mlp.linear_fc1.bias
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_bias
        module.decoder.layers.5.self_attention.linear_proj.weight
        module.decoder.layers.4.self_attention.linear_proj.weight
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.embedding.word_embeddings.weight
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.1.mlp.linear_fc2.weight
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.self_attention.linear_proj.bias
        module.decoder.layers.11.self_attention.linear_qkv.bias
        module.decoder.layers.11.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.linear_fc2.weight
INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=0.0001, min_lr=1e-05, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.1, fp16=True, bf16=False, params_dtype=torch.float16, use_precision_aware_optimizer=False, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=False, overlap_param_gather_with_optimizer_step=False, optimizer_cpu_offload=False, optimizer_offload_fraction=1.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=True, timers=<megatron.core.timers.Timers object at 0x7f144559ef00>, config_logger_dir='')
INFO:megatron.core.optimizer_param_scheduler:> learning rate decay style: cosine
WARNING: could not find the metadata file /root/workspace/checkpoint/GPT2_pretrain_WS1/latest_checkpointed_iteration.txt
    will not load any checkpoints and will start from random
(min, max) time across ranks (ms):
    load-checkpoint ................................: (0.18, 0.18)
[after model, optimizer, and learning rate scheduler are built] datetime: 2025-04-13 23:41:00 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      16000
    validation: 1680
    test:       80
INFO:megatron.core.datasets.blended_megatron_dataset_config:Let split_matrix = [(0, 0.949), (0.949, 0.999), (0.999, 1.0)]
> building train, validation, and test datasets for GPT ...
INFO:megatron.core.datasets.blended_megatron_dataset_builder:Building GPTDataset splits with sizes=(16000, 1680, 80) and config=GPTDatasetConfig(random_seed=1234, sequence_length=512, blend=(['/root/workspace/data/oscar-en-10k-meg-gpt_text_document'], None), blend_per_split=None, split='949,50,1', split_matrix=[(0, 0.949), (0.949, 0.999), (0.999, 1.0)], num_dataset_builder_threads=1, path_to_cache=None, mmap_bin_files=True, mock=False, tokenizer=<megatron.training.tokenizer.tokenizer._GPT2BPETokenizer object at 0x7f144532dc40>, mid_level_dataset_surplus=0.005, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask=True, drop_last_partial_validation_sequence=True, add_extra_token_to_sequence=True, object_storage_cache_path=None)
INFO:megatron.core.datasets.indexed_dataset:Load the _IndexReader from /root/workspace/data/oscar-en-10k-meg-gpt_text_document.idx
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence lengths
INFO:megatron.core.datasets.indexed_dataset:    Extract the sequence pointers
INFO:megatron.core.datasets.indexed_dataset:    Extract the document indices
INFO:megatron.core.datasets.indexed_dataset:> total number of sequences: 10000
INFO:megatron.core.datasets.indexed_dataset:> total number of documents: 10000
INFO:megatron.core.datasets.gpt_dataset:Build and save the GPTDataset train indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 55074
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 1
INFO:megatron.core.datasets.gpt_dataset:Build and save the GPTDataset valid indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 3090
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 1
INFO:megatron.core.datasets.gpt_dataset:Build and save the GPTDataset test indices
INFO:megatron.core.datasets.gpt_dataset:> total number of samples: 127
INFO:megatron.core.datasets.gpt_dataset:> total number of epochs: 2
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2025-04-13 23:41:00 
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (123.15, 123.15)
    train/valid/test-data-iterators-setup ..........: (114.59, 114.59)
training ...
Setting rerun_state_machine.current_iteration to 0...
[before the start of training step] datetime: 2025-04-13 23:41:00 
 [2025-04-13 23:41:12] iteration       10/    1000 | consumed samples:          160 | elapsed time per iteration (ms): 1147.6 | learning rate: 0.000000E+00 | global batch size:    16 | loss scale: 8388608.0 | number of skipped iterations:  10 | number of nan iterations:   0 |
Number of parameters in transformer block in billions:  0.08
Number of parameters in embedding layers in billions: 0.04
Total number of parameters in billions: 0.12
Number of parameters in most loaded shard in billions: 0.1236
Theoretical memory footprints: weight and optimizer=2121.85 MB
[Rank 0] (after 10 iterations) memory (MB) | allocated: 1464.33154296875 | max allocated: 1464.3408203125 | reserved: 1576.0 | max reserved: 1576.0
 [2025-04-13 23:41:20] iteration       20/    1000 | consumed samples:          320 | elapsed time per iteration (ms): 820.7 | learning rate: 4.000000E-05 | global batch size:    16 | lm loss: 1.072453E+01 | loss scale: 131072.0 | grad norm: 3.874 | num zeros: 71.0 | number of skipped iterations:   6 | number of nan iterations:   0 |
 [2025-04-13 23:41:28] iteration       30/    1000 | consumed samples:          480 | elapsed time per iteration (ms): 821.8 | learning rate: 9.000000E-05 | global batch size:    16 | lm loss: 1.038769E+01 | loss scale: 4096.0 | number of skipped iterations:   5 | number of nan iterations:   0 |
 [2025-04-13 23:41:37] iteration       40/    1000 | consumed samples:          640 | elapsed time per iteration (ms): 849.8 | learning rate: 9.998165E-05 | global batch size:    16 | lm loss: 9.992319E+00 | loss scale: 4096.0 | grad norm: 2.357 | num zeros: 881.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:41:46] iteration       50/    1000 | consumed samples:          800 | elapsed time per iteration (ms): 904.5 | learning rate: 9.991823E-05 | global batch size:    16 | lm loss: 9.236888E+00 | loss scale: 4096.0 | grad norm: 2.285 | num zeros: 1894.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (3274.73, 3274.73)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
----------------------------------------------------------------------------------------------
 validation loss at iteration 50 | lm loss value: 8.889540E+00 | lm loss PPL: 7.255679E+03 | 
----------------------------------------------------------------------------------------------
 [2025-04-13 23:41:57] iteration       60/    1000 | consumed samples:          960 | elapsed time per iteration (ms): 837.6 | learning rate: 9.980959E-05 | global batch size:    16 | lm loss: 8.638345E+00 | loss scale: 4096.0 | grad norm: 2.221 | num zeros: 949.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:42:06] iteration       70/    1000 | consumed samples:         1120 | elapsed time per iteration (ms): 840.3 | learning rate: 9.965582E-05 | global batch size:    16 | lm loss: 8.194965E+00 | loss scale: 4096.0 | grad norm: 1.803 | num zeros: 1850.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:42:14] iteration       80/    1000 | consumed samples:         1280 | elapsed time per iteration (ms): 849.5 | learning rate: 9.945709E-05 | global batch size:    16 | lm loss: 7.928842E+00 | loss scale: 4096.0 | grad norm: 1.466 | num zeros: 960.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:42:23] iteration       90/    1000 | consumed samples:         1440 | elapsed time per iteration (ms): 832.5 | learning rate: 9.921360E-05 | global batch size:    16 | lm loss: 7.656038E+00 | loss scale: 4096.0 | grad norm: 1.123 | num zeros: 524.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:42:31] iteration      100/    1000 | consumed samples:         1600 | elapsed time per iteration (ms): 869.1 | learning rate: 9.892558E-05 | global batch size:    16 | lm loss: 7.514787E+00 | loss scale: 4096.0 | grad norm: 1.082 | num zeros: 298.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (2052.52, 2052.52)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
-----------------------------------------------------------------------------------------------
 validation loss at iteration 100 | lm loss value: 7.398865E+00 | lm loss PPL: 1.634128E+03 | 
-----------------------------------------------------------------------------------------------
saving checkpoint at iteration     100 to /root/workspace/checkpoint/GPT2_pretrain_WS1 in torch_dist format
/root/workspace/Megatron-LM/megatron/core/transformer/transformer_layer.py:376: UserWarning: TransformerLayer._get_layer_offset is deprecated.Please use get_transformer_layer_offset instead.
  warnings.warn(
  successfully saved checkpoint from iteration     100 to /root/workspace/checkpoint/GPT2_pretrain_WS1 [ t 1/1, p 1/1 ]
(min, max) time across ranks (ms):
    save-checkpoint ................................: (3413.14, 3413.14)
 [2025-04-13 23:42:47] iteration      110/    1000 | consumed samples:         1760 | elapsed time per iteration (ms): 1069.8 | learning rate: 9.859334E-05 | global batch size:    16 | lm loss: 7.424495E+00 | loss scale: 4096.0 | grad norm: 1.209 | num zeros: 227.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:42:55] iteration      120/    1000 | consumed samples:         1920 | elapsed time per iteration (ms): 774.3 | learning rate: 9.821720E-05 | global batch size:    16 | lm loss: 7.426577E+00 | loss scale: 4096.0 | grad norm: 1.023 | num zeros: 309.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:43:03] iteration      130/    1000 | consumed samples:         2080 | elapsed time per iteration (ms): 792.7 | learning rate: 9.779754E-05 | global batch size:    16 | lm loss: 7.312885E+00 | loss scale: 4096.0 | grad norm: 2.013 | num zeros: 227.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:43:12] iteration      140/    1000 | consumed samples:         2240 | elapsed time per iteration (ms): 860.2 | learning rate: 9.733479E-05 | global batch size:    16 | lm loss: 7.211839E+00 | loss scale: 4096.0 | grad norm: 0.925 | num zeros: 262.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:43:25] iteration      150/    1000 | consumed samples:         2400 | elapsed time per iteration (ms): 1296.2 | learning rate: 9.682942E-05 | global batch size:    16 | lm loss: 7.157401E+00 | loss scale: 4096.0 | grad norm: 0.996 | num zeros: 234.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (2054.12, 2054.12)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
-----------------------------------------------------------------------------------------------
 validation loss at iteration 150 | lm loss value: 7.269834E+00 | lm loss PPL: 1.436311E+03 | 
-----------------------------------------------------------------------------------------------
 [2025-04-13 23:43:35] iteration      160/    1000 | consumed samples:         2560 | elapsed time per iteration (ms): 822.7 | learning rate: 9.628192E-05 | global batch size:    16 | lm loss: 7.211591E+00 | loss scale: 4096.0 | grad norm: 2.791 | num zeros: 178.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:43:43] iteration      170/    1000 | consumed samples:         2720 | elapsed time per iteration (ms): 839.6 | learning rate: 9.569286E-05 | global batch size:    16 | lm loss: 7.135952E+00 | loss scale: 4096.0 | grad norm: 2.125 | num zeros: 209.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:43:52] iteration      180/    1000 | consumed samples:         2880 | elapsed time per iteration (ms): 867.7 | learning rate: 9.506283E-05 | global batch size:    16 | lm loss: 7.097198E+00 | loss scale: 4096.0 | grad norm: 1.798 | num zeros: 152.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:44:05] iteration      190/    1000 | consumed samples:         3040 | elapsed time per iteration (ms): 1265.2 | learning rate: 9.439245E-05 | global batch size:    16 | lm loss: 7.026772E+00 | loss scale: 4096.0 | grad norm: 1.328 | num zeros: 313.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:44:14] iteration      200/    1000 | consumed samples:         3200 | elapsed time per iteration (ms): 901.4 | learning rate: 9.368241E-05 | global batch size:    16 | lm loss: 7.102211E+00 | loss scale: 4096.0 | grad norm: 1.228 | num zeros: 379.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (1945.91, 1945.91)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
-----------------------------------------------------------------------------------------------
 validation loss at iteration 200 | lm loss value: 7.212789E+00 | lm loss PPL: 1.356671E+03 | 
-----------------------------------------------------------------------------------------------
saving checkpoint at iteration     200 to /root/workspace/checkpoint/GPT2_pretrain_WS1 in torch_dist format
  successfully saved checkpoint from iteration     200 to /root/workspace/checkpoint/GPT2_pretrain_WS1 [ t 1/1, p 1/1 ]
(min, max) time across ranks (ms):
    save-checkpoint ................................: (1627.79, 1627.79)
 [2025-04-13 23:44:26] iteration      210/    1000 | consumed samples:         3360 | elapsed time per iteration (ms): 832.4 | learning rate: 9.293342E-05 | global batch size:    16 | lm loss: 6.966640E+00 | loss scale: 4096.0 | grad norm: 1.306 | num zeros: 252.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:44:35] iteration      220/    1000 | consumed samples:         3520 | elapsed time per iteration (ms): 970.0 | learning rate: 9.214623E-05 | global batch size:    16 | lm loss: 6.980472E+00 | loss scale: 4096.0 | grad norm: 0.963 | num zeros: 331.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:44:45] iteration      230/    1000 | consumed samples:         3680 | elapsed time per iteration (ms): 992.3 | learning rate: 9.132164E-05 | global batch size:    16 | lm loss: 7.032919E+00 | loss scale: 4096.0 | grad norm: 1.731 | num zeros: 366.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:44:54] iteration      240/    1000 | consumed samples:         3840 | elapsed time per iteration (ms): 842.4 | learning rate: 9.046048E-05 | global batch size:    16 | lm loss: 6.931380E+00 | loss scale: 4096.0 | grad norm: 1.234 | num zeros: 322.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2025-04-13 23:45:02] iteration      250/    1000 | consumed samples:         4000 | elapsed time per iteration (ms): 824.3 | learning rate: 8.956362E-05 | global batch size:    16 | lm loss: 6.923337E+00 | loss scale: 4096.0 | grad norm: 1.479 | num zeros: 430.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (1731.06, 1731.06)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
-----------------------------------------------------------------------------------------------
 validation loss at iteration 250 | lm loss value: 7.083469E+00 | lm loss PPL: 1.192097E+03 | 
-----------------------------------------------------------------------------------------------
 [2025-04-13 23:45:13] iteration      260/    1000 | consumed samples:         4160 | elapsed time per iteration (ms): 973.8 | learning rate: 8.863195E-05 | global batch size:    16 | lm loss: 6.939181E+00 | loss scale: 4096.0 | grad norm: 0.991 | num zeros: 305.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```





## LLAMA模型案例复现（搁置）

llama需要向脸书申请才可以下载，暂且搁置

```sh
mkdir -p /root/workspace/data

# LLAMA
mkdir -p /root/workspace/tokenizers/Llama2Tokenizer && cd /root/workspace/tokenizers/Llama2Tokenizer
wget http://stonezhang.cn/tokenizers/Llama2Tokenizer/tokenizer.model
```

```
# llama模型
export BASE_PATH=/root/workspace
python /root/workspace/Megatron-LM/tools/preprocess_data.py \
       --input ${BASE_PATH}/data/oscar-en-10k.jsonl \
       --output-prefix ${BASE_PATH}/data/oscar-en-10k-meg-llama\
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${BASE_PATH}/tokenizers/Llama2Tokenizer/tokenizer.model \
       --workers 16 \
       --append-eod
```



## 手写AI案例复现

### 准备工作

```sh
mkdir -p /root/workspace/data

# GPT
cd /root/workspace && mkdir -p ./tokenizers/GPT2_tokenizer && cd tokenizers/GPT2_tokenizer
```

### 数据处理

将0.csv", "10000.csv", "20000.csv", "30000.csv", "40000.csv", "50000.csv 合并成一个文件共60000条，并且每一行都是一个json处理

```sh
import pandas as pd
import json
from tqdm import tqdm

# 1. 定义文件路径和输出路径
input_dir = "D:/gx/Desktop/Megatron-LM/shouxieai/data/"
output_file = "D:/gx/Desktop/Megatron-LM/shouxieai/data/merged_json.json"

# 2. 获取所有CSV文件
csv_files = ["0.csv", "10000.csv", "20000.csv", "30000.csv", "40000.csv", "50000.csv"]

# 3. 合并文件并转换为JSON格式
with open(output_file, 'w', encoding='utf-8') as f_out:
    for file in tqdm(csv_files):
        # 读取每个CSV文件
        df = pd.read_csv("D:/gx/Desktop/Megatron-LM/shouxieai/data/" + file, encoding="utf-8")
        
        # 确保列名为'content'
        if 'content' not in df.columns:
            df.columns = ['content']
            
        # 逐行写入JSON
        for text in df['content']:
            json.dump({"text": str(text)}, f_out, ensure_ascii=False)
            f_out.write('\n')

print(f"合并完成！输出文件：{output_file}")

# 修复：读取文件时指定UTF-8编码
print(f"总行数：{sum(1 for line in open(output_file, encoding='utf-8'))}")
```

### 配置文件准备

最好下载到本地在上传 wget下载有问题

#### 词汇文件 vocab.json

https://huggingface.co/openai-community/gpt2/blob/main/vocab.json

#### 合并表 merges.txt

 https://huggingface.co/openai-community/gpt2/blob/main/merges.txt

### 预处理数据

### 程序

tools/preprocess_data.py  使用以下命令将数据 tokenize、shuffle 进行训练:

```sh
# GPT模型
export BASE_PATH=/root/workspace
python ${BASE_PATH}/Megatron-LM/tools/preprocess_data.py \
       --input ${BASE_PATH}/data_json/merged_json.json \
       --output-prefix ${BASE_PATH}/data_json/merged_json\
       --tokenizer-type GPT2BPETokenizer \
       --vocab-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json \
       --merge-file ${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt \
       --workers 16 \
       --append-eod
```

workers 选项指的是预处理中使用的线程数量。

/workspace/data/oscar-en-10k-meg-gpt_text_document.bin 和 /workspace/data/oscar-en-10k-meg-gpt_text_document.idx 两个文件用于训练。

执行记录：

```sh
Opening /root/workspace/data_json/merged_json.json
Time to startup: 0.2639496326446533
Processed 1000 documents (1518.8785014238906 docs/s, 1.3638337567826735 MB/s).
Processed 2000 documents (1698.8835733429896 docs/s, 1.517577147986175 MB/s).
Processed 3000 documents (1715.040422243915 docs/s, 1.5571803627793144 MB/s).
Processed 4000 documents (1677.6716053861594 docs/s, 1.5569560027111191 MB/s).
Processed 5000 documents (1683.963851923418 docs/s, 1.5630442574972314 MB/s).
。。。。。
Processed 58000 documents (1824.6533322186533 docs/s, 1.7109933463169662 MB/s).
Processed 59000 documents (1816.6102070023255 docs/s, 1.7034835416896161 MB/s).
Processed 60000 documents (1830.4762061595845 docs/s, 1.715579540268096 MB/s).
```

处理完后会生成如下文件

```sh
root@autodl-container-7bda11a2fa-7c0b7ce3:~/workspace/data_json# pwd
/root/workspace/data_json
root@autodl-container-7bda11a2fa-7c0b7ce3:~/workspace/data_json# ls
merged_json.json  merged_json_text_document.bin  merged_json_text_document.idx
```

### 训练

参照老师给出的模型配置修改

```sh
#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2 # 每节点GPU数量
MASTER_ADDR=localhost # 主节点地址（单机默认为localhost）
MASTER_PORT=6000 # 主节点端口
NUM_NODES=1 # 节点总数（单机训练）
NODE_RANK=0 # 当前节点rank 
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))  # 全局GPU数量

# Paths
BASE_PATH=/root/workspace
SRC_PATH=/root/workspace/Megatron-LM/pretrain_gpt.py

# Log
LOG_NAME=GPT2_pretrain_WS${WORLD_SIZE}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

# Data 
DATA_PATH=${BASE_PATH}/data_json/merged_json_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

# Save Model
CHECKPOINT_PATH=${BASE_PATH}/checkpoint/${LOG_NAME}
mkdir -p ${CHECKPOINT_PATH}

# Tokenizer files
VOCAB_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/vocab.json
MERGE_FILE=${BASE_PATH}/tokenizers/GPT2_tokenizer/merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE  # 每节点进程数
    --nnodes $NUM_NODES  # 总节点数
    --master_addr $MASTER_ADDR  # 主节点IP
    --master_port $MASTER_PORT  # 主节点端口
)

GPT_MODEL_ARGS=(
    --num-layers 12   # 减少层数
    --hidden-size 768   # 减少隐藏层维度
    --num-attention-heads 12  # 减少注意力头数量
    --seq-length 512   # 减少序列长度
    --max-position-embeddings 512    
    --attention-backend auto # 自动选择最优注意力实现
)

TRAINING_ARGS=(
    --micro-batch-size 2  # 每个GPU的batch大小
    --global-batch-size 16  # 全局batch大小（需梯度累积）
    --train-iters 1000  # 总训练步数
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16 # 混合精度训练
    --lr 0.0001  # 学习率
    --lr-decay-style cosine  # 余弦退火
    --min-lr 0.00001
    --lr-warmup-fraction .01 
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2  # 张量并行度1
    --pipeline-model-parallel-size 1  # 流水线并行度1
)

DATA_ARGS=(
    --data-path $DATA_PATH   # 预处理数据路径
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1  # 训练/验证/测试集比例
    # --mock-data       # 启用模拟数据模式（可选）
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 100 
    --eval-interval 50 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 5
)

# ENGINE_ARGS=(
#     --transformer_impl transformer_engine 
# )

# 通过torchrun启动分布式训练
torchrun ${DISTRIBUTED_ARGS[@]} ${SRC_PATH} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    # ${ENGINE_ARGS[@]}
```

结果：

```
------------------------------------------------------------------------------------------------
 validation loss at iteration 2450 | lm loss value: 2.850238E+00 | lm loss PPL: 1.729190E+01 | 
------------------------------------------------------------------------------------------------
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
 [2025-04-14 18:16:33] iteration     2500/    2500 | consumed samples:        40000 | elapsed time per iteration (ms): 600.8 | learning rate: 1.001599E-05 | global batch size:    16 | lm loss: 2.895528E+00 | loss scale: 16384.0 | grad norm: 1.101 | num zeros: 2210.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
(min, max) time across ranks (ms):
    evaluate .......................................: (1376.49, 1377.00)
------------------------------------------------------------------------------------------------
 validation loss at iteration 2500 | lm loss value: 2.866578E+00 | lm loss PPL: 1.757677E+01 | 
------------------------------------------------------------------------------------------------
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
[after training is done] datetime: 2025-04-14 18:16:35 
saving checkpoint at iteration    2500 to /root/workspace/checkpoint/GPT2_pretrain_WS2 in torch_dist format
  successfully saved checkpoint from iteration    2500 to /root/workspace/checkpoint/GPT2_pretrain_WS2 [ t 1/2, p 1/1 ]
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
Evaluating on 80 samples
Evaluating iter 1/5
Evaluating iter 2/5
Evaluating iter 3/5
Evaluating iter 4/5
Evaluating iter 5/5
(min, max) time across ranks (ms):
    evaluate .......................................: (1381.65, 1382.51)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
------------------------------------------------------------------------------------------------------------------
 validation loss at iteration 2500 on validation set | lm loss value: 2.925279E+00 | lm loss PPL: 1.863942E+01 | 
------------------------------------------------------------------------------------------------------------------
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
Evaluating on 80 samples
Evaluating iter 1/5
Evaluating iter 2/5
Evaluating iter 3/5
Evaluating iter 4/5
Evaluating iter 5/5
(min, max) time across ranks (ms):
    evaluate .......................................: (1362.15, 1362.90)
WARNING:megatron.core.rerun_state_machine:Setting RerunStateMachine mode RerunMode.DISABLED
------------------------------------------------------------------------------------------------------------
 validation loss at iteration 2500 on test set | lm loss value: 2.867433E+00 | lm loss PPL: 1.759179E+01 | 
------------------------------------------------------------------------------------------------------------
```





### 模型准备

bert-base-chinese代码：

https://huggingface.co/google-bert/bert-base-chinese

```sh
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
```

模型保存地址：/root/.cache/huggingface/hub/models--bert-base-chinese

```sh
cp /root/.cache/huggingface/hub/models--bert-base-chine
se/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/* /root/workspace/model 
```



## 参考资料

https://huggingface.co/blog/zh/megatron-training