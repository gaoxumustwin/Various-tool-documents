# deepspeed

**DeepSpeed**，这是微软开发的一个**深度学习优化库**，专门用于**大规模模型训练**和**推理加速**。它通过一系列先进技术（如 ZeRO 优化、混合精度训练、梯度检查点等）显著提升训练效率，并降低显存占用。

DeepSpeed底层封装的还是torchrun，更接近底层；威震天也是封装的torchrun；

主节点启动的时候，会自动去调用其余的节点启动；

多机的环境是可以不一样的，包括显卡，cuda版本等等

多机是使用局域网进行通信的

- 数据并行：分发数据，每张卡算完后再把结果汇总
- 模型并行：切割数据

官方地址仓库：https://github.com/deepspeedai/DeepSpeed

官网：https://www.deepspeed.ai/  （先看Getting Started）



## 名词

ip地址：机器的地址

端口：机器留给应用程序的，应用程序的通信入口（门牌号），端口可变，也可写死，端口号的上限：65536

rank: 显卡的绝对下标，排名



## 环境配置

### 局域网通信

1. 前置条件是：局域网组网，或者广域网的访问速度非常快才可以
2. 要确保多机可以互相访问，相互ping通
3. 必须确保直连才可以进行数据传输，所以需要配置免密的ssh

### 准备工作

我们使用deepspeed是基于llamafactory的，这是因为我们不是从去写一个模型或写一个训练代码，而是基于llamafactory，llamafactory本身是支持deepspeed

**多机都要准备llamafactory，并配置好环境，并且llamafactory的版本和环境必须是一致的（例如：python版本、虚拟环境名称），而且环境的目录都必须是一样的（例如：conda的环境和目录地址）**

torchrun是启动多个进程，去跑多卡，分配不同的进程，llamafactory是对torchrun的封装，已经写好了数据分发逻辑，GPU的工作逻辑，llamafactory训练的所有入口都指向的是run_exp()函数，其多机多卡是调用torchrun去启动的；

deepspeed底层封装的还是torchrun，所以我们可以使用deepseed去替换llamafactory的torchrun，所以llamafactory支持使用deepspeed去进行多机多卡的训练  （llamafactory官网有写）

### 安装

**deepspeed只支持Linux**

安装：

```sh
pip install deepspeed==0.15.0  
```

安装的更多细节：https://www.deepspeed.ai/tutorials/advanced-install/

- DeepSpeed还支持AMD 、Intel Xeon CPU、Intel Data Center Max Series XPU、Intel Gaudi HPU、华为Ascend NPU等，请参阅[加速器设置指南](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/)



## 文件介绍

使用deepspeed的配置文件，LLaMA-Factory已经帮我们写好，在LLaMA-Factory的LLaMA-Factory\LLaMA-Factory\examples\deepspeed下面



## 文件修改

4.22跑deepspeed还正常，4.23llamafactory就更新了，多了一个read_args函数,我使用下面的方式运行模型：

```sh
deepspeed /root/workspace/LLaMA-Factory/src/train.py train /root/workspace/LLaMA-Factory/examples/train_lora/qwen2_7B_lora_sft.yaml
```

由于最后一个参数是配置文件，所以我需要修改一些read_args函数，如下所示：

```sh
def read_args(args: Optional[Union[dict[str, Any], list[str]]] = None) -> Union[dict[str, Any], list[str]]:
    r"""Get arguments from the command line or a config file."""
    if args is not None:
        return args

    # if len(sys.argv) > 2 and (sys.argv[-1].endswith(".yaml") or sys.argv[-1].endswith(".yml")):
    #     override_config = None  # 命令行覆盖参数（如 key=value）
    #     dict_config = yaml.safe_load(Path(sys.argv[-1]).absolute().read_text())  # 加载 YAML 文件
    #     return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    # elif sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
    #     override_config = OmegaConf.from_cli(sys.argv[2:])
    #     dict_config = yaml.safe_load(Path(sys.argv[1]).absolute().read_text())
    #     return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    # elif sys.argv[1].endswith(".json"):
    #     override_config = OmegaConf.from_cli(sys.argv[2:])
    #     dict_config = json.loads(Path(sys.argv[1]).absolute().read_text())
    #     return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    # else:
    #     return sys.argv[1:]
     # 处理 DeepSpeed 注入的 --local_rank 等参数
    argv = [arg for arg in sys.argv[1:] if not arg.startswith("--")]  # 过滤掉 -- 开头的参数
    
    if len(argv) >= 1 and (argv[-1].endswith(".yaml") or argv[-1].endswith(".yml")):
        # 情况1: 最后一个参数是 YAML 文件 (适用于 deepspeed 调用方式)
        override_config = None  # 这里不处理命令行覆盖，由后续流程处理
        dict_config = yaml.safe_load(Path(argv[-1]).absolute().read_text())
        # 修复：当override_config为None时直接返回dict_config
        return dict_config if override_config is None else OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(argv) >= 1 and (argv[0].endswith(".yaml") or argv[0].endswith(".yml")):
        # 情况2: 第一个参数是 YAML 文件 (原始调用方式)
        override_config = OmegaConf.from_cli(argv[1:])
        dict_config = yaml.safe_load(Path(argv[0]).absolute().read_text())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    elif len(argv) >= 1 and argv[0].endswith(".json"):
        # 情况3: JSON 配置文件
        override_config = OmegaConf.from_cli(argv[1:])
        dict_config = json.loads(Path(argv[0]).absolute().read_text())
        return OmegaConf.to_container(OmegaConf.merge(dict_config, override_config))
    else:
        # 情况4: 直接返回参数
        return argv if argv else sys.argv[1:]
```





## 单机训练

要先确保LLaMA-Factory能够跑起来

LLaMA-Factory的deepspeed入口很多，带run_exp()即可，我们以train.py为例：

```sh
deepspeed /path/to/LLaMA-Factory/src/train.py train /path/to/my_train.yaml

deepspeed /root/workspace/LLaMA-Factory/src/train.py train /root/workspace/LLaMA-Factory/examples/train_lora/qwen2_7B_lora_sft.yaml

# train是模式因为LLaMA-Factory不同的模式走不同的代码
# my_train.yaml是训练的各种配置参数  可以不去指定 但是不指定的话需要讲各种参数指定非常多
# 	里面的路径必须是绝对路径，这是因为多机多卡，不同的机器路径是不一样的，相对路径的话会找不到，指定死绝对路径可以直接找到  注意:多机多卡多进程一定要绝对路径   
#   里面要加上save_only_model: true  deepspeed 的配置相关
#   需要在dataset处 加上dataset_dir：指定文件的绝对路径地址前缀LLaMA-Factory的data文件夹的地址  让其能找到LLaMA-Factory的datainfo文件
```

LLaMA-Factory给什么参数，deepspeed就给什么参数就可以运行起来

deepspeed的其它入口：

- LLaMA-Factory\src\llamafactory\launcher.py
- LLaMA-Factory\src\llamafactory\cli.py



## 多机训练 

单机训练能跑通是多机训练可以跑通的前提

讲前面my_train.yaml中的deepspeed的配置文件路径给注释掉，其实还是可以跑（代码中有默认处理，会指定默认的配置），我们手动的去指定deepspeed的配置文件目前会发生问题，在外面设定有问题

```sh
deepspeed /path/to/LLaMA-Factory/src/train.py train /path/to/my_train.yaml 
```

在进行多机多卡之前需要讲单机训练的流程在多机上都复现跑通才可以，**GPU服务器上可以尝试保存一个自己的镜像**

### 配置SSH免密连接 （要熟悉）

#### 生成SSH密钥对

在本地机器上生成密钥对（公钥和私钥）：

```bash
ssh-keygen 
```

*.pub是公钥

#### 将公钥上传到远程服务器

**如果想让A服务器连接到B服务器，就需要在B的服务器上加上A的公钥**

需要将公钥放到~/.ssh中的authorized_keys文件中

登录B服务器，编辑

```bash
mkdir -p ~/.ssh  # 确保目录存在
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
# "粘贴公钥内容" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys  # 关键权限设置
```

自己SSH连接自己也需要给密码，所以一般也把自己的公钥也放到authorized_keys中  **也需要配置上，因为DeepSpeed有时候会用到自己连接自己**

#### 测试免密登录

```bash
ssh A的name@A的IP
```

如果配置正确，将直接登录，无需密码。

#### 优化连接

目前的连接是你去连接别人需要加用户名@，如果不想加可以再配置一个config(`~/.ssh/config`中添加别名)

```bash
cd ~/.ssh
vim config
```

设置以下的内容可以简化连接命令：

```sh
Host remote_server_ip
	User username
```

此时可以直接通过ssh A的IP就可以连接上服务器

### 多机启动文件

需要创建一个hsotfile

```
worker-1的ip地址 slots-4  # 机器1有多少张卡
worker-2的ip地址 slots-4  # 机器2有多少张卡
```

此时可以使用下面的命令执行：

```sh
deepspeed --hostfile=/path/to/hsotfile /path/to/LLaMA-Factory/src/train.py train /path/to/my_train.yaml 
```

- 如果出现问题的时候可以先把worker-1给注释掉，能跑再注释worker-2，再尝试，如果两台都可以那就是通信的问题

- 如果出现了TCP、addr、Scoket等都是网络连接到问题

- 可能出现端口占用问题

  查看端口占用

```
lsof -i:端口号
```

​	如果没有任何的输出就不是被端口占用

- **多机需要指定网卡，服务器有很多很多的网卡，所以我们需要指定使用哪个网卡**

```
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0 # 指定一个网卡
```

需要存储到**家目录下面的.deepspeed_env文件中，并且所有的多机都要去设置** 

- 注意pytorch版本要一致   出现过torch=2.4 和 torch=2.6的时候运行不了的问题
- **注意：配置文件如果A机器修改了，则B机器必须也修改为一模一样的**





出现过单机多卡4张3090训练qwen1.5B，四张的显存都只占用30%的情况

当时参数如下：

```
per_device_train_batch_size: 1 # 每个 GPU 的训练批次大小（batch size）。
gradient_accumulation_steps: 2 # 梯度累积步数。 当 per_device_train_batch_size 较小时，可以通过梯度累积模拟更大的全局批次大小。
# 全局批次大小 = per_device_train_batch_size × gradient_accumulation_steps × GPU 数量。
learning_rate: 1.0e-5 # 学习率。
num_train_epochs: 3.0 #  训练的总轮数（epochs）。
lr_scheduler_type: cosine # 学习率调度器类型。 使用余弦退火（cosine annealing）策略调整学习率，可以帮助模型在训练后期更平稳地收敛。
warmup_ratio: 0.1 # 学习率预热比例。
bf16: true # 启用 bfloat16 精度训练。	
ddp_timeout: 180000000 # 分布式数据并行（DDP）超时时间（单位：秒）。
resume_from_checkpoint: null # 是否从检查点恢复训练。
```

通过调整 `train_batch_size`=16 和 `gradient_accumulation_steps` =2 来增加每个 GPU 的计算负载，此时四张GPU跑满，并且训练速度由原来的3个epoch 30分钟到3个epoch 31秒



