# SWIFT 框架

## 简介

**SWIFT** （Scalable and Well-Integrated Fine-Tuning）是一个用于大语言模型（LLM）高效微调的开源框架，轻量级训练框架，主要由阿里巴巴达摩院开发。它的目标是为用户提供一个简单、灵活且高效的工具，用于对预训练模型进行微调，使其适应特定任务或领域。**SWIFT** 与 ModelScope（魔搭）深度集成,能快速微调多种预训练模型，并且方便下载和管理模型。

github仓库：https://github.com/modelscope/ms-swift

帮助文档：https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html

## SWIFT 的安装

- 环境推荐

| 范围         | 推荐         | 备注        |                             |
| ------------ | ------------ | ----------- | --------------------------- |
| python       | >=3.9        | 3.10        |                             |
| cuda         |              | cuda12      | 使用cpu、npu、mps则无需安装 |
| torch        | >=2.0        |             |                             |
| transformers | >=4.33       | 4.51        |                             |
| modelscope   | >=1.19       |             |                             |
| peft         | >=0.11,<0.16 |             |                             |
| trl          | >=0.13,<0.17 | 0.16        | RLHF                        |
| deepspeed    | >=0.14       | 0.14.5      | 训练                        |
| vllm         | >=0.5.1      | 0.7.3/0.8.3 | 推理/部署/评测              |
| lmdeploy     | >=0.5        | 0.7.2.post1 | 推理/部署/评测              |
| evalscope    | >=0.11       |             | 评测                        |

- python安装

```bash
pip install ms-swift -U
```

- 源码安装

```sh
# pip install git+https://github.com/modelscope/ms-swift.git

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```



## 其它用到的库

```bash
pip install transformers
```



## 数据

数据格式参考：

https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html

参考ms-swift内置的[dataset_info.json](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/data/dataset_info.json)）（[ms-swift](https://github.com/modelscope/ms-swift/tree/main)/[swift](https://github.com/modelscope/ms-swift/tree/main/swift)/[llm](https://github.com/modelscope/ms-swift/tree/main/swift/llm)/[dataset](https://github.com/modelscope/ms-swift/tree/main/swift/llm/dataset)/data/dataset_info.json）。该方案使用AutoPreprocessor预处理函数将数据集转换为标准格式。dataset_info.json文件中包含了数据集元信息的list，以下为一些例子：

```
[
  {
    "ms_dataset_id": "xxx/xxx"
  },
  {
    "dataset_path": "<dataset_dir/dataset_path>"
  },
  {
    "ms_dataset_id": "<dataset_id>",
    "subsets": ["v1"],
    "split": ["train", "validation"],
    "columns": {
      "input": "query",
      "output": "response"
    }
  },
  {
    "ms_dataset_id": "<dataset_id>",
    "hf_dataset_id": "<hf_dataset_id>",
    "subsets": [{
      "subset": "subset1",
      "columns": {
        "problem": "query",
        "content": "response"
      }
    },
    {
      "subset": "subset2",
      "columns": {
        "messages": "_",
        "new_messages": "messages"
      }
    }]
  }
]
```

支持以下参数：

- ms_dataset_id: 参考DatasetMeta参数。
- hf_dataset_id: 参考DatasetMeta参数。
- dataset_path: 参考DatasetMeta参数。
- dataset_name: 参考DatasetMeta参数。
- subsets: 参考DatasetMeta参数。
- split: 参考DatasetMeta参数。
- columns: 在数据集进行预处理前，对数据集进行列名转换。



## 使用命令

### 训练

预训练：

```bash
# 8*A100
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model Qwen/Qwen2.5-7B \
    --dataset swift/chinese-c4 \
    --streaming true \
    --train_type full \
    --deepspeed zero2 \
    --output_dir output \
    --max_steps 10000 \
    ...
```

微调：

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --train_type lora \
    --output_dir output \
    ...
```

RLHF：

```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --train_type lora \
    --output_dir output \
    ...
```

**RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）** 是一种结合人类反馈和强化学习的技术，用于训练大规模语言模型（如 GPT、LLaMA 等）以生成更符合人类期望的内容。它是当前大语言模型对齐（Alignment）研究中的核心技术之一。

### 推理

```
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048

# LoRA
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapters swift/test_lora \
    --stream true \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048
```

### 界面推理

```
CUDA_VISIBLE_DEVICES=0 swift app \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048 \
    --lang zh
```

### 部署

```
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm
```

### 采样

```
CUDA_VISIBLE_DEVICES=0 swift sample \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sampler_engine pt \
    --num_return_sequences 5 \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#5
```

### 评测

```
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend lmdeploy \
    --eval_backend OpenCompass \
    --eval_dataset ARC_c
```

### 量化

```
CUDA_VISIBLE_DEVICES=0 swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quant_bits 4 --quant_method awq \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --output_dir Qwen2.5-7B-Instruct-AWQ
```

### 与魔塔社区的联动

使用swift可以将训练好的模型推送到model_hub上

下面是简单的实例代码：

```sh
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```





## 使用 swift 框架运行 qwen-7B 微调

**10分钟**在单卡3090上对Qwen2.5-7B-Instruct进行自我认知微调：

### 模型下载

```sh
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct')
```

### 训练

```
# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

- 如果要使用自定义数据集进行训练，你可以参考[这里](https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html)组织数据集格式，并指定`--dataset <dataset_path>`。
- `--model_author`和`--model_name`参数只有当数据集中包含`swift/self-cognition`时才生效。
- 如果要使用其他模型进行训练，你只需要修改`--model <model_id/model_path>`即可。
- 默认使用ModelScope进行模型和数据集的下载。如果要使用HuggingFace，指定`--use_hf true`即可。

### 推理

训练完成后，使用以下命令对训练后的权重进行推理：

- 这里的`--adapters`需要替换成训练生成的last checkpoint文件夹。由于adapters文件夹中包含了训练的参数文件`args.json`，因此不需要额外指定`--model`，`--system`，swift会自动读取这些参数。如果要关闭此行为，可以设置`--load_args false`。

```
# Using an interactive command line for inference.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# merge-lora and use vLLM for inference acceleration
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
```

### 上传到ModelScope

使用以下命令将模型推送到ModelScope：

```
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```



## 使用 swift 框架运行 qwen-0.5B 在定义数据集上进行dpo微调

### 模型下载

```sh
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')

# /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
```

### 预训练

```bash
# 多卡
# NPROC_PER_NODE=2 \
# CUDA_VISIBLE_DEVICES=0，1\

# 单卡
CUDA_VISIBLE_DEVICES=0 \
swift sft \
--model /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \ 
--dataset AI-ModelScope/alpaca-gpt4-data-zh \ # 会自己去下载数据集
--train_type lora \
--output_dir /root/workspace/output 
```

**模型和数据不指定的本地的话会自己去魔塔社区下载**

### 运行过程

```
root@autodl-container-3ec3448547-ea564932:~# CUDA_VISIBLE_DEVICES=0 \
swift sft \
--model /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--dataset AI-ModelScope/alpaca-gpt4-data-zh \
--train_type lora \
--output_dir output \
> ^C
root@autodl-container-3ec3448547-ea564932:~# CUDA_VISIBLE_DEVICES=0 \
swift sft \
--model /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--dataset AI-ModelScope/alpaca-gpt4-data-zh \
--train_type lora \
--output_dir /root/workspace/output 
run sh: `/root/miniconda3/bin/python /root/workspace/ms-swift/swift/cli/sft.py --model /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct --dataset AI-ModelScope/alpaca-gpt4-data-zh --train_type lora --output_dir /root/workspace/output`
[INFO:swift] Successfully registered `/root/workspace/ms-swift/swift/llm/dataset/data/dataset_info.json`.
[INFO:swift] rank: -1, local_rank: -1, world_size: 1, local_world_size: 1
[INFO:swift] Loading the model using model_dir: /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
[INFO:swift] Setting torch_dtype: torch.bfloat16
[INFO:swift] Setting args.lazy_tokenize: False
[INFO:swift] Setting args.gradient_accumulation_steps: 16
[INFO:swift] Setting args.dataloader_num_workers: 1
[INFO:swift] output_dir: /root/workspace/output/v0-20250422-101811
[INFO:swift] Global seed set to 42
[INFO:swift] args: TrainArguments(
_n_gpu=-1,
acc_steps=1,
acc_strategy=token,
accelerator_config={'dispatch_batches': False},
adafactor=False,
adalora_beta1=0.85,
adalora_beta2=0.85,
adalora_deltaT=1,
adalora_init_r=12,
adalora_orth_reg_weight=0.5,
adalora_target_r=8,
adalora_tfinal=0,
adalora_tinit=0,
adam_beta1=0.9,
adam_beta2=0.95,
adam_epsilon=1e-08,
adapter_act=gelu,
adapter_length=128,
adapters=[],
add_version=True,
attn_impl=None,
auto_find_batch_size=False,
average_tokens_across_devices=False,
batch_eval_metrics=False,
bf16=True,
bf16_full_eval=False,
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_quant_storage=None,
bnb_4bit_quant_type=nf4,
bnb_4bit_use_double_quant=True,
boft_block_num=0,
boft_block_size=4,
boft_dropout=0.0,
boft_n_butterfly_factor=1,
check_model=True,
ckpt_dir=None,
columns={},
create_checkpoint_symlink=False,
custom_dataset_info=[],
custom_register_path=[],
data_seed=42,
dataloader_drop_last=False,
dataloader_num_workers=None,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
dataset=['AI-ModelScope/alpaca-gpt4-data-zh'],
dataset_num_proc=1,
dataset_shuffle=True,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_timeout=1800,
debug=None,
deepspeed=None,
device_map=None,
disable_tqdm=None,
do_eval=False,
do_predict=False,
do_train=False,
download_mode=reuse_dataset_if_exists,
enable_cache=False,
eval_accumulation_steps=None,
eval_datasets=[],
eval_datasets_args=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_generation_config=None,
eval_limit=None,
eval_on_start=False,
eval_steps=500,
eval_strategy=steps,
eval_use_evalscope=False,
eval_use_gather_object=False,
external_plugins=[],
fourier_n_frequency=2000,
fourier_scaling=300.0,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
freeze_aligner=True,
freeze_llm=False,
freeze_parameters=[],
freeze_parameters_ratio=0.0,
freeze_vit=True,
fsdp=,
fsdp_config=None,
fsdp_min_num_params=0,
fsdp_num=1,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
galore_cos_threshold=0.4,
galore_gamma_proj=2,
galore_optim_per_parameter=False,
galore_proj_bits=4,
galore_proj_group_size=256,
galore_proj_quant=False,
galore_proj_type=std,
galore_quantization=False,
galore_queue_size=5,
galore_rank=128,
galore_scale=1.0,
galore_target_modules=None,
galore_update_proj_gap=50,
galore_with_embedding=False,
generation_config=None,
generation_max_length=None,
generation_num_beams=None,
gradient_accumulation_steps=None,
gradient_checkpointing=True,
gradient_checkpointing_kwargs=None,
greater_is_better=False,
group_by_length=False,
half_precision_backend=auto,
hqq_axis=None,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_args_error=False,
ignore_data_skip=False,
include_for_metrics=[],
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
init_weights=True,
interleave_prob=None,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
lazy_tokenize=False,
learning_rate=0.0001,
length_column_name=length,
lisa_activated_layers=0,
lisa_step_interval=20,
llamapro_num_groups=None,
llamapro_num_new_blocks=4,
load_args=False,
load_best_model_at_end=False,
load_data_args=False,
load_dataset_config=None,
local_rank=-1,
local_repo_path=None,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=/root/workspace/output/v0-20250422-101811/runs,
logging_first_step=True,
logging_nan_inf_filter=True,
logging_steps=5,
logging_strategy=steps,
logprobs=False,
lora_alpha=32,
lora_bias=none,
lora_dropout=0.05,
lora_dtype=None,
lora_ga_batch_size=2,
lora_ga_direction=ArB2r,
lora_ga_iters=2,
lora_ga_max_length=1024,
lora_ga_scale=stable,
lora_ga_stable_gamma=16,
lora_modules=[],
lora_rank=8,
lorap_lr_ratio=None,
loss_scale=default,
loss_type=None,
lr_scheduler_kwargs=None,
lr_scheduler_type=cosine,
max_grad_norm=1.0,
max_length=None,
max_memory={},
max_new_tokens=64,
max_pixels=None,
max_steps=-1,
metric=None,
metric_for_best_model=loss,
metric_warmup_step=0,
model=/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct,
model_author=[None, None],
model_kwargs={},
model_name=[None, None],
model_revision=None,
model_type=qwen2_5,
modules_to_save=[],
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
norm_bbox=None,
num_beams=1,
num_labels=None,
num_train_epochs=3.0,
optim=adamw_torch,
optim_args=None,
optim_target_modules=None,
optimizer=None,
output_dir=/root/workspace/output/v0-20250422-101811,
overwrite_output_dir=False,
packing=False,
padding_side=right,
past_index=-1,
per_device_eval_batch_size=1,
per_device_train_batch_size=1,
predict_with_generate=False,
prediction_loss_only=False,
problem_type=None,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
quant_bits=None,
quant_method=None,
ray_scope=last,
reft_args=None,
reft_intervention_type=LoreftIntervention,
reft_layer_key=None,
reft_layers=None,
reft_rank=4,
remove_unused_columns=True,
repetition_penalty=None,
report_to=['tensorboard'],
response_prefix=None,
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
resume_only_model=False,
rope_scaling=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=500,
save_strategy=steps,
save_total_limit=None,
seed=42,
sequence_parallel_size=1,
shuffle_buffer_size=1000,
skip_memory_metrics=True,
sortish_sampler=False,
split_dataset_ratio=0.01,
stop_words=[],
stopping_strategy=first_exhausted,
stream=False,
streaming=False,
strict=False,
swanlab_exp_name=None,
swanlab_mode=cloud,
swanlab_project=None,
swanlab_token=<SWANLAB_TOKEN>,
swanlab_workspace=None,
system=None,
target_modules=['all-linear'],
target_regex=None,
task_type=causal_lm,
temperature=0.0,
template=qwen2_5,
template_backend=swift,
tf32=None,
tools_prompt=react_en,
top_k=None,
top_logprobs=None,
top_p=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_dtype=torch.bfloat16,
torch_empty_cache_steps=None,
torchdynamo=None,
tp_size=0,
tpu_metrics_debug=False,
tpu_num_cores=None,
train_dataloader_shuffle=True,
train_type=lora,
trainable_parameters=[],
truncation_strategy=delete,
tuner_backend=peft,
use_chat_template=True,
use_cpu=False,
use_dora=False,
use_galore=False,
use_hf=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_liger_kernel=False,
use_mps_device=False,
use_rslora=False,
use_swift_lora=False,
val_dataset=[],
val_dataset_shuffle=False,
vera_d_initial=0.1,
vera_dropout=0.0,
vera_projection_prng_key=0,
vera_rank=256,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.1,
zero_hpz_partition_size=None,
)
[INFO:swift] Loading the model using model_dir: /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct
[INFO:swift] model_kwargs: {'device_map': 'cuda:0'}
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
[INFO:swift] model.hf_device_map: {'': device(type='cuda', index=0)}
[INFO:swift] model_info: ModelInfo(model_type='qwen2_5', model_dir='/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=torch.bfloat16, max_model_len=32768, quant_method=None, quant_bits=None, rope_scaling=None, config=Qwen2Config {
  "_attn_implementation_autoset": true,
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 896,
  "initializer_range": 0.02,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "pad_token_id": 151643,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
, task_type='causal_lm', num_labels=None)
[INFO:swift] model.generation_config: GenerationConfig {
  "bos_token_id": 151643,
  "eos_token_id": [
    151645,
    151643
  ],
  "max_new_tokens": 64,
  "pad_token_id": 151643,
  "repetition_penalty": 1.1
}

[INFO:swift] default_system: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
[INFO:swift] Start time of running main: 2025-04-22 10:18:12.568438
[INFO:swift] Downloading the dataset from ModelScope, dataset_id: AI-ModelScope/alpaca-gpt4-data-zh
[WARNING:modelscope] Use trust_remote_code=True. Will invoke codes from alpaca-gpt4-data-zh. Please make sure that you can trust the external codes.
[WARNING:modelscope] Use trust_remote_code=True. Will invoke codes from AI-ModelScope/alpaca-gpt4-data-zh. Please make sure that you can trust the external codes.
[WARNING:modelscope] Use trust_remote_code=True. Will invoke codes from AI-ModelScope/alpaca-gpt4-data-zh. Please make sure that you can trust the external codes.
[WARNING:modelscope] Use trust_remote_code=True. Will invoke codes from AI-ModelScope/alpaca-gpt4-data-zh. Please make sure that you can trust the external codes.
Downloading [README.md]: 100%|█████████████████████████████████| 1.23k/1.23k [00:00<00:00, 688kB/s]
[INFO:modelscope] storing https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/alpaca-gpt4-data-zh/repo?Source=SDK&Revision=master&FilePath=README.md&View=False in cache at /root/.cache/modelscope/hub/datasets/0e5068922103f4f9417b7739909a0715b01750431f2038e5f4e9d7dcdd6cdcaa
[INFO:modelscope] creating metadata file for /root/.cache/modelscope/hub/datasets/0e5068922103f4f9417b7739909a0715b01750431f2038e5f4e9d7dcdd6cdcaa
Downloading data: 31.8MB [00:02, 13.7MB/s]
[INFO:modelscope] storing https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/alpaca-gpt4-data-zh/repo?Source=SDK&Revision=master&FilePath=train.csv in cache at /root/.cache/modelscope/hub/datasets/downloads/ee3959cc16ee530c43270b123e2d8694a153a70d1b9a10d1e697df701b3fd791
[INFO:modelscope] creating metadata file for /root/.cache/modelscope/hub/datasets/downloads/ee3959cc16ee530c43270b123e2d8694a153a70d1b9a10d1e697df701b3fd791
Generating train split: 48818 examples [00:00, 86086.76 examples/s]
[INFO:swift] create tmp_dir: /root/.cache/modelscope/hub/tmp/hf_datasets-62njnesv
Map: 100%|█████████████████████████████████████████| 48818/48818 [00:00<00:00, 62507.47 examples/s]
[INFO:swift] train_dataset: Dataset({
    features: ['messages'],
    num_rows: 48330
})
[INFO:swift] val_dataset: Dataset({
    features: ['messages'],
    num_rows: 488
})
[INFO:swift] The split dataset from the training set will be saved at: /root/workspace/output/v0-20250422-101811/val_dataset.jsonl.
Map: 100%|██████████████████████████████████████████| 48330/48330 [00:46<00:00, 1032.85 examples/s]
Map: 100%|██████████████████████████████████████████████| 488/488 [00:00<00:00, 1028.84 examples/s]
[INFO:swift] [INPUT_IDS] [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 114388, 99601, 104169, 101214, 102349, 1773, 151645, 198, 151644, 77091, 198, 87026, 85106, 99553, 111720, 57191, 114591, 3837, 35946, 101901, 102104, 87026, 1773, 109194, 87026, 85106, 56007, 99245, 11319, 151645]
[INFO:swift] [INPUT] <|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
请您现在给我您的答案。<|im_end|>
<|im_start|>assistant
您需要提供一个问题或情境，我才能回答您。请问您需要问什么？<|im_end|>
[INFO:swift] [LABELS_IDS] [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 87026, 85106, 99553, 111720, 57191, 114591, 3837, 35946, 101901, 102104, 87026, 1773, 109194, 87026, 85106, 56007, 99245, 11319, 151645]
[INFO:swift] [LABELS] [-100 * 35]您需要提供一个问题或情境，我才能回答您。请问您需要问什么？<|im_end|>
Map: 100%|██████████████████████████████████████████| 48330/48330 [00:05<00:00, 8974.26 examples/s]
[INFO:swift] Dataset Token Length: 172.068281±93.875049, min=36.000000, max=866.000000, size=48330
Map: 100%|██████████████████████████████████████████████| 488/488 [00:00<00:00, 8500.19 examples/s]
[INFO:swift] Dataset Token Length: 173.094262±91.890054, min=43.000000, max=473.000000, size=488
[INFO:swift] The TrainArguments will be saved in: /root/workspace/output/v0-20250422-101811/args.json
[INFO:swift] lora_config: LoraConfig(task_type='CAUSAL_LM', peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct', revision=None, inference_mode=False, r=8, target_modules={'k_proj', 'q_proj', 'up_proj', 'down_proj', 'gate_proj', 'o_proj', 'v_proj'}, exclude_modules=None, lora_alpha=32, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=[], init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', trainable_token_indices=None, loftq_config={}, eva_config=None, corda_config=None, use_dora=False, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False), lora_bias=False, lora_dtype=None, lorap_lr_ratio=None, lorap_emb_lr=1e-06)
[INFO:swift] model: PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Qwen2ForCausalLM(
      (model): Qwen2Model(
        (embed_tokens): Embedding(151936, 896)
        (layers): ModuleList(
          (0-23): 24 x Qwen2DecoderLayer(
            (self_attn): Qwen2Attention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=896, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=896, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=128, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=128, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=128, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=128, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=896, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=896, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): Qwen2MLP(
              (gate_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=4864, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=4864, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (up_proj): lora.Linear(
                (base_layer): Linear(in_features=896, out_features=4864, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=896, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=4864, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=4864, out_features=896, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.05, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4864, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=896, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((896,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((896,), eps=1e-06)
        (rotary_emb): Qwen2RotaryEmbedding()
      )
      (lm_head): Linear(in_features=896, out_features=151936, bias=False)
    )
  )
)
[INFO:swift] model_parameter_info: PeftModelForCausalLM: 498.4319M Params (4.3991M Trainable [0.8826%]), 0.0000M Buffers.
/root/workspace/ms-swift/swift/trainers/mixin.py:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Seq2SeqTrainer.__init__`. Use `processing_class` instead.
  super().__init__(
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
[INFO:swift] The logging file will be saved in: /root/workspace/output/v0-20250422-101811/logging.jsonl
{'loss': 1.60696292, 'token_acc': 0.63554502, 'grad_norm': 1.97572899, 'learning_rate': 0.0001, 'memory(GiB)': 2.44, 'train_speed(iter/s)': 0.250477, 'epoch': 0.0, 'global_step/max_steps': '1/9060', 'percentage': '0.01%', 'elapsed_time': '3s', 'remaining_time': '9h 30m 16s'}
。。。。。
{'loss': 1.66180363, 'token_acc': 0.59449748, 'grad_norm': 1.06889856, 'learning_rate': 9.965e-05, 'memory(GiB)': 5.15, 'train_speed(iter/s)': 0.302867, 'epoch': 0.11, 'global_step/max_steps': '340/9060', 'percentage': '3.75%', 'elapsed_time': '18m 42s', 'remaining_time': '7h 59m 46s'}
Train:   4%|█▉                                                | 344/9060 [18:55<7:58:06,  3.29s/it]
```

