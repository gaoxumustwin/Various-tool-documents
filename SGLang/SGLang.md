# SGLang

## 简介

SGLang的官网：https://docs.sglang.ai/

SGLang的github地址：https://github.com/sgl-project/sglang

## 环境配置

**python : 3.10**

**cuda : 12.1**

默认情况下，你可以通过 pip 在新环境中安装 `sglang` ：

```sh
pip install "sglang[all]>=0.4.6.post1"
# 安装完成后的pip sglang  0.4.6.post2
# 其对应的torch版本：
#torch                     2.6.0
#torchao                   0.10.0
#torchvision               0.21.0
```

如果在安装过程中遇到问题，请随时查阅官方安装文档（[链接](https://docs.sglang.ai/start/install.html)）

## 运行指令

运行命令如下所示：

```sh
python -m sglang.launch_server --model-path /mnt/d/gx/Desktop/Model/Qwen3-0___6B
```

运行日志：

```sh
(sglang) gx@DESKTOP-1OPLDI5:/mnt/e/ubuntu20.04$ python -m sglang.launch_server --model-path /mnt/d/gx/Desktop/Model/Qwen3-0___6B
[2025-05-02 16:13:35] server_args=ServerArgs(model_path='/mnt/d/gx/Desktop/Model/Qwen3-0___6B', tokenizer_path='/mnt/d/gx/Desktop/Model/Qwen3-0___6B', tokenizer_mode='auto', skip_tokenizer_init=False, enable_tokenizer_batch_encode=False, load_format='auto', trust_remote_code=False, dtype='auto', kv_cache_dtype='auto', quantization=None, quantization_param_path=None, context_length=None, device='cuda', served_model_name='/mnt/d/gx/Desktop/Model/Qwen3-0___6B', chat_template=None, completion_template=None, is_embedding=False, revision=None, host='127.0.0.1', port=30000, mem_fraction_static=0.88, max_running_requests=None, max_total_tokens=None, chunked_prefill_size=2048, max_prefill_tokens=16384, schedule_policy='fcfs', schedule_conservativeness=1.0, cpu_offload_gb=0, page_size=1, tp_size=1, pp_size=1, max_micro_batch_size=None, stream_interval=1, stream_output=False, random_seed=159989863, constrained_json_whitespace_pattern=None, watchdog_timeout=300, dist_timeout=None, download_dir=None, base_gpu_id=0, gpu_id_step=1, log_level='info', log_level_http=None, log_requests=False, log_requests_level=0, show_time_cost=False, enable_metrics=False, decode_log_interval=40, api_key=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, dp_size=1, load_balance_method='round_robin', ep_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', lora_paths=None, max_loras_per_batch=8, lora_backend='triton', attention_backend=None, sampling_backend='flashinfer', grammar_backend='xgrammar', speculative_algorithm=None, speculative_draft_model_path=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, disable_radix_cache=False, disable_cuda_graph=False, disable_cuda_graph_padding=False, enable_nccl_nvls=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_multimodal=None, disable_overlap_schedule=False, enable_mixed_chunk=False, enable_dp_attention=False, enable_ep_moe=False, enable_deepep_moe=False, deepep_mode='auto', enable_torch_compile=False, torch_compile_max_bs=32, cuda_graph_max_bs=8, cuda_graph_bs=None, torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, allow_auto_truncate=False, enable_custom_logit_processor=False, tool_call_parser=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through_selective', flashinfer_mla_disable_ragged=False, warmups=None, moe_dense_tp_size=None, n_share_experts_fusion=0, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, debug_tensor_dump_output_folder=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_bootstrap_port=8998, disaggregation_transfer_backend='mooncake', disaggregation_ib_device=None)
[2025-05-02 16:13:41] Attention backend not set. Use flashinfer backend by default.
[2025-05-02 16:13:41] Init torch distributed begin.
[2025-05-02 16:13:42] Init torch distributed ends. mem usage=0.00 GB
[2025-05-02 16:13:42] Load weight begin. avail mem=3.23 GB
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:50<00:00, 50.19s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:50<00:00, 50.19s/it]

[2025-05-02 16:14:33] Load weight end. type=Qwen3ForCausalLM, dtype=torch.bfloat16, avail mem=2.03 GB, mem usage=1.20 GB.
[2025-05-02 16:14:33] KV Cache is allocated. #tokens: 15387, K size: 0.82 GB, V size: 0.82 GB
[2025-05-02 16:14:33] Memory pool end. avail mem=0.00 GB
2025-05-02 16:14:33,560 - INFO - flashinfer.jit: Prebuilt kernels not found, using JIT backend
[2025-05-02 16:14:33] Capture cuda graph begin. This can take up to several minutes. avail mem=0.00 GB
[2025-05-02 16:14:33] Capture cuda graph bs [1, 2, 4, 8]
Capturing batches (avail_mem=0.00 GB):   0%|                                                      | 0/4 [00:00<?, ?it/s]2025-05-02 16:14:34,907 - INFO - flashinfer.jit: Loading JIT ops: batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False
2025-05-02 16:14:57,224 - INFO - flashinfer.jit: Finished loading JIT ops: batch_decode_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False
Capturing batches (avail_mem=0.00 GB): 100%|██████████████████████████████████████████████| 4/4 [00:24<00:00,  6.20s/it]
[2025-05-02 16:14:58] Capture cuda graph end. Time elapsed: 24.85 s. mem usage=0.00 GB. avail mem=0.00 GB.
[2025-05-02 16:14:58] max_total_num_tokens=15387, chunked_prefill_size=2048, max_prefill_tokens=16384, max_running_requests=2049, context_len=40960
[2025-05-02 16:14:59] INFO:     Started server process [1320]
[2025-05-02 16:14:59] INFO:     Waiting for application startup.
[2025-05-02 16:14:59] INFO:     Application startup complete.
[2025-05-02 16:14:59] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2025-05-02 16:15:00] INFO:     127.0.0.1:48030 - "GET /get_model_info HTTP/1.1" 200 OK
[2025-05-02 16:15:00] Prefill batch. #new-seq: 1, #new-token: 6, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0
2025-05-02 16:15:03,610 - INFO - flashinfer.jit: Loading JIT ops: batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False
2025-05-02 16:15:35,723 - INFO - flashinfer.jit: Finished loading JIT ops: batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False
[2025-05-02 16:15:37] INFO:     127.0.0.1:48036 - "POST /generate HTTP/1.1" 200 OK
[2025-05-02 16:15:37] The server is fired up and ready to roll!
```

默认情况下，**如果模型未指向有效的本地目录，它将从 Hugging Face Hub 下载模型文件。**要从 ModelScope 下载模型，请在运行上述命令之前设置以下内容：

```sh
export SGLANG_USE_MODELSCOPE=true
```

对于使用张量并行的分布式推理，操作非常简单：

```sh
python -m sglang.launch_server --model-path /mnt/d/gx/Desktop/Model/Qwen3-0___6B --tensor-parallel-size 4
```

上述命令将在 4 块 GPU 上使用张量并行。您应根据需求调整 GPU 的数量。

## API 服务

借助 SGLang ，构建一个与OpenAI API兼容的API服务十分简便，该服务可以作为实现OpenAI API协议的服务器进行部署。

**默认情况下，它将在 `http://localhost:30000` 启动服务器。**

您可以通过 `--host` 和 `--port` 参数来自定义地址。

### 基本用法

然后，您可以利用 [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) 来与Qwen进行对话：

curl

```sh
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/mnt/d/gx/Desktop/Model/Qwen3-0___6B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 32768
}'
```

或者您可以如下面所示使用 `openai` Python SDK中的 API 客户端：

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768,
)
print("Chat response:", chat_response)
```

虽然默认的采样参数在大多数情况下适用于思考模式，但建议根据您的应用调整采样参数，并始终将采样参数传递给 API。

### 思考与非思考模式

**Qwen3 模型会在回复前进行思考。这种行为可以通过硬开关（完全禁用思考）或软开关（模型遵循用户关于是否应该思考的指令）来控制。**

硬开关在 SGLang 中可以通过以下 API 调用配置使用。要禁用思考，请使用

```
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "Give me a short introduction to large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

或者您可以如下面所示使用 `openai` Python SDK中的 API 客户端：

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use SGLang's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=8192,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
print("Chat response:", chat_response)
```

要完全禁用思考，您可以在启动模型时使用[自定义聊天模板](https://qwen.readthedocs.io/zh-cn/latest/_downloads/c101120b5bebcc2f12ec504fc93a965e/qwen3_nonthinking.jinja)：

```
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --chat-template ./qwen3_nonthinking.jinja
```

该聊天模板会阻止模型生成思考内容，即使用户通过 `/think` 指示模型这样做。

建议为思考模式和非思考模式分别设置不同的采样参数。

### 解析思考内容

SGLang 支持将模型生成的思考内容解析为结构化消息：

```
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --reasoning-parser qwen3
```

响应消息除了包含 `content` 字段外，还会有一个名为 `reasoning_content` 的字段，其中包含模型生成的思考内容。

**请注意，此功能与 OpenAI API 规范不一致。**

**`enable_thinking=False` 可能与思考内容解析不兼容。如果需要向 API 传递 `enable_thinking=False`，请考虑禁用该功能。**

### 解析工具调用

SGLang 支持将模型生成的工具调用内容解析为结构化消息：

```
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --tool-call-parser qwen25
```

详细信息，请参阅[函数调用的指南](https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html)。

### 结构化/JSON输出

SGLang 支持结构化/JSON 输出。请参阅[SGLan文档](https://docs.sglang.ai/backend/structured_outputs.html#OpenAI-Compatible-API)。此外，还建议在系统消息或您的提示中指示模型生成特定格式。

### 部署量化模型

Qwen3 提供了两种类型的预量化模型：FP8 和 AWQ。

部署这些模型的命令与原始模型相同，只是名称有所更改：

```sh
# For FP8 quantized model
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B-FP8

# For AWQ quantized model
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B-AWQ
```

### 上下文长度

Qwen3 模型在预训练中的上下文长度最长为 32,768 个 token。为了处理显著超过 32,768 个 token 的上下文长度，应应用 RoPE 缩放技术。我们已经验证了 [YaRN](https://arxiv.org/abs/2309.00071) 的性能，这是一种增强模型长度外推的技术，可确保在长文本上的最佳性能。

SGLang 支持 YaRN，可以配置为

```
python -m sglang.launch_server --model-path Qwen3/Qwen3-8B --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'
```

备注

1. SGLang 实现了静态 YaRN，这意味着无论输入长度如何，缩放因子都保持不变，**这可能会对较短文本的性能产生影响。** 我们建议仅在需要处理长上下文时添加 `rope_scaling` 配置。还建议根据需要调整 `factor`。例如，如果您的应用程序的典型上下文长度为 65,536 个 token，则最好将 `factor` 设置为 2.0。

2. `config.json` 中的默认 `max_position_embeddings` 被设置为 40,960，SGLang 将使用该值。此分配包括为输出保留 32,768 个 token，为典型提示保留 8,192 个 token，这足以应对大多数涉及短文本处理的场景，并为模型思考留出充足空间。如果平均上下文长度不超过 32,768 个 token，我们不建议在此场景中启用 YaRN，因为这可能会降低模型性能。