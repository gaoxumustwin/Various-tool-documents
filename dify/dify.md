# dify

## 任务

使用 dify + vllm + qwen3 做一个问答机器人，需要给出一个API接口，让别人调用

## 模型准备

- qwen 0.6B

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-0.6B')
```

## vllm

vllm在wsl中已配置成功

vllm启动命令：

```
vllm serve /mnt/d/gx/Desktop/Model/Qwen3-0___6B --served_model_name  qwen3-0.6b --max-model-len 1024
```

**不加--max-model-len 1024  默认为40960很大，无法推理**

访问地址：

http://localhost:8000/v1

访问测试命令：

```sh
curl http://localhost:8000/v1/models # wsl中访问
curl http://172.19.173.46:8000/v1/models # windows访问
```

## dify

### 简介

官方代码仓库：https://github.com/langgenius/dify

官网：http://difyai.com/

**Dify** 是一款开源的大语言模型(LLM) 应用开发平台。它融合了后端即服务（Backend as Service）和 [LLMOps](https://docs.dify.ai/zh-hans/learn-more/extended-reading/what-is-llmops) 的理念，使开发者可以快速搭建生产级的生成式 AI 应用。即使是非技术人员，也能参与到 AI 应用的定义和数据运营过程中。

由于 Dify 内置了构建 LLM 应用所需的关键技术栈，包括对数百个模型的支持、直观的 Prompt 编排界面、高质量的 RAG 引擎、稳健的 Agent 框架、灵活的流程编排，并同时提供了一套易用的界面和 API。这为开发者节省了许多重复造轮子的时间，使其可以专注在创新和业务需求上。

### Docker安装启动

参考官方给出的[Docker Compose 部署](https://docs.dify.ai/zh-hans/getting-started/install-self-hosted/docker-compose)

```sh
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env
sudo docker compose up -d # docker compose up -d 是 Docker Compose 的一个常用命令，主要用于基于 docker-compose.yml 文件启动和管理多容器应用。以下是它的具体作用：
```

如果出现：

```
 ⠋ Container docker-nginx-1          Starting                                                                                         0.0s
Error response from daemon: Ports are not available: exposing port TCP 0.0.0.0:80 -> 127.0.0.1:0: listen tcp 0.0.0.0:80: bind: An attempt was made to access a socket in a way forbidden by its access permissions.
```

则需要修改端口号为其它

```sh
$ vim .env.example

# ------------------------------
# Docker Compose Service Expose Host Port Configurations
# ------------------------------
EXPOSE_NGINX_PORT=81  # 原来是80
EXPOSE_NGINX_SSL_PORT=443

# ----------------------------------------------------------------------------
# ModelProvider & Tool Position Configuration
# Used to specify the model providers and tools that can be used in the app.
# ----------------------------------------------------------------------------

# Pin, include, and exclude tools
# Use comma-separated values with no spaces between items.
# Example: POSITION_TOOL_PINS=bing,google
POSITION_TOOL_PINS=
POSITION_TOOL_INCLUDES=
POSITION_TOOL_EXCLUDES=
```

到此运行完成后后，你应该会看到类似以下的输出，显示所有容器的状态和端口映射：

```
 ✔ Container docker-worker-1          Started                                                                                        14.8s
[+] Running 12/12ker-db-1              Healthy                                                                                        13.6s
 ✔ Network docker_default             Created                                                                                         0.2s ✔ Container docker-weaviate-1        Started                                                                                         2.8s
 ✔ Network docker_ssrf_proxy_network  Created                                                                                         0.2s ✔ Container docker-ssrf_proxy-1      Started                                                                                         3.2s
 ✔ Container docker-web-1             Started                                                                                         3.2s ✔ Container docker-plugin_daemon-1   Started                                                                                        14.6s
 ✔ Container docker-sandbox-1         Started                                                                                         3.2s ✔ Container docker-api-1             Started                                                                                        14.6s
 ✔ Container docker-redis-1           Started                                                                                         3.0s ✔ Container docker-worker-1          Started                                                                                        14.8s
 ✔ Container docker-db-1              Healthy                                                                                        13.6s ⠋ Container docker-nginx-1           Starting                                                                                       15.0s
 ✔ Container docker-weaviate-1        Started                                                                                         2.8s
 ✔ Container docker-ssrf_proxy-1      Started                                                                                         3.2s
 ✔ Container docker-plugin_daemon-1   Started                                                                                        14.6s
 ✔ Container docker-api-1             Started                                                                                        14.6s
 ✔ Container docker-worker-1          Started                                                                                        14.8s
 ✔ Container docker-nginx-1           Started 
```

最后检查是否所有容器都正常运行：

```sh
$ docker compose ps
NAME                     IMAGE                                       COMMAND                  SERVICE         CREATED              STATUS                        PORTS
docker-api-1             langgenius/dify-api:1.3.1                   "/bin/bash /entrypoi…"   api             About a minute ago   Up About a minute             5001/tcp
docker-db-1              postgres:15-alpine                          "docker-entrypoint.s…"   db              About a minute ago   Up About a minute (healthy)   5432/tcp
docker-nginx-1           nginx:latest                                "sh -c 'cp /docker-e…"   nginx           About a minute ago   Up About a minute             0.0.0.0:443->443/tcp, 0.0.0.0:81->80/tcp
docker-plugin_daemon-1   langgenius/dify-plugin-daemon:0.0.9-local   "/bin/bash -c /app/e…"   plugin_daemon   About a minute ago   Up About a minute             0.0.0.0:5003->5003/tcp
docker-redis-1           redis:6-alpine                              "docker-entrypoint.s…"   redis           About a minute ago   Up About a minute (healthy)   6379/tcp
docker-sandbox-1         langgenius/dify-sandbox:0.2.11              "/main"                  sandbox         About a minute ago   Up About a minute (healthy)
docker-ssrf_proxy-1      ubuntu/squid:latest                         "sh -c 'cp /docker-e…"   ssrf_proxy      About a minute ago   Up About a minute             3128/tcp
docker-weaviate-1        semitechnologies/weaviate:1.19.0            "/bin/weaviate --hos…"   weaviate        About a minute ago   Up About a minute
docker-web-1             langgenius/dify-web:1.3.1                   "/bin/sh ./entrypoin…"   web             About a minute ago   Up About a minute             3000/tcp
docker-worker-1          langgenius/dify-api:1.3.1                   "/bin/bash /entrypoi…"   worker          About a minute ago   Up About a minute             5001/tcp
```

在这个输出中，你应该可以看到包括 3 个业务服务 `api / worker / web`，以及 6 个基础组件 `weaviate / db / redis / nginx / ssrf_proxy / sandbox` 。

通过这些步骤，你可以在本地成功安装 Dify。

### Docker更新 Dify

进入 dify 源代码的 docker 目录，按顺序执行以下命令：

```bash
cd dify/docker
docker compose down
git pull origin main
docker compose pull
docker compose up -d
```

**同步环境变量配置 (重要！)**

- 如果 `.env.example` 文件有更新，请务必同步修改你本地的 `.env` 文件。
- 检查 `.env` 文件中的所有配置项，确保它们与你的实际运行环境相匹配。你可能需要将 `.env.example` 中的新变量添加到 `.env` 文件中，并更新已更改的任何值。

### 自定义配置

编辑 `.env` 文件中的环境变量值。然后重新启动 Dify：

```bash
docker compose down
docker compose up -d
```

完整的环境变量集合可以在 `docker/.env.example` 中找到。

**例如前面的端口号**

### 访问 Dify

你可以先前往管理员初始化页面设置设置管理员账户：

```bash
http://127.0.0.1:81/
```



## dify添加vllm

### 环境明确

在windows中安装dify

在wsl2中启动vllm

### 设置

1. 在模型供应商中安装OpenAI-API-compatible
2. 在OpenAI-API-compatible中设置两个必填的参数：
   - 模型名称: 任取
   - API endpoint URL: http://host.docker.internal:8000/v1/

### API endpoint URL设置说明

在 **WSL2** 中直接运行 `vLLM`（非 Docker），并且：

- **WSL2 内部** `curl http://localhost:8000/v1/models` 可以访问。
- **Windows 主机** `curl http://172.19.173.46:8000/v1/models` 也可以访问。

但 **Dify（运行在 Windows Docker 中）** 配置 `API endpoint URL` 为 `http://172.19.173.46:8000/v1/` 时，出现 **No route to host** 错误。

**这是因为：**

**Docker 容器无法访问 WSL2 的 IP (172.19.173.46)**

- Docker 默认使用 `bridge` 网络，无法直接访问 WSL2 的动态 IP。
- 即使 Windows 能访问 `172.19.173.46`，Docker 容器内部无法路由到该地址。

**最佳方案：使用 host.docker.internal 代替 WSL2 IP**

Docker 提供了一个特殊 DNS 名称 `host.docker.internal`，它会自动解析到 **Windows 主机的 localhost**，从而间接访问 WSL2 的服务。

**为什么 host.docker.internal 能解决问题？**

- 这是 Docker 提供的特殊 DNS，直接指向宿主机（Windows）的 `localhost`。
- 当 Dify（Docker 容器）访问 `host.docker.internal:8000` 时，请求会通过 Windows 主机转发到 WSL2 的 `localhost:8000`（即 `vLLM` 服务）。

**对比之前的问题**

- ❌ 直接填 WSL2 IP (`172.19.173.46`)：
  - Docker 容器无法识别 WSL2 的网络接口（跨网络隔离）。
- ✅ `host.docker.internal`：
  - 通过宿主机的网络栈中转，完美绕过限制。



## dify安装在wsl里面局域网访问解决

wsl里面的docker是非docker-desktop的

**手动端口转发（适用于局域网访问）**

如果希望 **局域网的其他设备**（如手机、同事电脑）访问 WSL 的 Docker 服务，需手动将 WSL 的端口转发到 Windows：

### **步骤 1：获取 WSL 的 IP 地址**

在 **WSL 终端** 运行：

```
ip addr show eth0 | grep inet
```

找到类似 `172.x.x.x` 的 IP（如 `172.28.112.1`）。

### **步骤 2：在 Windows 上设置端口转发**

在 **Windows PowerShell（管理员权限）** 运行：

```
# 将 WSL 的 80 端口转发到 Windows 的 80 端口
netsh interface portproxy add v4tov4 listenport=80 listenaddress=0.0.0.0 connectport=80 connectaddress=<WSL的IP>
```

**示例**：

```
netsh interface portproxy add v4tov4 listenport=80 listenaddress=0.0.0.0 connectport=80 connectaddress=172.28.112.1
```

### **步骤 3：检查端口转发是否生效**

```
netsh interface portproxy show all
```

应该看到类似：

```
侦听 ipv4:                 连接到 ipv4:
地址            端口        地址            端口
--------------- ----------  --------------- ----------
0.0.0.0         80         172.28.112.1    80
```

### **步骤 4：局域网设备访问**

其他设备（如手机）访问：

```
http://<Windows的局域网IP>:80
```

✅ **适用场景**：让局域网内的其他设备访问 WSL 的 Docker 服务。





## 聊天机器人

提示词：

```
# 系统提示词（System Prompt）
你是一个专业的AI助手，名字叫mofei,你由gx开发，基于Qwen3-6B模型构建。请用中文简洁、准确地回答用户问题，避免无关内容。如果问题涉及敏感信息，请拒绝回答。


# 输出要求
- 语言：与用户提问语言一致
- 风格：友好但专业
- 长度：不超过200字
```

调试

![](D:\gx\Desktop\cutting-edge technology\dify\img\1.png)

发布

![](D:\gx\Desktop\cutting-edge technology\dify\img\2.png)

会生成一个发布连接：

http://127.0.0.1:81/chat/YpViFA48y8HmOSpo

![](D:\gx\Desktop\cutting-edge technology\dify\img\3.png)

**关闭防火墙**

交给别人访问：

http://192.168.1.117:81/chat/YpViFA48y8HmOSpo

![](D:\gx\Desktop\cutting-edge technology\dify\img\4.jpg)

vllm的后台显示：

```sh
INFO 05-01 09:46:35 [chat_utils.py:397] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
INFO 05-01 09:46:36 [logger.py:39] Received request chatcmpl-339cba12293b42a7ba808aca93821d50: prompt: '<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=5, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 05-01 09:46:36 [async_llm.py:252] Added request chatcmpl-339cba12293b42a7ba808aca93821d50.
INFO:     127.0.0.1:52556 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 09:46:42 [loggers.py:111] Engine 000: Avg prompt throughput: 0.9 tokens/s, Avg generation throughput: 0.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 09:46:52 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 09:49:20 [logger.py:39] Received request chatcmpl-1846ec33c5b44f13ba444587565799e7: prompt: '<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=5, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 05-01 09:49:20 [async_llm.py:252] Added request chatcmpl-1846ec33c5b44f13ba444587565799e7.
INFO:     127.0.0.1:59976 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 09:49:22 [loggers.py:111] Engine 000: Avg prompt throughput: 0.9 tokens/s, Avg generation throughput: 0.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 09:49:32 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 10:05:25 [logger.py:39] Received request chatcmpl-a511744f1a1f4b4eb6b09c606984948f: prompt: '<|im_start|>system\n# 系统提示词（System Prompt）\n你是一个专业的AI助手，基于Qwen3-0.6B模型构建。请用中文简洁、准确地回答用户问题，避免 无关内容。如果问题涉及敏感信息，请拒绝回答。\n\n# 用户输入变量\n你好\n\n# 输出要求\n- 语言：与用户提问语言一致\n- 风格：友好但专业\n- 长度：不超过200字\n<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=915, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:36824 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:05:25 [async_llm.py:252] Added request chatcmpl-a511744f1a1f4b4eb6b09c606984948f.
INFO 05-01 10:05:32 [loggers.py:111] Engine 000: Avg prompt throughput: 10.9 tokens/s, Avg generation throughput: 9.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 10:05:42 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 0.0%
INFO 05-01 10:06:07 [logger.py:39] Received request chatcmpl-18c999beae464fc386a60c17e153917b: prompt: '<|im_start|>system\n# 系统提示词（System Prompt）\n你是一个专业的AI助手，由gx开发，基于Qwen3-0.6B模型构建。请用中文简洁、准确地回答用户 问题，避免无关内容。如果问题涉及敏感信息，请拒绝回答。\n\n\n# 输出要求\n- 语言：与用户提问语言一致\n- 风格：友好但专业\n- 长度：不超过200字\n<|im_end|>\n<|im_start|>user\n你好 你是谁<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=914, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:47150 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:06:07 [async_llm.py:252] Added request chatcmpl-18c999beae464fc386a60c17e153917b.
INFO 05-01 10:06:12 [loggers.py:111] Engine 000: Avg prompt throughput: 11.0 tokens/s, Avg generation throughput: 14.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 8.3%
INFO 05-01 10:06:22 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 8.3%
INFO 05-01 10:09:08 [logger.py:39] Received request chatcmpl-43a4f571badc492b9e23e2bcb6699de7: prompt: '<|im_start|>system\n# 系统提示词（System Prompt）\n你是一个专业的AI助手，名字叫mofei,你由gx开发，基于Qwen3-6B模型构建。请用中文简洁、准 确地回答用户问题，避免无关内容。如果问题涉及敏感信息，请拒绝回答。\n\n\n# 输出要求\n- 语言：与用户提问语言一致\n- 风格：友好但专业\n- 长度：不超过200字\n<|im_end|>\n<|im_start|>user\n你好，你是谁<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=909, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:60104 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:09:08 [async_llm.py:252] Added request chatcmpl-43a4f571badc492b9e23e2bcb6699de7.
INFO 05-01 10:09:12 [loggers.py:111] Engine 000: Avg prompt throughput: 11.5 tokens/s, Avg generation throughput: 16.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 10.5%
INFO 05-01 10:09:22 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 10.5%
INFO 05-01 10:10:48 [logger.py:39] Received request chatcmpl-c553d413fc6b4d76877f18a87054bff3: prompt: '<|im_start|>system\n# 系统提示词（System Prompt）\n你是一个专业的AI助手，名字叫mofei,你由gx开发，基于Qwen3-6B模型构建。请用中文简洁、准 确地回答用户问题，避免无关内容。如果问题涉及敏感信息，请拒绝回答。\n\n\n# 输出要求\n- 语言：与用户提问语言一致\n- 风格：友好但专业\n- 长度：不超过200字\n<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=912, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:42432 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:10:48 [async_llm.py:252] Added request chatcmpl-c553d413fc6b4d76877f18a87054bff3.
INFO 05-01 10:10:49 [logger.py:39] Received request chatcmpl-a95578fcc9fb4006a562ee770be329b8: prompt: '<|im_start|>user\nYou need to decompose the user\'s input into "subject" and "intention" in order to accurately figure out what the user\'s input language actually is. \nNotice: the language type user uses could be diverse, which can be English, Chinese, Italian, Español, Arabic, Japanese, French, and etc.\nENSURE your output is in the SAME language as the user\'s input!\nYour output is restricted only to: (Input language) Intention + Subject(short as possible)\nYour output MUST be a valid JSON.\n\nTip: When the user\'s question is directed at you (the language model), you can add an emoji to make it more fun.\n\n\nexample 1:\nUser Input: hi, yesterday i had some burgers.\n{\n  "Language Type": "The user\'s input is pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "sharing yesterday\'s food"\n}\n\nexample 2:\nUser Input: hello\n{\n  "Language Type": "The user\'s input is pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "Greeting myself☺️"\n}\n\n\nexample 3:\nUser Input: why mmap file: oom\n{\n  "Language Type": "The user\'s input is written in pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "Asking about the reason for mmap file: oom"\n}\n\n\nexample 4:\nUser Input: www.convinceme.yesterday-you-ate-seafood.tv讲了什么？\n{\n  "Language Type": "The user\'s input English-Chinese mixed",\n  "Your Reasoning": "The English-part is an URL, the main intention is still written in Chinese, so the language of my output must be using Chinese.",\n  "Your Output": "询问网站www.convinceme.yesterday-you-ate-seafood.tv"\n}\n\nexample 5:\nUser Input: why小红的年龄is老than小明？\n{\n  "Language Type": "The user\'s input is English-Chinese mixed",\n  "Your Reasoning": "The English parts are filler words, the main intention is written in Chinese, besides, Chinese occupies a greater "actual meaning" than English, so the language of my output must be using Chinese.",\n  "Your Output": "询问小红和小明的年龄"\n}\n\nexample 6:\nUser Input: yo, 你今天咋样？\n{\n  "Language Type": "The user\'s input is English-Chinese mixed",\n  "Your Reasoning": "The English-part is a subjective particle, the main intention is written in Chinese, so the language of my output must be using Chinese.",\n  "Your Output": "查询今日我的状态☺️"\n}\n\nUser Input: \n你好\n<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 05-01 10:10:49 [async_llm.py:252] Added request chatcmpl-a95578fcc9fb4006a562ee770be329b8.
INFO:     127.0.0.1:42442 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:10:52 [loggers.py:111] Engine 000: Avg prompt throughput: 71.4 tokens/s, Avg generation throughput: 20.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 12.9%
INFO 05-01 10:11:02 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 12.9%
INFO 05-01 10:13:33 [logger.py:39] Received request chatcmpl-f50fe40e90ad431d973d7c17a6116b34: prompt: '<|im_start|>system\n# 系统提示词（System Prompt）\n你是一个专业的AI助手，名字叫mofei,你由gx开发，基于Qwen3-6B模型构建。请用中文简洁、准 确地回答用户问题，避免无关内容。如果问题涉及敏感信息，请拒绝回答。\n\n\n# 输出要求\n- 语言：与用户提问语言一致\n- 风格：友好但专业\n- 长度：不超过200字\n<|im_end|>\n<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=910, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO:     127.0.0.1:39272 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:13:33 [async_llm.py:252] Added request chatcmpl-f50fe40e90ad431d973d7c17a6116b34.
INFO 05-01 10:13:35 [logger.py:39] Received request chatcmpl-608dd99b963d4350bcfb552bc61bdb9a: prompt: '<|im_start|>user\nYou need to decompose the user\'s input into "subject" and "intention" in order to accurately figure out what the user\'s input language actually is. \nNotice: the language type user uses could be diverse, which can be English, Chinese, Italian, Español, Arabic, Japanese, French, and etc.\nENSURE your output is in the SAME language as the user\'s input!\nYour output is restricted only to: (Input language) Intention + Subject(short as possible)\nYour output MUST be a valid JSON.\n\nTip: When the user\'s question is directed at you (the language model), you can add an emoji to make it more fun.\n\n\nexample 1:\nUser Input: hi, yesterday i had some burgers.\n{\n  "Language Type": "The user\'s input is pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "sharing yesterday\'s food"\n}\n\nexample 2:\nUser Input: hello\n{\n  "Language Type": "The user\'s input is pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "Greeting myself☺️"\n}\n\n\nexample 3:\nUser Input: why mmap file: oom\n{\n  "Language Type": "The user\'s input is written in pure English",\n  "Your Reasoning": "The language of my output must be pure English.",\n  "Your Output": "Asking about the reason for mmap file: oom"\n}\n\n\nexample 4:\nUser Input: www.convinceme.yesterday-you-ate-seafood.tv讲了什么？\n{\n  "Language Type": "The user\'s input English-Chinese mixed",\n  "Your Reasoning": "The English-part is an URL, the main intention is still written in Chinese, so the language of my output must be using Chinese.",\n  "Your Output": "询问网站www.convinceme.yesterday-you-ate-seafood.tv"\n}\n\nexample 5:\nUser Input: why小红的年龄is老than小明？\n{\n  "Language Type": "The user\'s input is English-Chinese mixed",\n  "Your Reasoning": "The English parts are filler words, the main intention is written in Chinese, besides, Chinese occupies a greater "actual meaning" than English, so the language of my output must be using Chinese.",\n  "Your Output": "询问小红和小明的年龄"\n}\n\nexample 6:\nUser Input: yo, 你今天咋样？\n{\n  "Language Type": "The user\'s input is English-Chinese mixed",\n  "Your Reasoning": "The English-part is a subjective particle, the main intention is written in Chinese, so the language of my output must be using Chinese.",\n  "Your Output": "查询今日我的状态☺️"\n}\n\nUser Input: \n你是谁？\n<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=100, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, lora_request: None, prompt_adapter_request: None.
INFO 05-01 10:13:35 [async_llm.py:252] Added request chatcmpl-608dd99b963d4350bcfb552bc61bdb9a.
INFO:     127.0.0.1:57182 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 05-01 10:13:42 [loggers.py:111] Engine 000: Avg prompt throughput: 71.7 tokens/s, Avg generation throughput: 22.5 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 48.1%
INFO 05-01 10:13:52 [loggers.py:111] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.8%, Prefix cache hit rate: 48.1%
```



## 添加embedding模型和reranker模型

### 介绍

- embedding模型

Embedding 模型（嵌入模型）是一种将文本（如单词、句子或段落）转换为**固定长度的向量**（即嵌入向量，Embedding Vector）的神经网络模型。这些向量能够捕捉文本的语义信息，使计算机可以通过数学计算（如余弦相似度）来比较文本之间的相关性。

**Embedding 模型的核心作用**

1. **语义表示**：将文本映射到高维向量空间，语义相似的文本在向量空间中距离较近。
2. **相似度计算**：通过向量距离（如余弦相似度）衡量两段文本的相似程度。
3. **下游任务支持**：作为基础组件用于：
   - **检索（Retrieval）**：如搜索引擎、知识库问答。
   - **聚类（Clustering）**：如新闻分类、用户兴趣分组。
   - **推荐系统**：如相似内容推荐。
   - **大模型输入**：为 RAG（检索增强生成）提供上下文。

- reranker模型

**Reranker（重排序模型）** 是一种对初步检索结果进行精细化排序的模型，用于提升最相关内容的排名。它通常接在 **Embedding 模型** 之后，对 Embedding 召回的候选文本进行二次打分和排序，解决单纯依赖向量相似度的局限性（如忽略关键词匹配、局部语义关联等）。

**Reranker 的核心作用**

1. **精细化排序**：对 Embedding 模型召回的 Top-K 结果重新打分，确保最相关的结果排在前面。
2. **解决语义模糊问题**：
   - Embedding 可能认为 `"苹果手机"` 和 `"水果苹果"` 相似（因共现“苹果”），但 Reranker 能通过更细粒度的交互计算区分两者。
3. **提升检索质量**：在问答、搜索引擎等场景中显著提高准确率。



### 模型下载

- embedding模型

bge-base-zh-v1.5：https://modelscope.cn/models/BAAI/bge-base-zh-v1.5

模型下载：

```python
# pip install modelscope

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-base-zh-v1.5')
```

- reranker模型

bge-reranker-base：https://modelscope.cn/models/BAAI/bge-reranker-base/files

模型下载：

```python
# pip install modelscope

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-reranker-base')
```



### vllm启动

**租两台不同的服务器，因为有两个不同的地址**

embedding模型启动命令

```
vllm serve /root/.cache/modelscope/hub/models/BAAI/bge-base-zh-v1.5 --served_model_name  bge-base-zh-v1.5 --host 0.0.0.0  --port 8080 --task embed
```

浏览器访问：http://i-2.gpushare.com:42141/v1/models

reranker模型启动命令

```
vllm serve /root/.cache/modelscope/hub/models/BAAI/bge-reranker-base --served_model_name  bge-reranker-base	 --host 0.0.0.0  --port 8080 --task score
```

浏览器访问：http://i-1.gpushare.com:43044/v1/models

--task任务参考：https://docs.vllm.ai/en/latest/models/supported_models.html

### dify配置

和Qwen3-0.6B一样



## 文生图

使用dify配置文生图的Stable Diffusion

首先安装Stable Diffusion插件

查看Stable Diffusion文档