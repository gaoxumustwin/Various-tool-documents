# Stable Diffusion

## 介绍

### **Stable Diffusion 是什么？**

**Stable Diffusion（稳定扩散）** 是一种 **文本生成图像（Text-to-Image）的AI模型**，由 **Stability AI**、CompVis 和 RunwayML 等团队联合开发。它基于 **潜在扩散模型（Latent Diffusion Model, LDM）**，能够根据用户输入的文本描述（Prompt）生成高质量的图像。

#### **核心特点**

- **开源免费**：模型代码和权重公开，允许本地部署和商用（需遵守许可证）。
- **高效推理**：相比早期扩散模型（如 DALL·E 2），它优化了计算效率，可在消费级GPU（如RTX 3060）上运行。
- **多功能**：支持多种任务：
  - **文生图（Text-to-Image）**：输入文本生成图像。
  - **图生图（Img2Img）**：基于参考图生成新图像。
  - **图像修复（Inpainting）**：局部修改图像内容。
  - **超分辨率（Upscaling）**：提升图像分辨率。

#### **技术基础**

- **扩散模型（Diffusion Model）**：通过逐步去噪生成图像。
- **潜在空间（Latent Space）**：在低维空间计算，减少显存占用。
- **CLIP 文本编码器**：将文本提示转换为模型可理解的向量。

------

### **Stable Diffusion WebUI 是什么？**

**Stable Diffusion WebUI** 是一个基于 **Gradio 框架** 开发的用户友好界面，由社区开发者（如 [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)）维护，用于简化 Stable Diffusion 的使用。它提供了图形化操作方式，无需编写代码即可调用模型功能。

#### **核心功能**

| 功能              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| **文生图/图生图** | 通过文本框和参数滑块控制生成过程。                      |
| **模型管理**      | 支持加载不同的模型（如 SD1.5、SDXL、动漫风格模型）。    |
| **插件扩展**      | 可安装 ControlNet（姿态控制）、LoRA（微调模型）等插件。 |
| **高级参数**      | 调整采样器（Euler、DPM++）、步数（Steps）、CFG 值等。   |
| **批量生成**      | 一次性生成多张图像并对比效果。                          |

#### **优势**

- **易用性**：适合非技术用户，无需命令行操作。
- **高度可定制**：支持自定义脚本、主题和第三方插件。
- **本地部署**：数据隐私性强，无需依赖云端服务。

------

### **Stable Diffusion 和 WebUI 的关系**

| 组件                      | 角色                                 |
| ------------------------- | ------------------------------------ |
| **Stable Diffusion 模型** | 核心AI模型，负责图像生成。           |
| **WebUI**                 | 交互界面，调用模型并提供可视化控制。 |

**类比**：

- Stable Diffusion 是“引擎”，WebUI 是“方向盘和仪表盘”。



## 环境

恒源云服务器：

- cuda 12.1

- python 3.11  （最好3.10）



## 安装

源码下载：

```sh
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git

# 切换到目录
cd stable-diffusion-webui

# 安装环境
pip install -r requirements.txt
pip install -r requirements_versions.txt
# pip install xformers
# 出现报错不用管
```



## 下载stable diffusion的模型

进入：https://modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5/files

或https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main

```
wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

wget https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt

# 镜像是从hugginface下载 

 mv v1-5-pruned-emaonly.* stable-diffusion-webui/models/Stable-diffusion
```

下载v1-5-pruned-emaonly.ckpt和v1-5-pruned-emaonly.safetensors放至stable-diffusion-webui/models/Stable-diffusion目录下，这个目录专门存放用于生成AI绘图的绘图元素的基础模型库。



## 启动stable-diffusion-webui

```sh
python launch.py --server-name 0.0.0.0 --port 8080 --ckpt v1-5-pruned-emaonly.safetensors  --api --listen
```

**这里要安装很多依赖，并且要git clone相关包和模型文件，需要科学上网，否则会失败**



## 端口连接

恒源云：8080的端口会映射到公网

stable-diffusion的原本连接为：http://127.0.0.1:7860

所以需要修改第78行的--port：/root/stable-diffusion-webui/modules/cmd_args.py为8080



## dify配置

### Agent启动

创建Agent

![](D:\gx\Desktop\cutting-edge technology\StableDiffusion\img\2_1.png)

添加std

![](D:\gx\Desktop\cutting-edge technology\StableDiffusion\img\2.png)

右侧上方的模型使用qwen3等思考模型的时候要将max_tokens设置为4096，不然会使得模型没有思考完，从而不去调用文生图；另一种措施是使用普通的qwen2.5的对话模型或智谱模型来让其不思考，直接调用文生图；



### 流式启动

参考：

https://blog.csdn.net/xhzq1986/article/details/146044090?ops_request_misc=%257B%2522request%255Fid%2522%253A%25228837efbd51aff52bd85060d82de2460f%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=8837efbd51aff52bd85060d82de2460f&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-146044090-null-null.142^v102^control&utm_term=Stable%20Diffusion%20%20dify%E9%85%8D%E7%BD%AE&spm=1018.2226.3001.4187

效果：

![](D:\gx\Desktop\cutting-edge technology\StableDiffusion\img\1.png)







