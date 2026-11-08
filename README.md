<a name="top"></a>

<div align="center">

<img src="assets/new-logo.png" alt="Nano-Infer by sunnyfish — 金色游鱼、太阳与 GPU 数据流主题横幅" width="100%">

<h1>Nano-Infer</h1>

<p><strong>从请求调度到 GPU 执行，读懂并运行 Qwen 推理。</strong></p>

<p>基于 nano-vllm 的轻量 LLM 推理引擎<br>
以清晰的 Python 实现，连接模型适配、缓存管理与 CUDA 执行优化。</p>

<p><code>Qwen3 Dense</code> &nbsp; / &nbsp; <code>Qwen3 MoE</code> &nbsp; / &nbsp; <code>Qwen3.5 Dense</code></p>

[![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-0EA5E9?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-8B5CF6?style=for-the-badge&logo=pytorch&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/Runtime-NVIDIA_CUDA-0891B2?style=for-the-badge&logo=nvidia&logoColor=white)](#requirements)
[![MIT License](https://img.shields.io/badge/License-MIT-D97706?style=for-the-badge)](LICENSE)

<p><strong>🐠 sunnyfish</strong> &nbsp; · &nbsp; Python API &nbsp; · &nbsp; CUDA Runtime</p>

[✨ 核心能力](#features) &nbsp; / &nbsp; [🧩 模型支持](#models) &nbsp; / &nbsp; [🚀 快速开始](#quick-start) &nbsp; / &nbsp; [💻 使用指南](#usage) &nbsp; / &nbsp; [⚙️ 运行方式](#architecture)

</div>

---

Nano-Infer 面向 **Qwen 系列模型的离线文本生成与推理机制学习**。项目将模型架构、通用算子、请求调度和 GPU 执行分开组织，让一次生成请求如何经过分块预填充、缓存分配和逐 token 解码，有清晰可读的实现路径。

通过 `LLM` 与 `SamplingParams` 即可提交批量请求。项目沿用上游的发行名 `nano-vllm` 和 Python 导入名 `nanovllm`。

下文提供安装、模型准备、批量生成和常用参数说明。

<a name="features"></a>

## ✨ 核心能力

| 能力 | 实现与价值 |
| :--- | :--- |
| 🧩 **多模型适配** | 通过统一注册入口构建 Qwen3 Dense、Qwen3 MoE 与 Qwen3.5 Dense，复用调度和执行基础设施。 |
| 🖼️ **图像与视频理解** | Qwen3.5 Dense 支持文本、单图、多图和本地视频输入，统一生成文本回答。 |
| ⚡ **迭代式请求调度** | 按 token 预算执行 chunked prefill 与逐步解码；缓存不足时抢占请求，并通过重新预填充恢复执行。 |
| 🗂️ **分页缓存与前缀复用** | 按块分配 KV cache，以完整块哈希和引用计数复用已有前缀，减少重复预填充；前缀复用限非 GDN 模型。 |
| 🔥 **CUDA 执行优化** | FlashAttention 处理注意力计算，Triton 写入 KV cache，部分算子使用 `torch.compile`；Qwen3 Dense 可启用解码 CUDA Graph。 |
| 🌊 **混合注意力状态管理** | Qwen3.5 结合门控全注意力与 Gated DeltaNet（GDN），分别维护 KV cache、卷积历史和 FP32 递归状态。 |
| 🔗 **单机张量并行** | 实现线性层与词表分片、NCCL 通信及多进程执行控制；具体适用范围见下表。 |

<a name="models"></a>

## 🧩 模型支持

模型由本地 Hugging Face 配置中的 `model_type` 自动选择。

| 模型架构 | 当前用途 | 运行方式与限制 |
| :--- | :--- | :--- |
| **Qwen3 Dense** | 离线文本生成，提供0.6B示例 | eager或Decode CUDA Graph；TP需满足模型维度的分片约束。 |
| **Qwen3 MoE** | 实验性文本适配 | 显式设置 `enforce_eager=True`；不提供专家并行（EP）。 |
| **Qwen3.5 Dense** | 实验性文本、图片和本地视频适配，提供0.8B统一示例 | 仅单GPU，自动使用eager，不提供跨请求前缀复用。 |

模型适配仍在完善，同系列不同 checkpoint 可能需要额外适配。建议从下方提供的单卡示例开始。

当前入口为同步、离线 Python API。Qwen3.5 Dense 支持图像和本地视频输入，具体用法见下文；尚未提供 Qwen3.5 MoE、张量并行及 CUDA Graph。当前引擎也未提供 HTTP 服务、流式生成、量化或投机解码。

<a id="quickstart"></a>
<a name="quick-start"></a>

## 🚀 快速开始

<a name="requirements"></a>

### 环境要求

- **平台：** Linux、NVIDIA CUDA GPU，以及兼容的驱动与 PyTorch CUDA 构建。
- **Python：** `>=3.10,<3.13`。
- **GPU 算子：** 设备、Torch 与 CUDA 版本需满足 [FlashAttention 的安装要求](https://github.com/Dao-AILab/flash-attention#installation-and-features)。从源码构建时需准备 CUDA Toolkit 与 C++ 编译器。
- **模型文件：** 本地目录需包含 Hugging Face 配置、tokenizer 及 `.safetensors` 权重；引擎不直接接受 Hub 模型 ID。

Qwen3.5 多模态依赖 `transformers>=5.22.1,<5.23`、匹配 Torch 构建的 torchvision、Pillow 和 PyAV。应从相同的 PyTorch CUDA wheel 来源安装匹配的 torch/torchvision。模型目录还需包含 Processor 配置和聊天模板。

### 01 · 安装项目

以下命令在 Linux Bash 中执行。先确认 `python3` 版本在支持范围内；已有仓库可跳过 clone 和 cd。

```bash
git clone https://github.com/SunnyFish333/nano-infer.git
cd nano-infer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch>=2.4.0' packaging ninja psutil
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --no-deps -e .
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

若已有适配设备的 Torch，可复用它；请确认上面的 CUDA 检查输出 `True`。`--no-build-isolation` 让 FlashAttention 构建使用当前环境中的 Torch；从源码构建还需要 CUDA Toolkit 和 C++ 编译器。`--no-deps` 跳过依赖解析，因此必须先安装 requirements。CPU 版 Torch 不能运行本项目的模型推理。

### 02 · 下载模型

以下两步仍在Linux Bash、仓库根目录和已激活GPU环境中执行，需要网络及模型空间；下载预期得到本地文件，运行预期打印生成结果。

使用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 将 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 下载到示例默认目录：

```bash
python -m pip install --upgrade huggingface_hub
hf download Qwen/Qwen3-0.6B --local-dir "$HOME/huggingface/Qwen3-0.6B"
```

### 03 · 运行文本生成

```bash
python example.py
```

[example.py](example.py) 展示两条请求的聊天模板处理、批量生成与结果打印。示例使用单卡 eager 模式，并显式设置上下文长度、批次 token 预算和显存比例。

<a name="usage"></a>

## 💻 使用指南

### 批量生成与逐请求采样

`generate()` 接受字符串列表或 token ID 列表。下面为两条请求分别设置生成长度；聊天模板由调用方应用。

```python
import os

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question in ["What is a KV cache?", "Explain tensor parallelism."]
    ]

    llm = LLM(
        path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.7,
    )
    params = [
        SamplingParams(temperature=0.6, max_tokens=64),
        SamplingParams(temperature=0.8, max_tokens=128),
    ]
    outputs = llm.generate(prompts, params)
    for output in outputs:
        print(output["text"])


if __name__ == "__main__":
    main()
```

以上Python示例在已安装GPU依赖且模型目录完整时，从仓库根目录保存为脚本执行，预期按输入顺序打印两条结果。每项包含 `text` 和 `token_ids`；单个 `SamplingParams` 共享策略，列表须与请求数量一一对应。

采样器要求温度大于 `1e-10`，不支持 `temperature=0` 贪心解码。`SamplingParams` 支持以下选项：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `temperature` | `1.0` | 正温度随机采样 |
| `max_tokens` | `64` | 每条请求最多生成的 token 数，需为正整数 |
| `ignore_eos` | `False` | 是否忽略 EOS，继续生成到输出预算 |

输入须非空、token ID 合法、参数列表与请求数量一致，且输入长度与生成预算之和不超过 `max_model_len`。超长输入会被拒绝，请先缩短输入或调整生成预算。

### Qwen3.5 文本推理

下载 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 后，运行专用示例：

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir "$HOME/huggingface/Qwen3.5-0.8B"
python example_qwen3_5.py
```

[example_qwen3_5.py](example_qwen3_5.py) 使用 `tensor_parallel_size=1`、`max_num_seqs=1` 和 `num_gdn_slots=2`。GDN 状态池独立占用显存，扩大并发时应同时考虑槽位容量与 KV cache 预算。

### Qwen3.5 图片与视频输入

同一个示例支持文本、图片和本地视频，不需要单独的 VL 模型入口：

```bash
python example_qwen3_5.py --mode text --question '介绍一下你自己'
python example_qwen3_5.py --mode image --image ./demo.jpg --question '图片里是什么？'
python example_qwen3_5.py --mode video --video ./demo.mp4 --question '描述视频内容'
```

图片模式可重复传入 `--image`。API 接受结构化消息，图片支持本地路径或 PIL Image；视频支持有帧率/时长元数据的本地文件，按上限采样，时间戳保留原视频帧索引：

```python
request = {"messages": [{"role": "user", "content": [
    {"type": "image", "image": "./demo.jpg"},
    {"type": "text", "text": "描述图片。"},
]}]}
outputs = llm.generate([request], SamplingParams(temperature=0.6, max_tokens=64))
```

图片和视频会先进行预处理，再与文本一起计入上下文长度。较大的图片或更多视频帧会占用更多输入 token；超出上下文预算时，可降低图片像素上限、减少视频帧数或缩短生成长度。

默认限制：每请求4张图片、1段视频、最多8个视频采样帧、采样2 FPS，单图/帧预处理面积上限262144像素，CPU视觉数据预算256 MiB。对应参数为 `max_images_per_prompt`、`max_videos_per_prompt`、`max_video_frames`、`video_fps`、`max_image_pixels` 和 `vision_cpu_cache_mb`。小显存示例使用65536像素上限。超出预算的请求会拒绝；视觉工作区在分配 KV cache 前预留。

### 常用运行参数

这些参数通过 `LLM(model_path, **kwargs)` 传入：

| 参数 | 默认值 | 使用说明 |
| --- | --- | --- |
| `max_model_len` | `4096` | 上下文上限，初始化时与模型位置上限取较小值；输入长度与生成预算之和须在该上限内 |
| `max_num_batched_tokens` | `16384` | 单轮 Prefill 的 token 预算；较长输入会分块处理，需为正整数 |
| `max_num_seqs` | `512` | 单轮调度的序列数上限；小显存建议从1或2开始 |
| `gpu_memory_utilization` | `0.9` | 总显存预算比例；必须留出权重、工作区和缓存空间 |
| `num_gdn_slots` | `64` | GDN 请求状态容量；Qwen3.5 小显存可从2开始 |
| `enforce_eager` | `False` | `True` 关闭执行器 CUDA Graph，局部 `torch.compile` 仍会运行；Qwen3 MoE需显式为True，GDN模型自动为True |
| `tensor_parallel_size` | `1` | 单机GPU数，配置范围1–8，还需设备数及分片维度合法；Qwen3.5仅支持1 |

显存不足时可先降低上下文长度、单轮 token 预算、并发数和 GDN 槽位；多模态请求还可降低图片像素上限和视频帧数。`enforce_eager=True` 可减少 Graph 捕获开销，但不能保证任意模型适配显存。每个独立进程建议只运行一个引擎。

<a name="architecture"></a>

## ⚙️ 运行方式

当前采用本地Python进程运行，通过 `LLM.generate()` 提交批量文本或结构化多模态请求。字符串和 token ID 输入由调用方应用聊天模板；结构化多模态消息由 Processor 处理模板和视觉输入。

运行前准备完整本地模型目录。输出按输入顺序返回，每项含 `text` 和 `token_ids`。调用方可自行选择生成预算及保存结果；当前没有 HTTP 服务或流式返回接口。

## 🤝 许可与致谢

项目采用 [MIT License](LICENSE)，基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 扩展，保留原作者版权声明。

感谢 PyTorch、Triton、FlashAttention、Hugging Face Transformers 与 Qwen 开源社区提供的基础设施和模型资源。

---

<div align="center">

<p>☀️ <strong>sunnyfish</strong> &nbsp; · &nbsp; 🐠 <strong>Nano-Infer</strong></p>
<p><a href="#top">↑ 回到顶部</a></p>

</div>
