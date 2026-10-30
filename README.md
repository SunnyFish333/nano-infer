<!-- Logo retained from the upstream Nano-vLLM project. -->
<p align="center">
  <img src="assets/logo.png" width="200" alt="Upstream Nano-vLLM logo">
</p>

<h1 align="center">Nano-Infer</h1>

<p align="center">
  面向多模型架构适配的轻量级 LLM 推理引擎
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%E2%80%933.12-3776AB?logo=python&logoColor=white" alt="Python 3.10–3.12"></a>
  <img src="https://img.shields.io/badge/Platform-Linux%20%2F%20CUDA-76B900?logo=nvidia&logoColor=white" alt="Linux and CUDA">
  <a href="#roadmap"><img src="https://img.shields.io/badge/Status-In%20Development-orange" alt="In development"></a>
</p>

<p align="center">
  <a href="#models">模型状态</a> ·
  <a href="#quickstart">快速开始</a> ·
  <a href="#architecture">代码结构</a> ·
  <a href="#benchmark">性能测试</a>
</p>

Nano-Infer 基于 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)，在通用调度、缓存管理与 GPU 执行框架上扩展不同模型架构。当前以 Qwen3 Dense 为基线，推进 Qwen3 MoE 验证与 Qwen3.5 适配。

## 功能

- **生成与调度**：离线批量生成，支持文本或 token ID 输入、Prefill 优先调度与 Chunked Prefill。
- **缓存管理**：分页 KV Cache、前缀复用与缓存不足时的请求抢占。
- **GPU 执行**：单机张量并行、合并投影、FlashAttention、Triton KV 写入与 Decode CUDA Graph。
- **模型扩展**：按 Hugging Face 配置中的 `model_type` 注册和选择架构实现。

<a id="models"></a>

## 模型状态

| 模型 | 状态 | 说明 |
| --- | --- | --- |
| Qwen3-0.6B | 基线实现 | 包含架构、权重加载与生成示例；待补充分支回归记录 |
| Qwen3-30B-A3B | 已接入，待验证 | MoE 架构已注册；待验证真实权重加载、生成与数值一致性 |
| Qwen3.5 Dense / MoE | 组件开发中 | 已添加位置编码、Norm 与门控投影；完整模型尚未接入 |

上表区分代码接入与运行验证；Qwen3.5 尚不提供完整推理入口。

当前 Qwen3 MoE 采用 TP，尚不包含 EP；验证时需设置 `enforce_eager=True`，关闭暂不兼容的 CUDA Graph 路径。

<a id="quickstart"></a>

## 快速开始

运行环境：Linux、NVIDIA GPU、兼容的 CUDA 与驱动，以及 Python 3.10–3.12。依赖见 [pyproject.toml](pyproject.toml)；当前未提供锁定版本的环境文件。

### 安装

```bash
git clone https://github.com/SunnyFish333/nano-infer.git
cd nano-infer
```

在独立 Python 环境中安装与设备匹配的 [CUDA 版 PyTorch](https://pytorch.org/get-started/locally/)，确认 `torch.cuda.is_available()` 返回 `True`，然后执行：

```bash
python -m pip install packaging ninja
python -m pip install flash-attn --no-build-isolation
python -m pip install -e .
python -m pip install huggingface_hub
```

FlashAttention 的硬件与构建要求见其[安装说明](https://github.com/Dao-AILab/flash-attention#installation-and-features)。

### 运行

以下使用 Qwen3-0.6B 基线模型：

```bash
hf download Qwen/Qwen3-0.6B --local-dir "$HOME/huggingface/Qwen3-0.6B"
python example.py
```

`example.py` 默认读取 `~/huggingface/Qwen3-0.6B/`。Python 包沿用上游名称 `nanovllm`，调用方式如下：

```python
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "解释一下 KV Cache。"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.6, max_tokens=128),
    )
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
```

每条结果包含 `text` 和 `token_ids`。当前采样器要求 `temperature > 1e-10`，不支持通过 `temperature=0` 启用贪心采样。

<a id="architecture"></a>

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `nanovllm/models/` | 模型注册、架构定义与权重映射 |
| `nanovllm/layers/` | 并行线性层、Attention、Norm、位置编码与采样 |
| `nanovllm/engine/` | 请求生命周期、调度、缓存管理与 GPU 执行 |
| `nanovllm/utils/` | 权重加载与执行上下文 |
| `example.py` | 离线生成示例 |
| `bench.py` | 批量生成吞吐量测试 |

模型实现组合通用层，并提供 `forward()` 与 `compute_logits()`。新增架构通过 `models/registry.py` 注册；输入协议、缓存布局或状态管理变化时，需要同步适配引擎。

<a id="roadmap"></a>

## 开发计划

- 完成 Qwen3 Dense 回归，以及 Qwen3 MoE 的权重加载、数值和 TP 验证。
- 实现 Qwen3.5 Gated DeltaNet、混合层状态管理与模型接入，再推进视觉输入支持。
- 在正确性验证后优化专家计算、CUDA Graph 和端到端性能。

欢迎提交模型适配、正确性修复与性能优化。新增模型请附目标 checkpoint、权重映射和复现步骤；性能改动请同时提供数值验证与相同负载下的对照结果。

<a id="benchmark"></a>

## 性能测试

准备好基线模型后运行：

```bash
python bench.py
```

脚本使用随机 token 请求，预热后统计整批生成的输出 token 吞吐量，不包含模型加载时间，也不代表单请求 Decode 速度或服务延迟。Qwen3 MoE 与 Qwen3.5 尚未发布本分支的性能结果。

报告结果时请注明 commit、模型与精度、GPU、依赖版本、TP 配置、输入输出长度和批量大小。

## 来源与许可证

本项目基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，沿用 [MIT License](LICENSE) 并保留上游版权声明；页首 Logo 来自上游项目。模型权重遵循各自仓库的许可证。
