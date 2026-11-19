# 实验代码使用说明

规范见 [README.md](README.md)。所有命令在仓库根目录执行；GPU 测量使用已安装项目依赖的 Linux/WSL2。

## 1. 查看测量计划

```bash
python -m benchmark plan --groups ABCDE
python -m benchmark.execution plan
python -m benchmark.prefix plan
python -m benchmark.mtp plan
python -m benchmark.continuous plan
python -m benchmark.capacity plan
```

plan 只打印 JSON，不加载模型、不下载数据、不创建结果目录。全量最多 132 个测量 trials、9552 次请求；E 的停止规则可能缩减执行量。配置在各目录 cases()，引擎参数在 common/schema.py；默认三轮，组内顺序轮转，E 按并发档位扫描。

## 2. 准备数据

先安装项目运行依赖；数据依赖可放在本目录：

```bash
python -m pip install --target benchmark/.deps --no-deps -r benchmark/requirements.txt
PYTHONPATH=benchmark/.deps python -m benchmark.data --model /path/to/Qwen3.5-0.8B
```

默认数据路径为 benchmark/datasets/v1/，不覆盖完整数据。中断后可用 --resume 复用完整原始文件。仅数据准备命令访问网络；run 不下载模型或数据。使用本地完整模型目录。

准备 WikiText 的 256/2048-token 输入、三档 prefix 请求及 seed、80 条 MT-Bench 首轮问题。输入 token IDs、源 revision、SHA256、聊天模板和 tokenizer 指纹归档。切换 tokenizer 后需重新准备数据。

## 3. 小样本性能试跑（可选）

```bash
python -m benchmark run --groups ABCD --smoke --repeats 1 --model /path/to/model --output benchmark-results/smoke
python -m benchmark.capacity run --smoke --repeats 1 --model /path/to/model --output benchmark-results/capacity-smoke
```

--smoke 每配置使用 8 条输入（至少覆盖 batch），E 只覆盖 C=1/2/4。用于检查运行、指标采集和估计耗时，不并入正式性能数据。

## 4. 正式实验

```bash
python -m benchmark.execution run --model /path/to/model --output benchmark-results/A
python -m benchmark.prefix run --model /path/to/model --output benchmark-results/B
python -m benchmark.mtp run --model /path/to/model --output benchmark-results/C
python -m benchmark.continuous run --model /path/to/model --output benchmark-results/D
python -m benchmark.capacity run --model /path/to/model --output benchmark-results/E
```

也可一次运行 A～E：

```bash
python -m benchmark run --groups ABCDE --model /path/to/model --output benchmark-results/all
```

--data 指定数据路径。默认三轮，每 trial 超时 1800 秒（含初始化和预热）；--timeout 可更改并归档。输出目录必须不存在；省略 --output 时创建带北京时间的独立目录。

每 trial 独立进程，GPU 串行使用。超时终止该 trial 进程组并保存状态；后续配置继续，E 遵循扫描停止规则。源码改变、损坏的原始记录或目标 Graph 未生效时，报告记录原因并返回非零。

每配置至少 5 个预热窗口，另覆盖采样器可执行 batch 的 dtype/device/inference-mode 形状；预热、编译、捕获及 seed 成本排除于正式窗口。容量规划降档后按实际可执行 batch 预热。

E 只在初始化规划缓存；若降低可驻留并发，保存实际容量并停止扩展。OOM、资源不足、实现错误和超时分开记录，不能把执行错误归为硬件极限。

## 5. 独立 MTP profiling

```bash
python -m benchmark.mtp run --profiling --repeats 1 --model /path/to/model --output benchmark-results/C-profile
```

固定前 8 条 MT-Bench 问题、256 输出、C0～C4。记录 draft/verify/sync/采样 CUDA 时间、逐步 GPU 区间和 host 开销。正式性能运行关闭细粒度计时。GPU 区间可能包含发射间隙；融合状态操作在 verify 内，嵌套区间不能相加。profiling 吞吐独立呈现。

## 6. 汇总与原始记录

```bash
python -m benchmark summarize benchmark-results/all
cd benchmark-results/all
sha256sum --quiet -c checksums.sha256
```

结果包含环境/模型/数据/源码指纹、配置、计划、源码快照、逐请求/批次/step/资源 JSONL、执行状态、汇总 CSV/JSON、report.md 和校验和。失败保留 partial-*，不计入成功吞吐。请求记录保存时间、长度、终止及命中量，不保存用于输出对照的生成 token IDs。

汇总从原始时间、提交数量和实际窗口重算，检查测量记录完整性、计划参数、命中量及源码归档。吞吐按总 tokens/总时间计算，不平均请求 tok/s。持续并发另报 Decode 调度阶段吞吐，不伪造纯 Decode 墙钟吞吐。正式性能比较要求三轮配对、目标执行模式生效及源码完整；试跑和 profiling 单列。

数据、.deps 和测量结果默认不提交 Git。
