# Nano Infer Benchmark

## 1. 范围与原则

本规范用于测量 Qwen3.5 Dense 单 GPU 离线同步引擎，包括三组核心实验和两组补充实验：普通推理 Graph/合批、Prefix Caching 热命中、MTP 窗口、真实问题持续并发、并发扩展与资源边界。前缀缓存实验对比关闭缓存与预填后的热命中。

范围不包括 HTTP/网络、开放到达流量、线上 QPS/SLO、多卡、多模态和模型任务质量。记录执行状态及性能测量条件。保留失败、负收益和资源限制，不筛选最优轮次。

## 2. 术语

| 名称 | 定义 |
|---|---|
| 固定批次 B | 同时入队 B 条请求，整批完成后提交下一批 |
| 持续并发 C | 至多 C 条已入队未完成请求，完成一条立即补一条，直到数据耗尽 |
| 实际 batch size | 一次模型执行处理的请求数；Prefill/Decode 分开记录 |
| 驻留请求 | 已获得工作缓存资源的请求，不等于全部提交请求 |
| 运行预热 | 完成模型加载、编译、Graph 捕获与算子预热 |
| 热命中 | 正式提交前，已有对应 KV/GDN 联合前缀缓存 |
| trial | 一个独立引擎进程执行一个配置的一轮测量 |

实际输入长度、实际输出长度、配置上下文上限分别记录。所有请求均执行必要的 Prefill；分别报告包含 Prefill 的总吞吐和 Decode 阶段吞吐。

## 3. 公共控制变量

基准使用 Linux / WSL2、单张 NVIDIA CUDA GPU 和 Qwen3.5-0.8B BF16。每批实验记录设备、模型和依赖版本，任一项发生变化时另建批次。

| 项目 | 规定 |
|---|---|
| 模型/设备 | 同一完整 checkpoint、BF16、单 GPU、TP=1 |
| 采样 | temperature=0；固定长度实验 ignore_eos=True；正常 EOS 独立标记 |
| 上下文上限 | 核心实验 max_model_len=4608；禁止超限和静默截断 |
| Prefill 预算 | 核心实验 max_num_batched_tokens=256；改变预算须另立配置 |
| 缓存 | KV block=256 tokens；gpu_memory_utilization=0.8 |
| 策略 | 禁用自适应投机、测量期间的自动缓存重规划 |
| 进程 | 每个 trial 独立进程；同 GPU 上实验串行 |
| 预热 | 至少 5 个与配置长度、批次及输出形状匹配的预热批次/请求 |
| 计时 | 同一进程单调时钟；不逐算子额外 CUDA synchronize |
| profiling | 正式吞吐关闭详细阶段 profiling；独立诊断运行开启 |

A/B 组建议统一 max_num_seqs=4、GDN slots=12；Graph 最大 batch=4，实际 B=1 或 4；eager 使用明确 eager 路径。C 组统一 max_num_seqs=1、Graph 最大 batch=1、GDN slots=12。记录最终 KV/GDN 容量、Graph 内存；容量差异若引起抢占/回退，不把结果简单归因于 Graph。

Graph 指当前阶段 CUDA Graph，不意味着 Prefill、采样、调度和状态管理全部图内执行。必须记录各阶段实际 replay、eligible、fallback 及原因。

初始化、编译、捕获、预热、分词和输入准备不计入正式窗口。预热完成后在 idle 状态清理逻辑缓存，保留固定 GPU 池与 Graph 保留槽。预热数据与计数不得进入测量。

运行器在完整窗口预热后，另外覆盖采样器 1～实际可执行 batch 上限的 dtype/device/inference-mode 形状；上限同时受请求配置、token 预算和已分配 GDN 工作槽限制。避免正常 EOS/排空阶段首次触发新 batch 编译，也避免容量规划降档后按过大的请求上限预热。该准备时间排除于测量。

## 4. 数据和清单

### 4.1 WikiText-2：受控性能负载

使用 Salesforce/wikitext 的 wikitext-2-raw-v1 test，固定 revision。合并非空行，用本地 tokenizer 编码后截取 token IDs。这是文章续写性能负载，不是任务质量评估。

- 每档准备 64 个确定性样本，保存 ID、切片位置、token IDs、长度和 SHA256。
- 对照配置使用相同输入；三轮重用同一清单，不把重复测量当新增独立样本。
- A/C 的普通片段不同，且显式关闭前缀缓存；不依赖文本差异保证无命中。
- B 组保存公共前缀 P、各独立后缀 S_i、seed 和请求关系；同轮后缀不重复。
- B 组数据准备时核查每批请求两两共享的完整块前缀恰好为 P；后缀不同不等于不会意外共享更多完整块。额外共用块或意外命中均须拒绝该输入构造并重新选片段。
- 不用随机 token 测 MTP 接受率，不把 token 数当字数。

### 4.2 MT-Bench：真实提示词

使用 HuggingFaceH4/mt_bench_prompts 全部 80 条首轮问题，固定 revision 和源顺序。保存 question ID、category、token IDs；本地聊天模板固定 add_generation_prompt=True、enable_thinking=False。只借用问题，不执行评审模型和质量评分。

C 组固定输出 256 tokens；D 组允许正常 EOS，两组分开归档。80 条问题重复三轮仍只有 80 个独立问题。

## 5. 核心实验矩阵

### A. 普通推理：执行模式与固定批次合批

固定 Greedy、MTP 关闭、Prefix Caching 关闭、固定输出长度。

| 编号 | B | 模式 |
|---|---:|---|
| A1 | 1 | eager |
| A2 | 1 | 阶段 CUDA Graph |
| A3 | 4 | eager |
| A4 | 4 | 阶段 CUDA Graph |

每种配置覆盖 WikiText-2 的输入/输出 **256/256、2048/256**，共 **8 个配置**。每配置每轮 64 请求，B=1 为 64 批、B=4 为 16 批，重复三轮。

比较 A2/A1、A4/A3 得 Graph 收益；A3/A1、A4/A2 得合批收益。核心输出：Decode 吞吐、含 Prefill 总吞吐、TTFT/TPOT/完成延迟 p50/p95、资源与实际 batch size。不同长度之间不算优化收益。

### B. Prefix Caching：关闭与预填后的热命中

固定 Graph、Greedy、MTP 关闭、输入 2048、输出 128。

| P 长度 | 独立 S 长度 | 预期复用比例 |
|---:|---:|---:|
| 512 | 1536 | 25% |
| 1024 | 1024 | 50% |
| 1792 | 256 | 87.5% |

每档覆盖 B=1/4、缓存关闭/热命中，共 **12 个配置**。每配置每轮 64 请求，重复三轮。

每个独立批次链路：

1. 引擎 idle，清空逻辑前缀缓存，重置测量计数；不重建已预热的 GPU 池。
2. 关闭组禁用复用，正式请求预期复用 0 tokens。
3. 热命中组先执行 **P+1 个独立 token**，输出 1 token；确认 P 边界的 KV 及 GDN READY 检查点已发布。
4. seed 完成后开始正式计时，提交 P+S_i；B=4 同时入队四条。
5. 每条实际恢复恰好 len(P) tokens；记录命中量并与关闭组比较 TTFT、吞吐及资源成本。
6. 下一批重新准备；不得借上批后缀获得意外命中。

当前默认 GDN 保留策略仅保留输入末尾附近的完整块检查点；P+1 的 seed 用来在所需边界建立检查点。不得假设完整 2048-token seed 会保留 512/1024 的中间状态。保留策略如改变，另立配置。

seed 成本单独记录，排除热命中性能窗口。结论必须标明“预先建立前缀后的热命中”，不能外推为首次请求收益。

核心输出：TTFT 降幅、总吞吐倍数、实际命中比例、逻辑 KV 共享与 GDN 检查点成本；同时观察 Decode/TPOT 是否退化。

### C. MTP：单请求窗口消融

固定单请求、Graph、Greedy、Prefix Caching 关闭、固定输出、关闭自适应投机。

| 编号 | 设置 |
|---|---|
| C0 | enable_mtp=False，常规 Decode 基线 |
| C1 | MTP k=1 |
| C2 | MTP k=2 |
| C3 | MTP k=3 |
| C4 | MTP k=4 |

每种设置覆盖三档：WikiText-2 **256/256、2048/256**（各 64 请求），MT-Bench 全部 80 首轮问题/固定输出 256。共 **15 个配置**，重复三轮。

- C0 不用“开启 MTP、k=0”替代；目标权重、输入、模板及计时口径一致。
- Scheduler.add() 对实际 MTP 文本请求设置 prefix_cacheable=False；MTP 与前缀复用分别测试，不设置联用配置。
- 核实 k=1/2/3/4 的 verify、draft/sync Graph 覆盖，特别检查 k=3；记录 fallback。
- 保存实际 proposed/accepted/committed，区分尾部窗口、EOS、容量缩减、k=0 和无 plan 回退。
- 独立 profiling 至少覆盖预先保存 ID 的 8 条 MT-Bench 问题，输出 256，覆盖 C0～C4；不把诊断耗时当正式吞吐。
- 核心输出：Decode 与总吞吐提升、TTFT/TPOT、接受率/接受长度/提交长度、分阶段开销和显存/投机槽。

## 6. 补充实验

### D. 真实问题、正常 EOS、持续并发

固定 Graph、普通 Decode、Prefix Caching 关闭，使用 80 条 MT-Bench 问题，输出上限 256、ignore_eos=False。维持 C=1、4，各重复三轮，共 **2 配置、6 trials、480 请求**。

完成一条立即补下一条；有限数据集的初始填充和末尾排空计入整段时间。保存实际在途、等待、驻留和执行 batch；目标 C 不代表全程实际 C。记录 EOS/长度终止数、实际输出量。

本组验证调度器处理不同长度和结束时间的能力，报告总吞吐、TTFT/TPOT/完成延迟 p50/p95、资源与回退。提交模式和停止条件不同，不与 A 的固定批次结果混合。本组不包含正常 EOS 下的 MTP 窗口对照。

### E. 并发扩展、吞吐饱和与资源边界

固定 Graph、普通 Decode、Prefix Caching 关闭、Greedy、WikiText-2 **2048/256**，持续并发。扫描 **C=1、2、4、8、16、32、64**；64 是搜索预算，不是预设引擎上限。

每档每轮至少 **max(64,4*C)** 条确定性输入，预先准备至 256 条并存档；重复三轮。相同档位三轮样本相同；高档位新增 ID、与前 64 样本的覆盖关系和样本数公开，不隐瞒负载组成变化。

每档新建进程，设置在途上限、调度与 Graph batch 上限。GDN slots 按目标并发规划，不固定为 12。按 C、上下文、Graph 保留槽和固定显存预算初始化规划 KV/GDN 池；若自动规划降低驻留容量，明确记录。只在测量前规划，测量期间固定。

提交 C 条不等于执行 C 条。记录实际驻留、Decode batch、等待及抢占；区分 OOM、GDN 槽不足、KV 不足、Graph 不覆盖、实现错误和超时。

预先声明的停止规则：

1. 不可恢复错误或资源不足，保存原因及失败档位后停止扩展；配置错误先修正，不冒充硬件极限。
2. 连续两次并发翻倍后，总生成吞吐汇总增幅均不足 5%，且三轮支持饱和判断，可停止；异常波动不据单轮判定。
3. 到 C=64 仍未触及边界，写“已测试至 64”，不得称最大并发为 64。

分别报告最大测试提交数、实际驻留/执行规模、吞吐饱和区域和资源失败边界。未指定延迟 SLO，不宣称满足 SLO 的最大容量；未定位硬件边界，不宣称 GPU 绝对极限。E 最多 7 档、21 trials，提前停止的后续档位记 not_run 及原因。

E 与 D 的输入、EOS 条件不同，不直接复用结果。不把容量扫描再乘上全部前缀比例、eager 或 MTP 窗口。

## 7. 指标规范

### 7.1 请求级

同进程 perf_counter() 记录 enqueue、first_schedule、first_token_commit、finish，以及实际输出 N。

| 指标 | 定义 |
|---|---|
| TTFT | 1000*(first_token_commit-enqueue)，ms，包含引擎内等待 |
| TPOT | 1000*(finish-first_token_commit)/(N-1)，ms/token；N<=1 为 null |
| 完成延迟 | 1000*(finish-enqueue)，ms |
| 初次调度等待 | 1000*(first_schedule-enqueue)，ms；不等于全部排队时间 |

报告 p50/p95，线性插值，保留逐轮数据。后续调度间隔/抢占等待单独计数。TPOT 分位数来自每请求平均值，不是逐 token ITL。MTP 成批提交不得均匀摊开伪造 ITL；保存每步提交量。TTFT 是 CPU 可见提交，不是首次 GPU kernel 完成或 HTTP 流式延迟。

### 7.2 批次与持续并发吞吐

| 指标 | 分子/分母 |
|---|---|
| 固定批次总生成吞吐 | 整批输出 tokens / 从提交前起点到最后请求完成的墙钟时间 |
| 多批汇总总吞吐 | 各批 tokens 总和 / 各批测量窗口耗时总和 |
| 持续并发总吞吐 | 整段输出 tokens / 从首次提交前起点到最后完成的墙钟时间 |
| 纯 Decode 吞吐 | 连续纯 Decode 阶段提交 tokens / 该阶段墙钟时间 |

**多并发禁止使用 sum(tokens)/sum(各请求延迟)，禁止直接平均请求 tok/s。** 持续并发的补充提交及间隙计入测量；固定批次之间的准备、seed、分词和预热不计入窗口。

固定同形状批次确认全部 Prefill 完成后才进入连续 Decode：从首个 Decode step 开始，到最终输出提交结束；分子只数此阶段实际提交的 tokens，通常每请求扣除 Prefill 首 token。期间若插入重算/抢占 Prefill，纯 Decode 值标不可用，不简单相减伪造纯阶段。

持续并发有 Prefill 插入，主指标是总生成吞吐；可另报“Decode 调度阶段吞吐”=Decode steps 提交 tokens/这些 steps 耗时之和。它只用于路径诊断，不是整段服务吞吐，不与纯 Decode 墙钟口径混合。

单请求摊销 Decode 口径如需展示应另列名称，不能搬到多并发。TPOT p50 不等于总吞吐倒数。

### 7.3 资源、调度、Graph 与 MTP

| 类别 | 必须记录 |
|---|---|
| 显存 | 捕获后基线、运行 allocated/reserved 峰值，标注重置时机 |
| 缓存容量 | KV/conv/SSM 字节数、KV 总块、GDN 总槽、规划输入和实际容量 |
| 逻辑占用 | 活跃 KV、空闲有效前缀 KV；GDN Working/Ready/Speculative/Graph/Free |
| 采样边界 | schedule、执行/提交、批次 idle；不能漏掉 idle 后宣称无保留量 |
| 调度 | 在途/等待/驻留、Prefill/Decode 实际 batch 与 tokens、抢占及回退原因 |
| 命中 | 每请求实际恢复 tokens/边界、恢复tokens/输入tokens；KV-only 不冒充联合恢复 |
| Graph | 各阶段/形状 eligible、replay、fallback、原因及 bucket/padding |
| Draft 接受率 | 验证接受 draft 总数 / 实际提出 draft 总数 |
| 平均接受长度 Z | 验证接受 draft 总数 / 实际提出至少一个 draft 的请求轮数 |
| 投机提交长度 | 上述真正投机轮实际提交 tokens / 同一轮数 |
| MTP 边界 | 验证接受、最终提交 accepted draft、EOS/长度截断分别统计 |
| 其他路径 | k=0、无 plan 回退、失败轮独立计数，不进入真正投机轮分母 |

GDN 状态独立峰值不能直接相加当总峰值，应由同一快照计算。步骤峰值不冒充 GPU 中间瞬时峰值；预分配池变小不直接等于缓存节省显存。详细 profiling 记录 draft、verify、sync、采样、状态 I/O 和主机调度/提交；嵌套 GPU 区间不能直接相加当总耗时，说明包含关系及未归因开销。

真正投机轮仅统计 proposed>0 的轮次；验证接受与最终提交分开记录。

### 7.4 收益和统计

- Graph 提升：100*(Q_graph/Q_eager-1)%；相同 B、负载。
- 合批倍数：Q_B4/Q_B1；相同模式、负载，不预设 4 倍。
- 前缀 TTFT 降幅：100*(1-T_hot/T_off)%；p50/p95 分别计算。
- 前缀总吞吐倍数：Q_hot/Q_off；明确包含 Prefill。
- MTP Decode 提升：100*(Q_k/Q_MTP_off-1)%；同一 Decode 定义。
- 并发增量：100*(Q_2C/Q_C-1)%；同时报告延迟和实际执行规模。

汇总吞吐先分别求 tokens、时间之和再相除，不平均各轮 tok/s。分位数可汇集请求，但报告独立样本数及重复轮次；保留三轮配对收益及范围，不只给最好值，不报告小样本 p99。

## 8. 执行顺序与测量要求

顺序：数据/身份 → 小样本性能试跑（可选） → A → B → C 正式及独立 profiling → D → E → 原始数据重算 → 校验和及报告。

Benchmark 负责吞吐、延迟、执行模式、调度和资源使用的测量。

试跑使用相同配置和较少请求，供估计耗时及检查采集流程，不能充当正式性能数据。每个 trial 独立进程、串行使用 GPU。崩溃、OOM、超时和损坏的测量记录必须如实记录；缺失的请求不能计入成功吞吐。超时阈值在运行前确定并归档。

保留输入长度、实际输出量、前缀命中量、实际 batch、Graph replay/fallback、计时及显存记录，以确认测得的确实是所声明的性能配置。

## 9. 执行规模

| 组 | 配置数 | 三轮 trial 数 | 正式请求数 |
|---|---:|---:|---:|
| A | 8 | 24 | 1536 |
| B | 12 | 36 | 2304 |
| C | 15 | 45 | 3120 |
| 核心合计 | **35** | **105** | **6960** |
| D | 2 | 6 | 480 |
| E | 最多 7 | 最多 21 | 最多 2112 |

A～D 合计 37 配置、111 trials、7440 请求。E 全跑为 3*(64*5+128+256)=2112 请求；全计划最多 **44 配置、132 trials、9552 请求**。不含预热、seed 和独立 profiling。E 由停止规则决定实际数量，须说明未执行档位。

## 10. 可复现产物

每次使用独立批次目录。日期/时间使用带 +08:00 偏移的北京时间；monotonic 仅用于进程内差值。

```text
benchmark-results/<run-id>/
  manifest.json             # 协议、环境、模型/源码指纹、Git状态
  config.json               # 实际参数、计时/profiling模式
  dataset-manifest.json     # revision、模板、ID清单、SHA256
  plan.json                 # 实验、对照、重复、顺序与停止规则
  source.zip                # 实际测量源码和依赖声明
  trials/<trial-id>/
    case.json
    requests.jsonl          # 时间、长度、终止、命中
    batches.jsonl           # 批次/持续并发窗口、tokens和耗时
    steps.jsonl             # 调度阶段、batch、Graph与回退
    resources.jsonl         # 含idle的带时间戳资源快照
    summary.json
    run.log
  profiling/                # 独立诊断配置和阶段数据
  execution-status.json     # measured/not_run/failed/timeout及原因
  summary.json
  summary.csv
  report.md
  checksums.sha256
```

记录 GPU/驱动、Python/Torch/CUDA/Triton/FlashAttention/Transformers、模型/模板/tokenizer/数据/源码 SHA256、Git HEAD 与脏状态。不能只用脏工作区 HEAD 代表源码；保存可重建快照。

报告含全部配置及未执行原因、条件、样本数、三轮波动、正负收益、命中和 Graph 覆盖、资源限制、复现命令与局限。声明写清模型/GPU/精度/输入/输出/B或C/缓存状态。B 写热命中，C 的 Y/Z 来自同一 k、同一负载。

## 11. 实现入口与运行门槛

所有入口仅在显式调用 run 时执行 GPU 实验。plan 和 --help 不下载数据、不加载模型。

| 目录 | 作用 |
|---|---|
| execution/ | A：普通推理 eager/Graph、固定批次 |
| prefix/ | B：关闭/热命中及指定边界 seed |
| mtp/ | C：关闭 MTP、k=1～4 和独立 profiling |
| continuous/ | D：正常 EOS 的闭环请求补充 |
| capacity/ | E：容量规划、扩展扫描与停止判定 |
| data/ | 固定 revision 下载、token 化及输入清单 |
| common/ | 配置、子进程、超时、采集、统计、报告 |

运行方法见 [USAGE.md](USAGE.md)。准备本地模型和输入清单后即可运行；可先执行小样本性能试跑。

独立 profiling 不计入正式测量数量。

阶段 profiling 的 CUDA 区间可能包含主机发射间隙，融合状态 I/O 属于 verify，不作为独立可相加的内核时间。

## 12. 性能覆盖范围

完成 A～E 后，可覆盖当前单 GPU 离线项目的 Graph、合批、持续并发、前缀复用、MTP、缓存/显存和并发资源限制。E 只能给测试预算内的扩展曲线及边界，不保证找到硬件绝对极限。

本规范不覆盖 eager 下全部前缀比例、MTP 多并发、MTP+Prefix 联用、开放到达流量、服务 QPS/SLO、p99、多卡、多模态和任务质量。这些结论需要独立实验，不得由本规范结果外推。
