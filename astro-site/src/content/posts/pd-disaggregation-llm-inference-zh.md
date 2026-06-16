---
title: "把 Prefill 和 Decode 拆开：聊聊大模型推理里的 PD 分离"
date: "2026-06-16"
description: "从 KV Cache 的物理瓶颈讲到 vLLM 与 SGLang 的落地差异，系统梳理 Prefill-Decode Disaggregation 为什么有用、什么时候值得上。"
author: "Jingchen Ni"
tags: ["LLM", "Inference", "Systems", "Serving"]
image: "/images/pd-disaggregation-llm-inference-cover.svg"
---

一次大模型推理，从 API 边界看像是一个操作，但落到 GPU 上，其实藏着两种性格几乎相反的负载。

**Prefill** 会吃下整段 prompt，并行处理所有输入 token，构造初始 KV cache，同时产出第一个输出 token。它喜欢大矩阵乘、足够大的 batch，以及把 GPU 算力拉满。**Decode** 则每次只往外吐一个 token。每一步的计算量很小，但都要反复从 HBM 里读取不断变大的 KV cache。它需要稳定的显存带宽、可预测的调度节奏，以及不要被突然插进来的长 prompt 打断。

把这两段工作放在同一块 GPU 上很方便，但这也意味着永远在妥协。Prefill 是 compute-bound，decode 是 memory-bound。Prefill 想要大 batch，decode 想要稳定的逐 token 节奏。**Prefill-Decode Disaggregation**，也就是 PD 分离，出发点很直接：既然首 token 的工作和下一个 token 的工作会压到不同的物理资源，那就把它们放到不同的 GPU 池里，分别部署、分别调优。

## Prefill 和 Decode 是两种负载

理解 PD 分离的第一步，是先别过早地把它们都叫作“一次推理”。一次请求至少有两个阶段。

Prefill 阶段里，完整 prompt 已经给定。模型可以并行计算 prompt 内所有位置的 attention，GPU 得到的是它最擅长的密集矩阵计算。长 prompt 当然也贵，动辄几百毫秒，但主要贵在“算”：大 GEMM、高 tensor core 利用率、天然适合 batching。

Decode 阶段完全不一样，因为生成是自回归的。在第 $t$ 步，模型只能产出下一个 token。为了算这一个 token，它要读取前面 $t$ 个 token 的 key 和 value。每步真正做的算术很少，但内存访问一点也不少。所以 decode 经常出现 tensor core 没吃满、HBM 带宽先卡住的情况。

一个最短的心智模型是：

| 阶段 | 主要操作 | 物理瓶颈 | 用户感知指标 |
|---|---|---|---|
| Prefill | 并行处理 prompt，生成 KV cache | FLOPs / 算力 | TTFT，也就是首 token 延迟 |
| Decode | 反复读取 KV cache，每步生成一个 token | HBM 带宽 | TPOT / ITL，也就是 token 间延迟 |

![Prefill 和 Decode 的负载分离](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/pd-disaggregation-llm-inference-workload-split.svg)

*图 1。同一次推理的两副面孔：Prefill 把算力打满，Decode 把显存带宽打满。*

我更喜欢的类比是仓库：prefill 像一次性搬一卡车货，重，但工作集中；decode 像反复跑腿，一次只拿一件，但每次都得扫一遍越来越长的历史。一个吃力气，一个吃来回搬运。

这不是概念上的细枝末节。它解释了为什么同一套 batching、placement 和 GPU 分配策略，很难同时服务好两个阶段。

## KV Cache 才是物理根源

Decode 之所以 memory-bound，根子在 KV cache。每处理一个 token，transformer 的每一层都要把对应的 key 和 value 存下来。一个常用的一阶公式是：

$$
\text{bytes} = 2 \times L \times H_{kv} \times D_h \times S \times B \times \text{dtype\_bytes}
$$

这里的 $2$ 代表 key/value 两份，$L$ 是层数，$H_{kv}$ 是 KV head 数，$D_h$ 是 head dimension，$S$ 是序列长度，$B$ 是 batch size。对一个固定模型和 dtype 来说，每 token 的 KV 大小几乎是常数，所以 KV 总量随上下文长度线性增长。

这也是为什么 MQA、GQA、MLA 对 serving 这么重要。它们本质上都在削这个公式里的项：

- **MQA** 让所有 query head 共享一个 KV head。
- **GQA** 按组共享 KV head，把 `num_kv_heads` 砍小。
- **MLA**，也就是 DeepSeek 一类模型用的路线，不再完整存每个 head 的 K/V，而是缓存压缩后的 latent 表示，用时再重建 attention 需要的信息。

拿 Qwen2.5-32B 的 BF16 配置算一笔实账。它有 64 层、40 个 query head、通过 GQA 保留 8 个 KV head，并且 `head_dim = 128`。每 token 的 KV cache 是：

$$
2 \times 64 \times 8 \times 128 \times 2 = 262{,}144 \text{ bytes} \approx 256 \text{ KB}
$$

这意味着一个 8K 上下文的单请求，光 KV cache 就大约 2 GB。32K 上下文就是大约 8 GB。如果它不用 GQA，而是普通 MHA 的 40 个 KV head，每 token KV 会变成 5 倍，大约 1.25 MB，8K 上下文单请求就要约 10 GB。

这串数字写成代码，就是公式的直接翻译：

```python
def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> int:
    # 2 = Key + Value 两份；其余每一项都对应模型或负载维度。
    return 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes


# Qwen2.5-32B with GQA: 64 层、8 个 KV heads、head_dim 128、BF16。
per_token = kv_cache_bytes(64, 8, 128, seq_len=1, batch_size=1)
print(per_token / 1024)                                  # 256.0 KB / token
print(kv_cache_bytes(64, 8, 128, 8192, 1) / 1024**3)     # 8K context = 2.0 GB
print(kv_cache_bytes(64, 8, 128, 32768, 1) / 1024**3)    # 32K context = 8.0 GB
```

现在把这个数字和 decode 接上。每生成一个 token，decode 都会追加新 token 的 K/V，然后让新的 query 去和全部历史 K/V 做 attention。示意代码里，这个访存模式一眼就能看出来：

```python
class KVCacheAttention(nn.Module):
    def forward(self, x, kv_cache=None):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        if kv_cache is not None:
            # Decode：把新 token 的 K/V 追加进不断增长的历史。
            k = torch.cat([kv_cache.k, k], dim=1)
            v = torch.cat([kv_cache.v, v], dim=1)

        kv_cache.update(k, v)

        # 1 个新 query 要和全部历史 K/V 做 attention。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ v
        return out, kv_cache
```

关键不只是 `torch.cat` 这行，而是 `k` 和 `v` 的长度随生成历史增长，并且每一步又要被重新读出来。上下文越长，要搬的数据越多，decode 越慢。

从 roofline 视角看，decode 的算术强度很低。粗略按矩阵-向量模式估算，可能只有 1 FLOP/byte 左右，远低于 H100 约 295 FLOP/byte 的 roofline 脊点，也就是 989 TFLOPS 除以 3.35 TB/s。Decode 远远落在带宽斜坡上；prefill 则因为有更大的矩阵-矩阵计算，更接近算力屋顶。

![Prefill 和 Decode 的 roofline 视角](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/pd-disaggregation-llm-inference-roofline.svg)

*图 2。Roofline 视角下，Decode 因反复整读 KV Cache 落在带宽斜坡，Prefill 更接近算力屋顶。*

所以 KV cache 在 PD 分离里有两重身份。它是 decode memory-bound 的原因，也是 prefill 池和 decode 池之间必须搬运的实体。KV cache 越小，decode 的带宽压力越小，P 到 D 的传输成本也越低。

## 混在一起调度为什么会互相伤害

现代推理引擎通常会做 **continuous batching**。它不是等一个固定 batch 全部结束，而是在运行过程中持续加入新请求，并逐步推进已有请求的 decode。这是提高利用率的大方向，但它也会把 prefill 和 decode 混在同一块 GPU 上。

最典型的问题是 **prefill 抢占 decode**。一个带着长 prompt 的新请求进来，它的 prefill 阶段想独占 GPU 做密集计算，可能持续几百毫秒。已经在流式输出的老请求，本来该稳定吐下一个 token，现在被迫等一拍。用户看到的不是“调度冲突”，而是输出突然顿了一下，也就是 ITL 或 TPOT 出现尖刺。

![长 prefill 导致 ITL 尖刺](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/pd-disaggregation-llm-inference-itl-spike.svg)

*图 3。长 prefill 一插队，正在 decode 的请求就被迫等待，token 间延迟出现尖刺。*

反过来也成立。如果调度器过度保护 decode 延迟，就不得不限制 prefill 的 batch 大小，结果 prefill 吞吐又被拖累。两种负载想要的调度形状完全不同：

- Prefill 想要更大的 chunk 和 batch，好把算力吃满。
- Decode 想要短而稳定的调度间隔，避免 token 间延迟抖动。
- 单块共享 GPU 会把两个 SLO 绑在一起，调一个就牺牲另一个。

**Chunked prefill** 是一条重要的不分离路线。它把长 prefill 切成小块，和 decode step 交错执行，避免长 prompt 一次性霸占 GPU。在单机或中等规模部署里，这经常很实用。但它本质上还是“同一块 GPU 上把时间切得更细”。Prefill 和 decode 依然共享硬件，依然在抢同一份资源。

PD 分离更激进。它认为问题不只是调度粒度不够细，而是两段负载本来就该属于不同资源池。

## PD 分离改变了什么

在 PD 分离的系统里，请求先进入 prefill worker 或 prefill GPU 组。Prefill 侧处理 prompt，生成 KV cache，并产出首 token。随后 KV cache 通过高速互联，比如 NVLink、RDMA、InfiniBand，传到 decode worker。Decode 侧接住 KV cache 后，继续自回归生成，并把后续 token 流式返回给用户。

![PD 分离流水线](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/pd-disaggregation-llm-inference-pipeline.svg)

*图 4。PD 分离架构：Prefill 池算出 KV Cache，经高速互联传给 Decode 池继续生成。*

这样一拆，两个端点的 SLO 就解耦了。Prefill 节点可以围绕 TTFT 和吞吐优化。它可以用更大的 prefill batch、更偏 prompt-heavy 的调度策略，以及更适合密集计算的并行方式。Decode 节点可以围绕 TPOT 和流式稳定性优化。它不再会被突然插进来的长 prompt 打断。

代价是 KV 传输。一个 prompt 的 KV cache 可能是几百 MB，也可能是多个 GB，取决于上下文长度、模型大小、dtype 和 KV 压缩方式。如果互联太慢，传输延迟就会吃掉分离带来的收益。这也是为什么 PD 分离最适合有高速互联的大规模 serving 集群，尤其是对延迟敏感、流量足够大、并且能让 P/D 两边都保持忙碌的场景。

一个干净的理解是：用一次 KV cache 传输的代价，换来 prefill 和 decode 不再共享同一套 GPU 调度。

## vLLM 和 SGLang：骨架相同，契约不同

vLLM 和 SGLang 的公开实现共享同一个大骨架：分开的 prefill 和 decode 实例，中间有 proxy 或 router 做配对和路由，KV cache 通过 RDMA 能力的传输系统跨节点移动。差别在于它们如何抽象“传输”和“握手”。

### vLLM：先做连接器抽象

vLLM 把 PD 分离放在可插拔的 **KVConnector** 抽象下面。核心引擎不把某一个传输后端写死，而是可以接不同的 KV 传输后端。公开材料里常见的路径包括 NIXL、Mooncake、LMCache，以及原生 xPyD 部署里的 P2P NCCL。

典型流程不难理解：

1. Proxy 选择一个 prefill 实例和一个 decode 实例，常见是 1P1D 配对。
2. Prompt 发给 prefill 侧，有些路径会把 `max_tokens` 调成只需要产出首 token。
3. Prefill worker 计算 KV cache。
4. Prefill 侧通过配置好的 connector 把 KV push 给 decode 侧。
5. Decode 侧接着生成，并流式返回剩余 token。

这种设计让 vLLM 更像“引擎 + connector 生态”。好处是后端选择多，能融入一个高速迭代的 serving engine。代价是成熟度和部署复杂度。公开文档里 PD/xPyD 的部分路径仍带实验性质，有些部署还会依赖 llm-d 或 Dynamo 这类外部编排。更准确地说，它主要是调 TTFT 和 ITL 的工具，不是自动提升吞吐的魔法开关。

### SGLang：内建组件和固定握手

SGLang 给出的 PD 形态更内建：proxy、prefill server、decode server，再配 `sglang_router` 做路由、负载均衡和容错。传输引擎通常是 Mooncake 或 NIXL，关键系统原语是 RDMA 和后台非阻塞传输。

最容易被忽略的差别在握手顺序。SGLang 的 PD 路径里，decode 侧会先准备好接收 KV cache 的目标显存。然后它再通知 prefill 侧开始计算，并把 KV 写入已经预留好的 decode 侧内存。也就是说，它不是“prefill 算完再找地方放”，而是“decode 先把坑占好，再让 prefill 来填”。

![vLLM 与 SGLang 的 PD 数据流](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/pd-disaggregation-llm-inference-frameworks.svg)

*图 5。vLLM 和 SGLang 的 P/D 拆分骨架相同，但 KV 传输契约和握手顺序不同。*

SGLang 还暴露了一些真实部署里很关键的工程能力：动态连接、非阻塞 KV 传输、P/D 异构 tensor parallelism，以及 RDMA 传输。异构 TP 很有价值，因为 prefill 和 decode 可能适合不同的并行切分。边界条件是，对非 MLA 模型来说，如果 P/D 的布局不完全一致，可能需要 staging buffer。

两者可以粗略对比如下：

| 维度 | vLLM | SGLang |
|---|---|---|
| 抽象方式 | 可插拔 `KVConnector` 生态 | 内建 proxy、prefill、decode、router |
| 常见传输后端 | NIXL、Mooncake、LMCache、P2P NCCL | Mooncake、NIXL |
| 控制流 | Prefill 算完后 push KV | Decode 先预留显存，再触发 prefill |
| 优势 | 后端选择多，容易融入引擎生态 | PD serving 拓扑和握手更显式 |
| Caveat | 部分 PD 路径仍偏实验 | 栈和协议更 opinionated |

Mooncake 值得单独提一句，因为它代表的是更宽的 KV-cache-centric 路线。它不是把传输只看成点对点拷贝，而是把集群里的 CPU、DRAM、SSD、RDMA 资源池化成全局 KV cache 层。中央调度器可以根据 KV 分布和负载放置请求，可以跨请求复用 prefix KV，也可以只计算未命中的增量。这和 DistServe、Splitwise 属于同一类 serving 设计：系统围绕 KV cache 这个中心对象来组织。

## 代价：传输和 P:D 调度

分离不是免费午餐。它把“同一块 GPU 上的两种负载互相抢资源”，换成了一组分布式系统问题。到底值不值得上，主要看两类成本。

第一类成本是跨节点 KV 传输。一个请求的 KV cache 可能有几百 MB 甚至上 GB，并且会随 prompt 长度、层数、KV head 数、batch size 和 dtype 线性增长。如果网络不能足够快地搬这块 KV，传输延迟就足以吃掉分离省下来的收益。所以生产里的 PD 工程会把大量心思花在分层传输、传输与计算 overlap、按需或选择性搬运，以及更聪明的 placement 上。

第二类成本是 P:D 比例。到底多少 GPU 做 prefill，多少 GPU 做 decode，不是拍脑袋定死的，因为它强依赖线上流量形态。

| 负载形态 | 典型场景 | 配比压力 |
|---|---|---|
| 长输入、短输出 | RAG、文档问答、长 context 摘要 | 更多 prefill capacity |
| 短输入、长输出 | Agent 长链推理、长篇生成 | 更多 decode capacity |
| 混合在线流量 | 真实生产流量 | 动态调整 P:D |

如果比例错了，一边排队，另一边闲置，分离收益就会打折。所以好的 PD 系统不仅需要快传输，还需要理解 P/D 两类瓶颈的 routing、admission control、placement 和 autoscaling。

实用判断很简单：大规模在线 serving、严格延迟 SLO、长上下文或混合流量、高速互联，这些条件越多，PD 分离越值得。小规模单机部署、互联带宽一般、流量不高的场景，continuous batching 加 chunked prefill 往往是更划算的工程解。

## 两个容易混的点

PD 分离是推理阶段优化。它不是 RL 后训练系统里常说的“训练-推理分离”。

在 RLHF、PPO、GRPO、DAPO 和 agentic RL 系统里，人们也会把 **rollout** 和 **training** 拆开。那是更高一层的架构：一个集群负责采样，另一个集群负责更新模型权重，中间还要处理 policy lag、replay buffer、partial rollout 和 off-policy correction。PD 分离可以放在 rollout 服务内部，因为 rollout 本质上是在做推理。但它和 rollout/train 分离不是一回事。前者发生在一次生成请求内部，后者发生在采样和参数更新之间。

普通 teacher-forcing 训练里也没有 PD 分离。SFT 和预训练已经知道完整 target 序列，forward pass 会并行计算所有位置，更像 prefill，而不是逐 token decode。只有当模型在自回归生成未知未来 token 时，decode 阶段才真正出现。PD 只会通过 RL 的 rollout 阶段，间接出现在训练流程里。

## 总结

PD 分离之所以成立，是因为它顺着 LLM 推理的物理形状走：prefill 是 compute-bound，decode 是 memory-bound，而 KV cache 既解释了两者为什么不同，也是两者分离之后必须跨边界搬运的东西。

| 判断维度 | 继续混合调度 | 使用 PD 分离 |
|---|---|---|
| 部署规模 | 单机或小集群 | 大规模 serving 集群 |
| 互联条件 | 普通或带宽有限 | RDMA、NVLink、InfiniBand 级别 |
| 主要手段 | Continuous batching、chunked prefill | P/D 资源池分离，加 KV 传输 |
| 最适合负载 | 中等上下文、延迟要求较宽松 | 长上下文、混合流量、严格 ITL |
| 主要风险 | Prefill 和 decode 互相干扰 | KV 传输和 P:D 调度复杂度 |

关键不是 PD 分离永远更好，而是当“搬一次 KV cache”的代价，小于“强迫两种相反负载共享同一套 GPU 调度”的代价时，它就开始划算。从 continuous batching 到 chunked prefill 再到完整 PD 分离，serving stack 的演进主线很清楚：不断承认 prefill 与 decode 的对立，并用部署能承受的方式把它们解耦。

---

*PD 分离的本质，是推理系统终于承认：首 token 的工作和下一个 token 的工作，根本不是同一种负载。*
