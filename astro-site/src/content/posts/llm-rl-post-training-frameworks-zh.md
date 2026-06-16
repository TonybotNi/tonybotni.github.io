---
title: "主流 LLM RL 后训练框架的系统设计对比"
date: "2026-06-16"
description: "从 rollout 架构、async 控制、partial rollout、权重同步和平台化取舍角度，对比 verl、SLIME、AReaL 与 ms-swift。"
author: "Jingchen Ni"
tags: ["LLM", "RLHF", "Systems", "Post-training"]
image: "/images/llm-rl-post-training-frameworks-cover.svg"
---

LLM RL 后训练里真正有意思的问题，已经不只是“该用哪个 objective”。PPO、GRPO、GSPO、DAPO、RLOO 以及各种变体当然重要，但到了 frontier 规模，系统往往在别的地方决定成败：rollout 吞吐、训练权重到推理侧的移动、长尾样本调度、stale policy 数据治理，以及围绕 policy 搭出来的 serving topology。

如果把比较范围严格限定为**大规模 RL 后训练的系统工程**，四个开源框架呈现出四条很清楚的设计路线。

- **verl** 是通用抽象路线：宽 worker/engine 模型、默认偏同步主路径、以及实验性的 fully async recipe。
- **SLIME** 是 backend-native 路线：Megatron + SGLang 深度耦合，把 router、PD disaggregation、delta weight sync、partial rollout、dynamic sampling、speculative decoding 都做成一等系统工具。
- **AReaL** 是 async-first 路线：rollout 与 training 完全解耦，rollout 可中断，trainer 持续消费 replay buffer，controller 显式治理 staleness。
- **ms-swift** 是产品平台路线：把 RL 吸收到完整模型生命周期平台里，提供 vLLM colocate/server mode、Megatron-Ray 编排、LoRA-only sync、mismatch diagnostics 和广泛模型覆盖。

我的结论不是某一个框架绝对胜出。更可能成为标准的是一种 **bounded-async disaggregated architecture**：AReaL 的 fully async 总体方向，叠加 SLIME 的 backend-native rollout 优化、verl 的 backend 抽象，以及 ms-swift 的产品级 mismatch correction、adapter sync 和 orchestration。

## 对比总表

| 框架 | 训推协同架构 | Sync / async 机制 | Partial rollout | 推理引擎与权重同步 | 分布式训练与调度 | 官方性能信号 | 相比 verl 的主要创新 | 适用场景与局限 |
|---|---|---|---|---|---|---|---|---|
| **verl** | 默认 worker 抽象是 `ActorRolloutRefWorker`，可共置 actor、rollout 与 optional reference。Critic、reward、standalone reference 可作为独立 `TrainingWorker`。实验性 `fully_async_policy` 可把 `Trainer` 与 `Rollouter` 按 node/GPU group 独立部署。 | 主流路径仍接近同步 PPO/GRPO：生成一个 batch、训练、同步权重、重复。`fully_async_policy` 提供 stream off-policy、async stale-sample、async + partial-rollout，并用 `staleness_threshold` 与 `trigger_parameter_sync_step` 控制新鲜度。 | 主要在 `fully_async_policy` 中支持，通过 `sleep()` / `resume()` 保存未完成 rollout，并在参数同步后继续。公开最清晰路径是 Megatron/FSDP + vLLM server。 | 一线后端包括 vLLM、SGLang、HF Transformers。Rollout weight loading 支持 `auto`、`megatron`、`dtensor`；rollout 可 `free_cache_engine`；fully async 用 NCCL `ParameterSynchronizer`；HybridFlow 核心是 `3D-HybridEngine` 做 train-generate resharding。 | 支持 FSDP、FSDP2、Megatron-LM。Worker 调度用 `RayWorkerGroup`；较新的 engine registry 还接入 Automodel、TorchTitan、VeOmni。 | HybridFlow 报告相对早期 baseline 有 1.53x-20.57x throughput 提升。Fully async recipe 报告 7B DAPO 在 32/64/128 GPUs 上约 2x 改进。没有统一公开 tokens/s 面板。 | 核心创新是宽泛可复用抽象：hybrid single-/multi-controller execution 与 `3D-HybridEngine`，而不是最深的 async。 | 适合需要同时保留 FSDP/Megatron 和 vLLM/SGLang 选择权的团队。局限是默认路径仍比 AReaL 更同步，fully async 仍更像 recipe。 |
| **SLIME** | 核心是 Megatron training、data buffer、SGLang router/engines rollout。多模型配置可把 actor/ref/reward 放到不同 router 或 server group；只有 `update_weights: true` 的 actor 类 group 接收更新。 | 基础流程是 batched rollout -> train。`train_async.py` 用 Ray async 重叠当前训练和下一轮 rollout。Fully async example 用后台 `AsyncRolloutWorker` 跨 rollout boundary 保持 in-flight queue。公开文档更强调 queue warmth 和 non-blocking scheduling，而不是形式化 staleness governance。 | 标准 dynamic sampling 路径中是一等能力：被中断样本进入 data buffer，下一轮恢复。但 fully async example 还没与 `ABORTED` trajectory 的同一恢复语义完全统一。 | 核心后端是 SGLang。vLLM-native 路径通过生态项目 vime 出现，而不是 SLIME 默认。Delta Weight Sync 支持 NCCL/disk、full/delta、sparse changed positions + values。另有 PD disaggregation、session affinity、speculative decoding、FP8 rollout、FP8 KV cache。 | 主栈是 Megatron + SGLang + Ray。SLIME 有意透传 Megatron 与 SGLang 原生参数，而不是藏在宽泛 backend abstraction 后。 | 公开资料强调大规模 examples、profiling、tracing、fault tolerance。Examples 覆盖 8xH100、64xH100、128xH100 级别，但没有标准化 throughput/GPU-utilization benchmark 表。 | 创新是更窄但更深的 Megatron + SGLang 路线：router、PD、delta sync、session affinity、dynamic sampling、partial rollout、speculative decoding 都贴近 serving backend。 | 适合已押注 Megatron + SGLang、需要长 context 或 agentic rollout 的团队。局限是 backend 选择窄，async/staleness 理论化程度不如 AReaL。 |
| **AReaL** | 设计上就是 fully asynchronous, disaggregated：Rollout Controller、Interruptible Rollout Workers、Reward Service、Trainer Workers、replay buffer、update-weight flow。项目还延伸到 online proxy 与 agentic continuous RL。 | Async 是默认哲学。`rollout.max_head_offpolicyness > 0` 启用 async，`=0` 退回 sync。Rollout 持续生成，trainer 攒够 batch 即更新，controller 优先消费旧 trajectory，并拒绝会违反 freshness 约束的新请求。 | 核心能力。Partial rollout 意味着单条 trajectory 可跨多个 policy version。论文术语 `interruptible generation`：新权重到来时中断 decode，丢弃旧 KV cache，用新权重重新 prefill，再继续生成。 | 一线后端是 SGLang 和 vLLM。论文多用 SGLang，部分大规模 case 用 vLLM。权重更新默认 NCCL，OOM 时可 disk fallback。稳定公开 PD disaggregation 还不如 SLIME 核心。 | 支持 Megatron、PyTorch FSDP、PyTorch Archon。Quickstart 支持 local/Ray；agent proxy workflow 支持 local/Slurm。Megatron/Archon 支持 PP/EP，FSDP 支持 1D sequence packing。 | 四者中公开 benchmark 最完整：最高 2.77x speedup，最高 2.57x effective throughput，近线性 scaling 到 512 GPUs，dynamic batching 约 30% gain，interruptible generation 在 1.5B/7B 上 12%/17% gain，8-GPU effective throughput 从 27.1k 到 52.0k。 | 创新是把 fully async 做成系统契约，而不是 optional recipe，同时包含 staleness control、interruptible generation、parallel reward service、online continuous RL。 | 适合长 CoT、代码验证、tool-using agents、重尾 decode。局限是 control plane 更复杂，部分 serving-side 深优化仍在演进。 |
| **ms-swift** | 主路线不是显式 actor/ref/critic service graph，而是 trainer + vLLM rollout acceleration + Megatron/Ray 工程化。GRPO 支持 colocate/internal 与 async external/server。Megatron-Ray 用 YAML 声明 train/rollout/teacher GPU group。 | 支持 synchronous / asynchronous vLLM inference acceleration。`async_generate` 使用上一轮更新后的模型采样，是保守 near-on-policy async，而不是 AReaL 式 replay-buffer controller。 | 在公开 RL 文档中没看到 first-class partial rollout。系统更强调 `dynamic_sample`、`overlong_filter`、`steps_per_generation`、`async_generate`、multi-turn scheduler 和 rollout mismatch correction。 | RL rollout 主要是 vLLM server 或 colocate mode。LoRA 可用 `vllm_enable_lora` 做 adapter-only sync；full-parameter server mode 可 bucketed sync；ZeRO-3 有 layer-wise gather 与 `move_model_batches` 类 OOM 缓解。平台级 inference/deploy/eval 还支持 SGLang 和 LMDeploy。 | 两层路线：HF/TRL 风格 GRPO/PPO/DPO + Deepspeed，以及 Megatron-SWIFT 的 TP/PP/CP/EP。Megatron + Ray GRPO/GKD 加入声明式资源分配和 colocate / dedicated rollout GPU group。 | 公开资料更强调功能矩阵和实验记录，而不是系统论文式 benchmark 表。 | 创新是产品化：模型/多模态覆盖、Megatron parallelism、Ray/YAML deployment、LoRA-only sync、router replay、training-inference mismatch diagnostics、multi-turn scheduler。 | 适合需要一个覆盖 SFT、preference、RL、eval、deploy、多模态 workflow 的完整模型平台。局限是 RL 系统拆解不如 verl/AReaL 清晰，async 深度不如 AReaL，rollout backend 优化不如 SLIME SGLang-native。 |

几个框架的高层 topology 用图看更直观：

![Framework topology comparison](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/llm-rl-post-training-frameworks-topologies.svg)

## verl：先做通用抽象

verl 的核心系统思想来自 HybridFlow：RLHF/post-training 是分布式 dataflow 问题，不是单个 trainer script。HybridFlow 用 hybrid single-controller / multi-controller orchestration 协调 distributed training 与 distributed generation，`3D-HybridEngine` 则专门处理 train-generate phase switch：actor 参数 resharding、避免冗余显存、减少通信。

在公开框架里，这套思想落到 engine-worker abstraction 上。`ActorRolloutRefWorker` 可以共置 actor、rollout 与 optional reference；critic、reward、standalone reference 可是独立 `TrainingWorker`。这让默认部署既能 colocate，也能做 disaggregated worker groups，同时保持统一心智模型。

默认 loop 仍是同步味道：rollout 产出 batch，reward/ref/critic 计算完成，PPO/GRPO update 运行，然后 actor 权重 export 或 reload 到 rollout engine。新的 `fully_async_policy` recipe 改变了这一点：Rollouter 生成样本并写入 `MessageQueue`；Trainer 按 `require_batches * ppo_mini_batch_size` 连续消费；训练推进后由 `ParameterSynchronizer` 做权重同步；`staleness_threshold` 限制旧样本比例。

关键在于，verl 的 async 路径不是重写框架，而是 worker abstraction 的扩展。这既是优势也是限制。它给研究团队一条从稳定同步训练迁移到 bounded async 的路径，但 async semantics 还没有像 AReaL 那样成为默认契约。

Partial rollout 是 verl 最近很值得看的系统能力。公开 recipe 在 parameter synchronization 周围使用 `sleep()` / `resume()`：未完成 rollout 可被暂停，权重同步后继续生成。这直接针对长尾样本 latency。Caveat 是公开支持最清晰的路径仍是 Megatron/FSDP + vLLM server；SGLang rollout 已存在，但 SGLang partial rollout 和 server-based RL 的公开成熟度较低。

当研究面很宽时，verl 是很好的选择：FSDP 或 Megatron，vLLM 或 SGLang，PPO/GRPO/GSPO/RLOO，colocate 或 disaggregate。它的限制也是通用性的代价：如果团队已经确定要最深的 SGLang-native rollout topology，SLIME 暴露更多 serving internals；如果团队想要 async-first governance，AReaL 更果断。

## SLIME：Backend-native 深优化

SLIME 的出发点不同：大规模 RL 的高性能主路径是 Megatron training + SGLang rollout，中间用 data buffer 连接。它不试图抽象掉 Megatron 和 SGLang，而是直接拥抱它们。

Data-buffer-centered 设计很关键。Training 从 buffer 消费样本；rollout 把生成数据和 reward/verifier feedback 写回 buffer；custom workflow、prompt initialization、tool use、agentic feedback 都可以接进去，而不需要把 trainer 改造成 serving system。

Rollout 侧明确是 serving-shaped。`--sglang-config` 可以把 actor、reference、reward 放到不同 router 或 server group。只有 `update_weights: true` 的 group 接收训练更新，reference/reward 可以 frozen。Placeholder groups 可以预留 GPU。PD disaggregation 可以拆 prefill 和 decode worker，并按 group 覆盖 TP、`chunked_prefill_size`、`mem_fraction_static`。

SLIME 的 async 要分层理解。基础层仍是 batched rollout/training。`train_async.py` 通过 Ray async 重叠当前训练和下一轮 rollout。Fully async example 用后台 `AsyncRolloutWorker` 跨 rollout boundary 保持 in-flight queue。这更像 pipeline 和 scheduling optimization，而不是 AReaL 那种形式化 staleness-controlled replay system。

Partial rollout 是 SLIME 主路径的一等能力。如果 dynamic sampling 或 reward filter 提前终止大量样本，`--partial-rollout` 会保存半生成样本并在下一轮恢复。自定义 `--buffer-filter-path` 可决定如何选择缓存样本。当 accepted prompt groups 足够时，也可以 early stop 当前 sampling round，不再等待慢样本或无用样本。这些细节对 invalid、overlong、zero-variance group 很多的 RL workload 很重要。

Caveat 是 fully async example 尚未完全共享同一套恢复语义。对于 `ABORTED` trajectory，公开文档描述的是重新入队，而不是 fully wired partial-rollout-style resume。因此准确评价是：主路径支持 partial rollout；fully async demo 还没和它完全统一。

SLIME 的 rollout optimization surface 是四者里最深的。它暴露 dynamic sampling、partial rollout、PD disaggregation、session-affinity routing、speculative decoding、BF16 training + FP8 inference、FP8 KV cache、SGLang-native 参数透传。Session affinity 对 multi-turn agents 尤其重要，因为同一 `session_id` 固定到同一 worker，可提升 prefix cache locality。Speculative decoding 路径也考虑 RL 中 draft/target distribution drift 会侵蚀收益，因此在线训练 draft model 或 MTP layers 也成为系统故事的一部分。

最有辨识度的是 Delta Weight Sync。它不是每步广播全模型，而是保留上次 broadcast 的 pinned-CPU snapshot，只传 changed byte positions 和 values。它支持 NCCL 与 disk transport，也考虑跨数据中心共享文件系统这类极端 disaggregated 场景。355B 级例子里约 3% delta density、约 5GB update 的量级说明了一点：训推物理解耦后，权重移动就是一等瓶颈。

SLIME 的强项也是边界。组织已经选择 Megatron + SGLang、想把 rollout serving 推到很深时，它非常合适；如果平台必须 backend-neutral，就没 verl 舒服。

## AReaL：Fully Async 作为系统契约

AReaL 是 async 方向最明确的系统表达。它的起点是同步 RL 的两个结构性瓶颈：generation 必须等待 batch 中最长序列；把 generation 分摊到更多设备后，每 GPU decode batch 可能变小，进而掉到 memory-IO-bound。AReaL 的答案是完全解耦 generation 与 training。

系统有四个核心组件：Rollout Controller、Interruptible Rollout Workers、Reward Service、Trainer Workers。Rollout workers 持续生成；reward services 打分；controller 把 trajectories 写入 replay buffer 并管理请求提交；trainer 攒够数据就更新，然后把新权重推回 rollout workers。

关键 primitive 是 `update_weights`。新参数到来时，rollout worker 可以中断正在进行的 generation，丢弃旧权重下的 KV cache，用新权重重新 prefill，再继续 decode 未完成样本。因此 AReaL 的 partial rollout 不是调度小技巧：一条 trajectory 可以跨多个 policy version，staleness/off-policyness 成为系统级不变量。

AReaL 的 async control 是显式的。`rollout.max_head_offpolicyness > 0` 启用 async，`=0` 回到 sync。Controller 跟踪 policy version 和生成样本，训练时优先消费旧 trajectory，并在新请求会违反 staleness limit 时拒绝提交。这不是简单“用上一轮模型样本训练”；queue submission、data consumption order、update cadence 都参与 freshness governance。

公开 benchmark 也很完整：最高 2.77x speedup，最高 2.57x effective throughput，近线性 scaling 到 512 GPUs，dynamic batching 约 30% gain，interruptible generation 在 1.5B/7B 上有 12%/17% gain。端到端训练小时数也具体：1.5B math 从 33.6h 到 14.8h，7B 从 52.1h 到 25.4h，14B coding 从 44.4h 到 21.9h，32B coding 从 46.4h 到 31.1h。8-GPU academic setup 中 effective throughput 从 27.1k 到 52.0k。

AReaL 还超出了普通 offline RL job。Online proxy 允许外部 agents、annotators 或 OpenAI-compatible clients 通过 gateway 交互；token-level interaction data 被收集进 RL buffer；数据足够后触发训练；updated model 再加载回 inference。这更接近 continuous RL service，而不是 batch training script。

Serving backend 足够成熟但还不完整。AReaL 支持 SGLang 和 vLLM，论文里大量使用 SGLang，部分大规模 setting 用 vLLM。权重更新默认 NCCL，OOM 时可 disk fallback。但稳定 PD disaggregation 还不像 SLIME 那样核心。成本是复杂度：采用 AReaL 也意味着采用 replay governance、staleness policy、interruption semantics 和更重的 control plane。

## ms-swift：RL 系统能力的平台化吸收

ms-swift 首先是完整大模型平台，而不只是 RL systems framework。它覆盖 training、inference、evaluation、quantization、deployment，支持 600+ LLM、400+ MLLM 以及大量 preference/RL 方法。缩小到 RL 系统视角，它更像 HF/TRL trainer + vLLM rollout acceleration + Megatron parallelism + Ray orchestration。

两个关键 deployment mode 是 colocate/internal 与 async external/server。Colocate mode 中 training 和 vLLM inference 共享 GPU，依靠 `sleep_level`、`offload_model`、`offload_optimizer`、`vllm_gpu_memory_utilization` 缓解显存压力。External mode 中，`swift rollout` 启动 vLLM server，training 连接它。Megatron-Ray 进一步用声明式方式配置 train、rollout、teacher 资源。

ms-swift 的 Ray 设计很务实。文档明确说，因为复用了许多 `transformers` 和 `trl` 实现，把所有东西拆成 veRL/ROLL 式 Ray roles 会不自然，也会让非 Ray workflow 被迫围绕 Ray 组织。因此它使用 decorator/function-level dispatch。结果是 RL service graph 没那么干净，但更容易接入现有训练平台。

它的 async 语义是保守的。`async_generate` 使用上一轮更新后的模型采样，引入的是一步 staleness，而不是宽 replay-buffer controller。该模式也不支持 multi-round scenarios。为了补偿 mismatch，ms-swift 暴露 `importance_sampling_level`、`rollout_importance_sampling_mode`、`rollout_importance_sampling_threshold`，以及 KL、PPL、chi-square、ESS 等 `rollout_correction` 指标。如果 AReaL 在 controller 层控制 freshness，ms-swift 更像在训练 objective 与 diagnostics 层暴露 mismatch。

Rollout 优化方面，ms-swift 更像参数丰富的生产平台，而不是系统论文。它提供 `dynamic_sample`、`max_resample_times`、`overlong_filter`、`steps_per_generation`、multi-turn scheduler、vLLM prefix caching、GPU sleep/offload、通过 `vllm_enable_lora` 做 adapter-only sync、router replay，以及新 Megatron GRPO/GKD 路径里的 LoRA-only sync。我没有在公开 RL 文档里看到可与 SLIME/AReaL 相比的 first-class partial rollout。

Backend 边界也要分清。平台层面，ms-swift 支持 vLLM、SGLang、LMDeploy 做 inference/deploy/eval。当前 RL rollout cluster support 中，vLLM 是主 external backend。所以如果问题是 SGLang-native RL rollout topology，ms-swift 不是最强答案；如果问题是把 RL 产品化到包含 Megatron、Ray、LoRA、多模态和部署的模型平台里，它就很有吸引力。

## 演进时间线

这条时间线说明，系统正在从同步 batch loop 走向 disaggregated、freshness-governed、backend-aware 的 RL 架构。

| 时间 | 代表事件 | 系统意义 |
|---|---|---|
| 2024-09 | HybridFlow / verl 提出 hybrid single-controller、multi-controller execution 与 `3D-HybridEngine`。 | RLHF post-training 从脚本工程进入 dataflow 与 execution-model 设计。 |
| 2025-02 至 2025-05 | AReaL v0.1/v0.2 与论文公开 fully async、interruptible rollout、staleness-aware training、decoupled PPO。 | 完全解耦 rollout/training 成为有 benchmark 支撑的公开系统方向。 |
| 2025 | SLIME 围绕 Megatron + SGLang + Data Buffer 发展，并加入 dynamic sampling、partial rollout、PD、delta sync、speculative decoding。 | Backend-native serving optimization 成为 RL training throughput 的核心。 |
| 2025-11 至 2026-06 | ms-swift 从 GRPO/vLLM rollout 扩展到 Megatron GRPO、Megatron-Ray GRPO/GKD、LoRA-only sync、router replay、多轮训练。 | RL 系统能力开始被吸收到完整模型平台。 |
| 2025-11 至 2026 | verl 发布 `fully_async_policy`；SLIME 公开 fully async rollout example；AReaL 延伸到 online proxy 和 agentic RL。 | Async rollout 成为多个框架都必须提供的能力，只是投入深度不同。 |

## Async Rollout 技术路线

到 2026 年中，这个设计空间已经比较清楚。

![Async rollout route taxonomy](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/llm-rl-post-training-frameworks-async-routes.svg)

| 路线 | 系统形态 | 优势 | 失效点 |
|---|---|---|---|
| Synchronous colocate | 训练和 rollout 共享进程或 GPU pool；生成、训练、同步、循环。 | 最容易复现、debug、保持 on-policy。 | 长尾 decode、长 CoT、多轮交互会在 batch barrier 浪费 GPU 时间。 |
| Synchronous disaggregated pipeline | 训练和 rollout 用不同 GPU pool，但仍按 batch round 推进。 | 训练/推理资源比例可独立调。 | 仍受 rollout round 和 weight-sync boundary 约束。 |
| Pipeline async with bounded staleness | 训练和 rollout overlap；rollout 可在 freshness 限制内跑在前面。 | 可能是近期最容易普及的路线。 | 需要 staleness threshold、partial rollout 语义和 mismatch diagnostics。 |
| Fully async / continuous RL | Rollout 持续流式生成，trainer 消费 replay buffer，controller 管 freshness 和在线数据。 | 长 CoT 与 agentic workload 收益最高。 | Control plane 与分布治理成本最高。 |

纯 synchronous colocate 不会消失。它仍会作为 baseline、debug mode 和小规模实验路径存在。但它很难成为 frontier 长期标准，因为样本长度分布正在变得更重尾。长 CoT、tool use、代码验证、多轮 agent 都会让 batch barrier 越来越贵。

研究前沿更可能走向 AReaL 式 fully async，并带严格 staleness/buffer governance。更广义的开源生态可能采用折中版本：训练与 rollout 物理解耦；允许有限 staleness；支持 partial/interruption/resume；推理引擎暴露 router/cache/PD/spec-decode controls；权重同步支持 full/delta/adapter-only；训练侧有 mismatch diagnostics；资源编排覆盖 Ray/Slurm/local。

## Open Questions

几个边界仍然要写清。第一，公开 benchmark surface 不均衡。AReaL 的系统 benchmark 最完整。verl 报告 HybridFlow 与 fully async recipe 收益，但没有统一 tokens/s dashboard。SLIME 与 ms-swift 有强 examples、features、profiling 资料，但标准化 benchmark 表较少。

第二，每个框架都有边界。AReaL 的 serving-side deep optimization，尤其稳定 PD disaggregation，还在演进。SLIME 的 fully async example 尚未把 `ABORTED` trajectory resume 与主 partial-rollout semantics 完全统一。verl 的 fully async 仍是 recipe 形态。ms-swift 的 RL rollout 路径仍以 vLLM 为中心，尽管平台层面支持更多 inference backend。

第三，在这组框架里，我没有看到 TensorRT-LLM 作为 first-class public RL rollout backend。它未来可能重要，但目前不应算作这四者的主流公开设计面。

## 总结

真正有用的比较不是“哪个框架实现了 GRPO”，而是哪个框架让哪种系统契约变得容易：通用抽象、backend-native rollout、async control，还是平台化落地。最可能成为标准的是 bounded async disaggregation：足够 async，能逃离 rollout barrier；但也足够受控，让 policy freshness、weight movement 和 distribution mismatch 始终可见。

| 维度 | verl | SLIME | AReaL | ms-swift |
|---|---|---|---|---|
| 系统哲学 | 通用抽象 | 窄栈深耦合 | Fully async contract | 产品平台 |
| Training backend | FSDP/FSDP2/Megatron | Megatron | Megatron/FSDP/Archon | HF/TRL、Deepspeed、Megatron-SWIFT |
| Rollout backend | vLLM/SGLang/HF | SGLang-native | SGLang/vLLM | RL rollout 以 vLLM 为中心 |
| Async 深度 | 可选 recipe | Pipeline overlap + async example | 默认架构 | 保守 previous-model overlap |
| Partial rollout | Fully async 路径 | 主路径一等能力 | interruption 核心 | 公开上不是一等能力 |
| Weight sync | Loader formats、NCCL、resharding | Delta sync、NCCL/disk、sparse updates | NCCL update、disk fallback | Full/bucketed、LoRA adapter-only |
| 最适合 | 灵活研究基础设施 | Backend-native rollout scaling | 长尾 agentic RL | 企业级模型平台 |

---

*最终胜出的 LLM RL stack，不会只靠一个漂亮 trainer，而是要把 rollout、freshness 和权重移动作为同一个系统管起来。*
