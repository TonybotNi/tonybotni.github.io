---
title: "A Systems Design Comparison of LLM RL Post-training Frameworks"
date: "2026-06-16"
description: "A deep systems comparison of verl, SLIME, AReaL, and ms-swift across rollout architecture, async control, partial rollout, weight sync, and platform trade-offs."
author: "Jingchen Ni"
tags: ["LLM", "RLHF", "Systems", "Post-training"]
image: "/images/llm-rl-post-training-frameworks-cover.svg"
---

The interesting question in LLM RL post-training is no longer only which objective to use. PPO, GRPO, GSPO, DAPO, RLOO, and their relatives matter, but at frontier scale the system usually fails or succeeds somewhere else: in rollout throughput, train-inference weight movement, long-tail sample scheduling, stale-policy governance, and the serving topology wrapped around the policy.

If we restrict the comparison to **large-scale RL post-training systems engineering**, four open frameworks reveal four distinct design philosophies.

- **verl** is the general abstraction route: a broad worker/engine model, a mostly synchronous default path, and experimental fully async recipes.
- **SLIME** is the backend-native route: Megatron + SGLang coupled deeply enough to expose router, PD disaggregation, delta weight sync, partial rollout, dynamic sampling, and speculative decoding as first-class systems tools.
- **AReaL** is the async-first route: rollout and training are fully disaggregated, rollout can be interrupted, trainers continuously consume a replay buffer, and a controller explicitly governs staleness.
- **ms-swift** is the product-platform route: RL is integrated into a full model lifecycle platform with vLLM colocate/server modes, Megatron-Ray orchestration, LoRA-only sync, mismatch diagnostics, and broad model coverage.

My conclusion is not that a single framework wins outright. The most plausible standard is a **bounded-async disaggregated architecture**: AReaL's fully async direction, SLIME's backend-native rollout optimization, verl's backend abstraction, and ms-swift's product-level mismatch correction, adapter sync, and orchestration.

## Comparison Matrix

| Framework | Training-inference architecture | Sync / async mechanism | Partial rollout | Inference engine and weight sync | Distributed training and scheduling | Public performance signal | Main innovation relative to verl | Best fit and limitation |
|---|---|---|---|---|---|---|---|---|
| **verl** | The default worker abstraction is `ActorRolloutRefWorker`, which can colocate actor, rollout, and optional reference. Critic, reward, and standalone reference can be separate `TrainingWorker`s. Experimental `fully_async_policy` can deploy `Trainer` and `Rollouter` on independent nodes or GPU groups. | The mainstream path remains close to synchronous PPO/GRPO: generate a batch, train, sync weights, repeat. `fully_async_policy` adds stream off-policy, async stale-sample, and async + partial-rollout modes, with `staleness_threshold` and `trigger_parameter_sync_step` controlling freshness. | Supported mainly through `fully_async_policy`, where unfinished rollout can be saved with `sleep()` / `resume()` and continued after parameter sync. The clearest public path is Megatron/FSDP + vLLM server. | First-class backends include vLLM, SGLang, and HF Transformers. Rollout weight loading supports `auto`, `megatron`, and `dtensor`; rollout can `free_cache_engine`; fully async uses NCCL `ParameterSynchronizer`; HybridFlow's core system idea is `3D-HybridEngine` for train-generate resharding. | Supports FSDP, FSDP2, and Megatron-LM. Worker orchestration uses `RayWorkerGroup`; newer engine registry paths also connect to Automodel, TorchTitan, and VeOmni. | HybridFlow reports 1.53x-20.57x throughput improvement over earlier baselines. The fully async recipe reports around 2x improvement for 7B DAPO on 32/64/128 GPUs. There is no unified public tokens/s dashboard. | The key innovation is broad reusable abstraction: hybrid single-/multi-controller execution plus `3D-HybridEngine`, not maximum async depth. | Best for teams that need FSDP/Megatron and vLLM/SGLang optionality. Limitation: the default path is still more synchronous than AReaL, and fully async remains recipe-shaped. |
| **SLIME** | The core is training with Megatron, a data buffer, and rollout with SGLang router/engines. Multi-model config can place actor/ref/reward behind separate routers or server groups; only actor-style groups with `update_weights: true` receive updates. | The base flow is batched rollout -> train. `train_async.py` overlaps current training with next rollout through Ray async. A fully async example keeps a background `AsyncRolloutWorker` and an in-flight queue across rollout boundaries. Public docs emphasize queue warmth and non-blocking scheduling more than formal staleness governance. | First-class in the standard dynamic sampling path: interrupted samples can enter the data buffer and be resumed in the next round. However, the fully async example is not fully unified with the same resume semantics for `ABORTED` trajectories. | Core backend is SGLang. The vLLM-native path appears through vime, an ecosystem project built on SLIME rather than SLIME's default. Delta Weight Sync supports NCCL/disk, full/delta, and sparse changed positions + values. SLIME also exposes PD disaggregation, session affinity, speculative decoding, FP8 rollout, and FP8 KV cache. | Main stack is Megatron + SGLang + Ray. SLIME intentionally passes native Megatron and SGLang parameters through instead of hiding them behind a broad backend abstraction. | Public material emphasizes large-scale examples, profiling, tracing, and fault tolerance. Examples cover 8xH100, 64xH100, and 128xH100 classes, but not a standardized throughput/GPU-utilization benchmark table. | The innovation is a narrower but deeper Megatron + SGLang route: router, PD, delta sync, session affinity, dynamic sampling, partial rollout, and speculative decoding all sit close to the serving backend. | Best for teams already committed to Megatron + SGLang and long-context or agentic rollout. Limitation: backend choice is narrower, and async/staleness theory is less explicit than AReaL. |
| **AReaL** | Designed as a fully asynchronous, disaggregated system with Rollout Controller, Interruptible Rollout Workers, Reward Service, Trainer Workers, replay buffer, and update-weight flow. The project also extends toward online proxy and agentic continuous RL. | Async is the default philosophy. `rollout.max_head_offpolicyness > 0` enables async; `=0` returns to sync. Rollout continuously generates, trainer updates once a batch is available, and the controller prioritizes stale trajectories while rejecting new requests that would violate freshness constraints. | A core capability. Partial rollout means a single trajectory may span multiple policy versions. In the paper's terminology, `interruptible generation` stops ongoing decode when new weights arrive, drops old KV cache, re-prefills under new weights, and continues generation. | First-class backends are SGLang and vLLM. The paper mainly uses SGLang, with vLLM in some larger cases. Weight update defaults to NCCL with disk fallback for OOM cases. Stable public PD disaggregation is not yet as central as in SLIME. | Supports Megatron, PyTorch FSDP, and PyTorch Archon. Quickstart supports local and Ray; agent proxy workflows support local and Slurm. Megatron/Archon support PP/EP; FSDP supports 1D sequence packing. | The most complete public benchmark story among the four: up to 2.77x speedup, up to 2.57x effective throughput, near-linear scaling to 512 GPUs, about 30% dynamic batching gain, 12%/17% interruptible-generation gain on 1.5B/7B, and 8-GPU effective throughput from 27.1k to 52.0k. | The innovation is making fully async the system contract rather than an optional recipe, including staleness control, interruptible generation, parallel reward service, and online continuous RL. | Best for long CoT, code verification, tool-using agents, and heavy-tailed decode. Limitation: the control plane is more complex, and some serving-side deep optimizations are still evolving. |
| **ms-swift** | The main route is not an explicit actor/ref/critic service graph. It is trainer + vLLM rollout acceleration + Megatron/Ray engineering. GRPO supports colocate/internal and async external/server deployment. Megatron-Ray declares train/rollout/teacher GPU groups in YAML. | Supports synchronous and asynchronous vLLM inference acceleration. `async_generate` samples from the previous updated model, so it is a conservative near-on-policy async rather than an AReaL-style replay-buffer controller. | I did not find a first-class public partial rollout interface in the RL docs. The system instead emphasizes `dynamic_sample`, `overlong_filter`, `steps_per_generation`, `async_generate`, multi-turn scheduler, and rollout mismatch correction. | RL rollout is primarily vLLM in server or colocate mode. LoRA training can use `vllm_enable_lora` for adapter-only sync; full-parameter server mode can use bucketed sync; ZeRO-3 has layer-wise gather and `move_model_batches` style OOM mitigations. Platform-level inference/deploy/eval also supports SGLang and LMDeploy. | Two layers: HF/TRL-style GRPO/PPO/DPO with Deepspeed, and Megatron-SWIFT with TP/PP/CP/EP. Megatron + Ray GRPO/GKD adds declarative resource allocation and colocate or dedicated rollout GPU groups. | Public material emphasizes feature matrices and experiment records rather than paper-style system benchmark tables. | The innovation is productization: model/multimodal coverage, Megatron parallelism, Ray/YAML deployment, LoRA-only sync, router replay, training-inference mismatch diagnostics, and multi-turn scheduler. | Best for teams that need one full model platform across SFT, preference tuning, RL, evaluation, deployment, and multimodal workflows. Limitation: RL system decomposition is less clean, async depth is lower than AReaL, and rollout backend optimization is less SGLang-native than SLIME. |

The high-level topology differences are easier to see visually:

![Framework topology comparison](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/llm-rl-post-training-frameworks-topologies.svg)

## verl: General Abstraction First

verl's core system idea comes from HybridFlow: RLHF/post-training is a distributed dataflow problem, not a single trainer script. HybridFlow uses hybrid single-controller and multi-controller orchestration to coordinate distributed training and distributed generation, and `3D-HybridEngine` specifically targets the train-generate phase switch: resharding actor parameters, avoiding redundant memory, and reducing communication.

In the public framework, this becomes the engine-worker abstraction. `ActorRolloutRefWorker` can colocate actor, rollout, and optional reference; critic, reward, and standalone reference can be independent `TrainingWorker`s. This makes the default deployment flexible enough to run colocated or disaggregated worker groups while preserving a unified mental model.

The default loop is still synchronous in spirit: rollout produces a batch, reward/ref/critic computations complete, PPO/GRPO updates run, and actor weights are exported or reloaded into the rollout engine. The newer `fully_async_policy` recipe changes this: Rollouters generate samples and push them into a `MessageQueue`; Trainers continuously consume batches according to `require_batches * ppo_mini_batch_size`; `ParameterSynchronizer` triggers weight sync after some training progress; `staleness_threshold` limits stale samples.

The important point is that verl's async path is not a total rewrite of the framework. It is an extension of the worker abstraction. That is both its strength and its limit. It gives research teams a migration path from stable synchronous training toward bounded async. But it also means the async semantics are not yet the default contract in the same sense as AReaL.

Partial rollout is the most interesting recent systems feature in verl. The public recipe uses `sleep()` / `resume()` around parameter synchronization: unfinished rollout can be suspended, weights can be synced, and generation can continue. This directly targets long-tail sample latency. The caveat is that the clearest public support path is still Megatron/FSDP + vLLM server; SGLang rollout exists, but SGLang partial rollout and server-based RL remain less mature publicly.

verl is the best choice when the research surface is broad: FSDP or Megatron, vLLM or SGLang, PPO/GRPO/GSPO/RLOO, colocate or disaggregate. Its limitation is the price of that generality: if the team already knows it wants the deepest SGLang-native rollout topology, SLIME will expose more serving internals; if the team wants async-first governance, AReaL is more decisive.

## SLIME: Backend-native Depth

SLIME starts from a different premise: for large-scale RL, the highest-value path is Megatron training plus SGLang rollout, with a data buffer between them. It does not try to abstract away Megatron and SGLang. It leans into them.

That data-buffer-centered design is important. Training consumes samples from the buffer; rollout writes generated data and reward/verifier feedback back into the buffer; custom workflows, prompt initialization, tool use, and agentic feedback can all be integrated without turning the trainer into a serving system.

The rollout side is explicitly serving-shaped. `--sglang-config` can place actor, reference, and reward behind different routers or server groups. Only groups with `update_weights: true` receive training updates, while reference and reward can remain frozen. Placeholder groups can reserve GPUs. PD disaggregation can split prefill and decode workers, and group-specific TP, `chunked_prefill_size`, and `mem_fraction_static` can be overridden.

SLIME's async story is best understood in layers. The base layer is still batched rollout/training. `train_async.py` overlaps the current training step with the next rollout through Ray async. The fully async example uses a background `AsyncRolloutWorker` to keep an in-flight queue across rollout boundaries. This is more of a pipeline and scheduling optimization than AReaL's formal staleness-controlled replay system.

Partial rollout is a first-class feature in SLIME's main path. If dynamic sampling or reward filters terminate many samples early, `--partial-rollout` stores half-finished samples and resumes them next round. A custom `--buffer-filter-path` can decide how cached samples are selected. A pragmatic early-stop policy can also stop a sampling round once enough accepted prompt groups exist, rather than waiting for slow or useless samples. This is exactly the kind of detail that matters for invalid, overlong, or zero-variance groups.

The caveat is that SLIME's fully async example does not fully share the same recovery semantics. For `ABORTED` trajectories, public docs describe re-enqueueing rather than fully wired partial-rollout-style resume. So the accurate assessment is: main path supports partial rollout; fully async demo is not yet unified with it.

SLIME's rollout optimization surface is the deepest of the four. It exposes dynamic sampling, partial rollout, PD disaggregation, session-affinity routing, speculative decoding, BF16 training with FP8 inference, FP8 KV cache, and SGLang-native parameter passthrough. Session affinity is especially relevant for multi-turn agents because it keeps the same `session_id` on the same worker and improves prefix cache locality. The speculative decoding path is also RL-aware: draft/target distribution drift can erode gains, so online training of draft models or MTP layers becomes part of the system story.

The most distinctive feature is Delta Weight Sync. Instead of broadcasting the full model every step, delta sync keeps a pinned-CPU snapshot of the last broadcast and transmits changed byte positions and values. It supports NCCL and disk transports, and the docs discuss extreme disaggregated cases such as shared filesystems across data centers. The 355B-scale example, with roughly 3% delta density and around 5GB updates, makes the point: once training and rollout are physically separate, weight movement is a first-class bottleneck.

SLIME's strength is also its boundary. It is excellent if the organization has chosen Megatron + SGLang and wants to push rollout serving hard. It is less ideal if the platform must remain backend-neutral.

## AReaL: Fully Async as the System Contract

AReaL is the most explicit statement of the future async direction. Its starting point is that synchronous RL has two structural bottlenecks: generation waits for the longest sequence in a batch, and distributing generation over more devices can shrink per-GPU decode batches until the workload becomes memory-IO-bound. AReaL's answer is to decouple generation and training completely.

The system has four central pieces: Rollout Controller, Interruptible Rollout Workers, Reward Service, and Trainer Workers. Rollout workers continuously generate. Reward services score outputs. The controller stores trajectories into a replay buffer and governs request submission. Trainers update once enough data is available, then push new weights back to rollout workers.

The critical primitive is `update_weights`. When new parameters arrive, rollout workers can interrupt ongoing generation, discard KV cache computed under the old weights, re-prefill under the new weights, and continue decoding unfinished samples. This is why partial rollout in AReaL is not a scheduling trick: one trajectory can cross multiple policy versions, so staleness and off-policyness become system-level invariants.

AReaL's async control is explicit. `rollout.max_head_offpolicyness > 0` enables async; `=0` reverts to sync. The controller tracks policy versions and generated samples, prioritizes old trajectories during training, and rejects new requests if accepting them would violate staleness limits. This is not merely "train on the previous model's samples." Queue submission, data consumption order, and update cadence all participate in governing freshness.

The public benchmark story is unusually complete. Reported results include up to 2.77x speedup, up to 2.57x effective throughput, near-linear scaling to 512 GPUs, about 30% dynamic batching gain, and 12%/17% interruptible-generation gains for 1.5B/7B models. End-to-end training hours are concrete: 1.5B math drops from 33.6h to 14.8h, 7B from 52.1h to 25.4h, 14B coding from 44.4h to 21.9h, and 32B coding from 46.4h to 31.1h. In an 8-GPU academic setup, effective throughput rises from 27.1k to 52.0k.

AReaL also points beyond offline RL jobs. Its online proxy lets external agents, annotators, or OpenAI-compatible clients interact through a gateway; token-level interaction data can be collected into the RL buffer; training can trigger once enough data accumulates; updated models are loaded back into inference. That is closer to a continuous RL service than to a batch training script.

The serving backend story is solid but not complete. AReaL supports SGLang and vLLM, uses SGLang heavily in the paper, and uses vLLM in some larger settings. Weight update is NCCL by default, with disk fallback for OOM cases. But stable PD disaggregation is not yet as central as in SLIME. The framework's cost is complexity: adopting AReaL means adopting replay governance, staleness policy, interruption semantics, and a heavier control plane.

## ms-swift: Platform Absorption of RL Systems

ms-swift differs because it is first a full large-model platform, not only an RL systems framework. It covers training, inference, evaluation, quantization, deployment, 600+ LLMs, 400+ MLLMs, and many preference/RL methods. When narrowed to RL systems, it is best understood as HF/TRL trainer + vLLM rollout acceleration + Megatron parallelism + Ray orchestration.

The two key deployment modes are colocate/internal and async external/server. In colocate mode, training and vLLM inference share GPUs; memory pressure is managed with settings such as `sleep_level`, `offload_model`, `offload_optimizer`, and `vllm_gpu_memory_utilization`. In external mode, `swift rollout` starts a vLLM server and training connects to it. Megatron-Ray then provides declarative GPU grouping for train, rollout, and teacher resources.

ms-swift's Ray design is pragmatic. The docs note that because it reuses many `transformers` and `trl` implementations, decomposing everything into veRL/ROLL-style Ray roles would be unnatural and would overfit non-Ray workflows to Ray. Instead, ms-swift uses decorator/function-level dispatch. The result is less clean as an RL service graph, but easier to integrate into an existing training platform.

Its async semantics are conservative. `async_generate` samples from the model after the previous update, introducing one-step staleness rather than a broad replay-buffer controller. It also does not support multi-round scenarios in that mode. To compensate, ms-swift exposes mismatch diagnostics and correction knobs: `importance_sampling_level`, `rollout_importance_sampling_mode`, `rollout_importance_sampling_threshold`, and `rollout_correction` metrics such as KL, PPL, chi-square, and ESS. If AReaL controls freshness at the controller layer, ms-swift exposes mismatch in the training objective and diagnostics layer.

On rollout optimization, ms-swift is parameter-rich rather than systems-paper-like. It offers `dynamic_sample`, `max_resample_times`, `overlong_filter`, `steps_per_generation`, multi-turn scheduler, vLLM prefix caching, GPU sleep/offload, adapter-only sync through `vllm_enable_lora`, router replay, and LoRA-only sync in newer Megatron GRPO/GKD paths. I do not see first-class partial rollout comparable to SLIME or AReaL in the public RL docs.

The backend boundary is also important. At the platform level, ms-swift supports vLLM, SGLang, and LMDeploy for inference/deploy/eval. In current RL rollout cluster support, vLLM is the main external backend. So if the question is SGLang-native RL rollout topology, ms-swift is not the strongest answer. If the question is productizing RL inside a model platform with Megatron, Ray, LoRA, multimodal support, and deployment, ms-swift is compelling.

## Evolution Timeline

The timeline shows a shift from synchronous batch loops toward disaggregated, freshness-governed, backend-aware RL systems.

| Time | Representative event | System meaning |
|---|---|---|
| 2024-09 | HybridFlow / verl introduces hybrid single-controller and multi-controller execution plus `3D-HybridEngine`. | RLHF post-training moves from script engineering to dataflow and execution-model design. |
| 2025-02 to 2025-05 | AReaL v0.1/v0.2 and paper expose fully async, interruptible rollout, staleness-aware training, and decoupled PPO. | Fully decoupled rollout/training becomes a public, benchmarked systems direction. |
| 2025 | SLIME develops around Megatron + SGLang + Data Buffer and adds dynamic sampling, partial rollout, PD, delta sync, and speculative decoding. | Backend-native serving optimization becomes central to RL training throughput. |
| 2025-11 to 2026-06 | ms-swift expands from GRPO/vLLM rollout to Megatron GRPO, Megatron-Ray GRPO/GKD, LoRA-only sync, router replay, and multi-turn training. | RL systems capabilities start being absorbed into full model platforms. |
| 2025-11 to 2026 | verl releases `fully_async_policy`; SLIME publishes a fully async rollout example; AReaL extends toward online proxy and agentic RL. | Async rollout becomes a required capability across frameworks, though with different commitments. |

## Async Rollout Routes

By mid-2026, the design space is fairly clear.

![Async rollout route taxonomy](https://pic-1313147768.cos.ap-chengdu.myqcloud.com/Homepage/images/llm-rl-post-training-frameworks-async-routes.svg)

| Route | System shape | Strength | Failure mode |
|---|---|---|---|
| Synchronous colocate | Training and rollout share process or GPU pool; generate, train, sync, repeat. | Easiest to reproduce, debug, and keep on-policy. | Long-tail decode, long CoT, and multi-turn interaction waste GPU time at batch barriers. |
| Synchronous disaggregated pipeline | Train and rollout use separate GPU pools but still advance by batch rounds. | Resource ratios can be tuned independently. | Still waits at rollout rounds and weight-sync boundaries. |
| Pipeline async with bounded staleness | Training and rollout overlap; rollout may run ahead under freshness limits. | Likely the most widely adopted near-term route. | Needs staleness thresholds, partial rollout semantics, and mismatch diagnostics. |
| Fully async / continuous RL | Rollout streams continuously, trainers consume replay buffer, controller governs freshness and online data. | Highest upside for long CoT and agentic workloads. | Highest control-plane and distribution-governance burden. |

Pure synchronous colocate will remain useful as a baseline, debug mode, and small-scale experimental path. It is unlikely to be the long-term frontier standard because sample length distributions are becoming heavier-tailed. Long CoT, tool use, code verification, and multi-turn agents all make batch barriers more expensive.

The research frontier likely moves toward AReaL-like fully async systems with strict staleness and buffer governance. The broader open-source ecosystem likely adopts a bounded version: physical train/rollout separation, limited staleness, partial/interruption/resume, backend-native router/cache/PD/spec-decode controls, full/delta/adapter-only weight sync, mismatch diagnostics, and Ray/Slurm/local orchestration.

## Open Questions

Several limits remain. First, the public benchmark surface is uneven. AReaL gives the clearest systems benchmark story. verl reports HybridFlow and fully async recipe gains, but not a unified tokens/s dashboard. SLIME and ms-swift provide strong examples, features, and profiling material, but fewer standardized benchmark tables.

Second, each framework has an exposed boundary. AReaL's serving-side deep optimization, especially stable PD disaggregation, is still evolving. SLIME's fully async example has not fully unified `ABORTED` trajectory resume with its main partial-rollout semantics. verl's fully async path remains recipe-shaped. ms-swift's RL rollout path is still centered on vLLM despite broader platform inference support.

Third, I do not see TensorRT-LLM as a first-class public RL rollout backend across this group. It may become important later, but it should not be counted as part of the current mainstream public design surface.

## Summary

The useful comparison is not "which framework implements GRPO." It is which system contract each framework makes easy: abstraction, backend-native rollout, async control, or platform adoption. The likely standard is bounded async disaggregation: async enough to escape rollout barriers, but constrained enough to keep policy freshness, weight movement, and distribution mismatch visible.

| Dimension | verl | SLIME | AReaL | ms-swift |
|---|---|---|---|---|
| System philosophy | General abstraction | Narrow and deep backend coupling | Fully async contract | Product platform |
| Training backend | FSDP/FSDP2/Megatron | Megatron | Megatron/FSDP/Archon | HF/TRL, Deepspeed, Megatron-SWIFT |
| Rollout backend | vLLM/SGLang/HF | SGLang-native | SGLang/vLLM | vLLM-centered RL rollout |
| Async depth | Optional recipe | Pipeline overlap + async example | Default architecture | Conservative previous-model overlap |
| Partial rollout | Fully async path | Main path first-class | Core to interruption | Not first-class publicly |
| Weight sync | Loader formats, NCCL, resharding | Delta sync, NCCL/disk, sparse updates | NCCL update, disk fallback | Full/bucketed, LoRA adapter-only |
| Best fit | Flexible research infra | Backend-native rollout scaling | Long-tail agentic RL | Enterprise model platform |

---

*The winning LLM RL stack will not be the one with the fanciest trainer alone; it will be the one that controls rollout, freshness, and weight movement as one system.*
