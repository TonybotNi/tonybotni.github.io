---
title: "MG-RWKV: Multi-Grained Context-Aware RWKV for Temporal Forgery Localization"
author: "Jingchen Ni*, Cangjin Yu*, Dan Jiang, Quan Zhang, Keyu Lv, Shannan Yan, Linyue Pan, Ke Zhang, Chun Yuan"
date: "2026-07-01"
journal: "European Conference on Computer Vision (ECCV) 2026 (THU-A)"
external_url: "https://arxiv.org/abs/2607.00902"
image: "/images/mg-rwkv.png"
description: "A multi-granularity RWKV framework for Temporal Forgery Localization that achieves state-of-the-art performance with linear O(T) complexity."
tags: ["Temporal Forgery Localization", "RWKV", "Mixture of Experts"]
---

## Overview

As a co-first author, I proposed MG-RWKV, a multi-granularity framework for Temporal Forgery Localization (TFL) that leverages the data-dependent state evolution of RWKV to achieve efficient full-sequence processing with linear O(T) complexity. It introduces three core innovations: (1) a Bidirectional RWKV architecture that captures bidirectional temporal contexts without quadratic overhead; (2) a Multi-Granularity Mixture of Experts (MG-MoE) that performs dynamic routing over explicit temporal receptive fields, adaptively selecting granularities based on forgery duration to enhance decision interpretability; and (3) Cross-Granularity Consistency (CGC), which aligns adjacent feature pyramid levels through hierarchical scale-wise pairing and spatial boundary-aware weighting to reduce false positives in authentic regions. Extensive experiments on Lav-DF, TVIL, and Psynd demonstrate state-of-the-art performance with low computational cost.

## Links

- [PDF](https://arxiv.org/pdf/2607.00902)
- [arXiv](https://arxiv.org/abs/2607.00902)
