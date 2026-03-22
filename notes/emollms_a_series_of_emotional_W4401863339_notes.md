---
title: "EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis"
authors: ["Zhiwei Liu", "Kailai Yang", "Qianqian Xie", "Tianlin Zhang", "Sophia Ananiadou"]
year: 2024
publication_date: "2024-08-24"
doi: "10.1145/3637528.3671552"
openalex_id: "W4401863339"
citation_count: 53
status: Read
tags:
  - paper
  - emotional_llm
  - instruction_tuning
  - affective_regression
  - benchmark
---

# 📝 EmoLLMs: A Series of Emotional Large Language Models for Comprehensive Affective Analysis

> [!ABSTRACT] TL;DR
> This paper presents **EmoLLMs**, a series of instruction-tuned LLMs (based on LLaMA2, OPT, BLOOM) designed for comprehensive affective analysis, explicitly bridging **classification** and **regression** (e.g., emotion intensity, sentiment strength). Using a custom 234K instruction dataset (**AAID**) and a 14-dataset evaluation benchmark (**AEB**), EmoLLMs outperform standard open-source models and match the performance of ChatGPT/GPT-4 on generalized, non-Twitter datasets.

## 🔗 Quick Links
- **Parsed Text**: [[emollms_a_series_of_emotional_W4401863339_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1145/3637528.3671552)

## 📚 Reading Notes

### 1. Core Objectives
-   Standard LLM setups overlook **affective regression tasks** (predicting continuous scalar values for **sentiment strength** or **emotion intensity**), which provide fine-grained affective features for downstream systems.
-   Traditional fine-tuned PLMs (like BERT) lack complex compositionality context, dropping accuracy rapidly on multi-task bundles.
-   The absolute gap between open-source models and static annotators (VADER, TextBlob) is large; structured instruction sets can bridge this comprehensively.

### 2. Methodological Approach
#### (A) AAID (Affective Analysis Instruction Dataset)
-   Based on **SemEval-2018 Task 1** using 5 main bundles:
    -   `EI-reg`: Emotion Intensity Regression (range [0,1]).
    -   `EI-oc`: Ordinal Classification of Intensity (0: No emotion - 3: High amount).
    -   `V-reg`: Valence (Sentiment) Regression (range [0,1]).
    -   `V-oc`: Ordinal Classification of valence (range [-3, 3]).
    -   `E-c`: Multi-label classification (11 categories e.g., anger, joy).
-   Uses template scaling with **10 distinct prompts per task** for robustness.

#### (B) EmoLLMs Setup
-   Backbones: **LLaMA2-7B/13B**, OPT-13B, BLOOM-7B.
-   Tuning: Standard multi-task instruction following, treating the numerical scores and category descriptors as autoregressive text generation outputs (e.g., `"Intensity score: 0.896"`).
-   Optimization: 3 epochs, AdamW, early stopping thresholds with DeepSpeed support.

### 3. Key Findings
-   **SOTA Metrics**: EmoLLaMA-chat-13B performs best on internal test bundles (AEB-1), pushing `EI-reg` to **0.831** Pearson scores (exceeding leaderboard caps).
-   **Zero-shot Instability Without Tuning**: Zero-shot Falcon / standard LLaMA do not align mathematically to float scoring constraints; instruction headers force scalar conformity.
-   **Generalization Dynamics**: Models like EmoLLaMA-chat-7B outperform 13B variants in unseen benchmark groups (AEB-2, e.g., Amazon, Movies reviewers, GoEmotion), likely because larger models drift into overfitting during strict list-generation fine-tuning intervals.

### 4. Application Strategy
-   **Zero-shot vs Dense Demonstration cascades**: GPT-4 handles interval regressions well in zero-shot prompts but performs less reliably when generic few-shot index demos are forced, demonstrating high response sensitivity to demo structures for scalar tasks. Optimal cascades should use strict demonstration bounds or clean instruction guards.

## 🕸️ Relations
-   **Builds on**: [[SemEval-2018 Task 1]]; [[SentiBERT]]
-   **Relevant to**: [[Emotion Intensity]]; [[Multi-Task Learning]]; [[Soft Labels]]
-   **Pairs well with**:
  - [[LaERC-S: Improving LLM-based Emotion Recognition]] (Speaker characteristics integration)
  - [[InstructERC: Reforming Emotion Recognition in Conversation]] (Prompt demonstration framework comparisons)
