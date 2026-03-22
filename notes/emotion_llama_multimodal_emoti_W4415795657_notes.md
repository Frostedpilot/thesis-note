---
title: "Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning"
authors: ["Zebang Cheng", "Zhi-Qi Cheng", "Alexander G. Hauptmann", "Jun-Yan He", "Zheng Lian", "Yuxiang Lin", "Xiaojiang Peng"]
year: 2024
publication_date: "2024-11-20"
doi: "10.48550/arxiv.2411.13501"
openalex_id: "W4415795657"
citation_count: 0
status: Read
tags:
  - paper
  - multimodal
  - llm
  - instruction-tuning
  - reasoning
  - soft_labels
  - cross_modal
---

# 📝 Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning

> [!ABSTRACT] TL;DR
> **Emotion-LLaMA** aligns audio, visual, and textual modalities into a unified LLM backbone (**LLaMA-2**) using emotion-specific encoders. It introduces the **MERR dataset** (28k coarse, 4.4k fine-grained) auto-annotated via Action Units and unimodal models for description synthesis. The framework trains the model on both **emotional recognition** and **reasoning** (explanations), showcasing superior performance over general MLLMs on contextual and micro-expression understanding.

## 🔗 Quick Links
- **PDF**: [[papers/emotion_llama_multimodal_emoti_W4415795657/emotion_llama_multimodal_emoti_W4415795657.pdf|Open Local PDF]]
- **Parsed Text**: [[emotion_llama_multimodal_emoti_W4415795657_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.48550/arxiv.2411.13501)

## 📚 Reading Notes

### 1. Core Objectives
- The paper addresses the inability of general MLLMs to process audio cues and recognize subtle facial micro-expressions.
- **Data Breakdown**: Introduces the **MERR dataset** consisting of **28,618 coarse-grained samples** (auto-labeled via instruction aligned descriptors) and **4,487 fine-grained samples** (refined by experts for reasoning).
- It shifts from basic feature fusion to **knowledge-level interaction**, focusing on detailed descriptions of facial behavior (AUs), audio context, and text descriptions.

### 2. Architecture and Data Construction
- **Multi-view Encoder**: Combines **HuBERT** (Audio), **MAE** (Facial Static), **VideoMAE** (Facial Dynamics), and **EVA** (Global Scene Context using full 448x448 frames).
- **Peak Frame AU Trigger**: Uses OpenFace to select frames with maximum Action Unit (AU) intensity sums. AUs (e.g., AU-05 eyes widened) are mapped directly to text before synthesis.
- **Linear Projections**: Transforms modality features into a shared space aligned with **LLaMA-2-chat (7B)** tokens.
- **Efficiency Layer**: Fine-tuned using **LoRA** ($W_q$ and $W_v$ matrices) totaling only **34 Million trainable parameters** (0.495% of the total model size).
- **Multi-Task Scheme**: Alternates/parallelizes **recognition** (Predict Label: choose from 9 aligned metrics) and **reasoning** (Explain Answer: analyze clues) using formatted prompt templates: `[INST] <VideoFeature> <AudioFeature> [Task] Prompt [/INST]`.

### 3. Key Findings
- **Superior Benchmark Performance**: Emotion-LLaMA achieved **F1 score of 0.9036 on MER2023-SEMI** and topped scores on EMER (**Clue Overlap: 7.83**).
- **Zero-Shot Generalization**: Achieved peak WAR score of **59.37% on DFEW**, surpassing ChatGPT-4V. Avoided generic safety-related drops (e.g., zero disgust reports common in general MLLMs).
- **Accurate Adjudication**: Qualitative comparison reveals that models lacking full multimodal description integrations (e.g., PandaGPT) incorrectly classify mixed expressions (e.g., covering dissatisfaction with a smile), while Emotion-LLaMA successfully utilizes audio tone details to infer underlying anger.
- **Complementary Encoding**: Ablation studies confirm that combining spatial, temporal, and context encoders yields the highest accuracy (HuBERT+MAE+VideoMAE+EVA = F1 0.8910 in ablation).

### 4. Application Strategy
- **Modality-Specific Description Chaining**: Replicating the MERR pipeline (Scene descriptive model + Speech tone describer) on-the-fly offers a mechanism to produce explanatory context sheets before feeding into reasoning pipelines for zero-shot tasks.

## 🕸️ Relations
- **Relevant to**: [[Emotion Recognition in Conversation (ERC)]], [[Instruction Tuning]], [[Multimodal Fusion]], [[Reasoning Layer]], [[Soft Labels]]
- **Pairs well with**:
  - [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (distributional supervision)
  - [[Third-Person Appraisal Agent: Simulating Human Emotional Reasoning]] (agentic reasoning layer)
  - [[LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics]] (context modeling)