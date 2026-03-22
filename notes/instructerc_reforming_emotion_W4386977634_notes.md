---
title: "InstructERC: Reforming Emotion Recognition in Conversation with Multi-task Retrieval-Augmented Large Language Models"
authors: ["Shanglin Lei", "Guanting Dong", "Xiaoping Wang", "Keheng Wang", "Sirui Wang", "Wang, Sirui"]
year: 2023
publication_date: "2023-09-21"
doi: "10.48550/arxiv.2309.11911"
openalex_id: "W4386977634"
citation_count: 30
status: Read
tags:
  - paper
  - architecture
  - pipeline
  - fine-tuning
  - llm
  - erc
  - soft_labels
  - multi-task
---

# 📝 InstructERC: Reforming Emotion Recognition in Conversation with Multi-task Retrieval-Augmented Large Language Models

> [!ABSTRACT] TL;DR
> **InstructERC** reformulates Emotion Recognition in Conversation (ERC) from a discriminative pipeline into a **generative Large Language Model (LLM)** framework. It utilizes a **retrieval template module** to incorporate supervision signals (instructions, history, label statements) alongside auxiliary tasks (Speaker Identification, Emotion Impact Prediction) for high-accuracy reasoning.

## 🔗 Quick Links
- **PDF**: [[papers/instructerc_reforming_emotion_W4386977634/instructerc_reforming_emotion_W4386977634.pdf|Open Local PDF]]
- **Parsed Text**: [[instructerc_reforming_emotion_W4386977634_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.21437/Interspeech.2024) *(Assuming template update if DOI differed in parsed vs text)*

## 📚 Reading Notes

### 1. Core Objectives
- Previous ERC models separate sentence feature encoding from context modeling. This framework advocates for an end-to-end generative paradigm using unified prompt designs.
- **Label Unification (UIME)**: Pioneeringly aligns diverse datasets using **The Feeling Wheel** into **9 unified classes**: *joyful, sad, neutral, mad, excited, powerful, fear, peaceful, disgust*.
- It addresses the reasoning gap found in generic zero-shot LLMs by providing **historical context windows** and **demonstration examples**.

### 2. Methodological Approach
#### (A) Retrieval Template Module
- **Components**: Instructions, historical dialogues, explicit label menus, and semantic demonstration retrieval via SBERT.
- **Window Constraints**: Context historical window ($w$) optimal at **12 turns** (long contexts aid stability, especially for IEMOCAP).
- **Inference Setup**: Uses **All-labels pairing** during inference (unrestricted), while training relies on exact-match hints to prevent noise.

#### (B) Emotional Auxiliary Alignment Tasks
Joint training weights auxiliary tasks at $\alpha = 0.1$:
- **Speaker Identification**: LLM parameters are preheated on this task to model character speech styles prior to fine-tuning.
- **Emotion Impact Prediction**: Models joint interplay mechanics by omitting the *current* utterance's text and predicting downstream triggers based strictly on history.
- **LoRA Configuration**: Dimension $r=16$, adapter layers inserted after self-attention, trained on causality buffers (e.g., LLaMA-2).

### 3. Key Findings
- **SOTA Performance**: LoRA fine-tuning on decoders hit SOTA benchmarks on **IEMOCAP (71.39%)** and **MELD (69.15%)**.
- **PEFT over Full Tuning**: Parameter-efficient LoRA beats all-parameters fine-tuning on dialog benchmarks. Full parameter tuning causes **overfitting within 1-3 epochs**, proving adapters prevent dilution in short-round environments (e.g., MELD).
- **Unified Robustness**: Evaluated on UIME using Total vs Ratio Mixing; Ratio mixing preserves class balances better, proving open-domain scaling keeps performance within 1-2% of single-domain peaks.

### 4. Application Strategy (Prompt-Based / Agentic)
- **RAG for Ambiguity Resolution**: Semantic matching can retrieve previous examples of high-ambiguity or sarcasm to prompt an LLMs reasoning layer when multi-modal labels conflict.
- **Auxiliary Multi-Task Schemes**: Auxiliary tasks structured as descriptions of Speaker intention or Modality Adjudication guide multi-task backbones on zero-shot inference setups.
- **Context-Window tuning**: Replicating a historic contextual threshold ensures memory weights aren't diluted over long multi-party dialogues.

## 🕸️ Relations
- **Builds on**: [[RoBERTa]], [[COSMIC]], [[Retrieval-Augmented Generation (RAG)]], [[LoRA]]
- **Relevant to**: [[Emotion Recognition in Conversation (ERC)]], [[Instruction Tuning]], [[Multi-task Learning]], [[Speaker Modeling]], [[Label Space Unification]], [[Soft Labels]]
- **Connects with**: [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (distributional supervision) and [[Emotion-LLaMA]] (multimodal backbones)