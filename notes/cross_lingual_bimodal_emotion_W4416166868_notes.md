---
title: "Cross-Lingual Bimodal Emotion Recognition with LLM-Based Label Smoothing"
authors: ["Elena Ryumina", "Alexandr Axyonov", "Timur Abdulkadirov", "Darya Koryakovskaya", "Dmitry Ryumin"]
year: 2025
publication_date: "2025-01-01"
doi: "10.3390/app15020701"
openalex_id: "W4416166868"
citation_count: 0
status: Read
tags:
  - paper
  - bimodal
  - cross-lingual
  - label-smoothing
  - mamba
  - soft_labels
  - biformer
---

# 📝 Cross-Lingual Bimodal Emotion Recognition with LLM-Based Label Smoothing

> [!ABSTRACT] TL;DR
> This study utilizes lightweight LLMs (specifically **Qwen3-4B**) as a **"teacher" to generate soft labels** for audio-text bimodal emotion recognition. The architecture integrates a **Mamba-based temporal encoder** with a **BiFormer fusion network** (Transformer-based cross-modal attention). Label smoothing is employed as a form of distribution-based supervision to capture latent emotional co-occurrences and handle ambiguous or compound emotions.

## 🔗 Quick Links
- **PDF**: [[papers/cross_lingual_bimodal_emotion_W4416166868/cross_lingual_bimodal_emotion_W4416166868.pdf|Open Local PDF]]
- **Parsed Text**: [[cross_lingual_bimodal_emotion_W4416166868_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.3390/app15020701)

## 📚 Reading Notes

### 1. Core Objectives
- The paper addresses the challenge of effectively fusing audio and text when emotional signals are subtle or **conflicting across modalities**.
- **Dataset Contrast**: Reconciles **MELD** (Noisy, short utterances: Avg 3.1s, 14 tokens) and **RESD** (Clean, lab-recorded, longer: Avg 6.0s, 24 tokens).
- **Label Alignment**: To enable cross-lingual joint training, emotions were aligned: *Joy* with *Happy*, and *Enthusiasm* with *Surprise*.
- It critiques reliance on hard labels, proposing addressable **soft distributions** provided by an LLM's semantic understanding to guide training.

### 2. Methodological Approach
#### (A) LLM as a Soft Label Teacher (LS-LLM)
- The method uses **lightweight LLMs** (with **Qwen3-4B** performing best at probability _p_=0.2) to generate context-aware "soft" target labels.
- **Prompt Constraints**: Enforced strict rules: output 5-decimal places, make values sum exactly to 1.00000, and strictly list comma-separated: neutral, happy, sad, anger, surprise, disgust, fear. **Assigning 1.0 was banned unless 100% unambiguous.**

#### (B) Fusion and Architecture
- **Mamba Temporal Encoding**: Utilized for both audio (Wav2Vec2.0 pre-trained on MSP-Podcast) and text (Jina-v3 fine-tuned on 30 langs) feature sequences. Mamba’s linear scaling efficiency excelled on the longer RESD sequences.
- **BiFormer Fusion Layer**: Applies cross-modal attention containing **5 Transformer layers** and **8 attention heads**. Matrix configuration swaps Queries ($Q$) and Keys/Values ($K, V$) between Audio and Text to enable dynamic cross-modal interaction.

### 3. Key Findings
- **Unimodal Disparities**: Wav2Vec2.0 consistently outperforms ExHuBERT (suggesting volume of training data beats domain breadth). For text, word-level tokenization (Jina-v3) beat character-level (CANINE-c).
- **Architecture Validation**: On short-fragmented MELD, BiFormer alone succeeded. On longer, more complex RESD, **BiGraphFormer** (relational pair modeling) achieved superior linear performance.
- **Quantitative Gains**: Applied together (`BiFormer + SDS + LS-LLM`), joint training achieved relative UAR gains of **11.23%** on MELD and **31.70%** on RESD over single-modality baselines.
- **Language-Specific Biases**: In English (MELD), emotions tend to be confused with *Anger* and *Happy*. In Russian (RESD), they are frequently misclassified as *Fear*, reflecting distinct cultural/prosodic profiles.

### 4. Application Strategy
- **Dynamic Gating**: Mechanisms inspired by BiFormer offer a framework for adjudicating modal conflicts (e.g., weighting audio higher if text confidence is low). **However, knowing how confident a LLM is is hard.**
- **Conflict Metric with Soft Labels**: **JS-Divergence** can be used to compare audio-visual predictions against LLM-generated soft text labels to identify high-conflict samples for trigger reasoning steps.
- **Cross-Lingual Thresholds**: Evaluating joint models on different languages can verify whether modal adjudication patterns remain universal or language-dependent.
- **Prompt Strategy**: **Assigning 1.0 was banned unless 100% unambiguous.** This can be useful.

## 🕸️ Relations
- **Relevant to**: [[Emotion Recognition in Conversation (ERC)]], [[Mamba]], [[Soft Labels]], [[Whisper]]
- **Connects with**: [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (shared interest in distribution-based labels)