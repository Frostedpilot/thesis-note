---
title: "LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics"
authors: ["Yumeng Fu", "Wu, Junjie", "Wang, Zhongjie", "Zhang, Meishan", "Shan, Lili", "Wu, Yulin", "Li, Bingquan"]
year: 2024
publication_date: "2024-03-12"
doi: "10.48550/arxiv.2403.07260"
openalex_id: "W4392781922"
citation_count: 5
status: Read
tags:
  - paper
  - erc
  - llm
  - speaker_characteristics
  - soft_labels
  - conflict_resolution
---

# 📝 LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics

> [!ABSTRACT] TL;DR
> **LaERC-S** proposes a two-stage fine-tuning pipeline to model dynamic **speaker characteristics** involving the mental state and behavior of interlocutors, rather than relying on static biographical data. It leverages LLM world knowledge to generate implicit reasoning cues like **oReact** (listener reaction), which serve as critical guides for sizing up downstream emotional trajectories.

## 🔗 Quick Links
- **PDF**: [[papers/laerc_s_improving_llm_based_em_W4392781922/laerc_s_improving_llm_based_em_W4392781922.pdf|Open Local PDF]]
- **Parsed Text**: [[laerc_s_improving_llm_based_em_W4392781922_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.48550/arxiv.2403.07260)

## 📚 Reading Notes

### 1. Core Objectives
- Standard ERC models focus on explicit utterance dependencies or static speaker variables, often missing deep interlocutor dynamics present in live dialogue.
- Resolving emotional ambiguity requires extracting **latent mental states** (intentions, reactions, effects) to adjudicate between contrasting expressions.
- **9 Pillars of Speaker Characteristics** (Aligned with ATOMIC-2020):
  1. *Mental State*: `oReact` (listener reaction), `xReact` (speaker reaction), `xIntent` (speaker intention).
  2. *Behavior*: `xNeed`, `xWant`, `oWant`, `xEffect`, `oEffect`.
  3. *Persona*: `xAttr` (static attributes).

### 2. Methodological Approach
#### (A) Speaker Characteristic Extraction
- Queries LLMs using explicit prompt engineering to generate descriptions of listener reactions (`oReact`).
- Phrasing matters: Incorporating the word **"potential"** (e.g., "reaction of potential listeners") provides a significant performance buffer over generic listeners.

#### (B) The Two-Stage Injection Pipeline
1. **Stage 1 (Injection)**: Instruction-tunes the LLM to generate situational speaker characteristics (e.g., descriptions of intention or reaction). **Batch Size**: 8.
2. **Stage 2 (Recognition)**: Adds the emotion label setup to predict final categorization leveraging the updated reasoning weights. **Batch Size**: 16.
- **Context Window**: Maintained at **12 turns** to ensure sufficient historical grounding on multi-party dynamics.

### 3. Key Findings
- **SOTA Performance**: LaERC-S achieves comprehensive superiority on all evaluation sets: **IEMOCAP (72.40%)**, **MELD (69.27%)**, and **EmoryNLP (42.08%)**. Beats InstructERC by 1.01% on IEMOCAP.
- **Element Priority**: The **oReact** (listener reaction) component yields the most significant performance lift. Persona trackers (`xAttr`) trigger the lowest gains, validating that **dynamic state tracking** is superior to static profile reading.
- **Cross-Dataset Robustness**: Mixed ratio scaling tracks safely with single-domain absolute peaks, proving speaker injection prevents rigid domain overfitting.

### 4. Application Strategy
- **Multi-Stage Intent Calibration**: Replicating a two-phase loop allows training models to first output **intent descriptions** before aggregating multiple visual-acoustic streams for the final label distribution.
- **Historic alignment Priors**: Tracking continuous speaker dialogue windows (optimal $w=12$) builds modular profiles for adjusting high-entropy fuse rates dynamically.

## 🕸️ Relations
- **Builds on**: [[InstructERC]]; BiosERC (mentioned); CEPT (mentioned)
- **Relevant to**: [[Prompt engineering]]; [[Agentic workflows]]; [[Speaker modeling]]; [[Emotion dynamics]], [[Soft Labels]], [[Conflict Resolution]]
- **Connects with**: [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (shared interest in soft labels) and [[Emotion-LLaMA]] (multimodal backbones)