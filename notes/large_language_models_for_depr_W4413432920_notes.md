---
title: "Large language models for depression recognition in spoken language integrating psychological knowledge"
authors: ["Yupei Li", "Shuaijie Shao", "Manuel Milling", "Björn W. Schuller"]
year: 2025
publication_date: "2025-08-22"
doi: "10.3389/fcomp.2025.1629725"
openalex_id: "W4413432920"
citation_count: 4
status: Read
tags:
  - paper
  - depression_recognition
  - knowledge_injection
  - multimodal_fusion
  - wav2vec
---

# 📝 LLMs for Depression Recognition Integrating Psychological Knowledge

> [!ABSTRACT] TL;DR
> This paper presents a two-stage approach applying LLMs (**LLaMA2-7B**) to multimodal depression detection on **DAIC-WOZ**. Stage 1 injects authoritative clinical knowledge from the **World Health Organization (WHO)** via 4,920 Q&A pairs (definitions, symptoms, critical thinking), reducing hallucinations. Stage 2 embeds acoustic paralinguistics using **Wav2Vec 2.0** aligned with text latent spaces through linear projections to estimate continuous **PHQ-8 scores**.

## 🔗 Quick Links
- **Parsed Text**: [[large_language_models_for_depr_W4413432920_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.3389/fcomp.2025.1629725)

## 📚 Reading Notes

### 1. Core Objectives
-   Typical AI mental health diagnostics lack clinical authenticity or integration with professional psychological expertise, reducing practitioner trust.
-   Unimodal text transcripts overlook crucial **paralinguistic cues** (pitch variability, speech rate, pause intervals) where depressive symptoms often manifest heavier than in dictionary-lexicon text filters.

### 2. Methodological Approach
#### (A) Two-Stage Pipeline
1.  **Stage 1: Psychology Knowledge Injection**:
    *   Filters entries from the **WHO** official medical classification site concerning mood disorders.
    *   Uses DeepSeek-V3 to construct **4,920 Q&A sets** structured into 5 logical nodes: Disorder definition, diagnosis rationale, common symptoms/manifestations, extended context, and critical thinking triggers.
    *   Trained on LLaMA-2 to absorb clinical thresholds directly inside prompt-follows to reduce hallucination cycles.
2.  **Stage 2: Audio feature projection**:
    *   Extracts acoustic items using **Wav2Vec 2.0** supporting prosodic emotion cues over strict text transcription wrappers (like Whisper).
    *   Aligns audio inputs through projection layers into LLaMA’s shared hiding space for consecutive interval scoring.

### 3. Key Findings
-   **Audio-superiority threshold**: pure audio models outperform pure text benchmarks because transcribed dictionary text fails to capture scale pitch drops.
-   **Knowledge Injection yields drops in error thresholds**: injects authoritative verification vectors that decrease Mean Absolute Error (MAE) and increases descriptive accuracy pass indices on pass@2 grading tests (scoring 8.20 vs 7.32 plain LLM).

### 4. Application Strategy
-   **Rigid adapter guards for factual conformity**: Stage 1's Q&A generation mirrors authoritative standard knowledge retrieval very well. This structured Q&A setup can be adapted into **dense prompt templates** or context embeddings retrieval or **prompt template retrieval** to secure agentic lookup nodes without full weight fine-tuning.
-   **Align weights into description guards**: To fit purely into prompt-agent structures (avoiding dimensional projection lockups), acoustic paralinguistics can be distilled into high-level textual descriptors (similar to DialogueLLM setups for video using ENRIE-BOT) that explicitly outline audio feature directly into the text buffer prompts.

## 🕸️ Relations
-   **Builds on**: [[PsychoLexLLaMA]]; [[Wav2Vec 2.0]]; [[DAIC-WOZ]]
-   **Relevant to**: [[Mental Health Analysis]]; [[Multimodal Fusion]]; [[Continuous Score Regression]]
-   **Pairs well with**:
  - [[DialogueLLM: Context and Emotion Knowledge-Tuned LLMs]] (Multimodal translation to descriptive buffers guidelines)
  - [[CDEA: Causality-Driven Dialogue Emotion Analysis]] (Causal triggers anchoring)
