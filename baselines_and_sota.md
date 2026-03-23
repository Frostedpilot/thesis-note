---
title: "Baselines and SOTA in LLM-Based Emotion Detection"
status: Compiled
tags:
  - benchmarks
  - sota
  - baselines
  - evaluation
---

# 📊 Baselines and SOTA in LLM-Based Emotion Recognition

This document summarizes the common baseline architectures, standard evaluation datasets, and the current State-of-the-Art (SOTA) performance thresholds extracted from the reviewed literature.

## 1. Standard Evaluation Benchmarks (Datasets)

The community evaluates models primarily across these core dialogue and multi-modal datasets:

| Dataset          | Modality                      | Characteristics                                                     | Key Reference                                                                                                                                 |
| :--------------- | :---------------------------- | :------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| **IEMOCAP**      | Multimodal (Video/Audio/Text) | Long dyadic conversations, requires deep history lookback.          | [InstructERC](notes/instructerc_reforming_emotion_W4386977634_notes.md), [LaERC-S](notes/laerc_s_improving_llm_based_em_W4392781922_notes.md) |
| **MELD**         | Multimodal (Video/Audio/Text) | Short utterances from *Friends* TV show, multiparty, high noise.    | [Cross-Lingual MELD](notes/cross_lingual_bimodal_emotion_W4416166868_notes.md)                                                                |
| **DailyDialog**  | Text-only                     | Daily conversations, shorter turns, clean topic clusters.           | [RECCON](notes/recognizing_emotion_cause_in_c_W3115793997_notes.md)                                                                           |
| **SemEval-2018** | Text-only                     | Multi-label classification & continuous intensity regression tasks. | [EmoLLM](notes/emollms_a_series_of_emotional_W4401863339_notes.md)                                                                            |
| **MOSEI**        | Multimodal                    | Continuous sentiment and discrete emotion intensity ratings.        | [Agent-Based Modular](notes/agent_based_modular_learning_f_W7115566960_notes.md)                                                              |



---

## 2. Common Baselines ans SOTAs

Modern backbones (using **LLaMA-2/3**, **Mistral**, or **GPT-4**) combined with adapter adapters (LoRA) or structured description piping set the current score aggregates:


### X. Control Group
-  **Zero-shot LLMs**: LLaMA-2/3, Mistral, GPT-4, Gemini-2.0, Qwen, ...
### A. Text-Centric dialogue ERC
*   **InstructERC** ([[notes/instructerc_reforming_emotion_W4386977634_notes|InstructERC]]): Fine-tuned LoRA on decoders incorporating retrieval cues.
    *   *IEMOCAP*: **71.39%**
    *   *MELD*: **69.15%**
*   **LaERC-S** ([[notes/laerc_s_improving_llm_based_em_W4392781922_notes|LaERC-S]]): Incorporating "Listener Reaction" descriptions into a two-phase loop.
    *   **IEMOCAP**: **72.40%** (beats InstructERC)
    *   **MELD**: **69.27%**
*   **CDEA (Causality Driven Dialogue)** ([[notes/cdea_causality_driven_dialogue_W4408833052_notes|CDEA]]): Pushing history window lookup thresholds.
    *   **IEMOCAP**: **73.26%** (Current peak in reviewing list)
    *   **MELD**: **69.34%**

### B. Continuous Regression & Intensities
*   **EmoLLM-13B** ([[notes/emollms_a_series_of_emotional_W4401863339_notes|EmoLLM]]): Pre-tuned autoregressive regression outputs.
    *   Hits **0.831 Pearson Score** on SemEval `EI-reg` intensity regression modeling, beating standing leaderboard architectures that rely on Sigmoid scaling.

### C. Multimodal Distillations (No custom fusion layers)
*   **DialogueLLM** ([[notes/dialoguellm_context_and_emotio_W4412602922_notes|DialogueLLM]]): Summarizes vision description frames via ERNIE-Bot into text streams:
    *   **MELD**: **71.90% F1** (proving written prompts outperform hidden layer concatenations by nearly 5%).
*   **Emotion-LLaMA** ([[notes/emotion_llama_multimodal_emoti_W4415795657_notes|Emotion LLaMA]]):
    *   Achieved **0.9036 F1** peak on MER2023-SEMI evaluation splits.

