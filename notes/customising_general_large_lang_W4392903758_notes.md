---
title: "Customising General Large Language Models for Specialised Emotion Recognition Tasks"
authors: ["Liyizhe Peng", "Zixing Zhang", "Tao Pang", "Jing Han", "Huan Zhao", "Hao Chen", "Björn W. Schuller"]
year: 2024
publication_date: "2024-03-18"
doi: "10.1109/icassp48485.2024.10447044"
openalex_id: "W4392903758"
citation_count: 12
status: Read
tags:
  - paper
  - llm_adaptation
  - peft
  - lora
  - p_tuning
  - emotion_recognition
---

# 📝 Customising General LLMs for Emotion Recognition Tasks

> [!ABSTRACT] TL;DR
> This paper investigates adapting a general-purpose LLM (**ChatGLM2-6B**) specifically for emotion recognition tasks using **Parameter-Efficient Fine-Tuning (PEFT)** techniques, specifically **P-Tuning v2** and **LoRA**. Across 6 diverse datasets (SST, MOSI, CH-SIMS, Friends, Mastodon, M3ED), adapted LLMs comfortably beat pre-adaptation models and often surpass specialized deep learning SOTA models, proving strong transferability with minimal resource bounds.

## 🔗 Quick Links
- **Parsed Text**: [[customising_general_large_lang_W4392903758_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1109/icassp48485.2024.10447044)

## 📚 Reading Notes

### 1. Core Objectives
-   General LLMs offer strong zero-shot and few-shot capabilities for sentiment tasks but extending them explicitly with few-shot examples scales inference costs quadratically. 
-   Full Fine-Tuning (FFT) demands prohibitive computational budgets.
-   The goal is to determine if **Parameter-Efficient Fine-Tuning (PEFT)** can morph a generalized LLM into an ERC specialist that rivals custom non-LLM specialized deep architectures.

### 2. Methodological Approach
#### (A) LLM Backbone
-   **ChatGLM2-6B**: Chosen for consumer-grade GPU viability (runs inference on 13GB VRAM using FP16). Prompts heavily structured dynamically format tasks e.g., "Classify the sentiment of the sentence to Emotion 1, Emotion 2...: \<input\>".

#### (B) Modal Adaptation Techniques (PEFT)
-   **P-Tuning v2**: Incorporates continuous prompts into *every layer* of the model (unlike v1 which only hits input embeddings), tuning 0.1%-3% of parameters. Has a stronger direct impact on model predictions for difficult tasks.
-   **LoRA (Low-Rank Adaptation)**: Freezes pre-trained weights and injects trainable low-rank decomposition matrices ($r=8$) into Transformer layers. Drops trainable parameter counts 10,000-fold vs full fine-tuning.

### 3. Key Findings
-   **Adapted superiority**: ChatGLM2 adapted models almost double their pre-adaptation accuracy in complex multi-class settings (e.g., dropping SST-5 errors significantly).
-   **Beating Custom Architectures**: On simpler binary/ternary tasks (MOSI, Mastodon, CH-SIMS), adapted ChatGLM2 outright outperforms the standing specialized SOTA models.
-   **Scaling complexity gap**: On datasets demanding rich dialogue context (like 7-class ERC in Friends and M3ED), adapted LLMs trailed SOTA because the authors purposefully restricted the prompt window strictly to singular utterances to test isolated limits.
-   **LoRA vs P-Tuning tradeoff**: LoRA dominates binary classification tasks, while P-Tuning v2 provides superior leverage across ternary and complex multi-class evaluations. 

### 4. Application Strategy (Prompt-Based / Agentic)
-   nothing

## 🕸️ Relations
-   **Builds on**: [[ChatGLM]]; [[LoRA]]; [[P-Tuning v2]]
-   **Relevant to**: [[Model Adaptation]]; [[Parameter-Efficient Fine-Tuning]]
-   **Pairs well with**:
  - [[DialogueLLM: Context and Emotion Knowledge-Tuned LLMs]] (Validating that adding $z \le 2$ contexts resolves the gaps left in this study's isolated utterance checks)
  - [[EmoLLMs: A series of Emotional Large Language Models]] (Comparing fine-tuning vs PEFT bounds)
