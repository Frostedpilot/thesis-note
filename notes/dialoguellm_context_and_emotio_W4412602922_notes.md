---
title: "DialogueLLM: Context and emotion knowledge-tuned large language models for emotion recognition in conversations"
authors: ["Yazhou Zhang", "Mengyao Wang", "Youxi Wu", "Prayag Tiwari", "Qiuchi Li", "Benyou Wang", "Jing Qin"]
year: 2025
publication_date: "2025-07-23"
doi: "10.1016/j.neunet.2025.107901"
openalex_id: "W4412602922"
citation_count: 17
status: Read
tags:
  - paper
  - emotional_llm
  - multimodal_fusion
  - lora
  - video_description
  - dialogue_context
---

# 📝 DialogueLLM: Context and Emotion Knowledge-Tuned LLMs for ERC

> [!ABSTRACT] TL;DR
> This paper presents **DialogueLLM**, an open-source emotional LLM based on **LLaMA2-7B** fine-tuned with Parameter-Efficient Fine-Tuning (**LoRA**) on multimodal datasets (MELD, IEMOCAP, EmoryNLP). To fuse modal signals within a text-only prompt budget, it converts video frames into **textual video descriptions** (via ERNIE Bot) and appends them as supplementary knowledge alongside dialogue contexts. It achieves SOTA F1 score vectors and aligns highly with human metrics in Emotional Intelligence (EQ) tests.

## 🔗 Quick Links
- **Parsed Text**: [[dialoguellm_context_and_emotio_W4412602922_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1016/j.neunet.2025.107901)

## 📚 Reading Notes

### 1. Core Objectives
-   General LLMs perform poorly on complex, subjective **emotion recognition in conversations (ERC)** tasks because they typically overlook speaker dependency context and multi-modal information (visual/acoustic).
-   DialogueLLM bypasses multi-modal layer architecture fusion demands by translating auxiliary sensory streams (videos) directly into **high-quality text summaries** suitable for text-only LLMs.

### 2. Methodological Approach
#### (A) Multimodal Description Datasets
-   Builds instruction templates across 24,304 utterances (MELD, IEMOCAP, EmoryNLP).
-   **Video Description Modules**: Video segments are split into frames and forwarded through visual-language pipelines (ERNIE Bot) to generate explicit textual descriptions of the interaction/activity (e.g., actor face expressions).
-   **Prompt Format Structure**: Fuses `Video Description`, sequential `Dialogue Context` (turns window $z$), and the target `Utterance` to compel generative emotion label classification.

#### (B) Fine-Tuning Pipeline
-   Backbone: **LLaMA2-7B**.
-   Tuning: **LoRA** adapter training ($r=4, \alpha=16$ with 2.1M trainable parameters).
-   Setup: 3-10 epochs, AdamW optimizer, high convergence efficiency (5 hours on a 40GB A100 GPU).

### 3. Key Findings
-   **Absolute SOTA Boosts**: DialogueLLM hits **71.90% F1** on MELD (vs 66.45% peak baseline SACL-LSTM) and **69.93%** on IEMOCAP.
-   **Description Benefits Ablation**: Removal of visual description text or contexts yields sharp accuracy drops across natural datasets, confirming that textualized summaries adequately replace custom hidden-layer cross-modality fusions.
-   **Context Calibration Bounds**: Accuracy peaks around turnovers of context turns $z=1$ to $z=2$. Continuing turns beyond that fails to boost accuracy and triggers scaling computation saturation.
-   **Human EQ Benchmarks**: DialogueLLM scores **109** on SECEU (Emotional Intelligence test vectors), surpassing standard LlaMA/Alpaca and approaching ChatGPT models with substantially smaller parameter weights.

### 4. Application Strategy
-   **Distilled Multimodal Representation (Descriptions Framework)**: DialogueLLM offers a clean strategy for agentic cascades—summarize audio/video sensors into compact text prompt statements rather than training embedding fusion layers. This keeps prompt pipelines composable without full-parameter adapter locks.
-   **Strict turn-limit windows**: Setting strict turns triggers limits density guards ($z \le 2$) might preventing high distraction ratios in dynamic chat agent memory weights.
## 🕸️ Relations
-   **Builds on**: [[DAG-ERC]]; [[InstructERC]]; [[DialogueRNN]]
-   **Relevant to**: [[Modal Translation]]; [[LoRA adapters]]; [[Emotional Intelligence Benchmark]]
-   **Pairs well with**:
  - [[CDEA: Causality-Driven Dialogue Emotion Analysis]] (Causal trigger paths)
  - [[Third-Person Appraisal Agent]] (Context reasoning node cascades)
