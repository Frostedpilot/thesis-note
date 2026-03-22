---
title: "CDEA: Causality-Driven Dialogue Emotion Analysis via LLM"
authors: ["Xue Zhang", "Mingjiang Wang", "Xuyi Zhuang", "Xiao Zeng", "Qiang Li"]
year: 2025
publication_date: "2025-03-25"
doi: "10.3390/sym17040489"
openalex_id: "W4408833052"
citation_count: 0
status: Read
tags:
  - paper
  - causality_driven
  - commonsense_knowledge
  - atomic
  - llm
  - dynamic_retrieval
  - auxiliary_tasks
---

# 📝 CDEA: Causality-Driven Dialogue Emotion Analysis via LLM

> [!ABSTRACT] TL;DR
> This paper introduces **CDEA**, a framework that explicitly identifies **emotional causes** between historical and target utterances using structured commonsense knowledge (**ATOMIC** for self/other paths). It feeds these causes along with dynamically retrieved experiences (via BERTScore) into an instruction-tuned LLM (**Llama2-7B** with **LoRA**). To refine representation, it enforces an **auxiliary task** forcing the LLM to generate semantic explanations of the emotion labels. The method achieves absolute SOTA boosts on IEMOCAP, MELD, and DailyDialog.

## 🔗 Quick Links
- **Parsed Text**: [[cdea_causality_driven_dialogue_W4408833052_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.3390/sym17040489)

## 📚 Reading Notes

### 1. Core Objectives
- ERC models often lack explicit assessment of **Forward & Backward Causality Symmetry** (how past creates emotions, and how reactions sustain them).
- High subjectivity leads to label inconsistency; identifying **objective triggers** stabilizes representation.
- Structured reasoning is needed to augment LLM contextual understanding by bridging explicit links to historical context to bypass noisy scaling.

### 2. Methodological Approach
#### (A) Sentiment Cause Sentence Acquisition
- Splits background logic into **Self-induced** and **Other-induced reasoning paths** referencing **ATOMIC**:
  - *Self*: `xReact`, `xEffect`, `xWant`
  - *Other*: `oReact`, `oEffect`, `oWant`
- Identifies $m$ Other-cause and $n$ Self-cause utterances prior to target $u_i$ based on feature similarity (RoBERTa + LSTM backends).

#### (B) Dynamic Retrieval & Augmented Prompting
- **Database Search**: Retrieves similar examples from EmoryNLP database using **BERTScore** to prevent rigid manual instructions.
- **BART Refinement**: Extracted causal sentences are fed to **BART** to generate continuous augmented explanations, repairing fragmented dialogue context.
- **Prompt Components**: Includes instruction, history index, BART-refined causes, task definitions, and retrieved examples.

#### (C) Auxiliary Semantic Task
- During LoRA fine-tuning, the generator must produce both the **Emotion Class** and its **Dictionary Explanation** (retrieved from SentiWordNet, e.g., *frustrated -> disappointedly unsuccessful*). 

### 3. Key Findings
- **SOTA Metrics**: CDEA + Llama pushes IEMOCAP to **73.26%**, MELD to **69.34%**, and DailyDialog to **64.59%**.
- **Window Calibration**: Found the optimal history window peaks at $w=15$ for long conversational datasets (IEMOCAP) and $w=10$ for brief structures (MELD).
- **LoRA stability**: Ablation without LoRA triggers full-parameter overfitting drops, proving parameter-efficient tuning prevents continuous text drifting.

### 4. Application Strategy (Prompt-Based / Agentic)
- **Modality-Specific Causality Allocation**: Self-cause versus Other-cause nodes map ideally to multimodal streams. Voice inflections usually correlate heavier with self-induced shifts (`xReact`), whereas text benchmarks track other-induced descriptions (`oReact`).
- **Clean Content Prompt Chains**: Using generators to "clean" raw multimodal nodes into continuous prompt sentences prevents LLM distraction on fragmented audio cues or noise artifacts.

## 🕸️ Relations
- **Builds on**: [[ATOMIC]]; [[COMET]]; [[DialogueCRN]]; [[CauAIN]]
- **Relevant to**: [[Emotion Recognition in Conversation]]; [[Causality loops]]; [[Soft Labels]]; [[Dynamic Retrieval]]
- **Pairs well with**:
  - [[InstructERC: Reforming Emotion Recognition in Conversation]] (Auxiliary tasks framework)
  - [[Third-Person Appraisal Agent]] (Causal chain feedback architecture)
