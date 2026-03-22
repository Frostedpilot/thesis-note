---
title: "AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models"
authors: ["Xin Hong", "Yuan Gong", "Vidhyasaharan Sethu", "Ting Dang"]
year: 2025
publication_date: "2025-03-12"
doi: "10.1109/ICASSP55128.2025.10903333"
openalex_id: "W4408347320"
citation_count: 0
status: Read
tags:
  - paper
  - erc
  - llm
  - uncertainty
  - soft_labels
  - icassp2025
---

# 📝 AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models

> [!ABSTRACT] TL;DR
> The AER-LLM framework addresses the "hard label" problem in Emotion Recognition in Conversation (ERC) by utilizing LLMs to generate **soft labels (probability distributions)**. Using zero-shot and few-shot prompting alongside retrospective dialogue context, the approach demonstrates that LLMs can effectively capture emotional ambiguity comparable to human perception. Metrics like **Bhattacharyya coefficients** and **JS Divergence** are employed to measure distribution alignment.

## 🔗 Quick Links
- **PDF**: [[papers/aer_llm_ambiguity_aware_emotio_W4408347320/aer_llm_ambiguity_aware_emotio_W4408347320.pdf|Open Local PDF]]
- **Parsed Text**: [[aer_llm_ambiguity_aware_emotio_W4408347320_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1109/ICASSP55128.2025.10903333)

## 📚 Reading Notes

### 1. The Core Problem
- The paper addresses the limitations of "one-hot" or majority-vote approaches in emotion recognition, noting that human emotions are inherently ambiguous and disagreeing reviews reflect genuine complexity.
- **Evaluation Datasets**:
  - **MSP-Podcast** and **IEMOCAP**: 4 continuous classes (neutral, angry, happy, sad); totaling ~4K utterances each.
  - **GoEmotions**: Fine-grained Reddit scope targeting high-ambiguity structures (admiration, gratitude, approval, amusement, neutral); weighted to select 210 highly ambiguous targets.

### 2. Methodology Analysis
#### (A) The Distributional Output
- **Backbone**: Evaluated using **Gemini-1.5-Flash** to take advantage of it's million-token context limit for deep dialogue lookback.
- **AER Dictionary Prompting**: The framework forces the LLM to output a dictionary summing to 1. Prompt structure is divided into **Background (BG)**, **Context (C)**, **Target Utterance (TU)**, **Task**, and strict **Output Constraints (OC)** (3 rules: dict format, sum to 1, no explanation).
- Incorporates **eGeMAPS** (88-dimensional acoustic features) translated explicitly into descriptive text sentences (e.g., "Fundamental Frequency: X") to maintain pure text input loops.

#### (B) Evaluation Metrics
The following distribution-aware metrics are utilized to measure fit rather than absolute accuracy triggers:
- **Bhattacharyya Coefficient (BC)**: Measures similarity/overlap of continuous distributions.
- **Expected Calibration Error (ECE)**: Gauges probability scaling accuracy.
- **Jensen-Shannon (JS) Divergence** & $R^2$: evaluating curve divergence and goodness-of-fit.

### 3. Key Findings
- **In-Context Learning Returns**: Including contextual conversational lookbacks improves ambiguous alignment. Increasing context from $M=0$ to $M=5$ triggers an average of **16% (JS), 28% (BC), 22% ($R^2$), and 19% (ECE) improvements**. Performance plateaus beyond context window lengths of 10-20.
- **Multimodal Boosting**: Adding eGeMAPS acoustic text prompts reliably reduces JS Divergence outperforming text-only branches, indicating successful non-semantic reasoning inside LLM prompts.
- **Entropy Inversion**: Performance demonstrates a continuous degradation as ground truth **entropy (ambiguity)** rises, mirroring the human ceiling for high-disagreement emotion inference.
- **Single-Label Parity**: Maximum probability outputs match state-of-the-art majority-vote benchmarks (InstructERC, etc.), showing the soft approach does not compromise hard prediction standards.

### 4. Application Strategy (Prompt-Based / Agentic)
- **Conflict Adjudication**: This soft-labeling approach can enable **cross-modal conflict resolution** in prompt-based or agentic systems. High entropy in outcomes can trigger a "re-appraisal" prompt layer to investigate sarcasm or modality conflict.
- **Evaluation Metric**: Metrics like **JS Divergence** can evaluate prompt performance or guide safe example selection for few-shot prompting, supporting a prompt-based architecture without requiring fine-tuning.
- **Uncertainty Trigger**: High uncertainty in predicted distributions can serve as a trigger condition for multi-agent workflows, activating specialized agents to analyze acoustic cues more deeply.

## 🕸️ Relations
- **Relevant to**: [[Uncertainty quantification]]; [[Expected Calibration Error (ECE)]]; [[Jensen-Shannon divergence]]; [[Bhattacharyya coefficient]]; [[eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set)]]
- **Connects with**: [[Emotion Recognition in Conversation (ERC)]] (shared focus on multi-turn context)