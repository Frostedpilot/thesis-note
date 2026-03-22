---
title: "From Text to Emotion: Unveiling the Emotion Annotation Capabilities of LLMs"
authors: ["Minxue Niu", "Mimansa Jaiswal", "Emily Mower Provost"]
year: 2024
publication_date: "2024-09-01"
doi: "10.21437/interspeech.2024-2282"
openalex_id: "W4402111569"
citation_count: 7
status: Read
tags:
  - paper
  - emotion_annotation
  - llm
  - evaluation
  - soft_labels
  - calibration
---

# 📝 From Text to Emotion: Unveiling the Emotion Annotation Capabilities of LLMs

> [!ABSTRACT] TL;DR
> This paper assesses GPT-4's performance as an emotion annotator against human standards across multiple datasets. A key finding is that human evaluators consistently prefer **GPT-4 annotations over original human labels** on disagreement sets. The study validates LLMs for capturing **emotional ambiguity** and filtering low-quality annotations to improve downstream model training.

## 🔗 Quick Links
- **PDF**: [[papers/from_text_to_emotion_unveiling_W4402111569/from_text_to_emotion_unveiling_W4402111569.pdf|Open Local PDF]]
- **Parsed Text**: [[from_text_to_emotion_unveiling_W4402111569_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.21437/interspeech.2024-2282)

## 📚 Reading Notes

### 1. Core Objectives
- The study challenges the assumption of a single "correct" ground truth, framing human disagreement not as noise, but as **valid emotional diversity** reflecting subjectiveness.
- **Evaluation Subsets**: Benchmarks 500 samples each from **ISEAR** (7 class), **SemEval** (11 class), **GoEmotions** (28 class), and **EmoBank** (3 VAD Continuous scales). **Samples used log inverse frequency weighting to ensure representation of rare classes.**
- It explores utilizing LLMs to generate high-quality annotations that can augment or filter existing crowd-sourced datasets.

### 2. Methodological Approach
#### GPT-4 Annotation Setups
The study compares two distinct prompting styles using a assigned **Persona** ("You are an emotionally-intelligent and empathetic agent"):
1. **Classification**: A fixed list of instructions requiring selections only where the model is "reasonably confident." Forces higher precision by allowing a "neutral" fallback.
2. **Generation**: Free-form descriptors which human evaluators prioritized over restricted classification sets due to nuanced alignment (particularly for low-class sets like ISEAR).

#### Evaluation Metrics
- **Controlled Disagreement evaluation**: Human evaluators preferred GPT-4 labels over original human annotations across all sets (**GoEmotions: 71.1%**, **SemEval: 68.2%**, **ISEAR: 62.3%**).
- **Regression correlations**: For continuous rating spaces (Emobank), GPT-4 identified relative valence direction accurately (**High Pearson Correlation: 0.764**), though absolute value averaging introduces integer-scale bias (higher MAE).

### 3. Key Findings
- **Adjudicated Superiority**: GPT-4 creates coherent target sets. BERT models trained on **GPT-4 labels** outperformed human-trained models on an adjudicated test set by a large margin (**0.524 vs. 0.392 Macro-F1**).
- **Ambiguity scaling**: GPT-4's efficacy scales well with large label spaces (28 classes), preventing the accuracy drop typically seen with human annotators facing higher cognitive fatigue.
- **Filtered yields**: Training on **filtered datasets** (where human and GPT-4 annotations agree) yields better accuracy on calibrated evaluation sets with 45% of the data volume.

### 4. Application Strategy
- **Soft Label Calibration setups**: Generating "reasonably confident" prompt lists allows synthesizing **consensus distributions** for training multi-label calibration frameworks.
- **Consistency-Based Filtering**: Creating samples where text-only LLMs and audio-visual backbones *agree* establishes a gold-standard dataset for training non-verbal alignment sub-modules.
- **Entropy thresholds as flags**: Higher label ambiguity (distribution flatness) can act as an agentic trigger to route visual streams to intensive multimodal adjudication sub-agents.

## 🕸️ Relations
- **Builds on**: [[Affective capabilities of LLMs]]; [[Prompting]]; [[Emotion Annotation]]; [[Emotion Recognition]]
- **Relevant to**: [[Emotion Recognition in Conversation|ERC]]; [[GoEmotions]]; [[SemEval 2017 Task 4]]; [[ISEAR]]; [[Calibration]]; [[Soft Labels]]