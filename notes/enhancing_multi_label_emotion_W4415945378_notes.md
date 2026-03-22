---
title: "Enhancing Multi-label Emotion Prediction through Rule-based Voting with LLM and BERT Variants"
authors: ["Minh Hieu Le", "Cong Phuoc Phan", "Thanh Tuan Nguyen", "Thi Thanh Sang Nguyen"]
year: 2025
publication_date: "2025-03-12"
doi: "10.21203/rs.3.rs-7501928/v1"
openalex_id: "W4415945378"
citation_count: 0
status: Read
tags:
  - paper
  - multi-label
  - ensemble
  - llm
  - bert
  - voting
  - soft_labels
  - conflict_resolution
---

# 📝 Enhancing Multi-label Emotion Prediction through Rule-based Voting with LLM and BERT Variants

> [!ABSTRACT] TL;DR
> This system utilizes a **rule-based voting aggregator** to combine Large Language Model (LLM) and BERT predictions for multi-label emotion analysis. The approach introduces **adaptive weighted voting** based on confidence scores and calibrates model outputs via a three-tier rule structure—**Total Weight, Vote Count, and Average Probability Threshold**—enabling robust adjudication of ambiguous or conflicting predictions.

## 🔗 Quick Links
- **PDF**: [[papers/enhancing_multi_label_emotion_W4415945378/enhancing_multi_label_emotion_W4415945378.pdf|Open Local PDF]]
- **Parsed Text**: [[enhancing_multi_label_emotion_W4415945378_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.21203/rs.3.rs-7501928/v1)

## 📚 Reading Notes

### 1. Core Objectives
- The framework addresses the challenge of combining predictions from **heterogeneous architectures** (12 BERT variants and generative LLMs) when multi-label outputs are ambiguous.
- **Benchmark Context**: Evaluated on **SemEval-2025 Task 11 (Track A)** featuring 5 emotions: Anger, Fear, Joy, Sadness, Surprise.
- **Class Imbalance**: Handles heavily skewed distribution where **Fear** dominates (58.2%) and **Anger** is sparse (12.0%).
- It leverages **soft weights** (calibrated confidence scores) to dynamically weight each model's contribution rather than relying on uniform ensemble stacking.

### 2. Methodological Approach
#### (A) Weight Assignment Mechanism
- **Model Foundations**: Uses twelve BERT variants for text probabilities and **Gemini-1.5-flash-001** for binary SFT branches.
- **Branch Strategy**: Compares Vanilla Multi-label, Task-Decomposed binary models ("Yes/No"), and Data-Augmented setups adding SemEval-2018 samples.
- **Discrete Vote Mapping**: Sigmoid vectors map onto standardized score intervals:
  - **[0.8 – 1.0] $\rightarrow$ +2** (Strong Confidence Presence)
  - **[0.6 – 0.8] $\rightarrow$ +1** (Moderate Presence)
  - **[0.4 – 0.6] $\rightarrow$ 0** (Uncertainty)
  - **[0.2 – 0.4] $\rightarrow$ -1** (Moderate Absence)
  - **[0.0 – 0.2] $\rightarrow$ -2** (Strong Absence)

#### (B) Three-Tier Hierarchical Voting
Predictions are adjudicated cascade-style:
1. **Total Weight**: Sums positive/negative model weights. If $> 0$ label present, $< 0$ absent.
2. **Vote Count**: Count of active models absolute direction ($N_{pos}$ vs $N_{neg}$) if sum is zero.
3. **Average Probability Threshold**: Soft average tie-breaking rule if absolute counts are also equal.

### 3. Key Findings
- **Ensemble Synergy**: The aggregator achieved **Macro F1 of 80.42%** and **Micro F1 of 82.33%**, beating individual DeBERTa by 9.8% and SFT Data-Augmented by 2.1%.
- **SFT vs ICL**: Fine-tuned SFT branches vastly outperform In-Context Learning (ICL) setups (using GPT-4o/DeepSeek V3) by 10-15%, verifying data volume beats zero-shot scaling for granular alignment.
- **Robust Category Scaling**: Excels in sparse categories, resolving data triggers to maintain solid accuracy across fear and low-frequency emotions (anger: 74.6%).

### 4. Application Strategy
- **Hierarchical Adjudication Logic**: A 3-tier rule structure can resolve **cross-modal conflicts** (e.g., text vs. audio) by using calibrated thresholds to determine which modality is more "certain."
- **Confidence-Weighted Soft Label Synthesis**: Weight assignment logic can serve to generate robust **soft labels** (consensus distributions) instead of simple averages, ensuring outlier models do not disproportionately bias the result.

## 🕸️ Relations
- **Builds on**: [[BERT]], [[DeBERTa]], [[Gemini]], [[ChatGPT-4o]], [[Multi-label Classification]]
- **Relevant to**: [[Emotion Recognition in Conversation (ERC)]], [[Prompt Engineering]], [[Ensembling]], [[Voting]], [[Soft Labels]]
- **Connects with**: [[AER-LLM]] (distributional labels) and [[Emotion-LLaMA]] (multimodal backbones)