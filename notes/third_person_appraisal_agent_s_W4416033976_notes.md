---
title: "Third-Person Appraisal Agent: Simulating Human Emotional Reasoning in Text with Large Language Models"
authors: ["Simin Hong", "Jun Sun", "Hongyang Chen"]
year: 2025
publication_date: "2025-01-01"
doi: "10.18653/v1/2025.findings-emnlp.1288"
openalex_id: "W4416033976"
citation_count: 0
status: Read
tags:
  - paper
  - agentic_workflow
  - appraisal_theory
  - erc
  - llm
  - soft_labels
  - conflict_resolution
---

# 📝 Third-Person Appraisal Agent: Simulating Human Emotional Reasoning in Text with Large Language Models

> [!ABSTRACT] TL;DR
> This paper proposes a **third-person cognitive appraisal agent** that simulates human emotional reasoning in text. By forcing an LLM to generate **structured appraisals** (evaluating goals, desires, and expectations) before making predictions, the framework aligns AI outcomes with human psychological processes. The **Counterfactual Reasoning** loop in the secondary phase proves highly effective for refining predictions under ambiguous conditions.

## 🔗 Quick Links
- **PDF**: [[papers/third_person_appraisal_agent_s_W4416033976/third_person_appraisal_agent_s_W4416033976.pdf|Open Local PDF]]
- **Parsed Text**: [[third_person_appraisal_agent_s_W4416033976_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.18653/v1/2025.findings-emnlp.1288)

## 📚 Reading Notes

### 1. Core Objectives
- Most existing ERC models rely on surface-level sentiment cues or static feature extraction, avoiding deep causal dynamics.
- Moving toward **structured emotional reasoning** requires understanding the connections between motivations, thoughts, and emotional expressions using **Cognitive Appraisal Theory**.
- **Reward metrics**: Critic evaluates appraised alignment weights using the **NRC-VAD lexicon** and **Circumplex Model** to bound Valence-Arousal scores.

### 2. Methodological Approach
#### (A) Three-Phase Appraisal Agentic Workflow
- **Phase 1 (Primary)**: Generator LLM infers initial emotions by evaluating context against speaker goals. Context window size is optimal at **$l=5$**.
- **Phase 2 (Secondary)**: Uses an Appraisal Evaluator LLM for feedback. Iterates Counterfactual Reasoning hypothesizing alternative responses until accurate or hitting iteration maximum $K$.
- **Phase 3 (Reappraisal)**: Reinforced Fine-Tuning (**ReFT**) using Actor-Critic loops. ReFT combines Action Reward and Critic Reward using grids-searched coefficients **$\alpha = 0.9$** and **$\beta = 0.45$**.

#### (B) Agent Tuning parameters
- LoRA parameters utilized include 4-bit adapters with rank **$r=16$**, utilizing pre-trained backbones like Mistral-7B-Instruct.

### 3. Key Findings
- **Counterfactual over Reflexion**: Counterfactual iterations yield a **30% increase** in accuracy over Reflexion loops, which only state errors without hypotheticals for correction.
- **Generalization Gains**: Generator model outperforms 12B baselines on unseen datasets like DailyDialog, validating structured appraisal benchmarks fit cross-domain scaling targets cleanly.
- **6-Dimensional Evaluation Metric Rating**: Assessment uses LLMs to score interpretability based on:
  1. *Sentiment Awareness*
  2. *Contextual Understanding*
  3. *Sensitivity to Emotional Causes*
  4. *Emotional Dynamics Responsiveness*
  5. *Motivational Understanding* (Highest offset gain for Generator)
  6. *Clarity and Coherence Assessment*

### 4. Application Strategy (Prompt-Based / Agentic)
- **Conflict Adjudication via Counterfactuals**: A **"Multimodal Counterfactual Phase"** can trigger when modalities yield conflicting distribution peaks (e.g., "If the speaker were actually sad, what goal would they be achieving?"). This aids in uncovering sarcasm.
- **Soft-Label Reward Calibration**: Critic Rewards can adapt to continuous distributions. Adjustments can reward the model based on **KL or JS-Divergence** between generated distributions and annotator soft labels.
- **Appraisal as a "Modality Weaver"**: Primary appraisals can explicitly formulate **Modality trust priors**—assessing whether a speaker’s inferred motivation suggests a mask that prioritizes acoustic branches over text.

## 🕸️ Relations
- **Builds on**: [[Lazarus’s Appraisal theory of emotion]]; [[Actor-Critic]]; [[ReFT]]
- **Relevant to**: [[Emotion Recognition in Conversation]]; [[Agentic workflows]]; [[Cross-modal conflict]]; [[Soft Labels]]; [[Conflict Resolution]]
- **Pairs well with**:
  - [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (distributional outputs)
  - [[LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics]] (speaker characteristics as latent evidence)
  - [[Emotion-LLaMA]] (multimodal backbones)