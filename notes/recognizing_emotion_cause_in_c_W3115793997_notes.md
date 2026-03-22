---
title: "Recognizing Emotion Cause in Conversations"
authors: ["Soujanya Poria", "Navonil Majumder", "Devamanyu Hazarika", "Deepanway Ghosal", "Rishabh Bhardwaj", "Samson Yu Bai Jian", "Pengfei Hong", "Romila Ghosh", "Abhinaba Roy", "Niyati Chhaya", "Alexander Gelbukh", "Rada Mihalcea"]
year: 2021
publication_date: "2021-09-01"
doi: "10.1007/s12559-021-09925-7"
openalex_id: "W3115793997"
citation_count: 11
status: Read
tags:
  - paper
  - dataset
  - emotion_cause_extraction
  - emotion_reasoning
  - dialogue_systems
---

# 📝 Recognizing Emotion Cause in Conversations (RECCON)

> [!ABSTRACT] TL;DR
> This paper shifts the focus from simple emotion *detection* (evidence) to discovering the *origin* (cause) of emotions in multi-turn dialogues. It introduces the RECCON dataset (over 1,000 dialogues from DailyDialog and IEMOCAP annotated with 10k utterance-causal span pairs) and formulates two NLP sub-tasks: Causal Span Extraction and Causal Emotion Entailment, baselined on RoBERTa and SpanBERT.

## 🔗 Quick Links
- **Parsed Text**: [[recognizing_emotion_cause_in_c_W3115793997_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1007/s12559-021-09925-7)

## 📚 Reading Notes

### 1. Core Objectives
-   Distinguish between **Emotion Evidence** (the expression of the emotion, "I am angry") and **Emotion Cause** (the stimuli, "because you broke my watch").
-   Identify the challenges of mapping causality in unstructured multi-party dialogues (e.g., Affective Primacy, complex co-references, and latent reasons).

### 2. Methodological Approach (The RECCON Dataset & Tasks)

#### Categorization of Causes
The authors define 5 predominant types of emotion causes:
1.  **No Context:** Cause is explicitly in the same utterance as the target emotion.
2.  **Inter-Personal Emotional Influence:** The cause belongs to the other speaker (e.g. *Trigger Events* or *Emotional Dependency*).
3.  **Self-Contagion:** A stable mood carries over from the participant's own past turn.
4.  **Hybrid:** Combination of 2 and 3.
5.  **Latent:** Cannot be pinned to an explicit textual span.

#### Dataset Statistics
-   Extracts subsets from **IEMOCAP** and **DailyDialog**.
-   Crucial Finding: In IEMOCAP, over **40% of emotion causes** lie at least 3 timestamps back in conversational history, necessitating long-term memory modeling.

#### Baseline Sub-tasks
-   **Causal Span Extraction**: Evaluated as a Machine Reading Comprehension (MRC) task (similar to SQuAD). They test a Fine-Tuned SpanBERT against RoBERTa.
-   **Causal Emotion Entailment**: Predict which specific historical utterances are responsible for the target emotion, scored as a binary text-pair or triplet classification task using RoBERTa Base/Large.

### 3. Key Findings & Trade-offs
-   **Context is King**: Supplying conversational context drastically improves performance over classifying isolated utterance metadata. SpanBERT performs better at span extraction (Macro F1: ~75%) but only when context is provided.
-   **Difficult Modeling**: The entailment problem is far from solved. RoBERTa Large hits peak Macro F1 scores around ~77% on DailyDialog and struggles significantly on IEMOCAP (~68%), proving that simply scaling a generic transformer does not resolve conversational causality limits. The complex neural baselines (like ECPE-2D, ECPE-MLL) often fail to beat simple RoBERTa.

### 4. Application Strategy
-   **Resolving Cross-Modal Conflict via Causal Spans**: When audio and text modalities present conflicting emotional signals (e.g., sarcastic tone vs. polite text), prompting an LLM Agent to extract the explicit *Causal Span* from the dialogue history can objectively ground the conflict resolution. If the agent identifies the cause as a past "Inter-Personal Trigger" (e.g., a previous insult), it can confidently output a negative soft label weighting, trusting the audio over the text.
-   **Prompt-Based Sequence Classification**: Instead of fine-tuning, utilize the RECCON taxonomy directly within the system prompt of the Emotion Inference Agent. Instructing the agent to first classify the *cause type* (Self-Contagion vs. Inter-Personal) before predicting the final continuous emotion soft label forces a structured Chain-of-Thought, preventing hallucinations when cross-modal inputs clash.

## 🕸️ Relations
-   **Builds on**: [[DailyDialog]]; [[IEMOCAP]]
-   **Relevant to**: [[Emotion Cause Pair Extraction (ECPE)]]; [[Machine Reading Comprehension]]
-   **Pairs well with**:
  - [[CDEA Causality Driven Dialogue]] (For building the directed acyclic graphs corresponding to RECCON pairs).
