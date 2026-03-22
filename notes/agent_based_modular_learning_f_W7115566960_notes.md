---
title: "Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems"
authors: ["Matvey Nepomnyaschiy", "Oleg Pereziabov", "Anvar Tliamov", "Stanislav Mikhailov", "Ilya Afanasyev"]
year: 2025
publication_date: "2025-12-02"
doi: "10.48550/arxiv.2512.10975"
openalex_id: "W7115566960"
citation_count: 0
status: Read
tags:
  - paper
  - multi_agent
  - modular_architecture
  - emotion_recognition
  - mosei
---

# 📝 Agent-Based Modular Learning for Multimodal Emotion Recognition

> [!ABSTRACT] TL;DR
> To bypass the computational bottlenecks and inflexibility of monolithic Multi-Modal Emotion Recognition (MER) models (which require full system retraining to update a single modality), this paper introduces a **Multi-Agent Supervisor framework**. Distinct modules (Vision/Audio/Text) act as autonomous agents passing normalized embeddings to a central orchestrator. Using a Ridge Regression Adapter, it bypasses costly retraining whenever a specific extractor is swapped out.

## 🔗 Quick Links
- **Parsed Text**: [[agent_based_modular_learning_f_W7115566960_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.48550/arxiv.2512.10975)

## 📚 Reading Notes

### 1. Core Objectives
-   Monolithic MER architectures (where text, vision, and audio are tightly coupled during early training) suffer from "catastrophic forgetting" and unmanageable retraining loops whenever one modality encoder becomes outdated.
-   The paper proposes loosely coupling the distinct sensory inputs into **independent agents**, coordinated by a "Supervisor Classifier".

### 2. Methodological Approach
#### (A) Autonomous Modality Agents
-   **Facial Emotion Detection (FED)**: YOLOv8-Face isolating frames passing into a frozen ResNet-50 block (512-dim output).
-   **Speech Emotion Recognition (SER)**: Passes WAV audio into *emotion2vec+* (256-dim output).
-   **Text Emotion Detection (TED)**: Orchestrates OpenAI Whisper Large V3 Turbo for transcription, sliding into FRIDA embeddings (768-dim output).
-   **Audio Event Detection (AED)**: CNN-14 on AudioSet providing *auxiliary side-information*. Crucially, AED is NOT treated as a primary fusion vector; it acts as a gatekeeper (a speech-presence tracker metadata tag) controlling downstream threshold confidences.

#### (B) Dimension Normalization & Adapter Transformation
-   Extractors output varying dimensions (512, 256, 768). The framework zeroes-pads or truncates them uniformly to 1024-dim per modality (totaling a 3072-dim concatenated sequence vector).
-   **Ridge Regression Adapter**: Maps arbitrary new pipeline embeddings directly back into the normalized CMU-MOSEI embedding space. This means downstream pre-trained classifiers expect identical spatial features even when the entire vision or audio backbone algorithm differs, massively saving processing time.

### 3. Key Findings
-   **Efficiency over pure SOTA**: Using CatBoost as the central orchestrator fusion, it scored **0.541 Accuracy and 0.534 weighted F1** on the 5-class MOSEI dataset. While not shattering accuracy records, the training loop dropped to *2.25 minutes* per 1000 iterations for the classifier/fusion step. The Ridge Adapter trained in *2-3 minutes*.
-   Replacing a modality encoder no longer breaks the overarching model; the adapter just realigns the localized shift in minutes instead of retraining the fusion weights across days.

### 4. Application Strategy
-  Nothing

## 🕸️ Relations
-   **Builds on**: [[MOSEI]]; [[Multi-Agent Systems]]; [[emotion2vec]]; [[Whisper Large V3 Turbo]]
-   **Relevant to**: [[Late Fusion Strategies]]; [[Adapter Tuning]]; [[Catastrophic Forgetting]]
-   **Pairs well with**:
  - [[Third-person Appraisal Agent for Synthetic Conversations]] (Validating modular extraction of discrete channels before a central decision agent resolves them)
