---
title: "Modeling Intra and Inter-modality Incongruity for Multi-Modal Sarcasm Detection"
authors: ["Hongliang Pan", "Zheng Lin", "Peng Fu", "Yatao Qi", "Weiping Wang"]
year: 2020
publication_date: "2020-01-01"
doi: "N/A"
openalex_id: "W4404412858"
status: Read
tags:
  - paper
  - sarcasm_detection
  - inter_modal_incongruity
  - co_attention
  - textual_visual_conflict
---

# 📝 Modeling Intra and Inter-modality Incongruity for Sarcasm Detection

> [!ABSTRACT] TL;DR
> This paper proposes a BERT-based framework that captures **cross-modal incongruity** (contradiction) for sarcasm detection. Instead of simply concatenating visual/textual features, it applies **inter-modality attention** (using text as Query, images as Key/Value) to force the model to look at visual sub-regions that contradict the text statement. It also models **intra-modality contradiction** within text (hashtags vs body) using an affinity co-attention matrix.

## 🔗 Quick Links
- **Parsed Text**: [[modeling_inter_modal_incongruo_W4404412858_parsed|View/Edit Parsed Source]]

## 📚 Reading Notes

### 1. Core Objectives
-   Existing multi-modal architecture setups typically concatenate or smooth-fuse feature weights, overlooking explicit **negation signals** and incongruity (contradictory vectors) fundamental to sarcasm/sentiment flips.
-   Concentrates on modeling both inter-modality (text vs image) and intra-modality (text sentence vs hashtag topics) contradiction buffers explicitly to drive accurate classification.

### 2. Methodological Approach
#### (A) Inter-Modality Attention (Cross-Modal)
-   **Text-Image Matching Layer**: Accepts encoded text representations as Query ($Q$), and visual regions (ResNet-152 layer) as Key ($K$) and Value ($V$).
-   Forces sequential triggers to generate **high attention values targeting image sub-nodes** that refute the text assertions (e.g., text states "Packed game", attention hits empty visual seats).
-   Stacks 3 layers for peak threshold weights.

#### (B) Intra-Modality Attention (Within-Text)
-   Fuses a **Co-attention Affinity Matrix** to model contradiction pointers between original text buffers and standalone hash variables (e.g., "Woke up at 5am #not").
-   Applies max-pooling operators directly on matrix columns to amplify discriminative incongruity highlights over sequential representations.

### 3. Key Findings
-   **SOTA F1 Boosts**: Hits **82.92% F1** on standard Twitter multi-modal grids, comfortably beating fine-tuned BERT standalone (80.22%) and concatenation benchmarks (Res-BERT 81.57%).
-   **Optical Characters Trigger (OCR)**: Combining text *found inside the images* leveraging basic reading nodes boosts peak F1 to **86.18%**, sealing OCR text elements as critical secondary negators.

### 4. Application Strategy
- Nothing
## 🕸️ Relations
-   **Builds on**: [[BERT]]; [[Co-attention matrix]]; [[Hierarchical Fusion Model]]
-   **Relevant to**: [[Incongruity Resolution]]; [[Negation detection]]; [[Optical Character Recognition]]
-   **Pairs well with**:
  - [[DialogueLLM: Context and Emotion Knowledge-Tuned LLMs]] (Translating multimodal conflicts to description buffers thresholds)
  - [[CDEA: Causality-Driven Dialogue Emotion Analysis]] (Tracing causal contradictions)
---
> [!IMPORTANT] Note on Metadata
> The title and authors were updated based on the actual parsed text contents provided in the workspace papers folder, reflecting Pan et al., rather than the initial template link.
