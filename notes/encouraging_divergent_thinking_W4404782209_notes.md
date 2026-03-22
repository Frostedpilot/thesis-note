---
title: "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate"
authors: ["Tian Liang", "Zhiwei He", "Wenxiang Jiao", "Xing Wang", "Yan Wang", "Rui Wang", "Yujiu Yang", "Shuming Shi", "Zhaopeng Tu"]
year: 2024
publication_date: "2024-05-30"
doi: "10.18653/v1/2024.emnlp-main.992"
openalex_id: "W4404782209"
citation_count: 151
status: Read
tags:
  - paper
  - multi_agent_debate
  - chain_of_thought
  - self_reflection
  - degeneration_of_thought
---

# 📝 Encouraging Divergent Thinking in LLMs through Multi-Agent Debate

> [!ABSTRACT] TL;DR
> Standard self-reflection capabilities in LLMs suffer from "Degeneration-of-Thought" (DoT) where models fail to spot their own initial errors. This paper proposes a **Multi-Agent Debate (MAD)** framework featuring distinct debater agents (Affirmative/Negative) operating in a "tit-for-tat" state and a designated Judge agent. MAD encourages divergent thinking and proves that zero-shot GPT-3.5 with debate can outperform standalone GPT-4 on complex reasoning.

## 🔗 Quick Links
- **Parsed Text**: [[encouraging_divergent_thinking_W4404782209_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.992)

## 📚 Reading Notes

### 1. Core Objectives
- Identify and solve the **Degeneration-of-Thought (DoT)** problem in LLM problem-solving.
- DoT is defined as: *Once an LLM establishes confidence in an incorrect answer, it cannot generate novel thoughts to repair the flaw via standard Self-Reflection.* Variables causing this include inherent text bias, rigidity (resistance to changing beliefs), and lack of external conflicting feedback.

### 2. Methodological Approach (MAD Framework)
The **Multi-Agent Debate (MAD)** framework splits the inference task into three distinct roles governed by specific prompt injections.

#### (A) Debaters (Affirmative and Negative)
-   $N$ debaters (typically $N=2$) independently express their thoughts in sequence.
-   Their Meta Prompts require a "tit for tat" stance: *"It’s not necessary to fully agree with each other’s perspectives, as our objective is to find the correct answer."*
-   *Constraint*: Giving prompts too hostile an instruction ("must disagree on EVERY point") causes polarization. Moderate "tit for tat" produced the best divergent logic.

#### (B) The Judge
-   A standalone agent role monitoring the debate history ($H$).
-   **Discriminative Mode**: Automatically decides if the correct solution has been obtained. If `True`, the judge triggers an **Adaptive Break** to early-stop the debate (saving compute tokens).
-   **Extractive Mode**: Extracts the final answer if the maximum sequence ceiling is reached. 

### 3. Key Findings & Trade-offs
-   **MAD Outperforms Zero-Shot Scaling**: GPT-3.5-Turbo using the MAD debate framework surpassed zero-shot GPT-4 on the Commonsense Machine Translation benchmarks, proving explicit multi-agent dialectics can beat brute parameter count.
-   **Adaptive Breaks are Critical**: The best outputs happen early (Iterations 1-2). Forcing agents to continue debating *after* uncovering the truth degrades the performance as they begin hallucinating reasons to disagree.
-   **Judge Bias**: If the judge model is an OpenAI model (GPT-4), it will actively favor the arguments presented by an OpenAI debater over an open-source debater (like Vicuna-13B), regardless of factual superiority. The judge is not impartial across architectures.

### 4. Application Strategy
-   **Cross-Modal Conflict Resolution**: In scenarios where audio and text provide conflicting emotional signals, a powerful architectural topology is treating each modality tracker as a "Debater". E.g., The Audio-Agent argues "The tone is sarcastic", the Text-Agent argues "The word choice is joyful". Sending both to a final Judge-Agent to evaluate and issue a floating-point "Soft Label" mimics the MAD framework to solve incongruity without fine-tuning.
-   **Compute Considerations**: MAD costs roughly 2.46$\times$ more tokens than standard Chain-of-Thought, but heavily increases accuracy on heavily counter-intuitive traps.

## 🕸️ Relations
-   **Builds on**: [[Chain-of-Thought Prompting]]; [[Self-Refine / Reflexion]]
-   **Relevant to**: [[Degeneration-of-Thought]]; [[LLM as a Judge Bias]]
-   **Pairs well with**:
  - [[Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems]] (Combining modal agents with a debate framework for conflict).
