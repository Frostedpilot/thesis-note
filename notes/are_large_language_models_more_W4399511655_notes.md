---
title: "Are Large Language Models More Empathetic than Humans?"
authors: ["Anuradha Welivita", "Pearl Pu"]
year: 2024
publication_date: "2024-06-07"
doi: "10.48550/arxiv.2406.05063"
openalex_id: "W4399511655"
citation_count: 5
status: Read
tags:
  - paper
  - empathy
  - evaluation
  - llm
  - conversational_ai
---

# 📝 Are Large Language Models More Empathetic than Humans?

> [!ABSTRACT] TL;DR
> This study investigates whether state-of-the-art LLMs can surpass humans in generating empathetic responses. Utilizing a **between-subjects** design with **1,000 participants**, the research compares human responses to those from **GPT-4**, **LLaMA-2-70B-Chat**, **Gemini-1.0-Pro**, and **Mixtral-8x7B-Instruct**. All tested LLMs outperformed the human baseline, with **GPT-4** demonstrating the highest performance (+31% in "Good" ratings).

## 🔗 Quick Links
- **PDF**: [[papers/are_large_language_models_more_W4399511655/are_large_language_models_more_W4399511655.pdf|Open Local PDF]]
- **Parsed Text**: [[are_large_language_models_more_W4399511655_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.48550/arxiv.2406.05063)

## 📚 Reading Notes

### 1. Core Objectives
- The paper evaluates LLMs beyond cognitive capabilities, focusing on **social-emotional intelligence** and empathetic response generation in comparison to human performance across 32 positive and negative emotions.
- **Source Data**: Evaluated using 2000 dialogue triggers from the **EmpatheticDialogues** dataset, balanced across 32 fine-grained emotions. Sample focuses strictly on the **first two dialogue turns** plus situation description.

### 2. Methodological Approach
#### (A) Between-Subjects Design
- **Rigor**: Leverages a **between-subjects** design with 5 separate groups of 200 participants (1000 total). This prevents **carry-over** or **order effects** where exposure to one model biases another.
- **Power Analysis**: Supported by G-Power to ensure statistical minimums (Minimum sample of 253 required; 1000 achieved for high-confidence significance testing).

#### (B) Structured Empathy Prompting
- Models were prompted with strict definitions addressing three dimensions of empathy (Cognitive, Affective, Compassionate).
- **Prompt Constraints**: Instructions enforced a specific word count limit: average of **28 words** and a maximum of **97 words** to ensure concise output.

#### (C) 3-Point vs 5-Point Scale Selection
- A **3-point scale** (Bad, Okay, Good) was objectively preferred over a 5-point scale due to higher inter-rater agreement. 
- **Metrics**: 3-point achieved fair agreement (**Cohen's Kappa = 0.2817**) while 5-point triggered poor agreement (**Kappa = 0.1813**) and lower correlation with automatic tools like EPITOME.

### 3. Key Findings
- **LLM Dominance**: All evaluated LLMs significantly outperformed the human benchmark.
- **Rankings**: **GPT-4** leads (+31% "Good" ratings), followed by LLaMA-2 (+24%), Mixtral-8x7B (+21%), and Gemini-Pro (+10%).
- **Finer Granularity Success**:
  - **Positive Emotions**: Major LLM gains in *Impressed*, *Surprised*, *Grateful*, *Proud*, and *Confident*.
  - **Negative Emotions**: narrower margin. Statistical gains were only observed for *Afraid*, *Apprehensive*, *Anxious*, and *Annoyed*.
- **The Negative Gap Constraint**: Evaluators judge responses to negative emotions with higher scrutiny; LLMs still beat crowdsourced humans here, but generally struggle to master empathetic nuance standard in trained therapists.

### 4. Application Strategy
- **Emotion-Targeted Prompting**: Performance disparities in negative emotions suggest that prompt engineering with tailored instructions or few-shot demonstrations could enhance response quality.

## 🕸️ Relations
- **Relevant to**: [[Empathetic response generation]]; [[Human evaluation]]; [[Emotion Recognition in Conversation]]
- **Connects with**: [[AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models]] (multimodal and distribution-based approaches)