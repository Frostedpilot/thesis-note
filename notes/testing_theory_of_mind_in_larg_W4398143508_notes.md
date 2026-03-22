---
title: "Testing theory of mind in large language models and humans"
authors: ["James W. A. Strachan", "Dalila Albergo", "Giulia Borghini", "Oriana Pansardi", "Eugenio Scaliti", "Saurabh Gupta", "K. B. Saxena", "Alessandro Rufo", "Stefano Panzeri", "Guido Manzi", "Michael S. A. Graziano", "Cristina Becchio"]
year: 2024
publication_date: "2024-05-20"
doi: "10.1038/s41562-024-01882-z"
openalex_id: "W4398143508"
citation_count: 186
status: Read
tags:
  - paper
  - theory_of_mind
  - cognitive_evaluation
  - prompt_engineering
---

# 📝 Testing theory of mind in large language models and humans

> [!ABSTRACT] TL;DR
> This paper subjects prominent LLMs (GPT-4, GPT-3.5, and LLaMA2-70B) and 1,907 human participants to a rigorous psychological test battery measuring Theory of Mind (ToM) capabilities. It tests false beliefs, irony, indirect requests (hinting), and faux pas. While GPT-4 matches or exceeds humans on most tests, it famously struggles with Faux Pas due to "hyperconservatism"—an RLHF-induced refusal to commit to attributing ignorance without explicit evidence, highlighting a massive gap between cognitive competence and behavioral performance.

## 🔗 Quick Links
- **Parsed Text**: [[testing_theory_of_mind_in_larg_W4398143508_parsed|View/Edit Parsed Source]]
- **Online**: [DOI](https://doi.org/10.1038/s41562-024-01882-z)

## 📚 Reading Notes

### 1. Core Objectives
-   Evaluate if the apparent Theory of Mind reasoning exhibited by modern Large Language Models is grounded in actual robust inference capability or simply shallow heuristics and test-data memorization.
-   Move beyond singular test evaluations (e.g., just the False Belief test) to a comprehensive testing battery indexing multiple facets of mentalistic tracking (misdirection, irony, faux pas, indirect requests).
-   Distinguish between **cognitive competence** (the ability to compute complex mentalistic inferences) and **performance** (the threshold required to output those inferences under uncertainty).

### 2. Methodological Approach (ToM Battery & Controls)
The study translates standardized human psychological exams into zero-shot prompts for LLMs and generates completely novel semantic variants to control for pre-training data memorization.

-   **The Battery of Tests**:
    1.  **False Belief**: Does character B know that character A moved an object?
    2.  **Irony**: Differentiating literal from non-literal intent based on context.
    3.  **Hinting Task**: Understanding indirect speech requests (e.g., "It's a bit cold in here" = "Close the window").
    4.  **Strange Stories**: Advanced mentalizing about misdirection, lying, and second-order beliefs.
    5.  **Faux Pas**: Recognizing when someone accidentally offends another due to a lack of knowledge/memory constraint.
-   **The Belief Likelihood Variant**: To debug GPT-4's failure on the Faux Pas test, the prompt "Did they know..." was rewritten to "Is it *more likely* that they knew or didn't know...". This reduces the threshold of absolute certainty required to answer.

### 3. Key Findings & Trade-offs
-   **Supra-Human Performance**: GPT-4 outperformed the human baselines in recognizing Irony, Hinting, and Strange Stories.
-   **The Faux Pas Failure**: GPT-4 and GPT-3.5 failed the standard Faux Pas test. They successfully identified that the victim would feel insulted, but when asked if the speaker *knew* what they said was offensive, the models responded that "there was not enough information provided to be sure."
-   **Hyperconservatism vs. Incompetence**: When tested on the "Belief Likelihood Variant" (asking what was *more likely*), GPT-4 scored perfectly (100%). This proved the model possessed the underlying **competence** to infer the hidden mental state, but RLHF safety alignment and hallucination mitigation caused **hyperconservatism**—a reluctance to output a high-confidence claim without explicit textual proof.
-   **Illusory Competence in Open Weights**: LLaMA2-70B was the only model to beat humans on the standard Faux Pas test. However, control tests proved this was merely an algorithmic bias toward assuming ignorance in all situations, rather than actually integrating context.

### 4. Application Strategy
-   **Prompt Instructions for Bypassing Epistemic Caution**: When resolving cross-modal conflict using agentic soft labels (e.g., predicting an emotion when Audio provides anger and Text provides neutral), standard zero-shot prompts might trigger the same "hyperconservatism" seen in the Faux Pas test ("There is not enough information to decide which modality is right"). By rewording the meta-prompt to aggressively authorize probabilistic reasoning (e.g., *"Based on the trajectory of the dialogue state, what is the MOST LIKELY underlying emotion driving this contradiction?"*), the agent is forced past the RLHF safety threshold to actually leverage its ToM capability and output a decisive continuous variable for the label.
-   **Third-Person Belief Anchoring**: Use the paper's *Strange Stories* taxonomy to explicitly instruct LLM evaluators to represent second-order mental states in conflict resolution. Before an agent assigns a cross-modal soft label, it can be prompted to articulate what Speaker A *thinks* Speaker B is currently feeling, adding an interpretable causal edge strictly derived from the text.

## 🕸️ Relations
-   **Builds on**: [[Machine Reading Comprehension]]
-   **Relevant to**: [[CDEA Causality Driven Dialogue]] (For second-order belief structures).
-   **Useful for**: Creating probabilistic prompts for the orchestrator loops of [[Agent-Based Modular Learning for Multimodal Emotion Recognition]].
