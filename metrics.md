# Evaluation Metrics in Emotion Recognition
 
 This file summarizes the core quantitative and qualitative metrics used to evaluate Large Language Models (LLMs) and Multimodal models for emotion analysis in this workspace.
 
 ---
 
 ## 1. Categorical Classification (Discrete Emotions)
 
 These metrics assess a model's ability to classify an utterance or video sequence into a fixed set of categories (e.g., Happy, Sad, Anger, Neutral).
 
 *   **F1-Score (Macro / Weighted)**: Harmonic mean of Precision and Recall. Essential for imbalanced datasets like **MELD** and **IEMOCAP** where certain emotions (e.g., Neutral, Joy) outnumber rarer ones (e.g., Fear, Disgust).
 *   **Accuracy (Acc / Weighted Accuracy)**: Ratio of correctly predicted samples. supplement by F1 intervals.
 ---
 
 ## 2. Soft Labels & Distributional Metrics
 
 Evaluates a model's ability to output a **probability distribution** or continuous commitment rather than a single hard label. This is crucial for capturing human ambiguity and subjective interpretation.
 
 *   **Jensen-Shannon Divergence (JSD)**:
     *   *Definition*: A symmetric measure of the similarity between two probability distributions (prediction vs. human consensus).
     *   *Use Case*: **AER-LLM** uses JSD to calibrate ambiguity buckets. Low JSD scores indicate the model matches the statistical spread of multiple human raters.
 *   **Kullback-Leibler Divergence (KLD)**:
     *   *Definition*: Measures how one probability distribution diverging from a second, expected probability distribution.
     *   *Use Case*: Standard for testing continuous alignment weights inside modular soft-label adapters.
 *   **Bhattacharyya Distance**:
     *   *Definition*: Measures the overlap between two statistical samples or distributions.
     *   *Use Case*: Compressing distance weights inside contrastive memory or ambiguity calibration thresholds.
 *   **Expected Calibration Error (ECE)**:
     *   *Definition*: Evaluates the continuous absolute agreement between prediction confidence and actual accuracy.
     *   *Use Case*: Essential for validating if the soft label probability (e.g., 0.8 joy) correctly correlates with the 80% likelihood of that prediction holding.
