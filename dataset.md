# Datasets for Multimodal Emotion Recognition
 
 This file provides a descriptive breakdown of core datasets used for Large Language Model (LLM) and Multimodal Small model training and evaluation in this workspace.
 
 ---
 
 ## 1. IEMOCAP (Interactive Emotional Dyadic Motion Capture)
 
 *   **Modality**: Multimodal (Video, Audio, Text, Motion Capture of Face/Hand)
 *   **Size**: ~12 hours of recorded audiovisual data across 5 dyadic sessions (10 actors: 5 male, 5 female).
 *   **Taxonomy**: 
     *   *Discrete*: Anger, Sadness, Happiness, Surprise, Fear, Disgust, Frustration, Excited, Neutral.
     *   *Dimensional*: Valence, Activation, Dominance (VAD scores on continuous scales).
 *   **Key Characteristic**: Improvised or scripted dyadic interactions. Requires **long-history context tracking** to resolve ambiguity (e.g., distinguishing "excited" from "frustration" using actor prior history).
 
 ---
 
 ## 2. MELD (Multimodal EmotionLines Dataset)
 
 *   **Modality**: Multimodal (Video, Audio, Text)
 *   **Size**: ~13,000 utterances extracted from the popular TV show *Friends*.
 *   **Taxonomy**: 
     *   *Emotions*: Anger, Disgust, Fear, Joy, Sadness, Surprise, Neutral.
     *   *Sentiment*: Positive, Negative, Neutral.
 *   **Key Characteristic**: **Multi-party dialogue** setups. Contains high background noise, frequent speaker overlaps, and sarcastic peaks. Typically yields lower absolute F1 scores than IEMOCAP due to its open-domain, multi-speaker noise levels.
 
 ---
 
 ## 3. CA-MER (Conflict-Aware Multimodal Emotion Reasoning)
 
 *   **Modality**: Multimodal (Video, Audio, Text)
 *   **Key Targets**:
     *  Construct from **MER-2023** dataset, with the 6 common emotion and the neutral emotion
     *  Specifically designed to evaluate MLLMs under emotion conflicts.
 *   **Key Characteristic:** Contain 3 set: audio-aligned, vision-aligned which refer to the modality containing true label, and at least one of the other 2 contain wrong label, and consistent, which all 3 modality contain true label.
