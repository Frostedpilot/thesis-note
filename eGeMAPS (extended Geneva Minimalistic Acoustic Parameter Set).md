

## Definition

The **extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS)** is a standardized set of acoustic features widely used in voice research and affective computing. Developed to establish a common baseline for acoustic feature extraction, it selects a minimalistic, theoretically grounded set of parameters (88 features) that are highly correlated with human paralinguistic behaviors, rather than relying on brute-force extraction of thousands of features.

Features are typically extracted using standard audio processing toolkits like [[https://github.com/audeering/opensmile|OpenSMILE]].

## Core Feature Categories

The 88 features in eGeMAPS are statistically aggregated over time windows and fall into three primary categories:

- **Frequency related parameters:** Pitch ($F_0$), jitter, and formants (which capture vocal tract resonances).
    
- **Energy/Amplitude related parameters:** Loudness, shimmer, and harmonics-to-noise ratio (HNR).
    
- **Spectral parameters:** Alpha ratio, Hammarberg index, spectral slopes, and Mel-Frequency Cepstral Coefficients ([[MFCCs]]).
    

## Role in LLM-Based Emotion Detection

While Large Language Models (LLMs) excel at extracting semantic meaning from text, they inherently lack access to the paralinguistic cues of speech (the "how" something was said, rather than the "what"). eGeMAPS bridges this gap in multimodal emotion recognition:

- **Multimodal Fusion:** In modern affective AI, an audio signal is processed to extract eGeMAPS features, which form a dense acoustic vector. This vector is then fused with the LLM's text embeddings (via cross-attention mechanisms or simple concatenation after a projection layer).
    
- **Contextualizing Sarcasm and Nuance:** Text alone might read as positive ("Oh, great job"), but eGeMAPS parameters indicating low pitch variance and heavy vocal fry can signal to the LLM that the utterance is actually sarcastic or frustrated.
    
- **Prompting with Acoustic Tokens:** In some advanced architectures, quantized eGeMAPS features are converted into discrete "acoustic tokens" and appended to the text prompt, allowing a standard text-based LLM to reason over acoustic variations directly.
    

## Key Advantages in Modern AI

- **Interpretability:** Unlike deep embeddings from raw audio (e.g., wav2vec 2.0 representations), eGeMAPS features have direct physical and psychological interpretations (e.g., high pitch variability equates to high arousal).
    
- **Dimensionality:** Its small, fixed size (88 dimensions) makes it highly efficient to compute and fuse with large transformer architectures without causing dimensionality bottlenecks.
    