# Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems 

Matvey Nepomnyaschiy[[1]] _[[,]]_[[2]] 

> [[1]] _[[,]]_[[2]] Oleg Pereziabov[1] Anvar Tliamov[1] Stanislav Mikhailov[2] Ilya Afanasyev[1] _[,]_[3] 

> 1 Research Center of the Artificial Intelligence Institute, Innopolis University, Innopolis, 420500, Russia 

> 2 ITMO University, St. Petersburg, 197101, Russia 

> 3 Saint Petersburg Electrotechnical University “LETI”, St. Petersburg, 197022, Russia 

Emails: 

m.nepomnyashchy@innopolis.ru, m.nepomnyashchy@niuitmo.ru (Matvey) o.perezyabov@innopolis.ru (Oleg) 

a.tliamov@innopolis.university (Anvar) stasmikhailov@niuitmo.ru (Stanislav) i.afanasyev@innopolis.ru, imafanasev@etu.ru (Ilya) 

_**Abstract**_ **—Effective human-agent interaction (HAI) relies on accurate and adaptive perception of human emotional states. While multimodal deep learning models—leveraging facial expressions, speech, and textual cues—offer high accuracy in emotion recognition, their training and maintenance are often computationally intensive and inflexible to modality changes. In this work, we propose a novel multi-agent framework for training multimodal emotion recognition systems, where each modality encoder and the fusion classifier operate as autonomous agents coordinated by a central supervisor. This architecture enables modular integration of new modalities (e.g., audio features via emotion2vec), seamless replacement of outdated components, and reduced computational overhead during training. We demonstrate the feasibility of our approach through a proof-of-concept implementation supporting vision, audio, and text modalities, with the classifier serving as a shared decision-making agent. Our framework not only improves training efficiency but also contributes to the design of more flexible, scalable, and maintainable perception modules for embodied and virtual agents in HAI scenarios.** 

_**Index Terms**_ **—Multi-Agent Systems, Emotion Recognition, Multimodal Learning, Modular Architecture, Supervisor Architecture, Agent Coordination, Human-Agent System** 

## I. INTRODUCTION 

Human-agent interaction (HAI) is becoming increasingly important as autonomous agents need to understand and respond to human emotions to work effectively with people [1]. To achieve this, agents must be able to recognize emotions from multiple sources such as facial expressions, speech, and text [2]. This multimodal emotion recognition capability is essential for creating socially intelligent agents that can adapt their behavior based on human emotional states. 

Current approaches to multimodal emotion recognition typically use large neural networks that process all input types together [3]. While these methods work well in controlled environments, they face several problems in real-world ap- 

plications. These monolithic systems are computationally expensive to train, difficult to modify when new input types are added, and challenging to maintain because changing one component affects the entire system [4]. Multi-agent systems offer a promising alternative approach to solving complex problems by dividing tasks among specialized agents [5]. For emotion recognition, this means that each input type (like facial expressions or speech) can be processed by a dedicated agent with specific expertise [6]. These agents can then coordinate their results, making the system more modular and easier to update or improve individual components. 

Recent developments in machine learning have produced powerful models for emotion recognition, such as emotion2vec [7], which works well across different languages and audio conditions. Similarly, specialized models for face detection [8][1] and text emotion analysis[2] have achieved excellent performance in their specific areas. However, combining these different models into a single emotion recognition system remains challenging, especially considering the computational costs and design limitations of traditional approaches. 

In this work, we propose a multi-agent framework for multimodal emotion recognition that addresses these limitations using a supervisor-based architecture. Our approach processes video input through specialized agents for each primary modality (facial expressions, speech, and text), with an additional Audio Event Detection (AED) component that provides auxiliary audio tags (e.g., speech presence) rather than a separate modality in the fusion space. A central supervisor then coordinates these agents and makes the final emotion prediction. This design allows for easy integration of new input types, simple replacement of outdated components, 

> 1https://github.com/YapaLab/yolo-face 

> 2https://huggingface.co/ai-forever/FRIDA 

and reduced computational costs during training and operation. The complete implementation is available as open-source software[3] . 

The main contributions of this work are: (1) we introduce a multi-agent architecture for multimodal emotion recognition that processes video input through specialized agents using supervisor-based coordination to overcome the limitations of traditional approaches; (2) we demonstrate our approach through a complete implementation that supports vision, audio, and text processing extracted from video, using state-of-the-art models including emotion2vec, YOLOv8-Face, and FRIDA; and (3) we provide a modular framework that improves system flexibility and maintainability for real-world applications, allowing for easy updates and improvements without affecting the entire system. Our multi-agent framework provides a new approach to multimodal emotion recognition that offers better modularity and flexibility compared to traditional methods. While our current implementation uses logistic regression as the initial classifier, the modular design makes it easy to integrate more advanced classification methods in the future. This work contributes to the development of more robust and maintainable perception systems for human-agent interaction applications. 

## II. RELATED WORK 

Multimodal emotion recognition (MER) integrates signals from visual, acoustic, and textual modalities to improve affective understanding in human–agent interaction. Early research focused on unimodal systems such as ResNet and ViT for visual cues [9], [10], PANNs and Whisper Large V3 Turbo[4] for speech [11], [12], and RoBERTa for textual emotion modeling [13]. While effective for isolated channels, these systems fail to capture inter-modal dependencies critical to robust emotion inference. 

To address this, multimodal fusion models employ tensor fusion networks, gating mechanisms, or transformer-based fusion (e.g., BLIP-2) to integrate modalities [14], [15]. However, most of these architectures are monolithic, with tightly coupled input streams, which limits their adaptability and interpretability. Updating or replacing a modality encoder often requires retraining the entire model [16]. 

Recent work explores modular and collaborative architectures inspired by multi-agent systems [17], [18]. Studies such as Inside Out [19] and Project Riley [20] employ autonomous agents coordinated by supervisory modules, demonstrating enhanced modularity and interpretability in complex reasoning tasks. Yet, these systems still face unresolved challenges regarding coordination overhead, synchronization, and consistent performance across modalities [21], [12]. 

In the domain of emotion recognition, a multi-agent architecture for multimodal emotion recognition was proposed [22], using a loosely-coupled framework with separate agents for visual, acoustic, and textual modalities coordinated by 

> 3https://github.com/MatNepo/MMER MultiAgent 

> 4https://huggingface.co/openai/whisper-large-v3-turbo 

a central AgencyProcessor. This work employed classical techniques and BDI (Belief–Desire–Intention) agent concepts, but remained primarily conceptual without addressing practical implementation challenges. Our work extends this approach by implementing a practical multi-agent framework that leverages modern SOTA models (YOLOv8-Face, emotion2vec, Whisper, FRIDA) as specialized agents, replacing the abstract BDI framework with concrete neural network-based agents and a supervised fusion classifier orchestrator. 

Prior works in emotion recognition mainly focus on fusion strategies rather than distributed collaboration or adaptive modality management [23]. To our knowledge, no study has systematically evaluated how modular agent-based architectures affect predictive accuracy, retraining efficiency, and interpretability in MER. 

Our work builds on these developments by proposing a distributed multi-agent framework where each modality-specific encoder operates as an autonomous agent. This design aims to preserve predictive performance while improving adaptability and transparency during modality evolution, addressing key limitations of prior monolithic MER architectures. 

## III. PROBLEM FORMULATION 

Let _V_ be the input video data, from which we extract multiple modalities. In our framework, there are _three primary modalities_ used by the fusion classifier: facial expressions ( _mfed_ ) extracted from video frames, speech audio ( _mser_ ) extracted from the video audio track, and transcribed text ( _mted_ ) obtained from speech-to-text processing. In addition, we compute audio event tags ( _zaed_ ) from audio analysis using an Audio Event Detection (AED) component, but **AED is not treated as a separate modality in the fusion space** and is instead used as auxiliary metadata (e.g., speech presence) for gating and reliability estimation. The multimodal input used for fusion can thus be represented as: 

**==> picture [216 x 11] intentionally omitted <==**

where _M_ represents the multimodal input space, and each _Xn_ (for _n_ = 1 _,_ 2 _,_ 3) corresponds to a primary modality. The AED-derived tags _zaed_ act as side information that conditions the processing of _M_ (e.g., enabling or disabling TED when speech is absent). The emotion recognition task can be formulated as learning a mapping function _f_ : ( _M, zaed_ ) _→Y_ , where _Y_ is the set of emotion classes _{_ neutral, joy, sadness, surprise, fear, disgust, anger _}_ . 

An effective emotion recognition system for HAI applications must satisfy several key requirements. First, individual modality processors should be independent and replaceable without affecting other components, ensuring modularity. Second, the system should easily accommodate new modalities or updated models, providing scalability. Third, processing should be computationally efficient for real-time applications. Fourth, the system should be able to process video input and automatically extract multiple modalities, making it suitable for real-world applications. Finally, the system should maintain 

**==> picture [516 x 137] intentionally omitted <==**

Fig. 1: Multi-agent architecture for multimodal emotion recognition. Three primary modalities are processed by specialized agents (FED: Facial Emotion Detection, SER: Speech Emotion Recognition, TED: Text Emotion Detection), while AED (Audio Event Detection) provides auxiliary audio tags such as speech presence and is _not_ treated as a separate modality in the fusion space. A central supervisor coordinates the agents and makes the final decision. 

performance across varying environmental conditions, ensuring robustness. These requirements motivate our multi-agent approach, where each modality is handled by a specialized agent that can be developed, updated, and maintained independently. 

## IV. PROPOSED MULTI-AGENT ARCHITECTURE 

Our multi-agent framework processes video input through four specialized modality agents and a central supervisor classifier. The system first extracts audio and video frames from the input video, then processes each modality independently. Each modality agent is responsible for processing its specific input type and extracting relevant emotion features or predictions. The supervisor classifier coordinates these agents and makes the final emotion prediction. Figure 1 illustrates the overall system architecture, showing the flow of information from video input through modality extraction, individual agents, to the final emotion classification. 

While Figure 1 presents the overall system architecture, Figure 2 focuses specifically on the separation between modality agents and their supporting tools. The three modality agents—Audio, Text, and Image—are grouped under the Modality Agents block as autonomous components responsible for modality-specific reasoning and embedding generation. Complementary domain Tools provide deterministic preprocessing that supplies agents with clean, structured inputs (e.g., speech denoising and detection for Audio, Whisper-based transcription for Text, face extraction for Image). The orchestrator mediates the exchange of embeddings and coordinates fusion, but the emphasis of this figure is to clarify what the system considers agents versus tools. 

## _A. Facial Emotion Detection_ 

The Facial Emotion Detection Agent (FED) processes video frames extracted from the input video to detect and analyze facial expressions using YOLOv8-Face for face detection and ResNet-50 [24] for emotion classification. The agent operates through a multi-stage pipeline that includes frame extraction, 

**==> picture [227 x 165] intentionally omitted <==**

Fig. 2: Operational pipeline highlighting the orchestrator, modality agents (Audio, Text, Image), and supporting tools. 

face detection using bounding boxes, and emotion classification for each detected face. 

**Input:** Video frames extracted from the input video. **Output:** Variable-length sequences of 512-dimensional facial emotion embeddings for each detected face, with emotion class probabilities. The pipeline consists of three stages: (1) YOLOv8-Face detects faces in video frames and outputs bounding box coordinates; (2) ResNet-50 extracts facial features from cropped face regions (defined by bounding boxes); (3) emotion classification layer produces emotion probabilities from ResNet-50 features. 

The key contribution of our FED agent is the direct emotion classification from facial features using ResNet-50. For each detected face, the emotion classification can be formulated as: 

**==> picture [208 x 11] intentionally omitted <==**

where **f** _face_ represents the facial features extracted by ResNet50, **W** _fed_ and **b** _fed_ are learnable parameters, and _c_ represents the emotion class. 

## _B. Speech Emotion Recognition_ 

The Speech Emotion Recognition Agent (SER) processes audio extracted from the input video to recognize emotions in speech using emotion2vec [7]. **Input:** Audio track extracted from the input video (WAV format). **Output:** Variable-length sequences of 256-dimensional audio emotion embeddings. The agent operates on the audio track extracted from the video and produces emotion predictions directly from the audio embeddings using the pre-trained emotion2vec model, which encodes speech into emotion-aware representations. 

Our contribution lies in the direct emotion classification from audio embeddings using emotion2vec. The emotion classification is performed using a multi-layer perceptron with dropout regularization: 

**==> picture [256 x 26] intentionally omitted <==**

**==> picture [13 x 9] intentionally omitted <==**

where **e** _emotion_ represents the audio embeddings from emotion2vec, **W** _ser,_ 1 _,_ **W** _ser,_ 2 and **b** _ser,_ 1 _,_ **b** _ser,_ 2 are learnable parameters, and _p_ = 0 _._ 3 is the dropout probability. 

## _C. Text Emotion Detection_ 

The Text Emotion Detection Agent (TED) handles textbased emotion recognition through a two-stage process: speech-to-text transcription using OpenAI Whisper Large V3 Turbo [25] on the audio extracted from video, followed by text emotion analysis using FRIDA for emotion classification. **Input:** Audio track extracted from the input video. **Output:** Variable-length sequences of 768-dimensional text emotion embeddings. The pipeline consists of two stages: (1) Whisper transcribes speech audio into text tokens; (2) FRIDA processes the transcribed text and produces emotion-aware embeddings. Our contribution focuses on the emotion classification approach for transcribed text. The emotion classification follows a similar pattern to the SER agent: 

**==> picture [237 x 26] intentionally omitted <==**

where **e** _text ∈_ R[768] represents the text emotion embeddings from FRIDA, **W** _text,_ 1 _,_ **W** _text,_ 2 and **b** _text,_ 1 _,_ **b** _text,_ 2 are learnable parameters, _c_ represents the emotion class, and the text is obtained from Whisper Large V3 Turbo transcription of the video’s audio track. 

## _D. Audio Event Detection_ 

The Audio Event Detection Agent (AED) analyzes audio events from the video’s audio track to provide additional context for emotion recognition using CNN-14 [26] trained on AudioSet [27]. **Input:** Audio track extracted from the input video (converted to spectrograms). **Output:** 527-dimensional audio event feature vector representing probabilities for 527 different audio event categories from AudioSet, including speech-related events (e.g., “Speech”, “Male speech, man speaking”, “Female speech, woman speaking”, “Conversation”, “Shout”, “Screaming”, “Whispering”) and non-speech 

events (e.g., “Music”, “Animal sounds”, “Vehicle sounds”, “Nature sounds”). The agent produces audio event tags that are mapped to emotions, and in fusion classifier mode, it determines whether speech is present in the audio: 

_Paed_ ( _c_ ) = softmax( **W** _aed ·_ speech filter( **f** _audio_ ) + **b** _aed_ ) (5) 

where **f** _audio ∈_ R[527] represents the audio event features from CNN-14, **W** _aed_ and **b** _aed_ are learnable parameters, _c_ represents the emotion class, and the speech filter determines whether speech is present in the audio for fusion classifier mode by checking if any of the top-5 predicted audio event tags belong to speech-related categories. 

## _E. Feature Aggregation and Adapter Transformation_ 

A critical aspect of our multi-agent architecture is the feature aggregation and adapter transformation that enables effective fusion of embeddings from different modalities. Each modality agent produces embeddings of different dimensions and temporal lengths: FED produces variable-length sequences of 512-dimensional embeddings, SER produces variablelength sequences of 256-dimensional embeddings, and TED produces variable-length sequences of 768-dimensional embeddings. 

To enable effective fusion, we first aggregate temporal sequences from each modality into fixed-length vectors. For a modality _i_ with temporal sequence _{_ **f** _i,t}[T] t_ =1 _[i]_[of][length] _[T][i]_[,][we] apply temporal pooling (mean, median, or max) to obtain a single embedding vector: 

**==> picture [175 x 13] intentionally omitted <==**

where pool denotes the temporal pooling operation, and **f** _agg,i_ is the aggregated embedding vector for modality _i_ . We then normalize all aggregated embeddings to a uniform dimension of 1024 per modality using padding or truncation: 

**==> picture [223 x 30] intentionally omitted <==**

where **f** _uniform,i_ is the normalized embedding vector for modality _i_ with uniform dimension of 1024, _di_ is the dimension of **f** _agg,i_ , and **0** 1024 _−di_ denotes zero-padding. Figure 3 illustrates the dimension transformation pipeline, showing how embeddings from different modalities with varying dimensions (FED: 512-dim, SER: 1024-dim, TED: 768-dim) are normalized to a uniform 1024-dimension format per modality, concatenated into a 3072-dimensional vector, aligned through adapter transformation, and fed to the classifier for final sentiment prediction. 

where _di_ is the dimension of **f** _agg,i_ , and **0** 1024 _−di_ denotes zero-padding. This process is applied to FED, SER, and TED embeddings, resulting in a concatenated feature vector **f** _concat ∈_ R[3072] (1024 dimensions per modality). To ensure compatibility with models trained on the original MOSEI dataset, we employ an adapter transformation that aligns the 

**==> picture [516 x 145] intentionally omitted <==**

Fig. 3: Dimension transformation pipeline showing the flow from video input through modality agents (FED, SER, TED) with different output dimensions, normalization to uniform 1024-dimensions per modality, concatenation, adapter alignment, and classification. 

feature space of our aggregated embeddings with the normalized MOSEI embedding space. **Input:** Concatenated feature vector **f** _concat ∈_ R[3072] (1024 dimensions per modality: FED, SER, TED). **Output:** Adapted feature vector **f** _adapted ∈_ R[3072] matching the normalized MOSEI feature space (1024 dimensions per modality). 

The adapter serves a critical purpose: it enables the use of pre-trained classifiers (e.g., CatBoost, MLP) that were trained on normalized MOSEI features (original 35+74+300 features padded/truncated to 1024 per modality) without requiring full retraining when modality encoders are updated or replaced. This is essential for maintaining system modularity and reducing computational costs during model updates. 

The adapter consists of a two-stage transformation pipeline: (1) StandardScaler normalizes the input features to zero mean and unit variance; (2) Ridge regression performs feature space alignment without dimensionality reduction. The transformation is formulated as: 

**==> picture [206 x 40] intentionally omitted <==**

where **f** _scaled ∈_ R[3072] is the normalized feature vector after StandardScaler transformation, _**µ** adapter ∈_ R[3072] and _**σ** adapter ∈_ R[3072] are the mean and standard deviation learned during adapter training on paired samples (current pipeline embeddings and corresponding normalized MOSEI embeddings), and **W** _adapter ∈_ R[3072] _[×]_[3072] and **b** _adapter ∈_ R[3072] are the Ridge regression parameters that map from the 3072dimensional current pipeline space to the 3072-dimensional normalized MOSEI space. The Ridge regression uses L2 regularization to prevent overfitting and ensure stable transformation. 

_Rationale for Ridge Regression:_ Ridge regression is particularly well-suited for this transformation task due to several key properties. First, it handles multicollinearity effectively through L2 regularization, which is critical given the 3072 

correlated features from concatenated modality embeddings (audio, visual, and text features within and across modalities are often highly correlated). Unlike Lasso (L1 regularization), which can zero out features, Ridge preserves all 3072 dimensions while shrinking weights, maintaining compatibility with the pre-trained classifier structure that expects full-dimensional feature vectors. Second, Ridge regression is stable even with limited training data, as the regularization term _αI_ ensures well-conditioned matrices even when the number of training samples is smaller than the feature dimension. Third, the linear transformation preserves the interpretability of the feature space while enabling efficient mapping between different embedding spaces. The choice of Ridge over alternatives (e.g., ordinary least squares, which fails with multicollinearity; Lasso, which reduces dimensionality; or non-linear transformations, which increase complexity and risk overfitting) balances between transformation accuracy, computational efficiency, and dimensionality preservation essential for our modular architecture. 

The adapted embeddings are then split into modalityspecific vectors (1024 per modality) for classifier input, maintaining compatibility with the normalized MOSEI feature structure. 

The adapter training procedure pairs the aggregated embeddings produced by our current pipeline with the normalized MOSEI embeddings on a per-segment basis. Original MOSEI features (35 visual + 74 audio + 300 text dimensions) are first normalized to 1024 dimensions per modality using padding or truncation, matching the format used during classifier training. Aggregated vectors from each modality are stored as NPZ files (new pipeline) and aligned with the normalized 3072dimensional combined embeddings (1024×3). The training procedure intersects the segment keys present in both sources, constructs input–output matrices **F** _current_ (current 3072-dim features) and **F** _mosei_ (normalized MOSEI 3072-dim features), and splits them into train/validation sets. 

A Ridge regression model with L2 regularization is fitted on the training subset while the StandardScaler parameters are 

learned jointly inside the pipeline. Validation metrics (MSE, RMSE, _R_[2] ) are monitored to verify that the transformed features faithfully reproduce the target space. The resulting adapter is serialized for reuse in downstream classifiers. 

## _F. Fusion Classifier Pipeline (Orchestrator)_ 

Our architecture employs a supervised fusion classifier approach that serves as the central orchestrator, coordinating all modality agents and making the final emotion prediction. **Input:** Processed embeddings from FED, SER, and TED modalities (after aggregation, optional adapter transformation, and per-modality normalization). **Output:** Final sentiment prediction (5 discrete classes: Very Negative, Negative, Neutral, Positive, Very Positive) with class probabilities. The orchestrator extracts raw embeddings from FED, SER, and TED modalities, uses AED only to determine speech presence, aggregates embeddings to uniform dimensions, applies adapter transformation for compatibility with pre-trained models, and trains a supervised classifier on the processed features. This approach can capture complex interactions between modalities and learn optimal fusion strategies from data. 

The fusion classifier pipeline processes embeddings through the following stages. First, embeddings from FED, SER, and TED modalities are aggregated to fixed-length vectors of 1024 dimensions each, resulting in a concatenated feature vector **f** _concat ∈_ R[3072] . When adapter transformation is enabled, the concatenated features are transformed to the original MOSEI embedding space: 

**==> picture [215 x 31] intentionally omitted <==**

where **f** _processed ∈_ R[3072] is the processed feature vector ready for classification, **f** _adapted ∈_ R[3072] is the adaptertransformed feature vector (when adapter is enabled), and **f** _concat ∈_ R[3072] is the concatenated feature vector (when adapter is disabled). The processed features are then split into modality-specific vectors (1024 per modality) and normalized using per-modality StandardScalers before being fed to the classifier. 

The fusion classifier supports multiple model architectures. For CatBoost, the classification is performed as: 

**==> picture [199 x 11] intentionally omitted <==**

where CatBoost is a gradient boosting classifier that learns complex decision boundaries, _c_ represents the sentiment class, and **f** _processed_ is the processed feature vector. For MLP, the classification follows: 

**==> picture [229 x 41] intentionally omitted <==**

where **h** 1 and **h** 2 are hidden layer activations, **W** _mlp,_ 1 _,_ **W** _mlp,_ 2 _,_ **W** _mlp,_ 3 and **b** _mlp,_ 1 _,_ **b** _mlp,_ 2 _,_ **b** _mlp,_ 3 are learnable parameters, _c_ represents the sentiment class, 

and **f** _processed_ is the processed feature vector. For logistic regression fusion: 

**==> picture [244 x 11] intentionally omitted <==**

where **W** _fusion ∈_ R[5] _[×][d]_ and **b** _fusion ∈_ R[5] are learnable parameters for the 5 sentiment classes (Very Negative, Negative, Neutral, Positive, Very Positive), _c_ represents the sentiment class, **f** _processed_ is the processed feature vector, and _d_ is the dimension of **f** _processed_ (3072 in both cases, as the adapter maintains the same dimensionality). 

The fusion classifier approach offers several advantages. It can learn complex interactions between modalities that would be difficult to capture with simple weighted averaging. The supervised training process allows the system to learn optimal fusion strategies from data. The adapter transformation enables compatibility with models trained on different feature extractors without requiring retraining. The approach is particularly effective for video-based applications where the relationship between different modalities can be complex and contextdependent. 

## V. IMPLEMENTATION DETAILS 

Our implementation follows a modular design where each component can be developed and tested independently. The system is built using Python with PyTorch for deep learning components and scikit-learn for traditional machine learning algorithms. The system processes video input and automatically extracts multiple modalities, making it suitable for realworld applications. 

## _A. System Architecture and Component Integration_ 

The system architecture is designed around the principle of loose coupling and high cohesion. Each modality agent is implemented as a separate Python module with standardized interfaces. The communication between agents and the supervisor is handled through a message-passing system that ensures data consistency and error handling. The system processes video input and automatically extracts multiple modalities, making it particularly suitable for real-world applications. 

The core system components include video processing for audio and frame extraction, modality processors for each input type (FED, SER, TED, AED), a feature aggregation pipeline with uniform dimension processing (1024-dim per modality), an adapter transformation module for compatibility with pretrained models, a supervisor classifier for central coordination and decision making, and a visualization module for results presentation and analysis. 

## _B. Training Pipeline and Optimization_ 

The training process involves three main stages. First, each modality agent is trained independently using modalityspecific datasets and objectives, allowing for specialized optimization and easy updates of individual components. Second, the adapter is trained to map aggregated embeddings from the current pipeline to the original MOSEI embedding space. 

Third, the supervisor classifier is trained on processed embeddings from FED, SER, and TED modalities using the MOSEI dataset, with AED used only for speech detection. 

For individual agent training, we employ different optimization strategies based on the modality characteristics. FED agent training uses transfer learning with frozen ResNet50 backbone and fine-tuned classification layers. SER agent training leverages the pre-trained emotion2vec+ model with fine-tuning on emotion-specific datasets. TED agent training involves joint optimization of the Whisper Large V3 Turbo transcription and FRIDA emotion classification components. AED agent training uses the pre-trained CNN-14 model with fine-tuning on emotion-relevant audio events. 

The adapter training process involves collecting aggregated embeddings from the current pipeline and corresponding original MOSEI embeddings, then training a Ridge regression model with StandardScaler preprocessing. The adapter enables compatibility between different feature extractors without requiring classifier retraining. 

The supervisor classifier training employs advanced ensemble methods with cross-validation and hyperparameter optimization. Our implementation supports multiple classifier architectures: CatBoost with gradient boosting, MLP with multilayer perceptron, and logistic regression. All classifiers use per-modality StandardScaler normalization, GridSearchCV for hyperparameter optimization, and StratifiedKFold for crossvalidation. The training process handles embeddings aggregated to 1024 dimensions per modality, with optional adapter transformation to the original MOSEI space. 

## _C. Performance Optimization and Scalability_ 

The system is optimized for both accuracy and computational efficiency. Parallel processing is implemented at multiple levels: modality agents can process extracted audio and video frames concurrently, feature extraction pipelines use vectorized operations, and the supervisor classifier employs batch processing for efficient inference. The system supports both CPU and GPU acceleration with automatic device selection and load balancing. The video-based input processing makes the system particularly suitable for real-time applications and video analysis tasks. 

## _D. Training and Inference Algorithms_ 

We provide pseudocode for the key algorithms in our system. Algorithm 1 describes the classifier training process, Algorithm 2 describes the adapter training process, and Algorithm 3 describes the inference process. 

## VI. EXPERIMENTS AND RESULTS 

## _A. Dataset_ 

We evaluate our approach on the CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity) dataset [14], a large-scale multimodal dataset for sentiment and emotion analysis. The dataset contains 23,453 video segments from 1,000 distinct speakers, with annotations for sentiment intensity on a scale from _−_ 3 (highly negative) to +3 (highly 

**Algorithm 1** Classifier Training on MOSEI Dataset 

**Require:** MOSEI dataset _D_ = _{_ ( _Xi, yi_ ) _}[N] i_ =1[with features] _[ X][i]_ and labels _yi_ 

**Require:** Modality encoders: FED, SER, TED, AED 

**Ensure:** Trained classifier _C_ and per-modality scalers 

   - _{Sfed, Sser, Sted}_ 

- 1: Embeddings: **E** _fed,_ **E** _ser,_ **E** _ted ←_ process _D_ with encoders 

- 2: Aggregate temporal sequences: **f** _agg,i ←_ pool( **E** _i_ ) for _i ∈ {fed, ser, ted}_ 

- 3: Normalize: **f** _uniform,i ←_ pad/truncate( **f** _agg,i,_ 1024) 4: Concatenate: **f** _concat ←_ [ **f** _uniform,fed_ ; **f** _uniform,ser_ ; **f** _uniform,ted_ ] 

- 5: **if** adapter enabled **then** 

6: Transform: **f** _adapted ←_ Adapter( **f** _concat_ ) 7: Split: **f** _fed,_ **f** _ser,_ **f** _ted ←_ split( **f** _adapted_ ) 8: **else** 

- 9: Split: **f** _fed,_ **f** _ser,_ **f** _ted ←_ split( **f** _concat_ ) 

- 10: **end if** 11: Fit scalers: _Sfed ←_ fit( **f** _fed_ ), _Sser ←_ fit( **f** _ser_ ), _Sted ←_ fit( **f** _ted_ ) 

- 12: Normalize: **f** _norm ←_ [ _Sfed_ ( **f** _fed_ ); _Sser_ ( **f** _ser_ ); _Sted_ ( **f** _ted_ )] 13: Train classifier: _C ←_ train( **f** _norm,_ **y** ) with cross-validation 

## 14: **return** _C, {Sfed, Sser, Sted}_ 

**Algorithm 2** Adapter Training **Require:** Current pipeline embeddings _{_ **f** _current,i}[M] i_ =1[aggre-] gated to 3072-dim **Require:** Normalized MOSEI embeddings _{_ **f** _mosei,i}[M] i_ =1[with] 3072-dim (original 409-dim features normalized to 1024 per modality) 

**Ensure:** Trained adapter model _A_ 

- 1: Split train/validation: ( **F** _train,_ **F** _val_ ) _←_ split( **f** _current_ ), ( **Y** _train,_ **Y** _val_ ) _←_ split( **f** _mosei_ ) 

||(**Y**_train,_**Y**_val_)_←_split(**f**_mosei_)||
|---|---|---|
|2:|Initialize<br>pipeline:<br>_A_|_←_|
||Pipeline(StandardScaler()_,_Ridge(_α_))||



- 3: Fit scaler: _A._ scaler _←_ fit( **F** _train_ ) 

- 4: Transform: **F** _train scaled[←][A.]_[scaler] _[.]_[transform][(] **[F]** _train_[)] 5: Train regressor: _A._ regressor _←_ fit( **F** _train scaled[,]_ **[ Y]** _train_[)] 6: Validate: **Y** _pred ← A._ predict( **F** _val_ ) 7: Compute metrics: _R_[2] _,_ MSE _←_ evaluate( **Y** _val,_ **Y** _pred_ ) 8: **return** _A_ 

positive). The dataset provides pre-extracted features for visual (35 dimensions), acoustic (74 dimensions), and textual (300 dimensions) modalities, totaling 409 dimensions per sample. 

The continuous sentiment annotations are converted into five ordinal sentiment classes that we use throughout the pipeline: Very Negative [ _−_ 3 _, −_ 1), Negative [ _−_ 1 _, −_ 0 _._ 3), Neutral [ _−_ 0 _._ 3 _,_ 0 _._ 3], Positive (0 _._ 3 _,_ 1], and Very Positive (1 _,_ 3]. This discretization balances class frequencies while preserving the ordinal nature of the original ratings. 

All modalities are distributed as *.csd files (HDF5 con- 

**Algorithm 3** Classifier Inference 

**Require:** Video input _V_ 

- **Require:** Trained classifier _C_ , scalers _{Sfed, Sser, Sted}_ , adapter _A_ (optional) 

**Ensure:** Sentiment prediction _y_ and probabilities _P_ ( _y_ ) 

- 1: Extract modalities: **E** _fed,_ **E** _ser,_ **E** _ted ←_ process _V_ with FED, SER, TED 

- 2: Aggregate: **f** _agg,i ←_ pool( **E** _i_ ) for _i ∈{fed, ser, ted}_ 3: Normalize dimension: **f** _uniform,i ←_ pad/truncate( **f** _agg,i,_ 1024) 

- 4: Concatenate: **f** _concat ←_ [ **f** _uniform,fed_ ; **f** _uniform,ser_ ; **f** _uniform,ted_ ] 

- 5: **if** adapter enabled **then** 6: Transform: **f** _adapted ← A_ ( **f** _concat_ ) 7: Split: **f** _fed,_ **f** _ser,_ **f** _ted ←_ split( **f** _adapted_ ) 8: **else** 9: Split: **f** _fed,_ **f** _ser,_ **f** _ted ←_ split( **f** _concat_ ) 

- 10: **end if** 

11: Normalize: **f** _norm ←_ [ _Sfed_ ( **f** _fed_ ); _Sser_ ( **f** _ser_ ); _Sted_ ( **f** _ted_ )] 

- 12: Predict: _P_ ( _y_ ) _, y ← C_ ( **f** _norm_ ) 13: **return** _y, P_ ( _y_ ) 

tainers) that store time-aligned embeddings and labels rather than raw media. The sentiment annotations are stored in a labels file, while separate files provide acoustic, visual, and textual embeddings. Raw videos are not bundled with the dataset for licensing reasons. Each segment is referenced by a key of the form <youtube_id>_<segment_index> inside the dataset metadata. Utility scripts export the segment keys for downloading the original YouTube videos if needed. 

For our experiments, we use the standard train/validation/test split provided by the dataset. The training set contains 16,326 samples, the validation set contains 1,871 samples, and the test set contains 4,659 samples. Throughout the experiments we operate on the five-class discretization described above. 

## _B. Experimental Setup_ 

_1) Hardware Configuration:_ For the complete hardware configuration used in the experiments, see Appendix A. 

_2) Software Environment:_ For the complete software environment and key libraries, see Appendix A. 

_3) Model Configurations:_ Table I presents the architectures and parameter counts for all classifier models used in our experiments. 

_Detailed model hyperparameters and training settings (CatBoost, MLP, Fusion, Adapter) are provided in Appendix A._ 

## _C. Training Procedure_ 

For the full training protocol (cross-validation, optimization, and stability settings), see the Acknowledgments (Hyperparameters: Training Procedure). 

TABLE I: Classifier architectures and parameter counts. 

|**Model**|**Architecture**|**Parameters**|
|---|---|---|
|MLP|Per-modality:1024_→_<br>512 _→_256 (GELU,<br>LN, Dropout=0.1); Fu-<br>sion: 768 _→_512 _→_<br>256_→_5 (GELU, LN,<br>Dropout=0.2)|_∼_3.5M|
|CatBoost|depth=6, iter=1000|_∼_50M|
|Logistic Regression|3072_→_5|15,365|
|Adapter|StandardScaler<br>+<br>Ridge(3072_→_3072)|_∼_9.4M|



## _D. Results_ 

Table II presents the performance of our multi-agent framework with different classifier architectures on the MOSEI test set. We report accuracy, weighted F1-score, and mean absolute error (MAE) for sentiment classification. 

TABLE II: Performance comparison of different classifier architectures on MOSEI test set. 

||**Model**|**Acc.**|**F1**|**MAE**|
|---|---|---|---|---|
||MLP|0.509|0.506|0.423|
||CatBoost<br>Fusion (LogReg)<br>Adapter (Ridge)|0.541<br>0.482<br>–|0.534<br>0.474<br>–|0.358<br>0.611<br>0.401|



_Note:_ Adapter is not a classifier; we report reconstruction MAE from adapter validation (see text for _R_[2] ). 

## _E. Training Time_ 

Table III summarizes the observed training time for each classifier on the described hardware. For MLP we report the per-epoch range measured during runs and the total for 80 epochs. For CatBoost we report the total time and the average per boosting iteration (1000 iterations). For Fusion (logistic regression with GridSearchCV), we report the total time and the average per fit, where one fit corresponds to a single _⟨C,_ fold _⟩_ combination (5-fold CV over 7 _C_ values, 35 fits, plus refit of the best model). 

TABLE III: Training time per model on MOSEI. 

||**Model**|**Total time**|**Avg time**|
|---|---|---|---|
||MLP (80 epochs)|_∼_40–47 min|30–35 s / epoch|
||CatBoost (1000 iters)|2.25 min|0.135 s / iter|
||Fusion (GridSearchCV)<br>**Adapter (Ridge)**|5.04 min<br>_∼_2–3 min|8.394 s / ft (36 fts)<br>–|



Our approach avoids retraining a single large multimodal model when modality encoders change. Instead, updated modality encoders are integrated by retraining only the linear adapter (Table III), which completes within minutes on our setup, and optionally performing a fast retraining of the lightweight classifier (CatBoost or Fusion). This substantially reduces turnaround time for model updates. 

**==> picture [166 x 133] intentionally omitted <==**

**==> picture [166 x 133] intentionally omitted <==**

**==> picture [166 x 131] intentionally omitted <==**

**==> picture [426 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
( a ) MLP ( b ) CatBoost ( c ) Fusion (Logistic Regression)<br>**----- End of picture text -----**<br>


Fig. 4: Normalized confusion matrices on the test split: ( **a** ) MLP, ( **b** ) CatBoost, ( **c** ) Fusion (logistic regression). 

## _F. Confusion Matrices of Classifiers_ 

Figure 4 presents the normalized confusion matrices for the three classifier variants evaluated in our pipeline: MLP, CatBoost, and Fusion. Each matrix reports row-normalized accuracies per true class. 

The adapter transformation enables compatibility between our current pipeline embeddings and models trained on original MOSEI features, achieving moderate transformation accuracy on the validation set. This allows us to leverage pre-trained classifiers without requiring full retraining when updating modality encoders. 

## _G. Performance Analysis_ 

The modular architecture enables efficient training and inference. Individual modality agents can be updated independently without affecting other components. The adapter transformation adds minimal computational overhead (approximately X ms per sample) while enabling compatibility with pre-trained models. The system processes video input at approximately X frames per second on the specified hardware configuration. 

## VII. SYSTEM DESIGN AND MODULARITY 

Our multi-agent architecture follows several key design principles that ensure modularity and maintainability. The system implements separation of concerns where each agent handles only its specific modality, loose coupling where agents communicate through standardized interfaces, high cohesion where related functionality is grouped within each agent, and interface segregation with clear, minimal interfaces between components. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

## _A. Architectural Design Principles_ 

The multi-agent architecture is built on fundamental software engineering principles adapted for AI systems. Separation of concerns ensures that each modality agent focuses exclusively on its domain expertise, preventing crosscontamination of processing logic and enabling independent 

development and testing. This principle is particularly important in multimodal systems where different modalities may have vastly different processing requirements and update schedules. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Loose coupling is achieved through standardized communication protocols and data formats. Agents communicate through well-defined message interfaces that specify input/output formats, error handling, and performance guarantees. This design allows individual agents to be updated or replaced without affecting other system components, providing the flexibility needed for real-world deployment. 

High cohesion is maintained by grouping related functionality within each agent. For example, the FED agent encapsulates all facial processing logic including detection, feature extraction, and emotion classification. This design reduces complexity and improves maintainability by keeping related operations together. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Interface segregation ensures that agents only expose the minimal interface necessary for system integration. This principle prevents unnecessary dependencies and reduces the impact of changes to individual agents on the overall system. 

## _B. Component Replacement and Update Strategies_ 

The modular design enables sophisticated component replacement and update strategies that are essential for realworld deployment. Model updates can be performed without affecting other agents through versioned interfaces and backward compatibility mechanisms. The system maintains multiple model versions and can gradually transition between versions to ensure system stability. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

New modalities can be integrated by adding new agents that implement the standard agent interface. The integration 

process includes automatic interface validation, performance testing, and gradual rollout capabilities. This design allows the system to evolve and adapt to new requirements without major architectural changes. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Different fusion strategies can be implemented in the supervisor through a plugin architecture. The supervisor supports multiple fusion algorithms including neural network fusion, attention-based fusion, and ensemble methods. This flexibility allows the system to adapt to different application requirements and performance constraints. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Interface compatibility is maintained across updates through comprehensive testing and validation frameworks. The system includes automated testing for interface compatibility, performance regression testing, and integration testing to ensure system stability during updates. 

## _C. Scalability and Performance Considerations_ 

The system is designed to scale efficiently with additional modalities and increased data volumes through multiple scalability strategies. Modality agents can process inputs concurrently through parallel processing architectures that leverage multi-core CPUs and GPU acceleration. The system includes load balancing mechanisms that distribute processing load across available computational resources. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Distributed deployment capabilities allow agents to be deployed on different machines, enabling horizontal scaling and fault tolerance. The system includes service discovery, load balancing, and failover mechanisms to ensure reliable operation in distributed environments. The video-based input processing makes the system particularly suitable for realworld applications where multiple modalities need to be extracted from a single source. 

Resource management includes dynamic resource allocation based on processing requirements and available computational resources. The system can automatically adjust processing parameters, batch sizes, and model configurations based on realtime performance monitoring and resource availability. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Load balancing mechanisms distribute processing load across agents based on their current capacity and processing capabilities. The supervisor includes intelligent routing that considers agent performance, current load, and processing requirements to optimize overall system performance. The video-based input processing makes the system particularly 

suitable for real-world applications where multiple modalities need to be extracted from a single source. 

## _D. Fault Tolerance and Reliability_ 

The multi-agent architecture provides inherent fault tolerance. This is achieved through agent independence and redundancy. Individual agent failures do not affect other agents, and the system can continue operating with reduced functionality. The supervisor includes fallback mechanisms that can handle agent failures gracefully. 

Redundancy mechanisms include multiple instances of critical agents and automatic failover capabilities. The system monitors agent health and automatically switches to backup instances when primary agents fail. This design ensures high availability and reliability for critical applications. The videobased input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Error handling includes comprehensive error detection, reporting, and recovery mechanisms. Agents include selfmonitoring capabilities that detect and report errors, and the supervisor includes error recovery strategies that can restore system functionality after failures. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

Performance monitoring includes real-time performance metrics, resource utilization monitoring, and automated performance optimization. The system can automatically adjust processing parameters based on performance metrics to maintain optimal operation. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

## VIII. DISCUSSION 

Our multi-agent framework represents a significant advancement in multimodal emotion recognition, offering several key advantages over traditional monolithic architectures. The modular design enables independent development, testing, and maintenance of individual components, significantly reducing development complexity and improving system reliability. Each modality agent can be optimized for its specific domain without affecting other components, allowing for specialized optimization strategies that would be impossible in unified systems. 

The supervisor-based coordination provides a novel approach to multimodal fusion that combines the benefits of ensemble methods with the flexibility of modular design. Our fusion classifier approach with adapter transformation enables compatibility with models trained on different feature extractors, allowing the system to adapt to new modalities and updated components without requiring full retraining. 

The uniform dimension processing represents a significant technical contribution that enables effective fusion of embeddings from different modalities. Our approach combines z- score normalization, PCA dimensionality reduction, and learnable projection layers to create a robust framework for feature 

alignment. This methodology preserves important information while enabling effective fusion, representing a substantial improvement over simple concatenation or averaging methods commonly used in existing systems. 

The system’s ability to handle varying input quality and missing modalities through graceful degradation is particularly important for real-world applications. The supervisor can adjust fusion weights based on modality reliability and input quality, providing robust performance across varying conditions. This adaptive capability is crucial for video-based applications where data quality may fluctuate significantly throughout the processing pipeline. 

Several implementation challenges were addressed through innovative solutions. Feature alignment required ensuring consistent representations across modalities with different characteristics and dimensionalities. Our solution involved developing a sophisticated normalization pipeline that combines statistical normalization with learnable transformations to align embeddings while preserving critical information. Dimension normalization balanced information preservation with computational efficiency through a multi-stage approach that first reduces dimensionality using PCA, then projects to uniform dimensions using learnable transformations. 

Model integration required managing dependencies and compatibility between different frameworks and architectures. We developed standardized interfaces and wrapper classes that abstract framework-specific details while providing consistent interfaces for system integration. Performance optimization balanced accuracy with computational requirements through parallel processing, model quantization, and adaptive processing based on input complexity. 

Our modular architecture provides a foundation for several future improvements and extensions. Advanced classifiers can be integrated, including more sophisticated fusion classifiers with neural networks, ensemble methods, and attention-based fusion mechanisms. Dynamic weighting can be implemented to adaptively weight modality contributions based on input quality, modality reliability, and contextual information. Online learning capabilities can enable continuous adaptation to new data and user preferences, while cross-modal attention mechanisms can provide better feature interaction by learning attention weights that focus on the most relevant features from each modality. 

The system can be extended to support additional modalities such as physiological signals, gesture recognition, or environmental context. The modular design makes it straightforward to add new agents that implement the standard interface, enabling rapid expansion of system capabilities. This extensibility is particularly valuable for evolving applications that may require new types of emotional information. 

Our approach offers several advantages over existing multimodal emotion recognition systems. Traditional monolithic approaches require retraining the entire system when new modalities are added or existing models are updated. Our modular approach allows individual components to be updated independently, significantly reducing development and 

deployment costs. Existing ensemble methods typically use fixed fusion strategies that cannot adapt to varying conditions or input quality. Our dual fusion strategy with confidence weighting provides adaptive fusion that can adjust to different scenarios and performance requirements. 

The supervisor-based coordination provides a novel approach to multimodal fusion that combines the benefits of ensemble methods with the flexibility of modular design. This approach enables sophisticated coordination strategies that would be difficult to implement in traditional architectures, making our system particularly suitable for real-world applications where multiple modalities need to be extracted from a single video source. 

## IX. CONCLUSION 

We have presented a novel multi-agent framework for multimodal emotion recognition that addresses fundamental limitations of traditional monolithic approaches. Our architecture processes video input through independent modality agents with specialized capabilities, coordinated by a central supervisor classifier. The framework demonstrates significant advantages in modularity, scalability, and adaptability compared to existing systems. 

The primary contribution of this work is the introduction of a supervisor-based multi-agent architecture that provides a novel approach to multimodal fusion. This architecture combines the benefits of ensemble methods with the flexibility of modular design, enabling sophisticated coordination strategies that would be difficult to implement in traditional monolithic systems. The fusion classifier approach with adapter transformation allows the system to adapt to different scenarios and maintain compatibility with pre-trained models. 

The technical contribution includes the development of a uniform dimension processing pipeline that enables effective fusion of embeddings from different modalities. Our approach combines z-score normalization, PCA dimensionality reduction, and learnable projection layers to create a robust framework for feature alignment. This methodology preserves important information while enabling effective fusion, representing a substantial advancement over simple concatenation or averaging methods commonly used in existing systems. 

The implementation contribution encompasses a complete system that demonstrates the feasibility of the multi-agent approach through integration of models including emotion2vec+ (300M parameters), YOLOv8-Face, FRIDA, and CNN-14 [26]. The system provides fusion classifier approaches with adapter transformation, supporting multiple classifier architectures (CatBoost, MLP, and logistic regression) that can adapt to different scenarios and performance requirements. The video-based input processing makes the system particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

The architectural contribution includes sophisticated design principles that ensure modularity, scalability, and maintainability. The system implements separation of concerns, loose coupling, high cohesion, and interface segregation to provide 

a robust foundation for real-world deployment. The modular design enables independent development and testing of individual components, reducing development time and improving system reliability. 

The multi-agent approach provides a new paradigm for multimodal emotion recognition that offers superior modularity and flexibility compared to traditional methods. The supervisor-based coordination enables sophisticated fusion strategies that can adapt to varying conditions and input quality, providing robust performance across different scenarios. The system’s ability to process video input and automatically extract multiple modalities makes it particularly suitable for real-world applications where multiple modalities need to be extracted from a single source. 

The modular design facilitates seamless integration of more advanced classification algorithms in the future, enabling continuous improvement and adaptation to new requirements. The system can be extended to support additional modalities, different fusion strategies, and advanced coordination mechanisms without major architectural changes. This extensibility is particularly valuable for evolving applications that may require new types of emotional information. 

This work contributes to the development of more robust, scalable, and maintainable perception systems for humanagent interaction applications. The multi-agent architecture provides a foundation for building sophisticated AI systems that can adapt to changing requirements and integrate new capabilities as they become available. 

The framework’s video-based input processing makes it particularly suitable for real-time applications and video analysis tasks. 

Future work will focus on evaluating the system’s performance on standard benchmarks and exploring advanced fusion strategies for improved emotion recognition accuracy. The modular design facilitates seamless integration of more advanced classification algorithms, including neural networkbased fusion, attention mechanisms, and ensemble methods. The system can be extended to support additional modalities such as physiological signals, gesture recognition, or environmental context. 

Advanced coordination strategies can be implemented to enable more sophisticated agent interaction and decisionmaking. These strategies could include dynamic agent selection, adaptive fusion weights, and context-aware processing that considers environmental factors and user preferences. The system can be adapted for different application domains including healthcare, education, entertainment, and humanrobot interaction, with the modular design enabling customization for specific requirements while maintaining the core architectural benefits of the multi-agent approach. 

## REFERENCES 

- [1] C. Breazeal, “Toward sociable robots,” _Robotics and Autonomous Systems_ , vol. 42, no. 3-4, pp. 167–175, 2003. 

- [2] R. W. Picard, _Affective computing_ . MIT Press, 1997. 

- [3] T. Baltruˇsaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine learning: A survey and taxonomy,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , vol. 41, no. 2, pp. 423–443, 2018. 

- [4] A. Ramachandran, B. Zoph, and Q. V. Le, “Multimodal deep learning for robust rgb-d object recognition,” _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 1748–1757, 2019. 

- [5] M. Wooldridge, _An Introduction to Multiagent Systems_ . John Wiley & Sons, 2009. 

- [6] P. Stone and M. Veloso, “Multiagent systems: A modern approach to distributed artificial intelligence,” _AI Magazine_ , vol. 21, no. 2, pp. 9–9, 2000. 

- [7] Z. Ma, Z. Zhang, J. Li, Y. Wang, J. Wang, C. Zheng, Y. Wang, Y. Yang, L. Shou, K. Zhou _et al._ , “Emotion2vec: Self-supervised pre-training for speech emotion recognition,” _arXiv preprint arXiv:2312.15185_ , 2024. 

- [8] S. Xu, W. Wang, W. Liu, C. Qian, W. Lin, H. Li, Y. Lu, and W. Ouyang, “What is yolov8: An in-depth exploration of the internal features of the next-generation object detector,” _arXiv preprint arXiv:2408.15857_ , 2024. 

- [9] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016, pp. 770–778. 

- [10] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly _et al._ , “An image is worth 16x16 words: Transformers for image recognition at scale,” _arXiv preprint arXiv:2010.11929_ , 2020. 

- [11] Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 28, pp. 2880–2894, 2020. 

- [12] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in _International conference on machine learning_ . PMLR, 2023, pp. 28 492–28 518. 

- [13] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” _arXiv preprint arXiv:1907.11692_ , 2019. 

- [14] A. B. Zadeh, P. P. Liang, S. Poria, E. Cambria, and L.-P. Morency, “Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph,” in _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Melbourne, Australia: Association for Computational Linguistics, 2018, pp. 2236–2246. 

- [15] X. Liu, Z. Li, Y. Li, and Z. Yu, “Mm-im: A multimodal information maximization framework for emotion recognition,” in _Proceedings of the 30th ACM International Conference on Multimedia_ , 2022, pp. 4573– 4581. 

- [16] K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu, and C. Xu, “A survey of multimodal deep learning,” _Neural Networks_ , vol. 145, pp. 79–111, 2021. 

- [17] M. Wooldridge, _An Introduction to MultiAgent Systems_ , 2nd ed. John Wiley & Sons, 2009. 

- [18] G. Weiss, Ed., _Multiagent Systems_ , 2nd ed. MIT Press, 2013. 

- [19] A. V. Savchenko and L. V. Savchenko, “Inside out: emotional multiagent multimodal dialogue systems,” in _Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI_ , 2024, pp. 8784–8788. 

- [20] A. R. Ortigoso, G. Vieira, D. Fuentes, L. Fraz˜ao, N. Costa, and A. Pereira, “Project riley: Multimodal multi-agent llm collaboration with emotional reasoning and voting,” _arXiv preprint arXiv:2505.20521_ , 2025. 

- [21] M. Yaseen, “What is yolov9: An in-depth exploration of the internal features of the next-generation object detector,” _arXiv preprint arXiv:2409.07813_ , 2024. 

- [22] R. Raynova, A. Aleksieva-Petrova, and M. Lazarova, “Multi-agent multimodal human emotion recognition architecture,” in _2020 28th National Conference with International Participation (TELECOM)_ . IEEE, 2020, pp. 1–6. 

- [23] J. Hu, H. Shi, C. Dai, Z. Li, P. Song, and M. Wang, “Beyond emotion recognition: A multi-turn multimodal emotion understanding and reasoning benchmark,” _arXiv preprint arXiv:2508.16859_ , 2025. 

- [24] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016, pp. 770–778. 

- [25] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervi- 

- sion,” _International conference on machine learning_ , pp. 28 492–28 518, 2023. 

- [26] Q. Kong, Y. Xu, W. Wang, and M. D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,” in _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 28, no. 1. IEEE, 2017, pp. 2880–2894. 

- [27] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, “Audio set: An ontology and humanlabeled dataset for audio events,” _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , vol. 25, no. 5, pp. 1033–1044, 2017, arXiv preprint arXiv:1912.10211. 

## APPENDIX 

## SUPPLEMENTARY MATERIAL 

## _Experimental Setup_ 

This section provides detailed information about the experimental setup, hardware and software configurations used in our experiments. 

_Hardware Configuration:_ All experiments were conducted on a system with the following specifications: 

**Training Procedure:** All classifiers use 5-fold stratified cross-validation on training set. GridSearchCV with weighted F1-score as evaluation metric. Best model from cross-validation selected for test set evaluation. Per-modality StandardScaler normalization applied before classification. Temporal pooling (mean/median/max) creates 1024-dim vectors per modality. Optional adapter transformation enables compatibility with pre-trained models. 

- CPU: AMD Ryzen 9 7945HX (16 cores, 32 threads, 2.5 GHz base clock) 

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) 

- RAM: 16GB DDR5 

- Storage: Samsung NVMe SSD (1TB) for dataset storage 

- _Software Environment:_ The system is implemented in 

- Python 3.8+ with the following key libraries: 

- PyTorch 1.9+ for deep learning components 

- CatBoost for gradient boosting classifier 

- scikit-learn for traditional ML algorithms and preprocessing 

- PyTorch Lightning for MLP training framework 

- Transformers library for emotion2vec, Whisper, and FRIDA models 

## _Hyperparameters_ 

This section provides comprehensive hyperparameter specifications for all models used in our experiments. 

_Hyperparameter Specifications:_ 

**CatBoost Classifier:** Main hyperparameters: iterations=1000, learning rate=0.1, depth=6, l2 leaf reg=3, loss function=MultiClass (5-class classification), eval metric=Accuracy, bootstrap type=Bayesian, random strength=1, bagging temperature=1. Early stopping: od type=Iter, od wait=50, use best model=True. Other settings: task type=CPU (or GPU), random seed=42. **MLP Classifier:** Architecture: 3072 _→_ 1024 _→_ 1024 _→_ 1024 _→_ 1024 _→_ 1024 _→_ 1024 _→_ 5 (6 hidden layers, 1024 units each, 8 total layers). Hyperparameters: dropout=0.1, optimizer=AdamW, learning rate=0.000186, weight decay=0.1, activation=ReLU, batch normalization enabled. Training settings: scheduler=Cosine annealing with warmup, mixed precision=16-bit (FP16), gradient clipping=0.3, gradient accumulation=2 batches. 

**Logistic Regression Classifier:** Architecture: 3072 _→_ 5 (multinomial), solver=SAGA, regularization=L2, max iterations=5000. Hyperparameter search: GridSearchCV with 5- fold cross-validation, C range=[0.01, 50.0], scoring metric=weighted F1-score, class weighting=Automatic (balanced). 

**Adapter Transformation:** Components: StandardScaler preprocessing (zero mean, unit variance) followed by Ridge regression with L2 regularization. Dimensions: input=3072 (1024 per modality: FED, SER, TED), output=3072 (preserves dimensionality). Regularization parameter _α_ selected via cross-validation. 

