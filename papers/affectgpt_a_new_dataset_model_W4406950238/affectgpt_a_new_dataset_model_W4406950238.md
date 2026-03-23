# **AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models** 

**Zheng Lian**[1] **Haoyu Chen**[2] **Lan Chen**[1] **Haiyang Sun**[3] **Licai Sun**[2] **Yong Ren**[1] **Zebang Cheng**[4] **Bin Liu**[1] **Rui Liu**[5] **Xiaojiang Peng**[6] **Jiangyan Yi**[7] **Jianhua Tao**[7 8] 

## **Abstract** 

The emergence of multimodal large language models (MLLMs) advances multimodal emotion recognition (MER) to the next level—from naive discriminative tasks to complex emotion understanding with advanced video understanding abilities and natural language description. However, the current community suffers from a lack of large-scale datasets with intensive, descriptive emotion annotations, as well as a multimodalcentric framework to maximize the potential of MLLMs for emotion understanding. To address this, we establish a new benchmark for MLLMbased emotion understanding with a novel dataset (MER-Caption) and a new model (AffectGPT). Utilizing our model-based crowd-sourcing data collection strategy, we construct the largest descriptive emotion dataset to date (by far), featuring over 2K fine-grained emotion categories across 115K samples. We also introduce the AffectGPT model, designed with pre-fusion operations to enhance multimodal integration. Finally, we present MER-UniBench, a unified benchmark with evaluation metrics tailored for typical MER tasks and the free-form, natural language output style of MLLMs. Extensive experimental results show AffectGPT’s robust performance across various MER tasks. We have released both the code and the dataset to advance research and development in emotion understanding: https://github.com/zeroQiaoba/AffectGPT. 

> 1Institute of Automation, Chinese Academy of Sciences 

> 2CMVS, University of Oulu 3Shanghai Jiao Tong University 4Shenzhen University 5Inner Mongolia University 6Shenzhen Technology University[7] Department of Automation, Tsinghua University[8] Beijing National Research Center for Information Science and Technology, Tsinghua University. Correspondence to: Zheng Lian _<_ lianzheng2016@ia.ac.cn _>_ , Jianhua Tao _<_ jhtao@tsinghua.edu.cn _>_ . 

_Proceedings of the 41[st] International Conference on Machine Learning_ , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). 

## **1. Introduction** 

Emotions encapsulate human intentions, and accurately recognizing emotional states is essential for enhancing humancomputer interaction experiences (Minsky, 1988). Emotions can be conveyed through various human behaviors in different forms, giving rise to the task of multimodal emotion recognition (MER), which integrates multimodal information (e.g., audio, video, and text) to evaluate human emotional states. As a critical area in artificial intelligence, MER has broad applications, ranging from education (Schutz, 2007) and psychological counseling (Liu et al., 2021) to empathic embodied robots (Spezialetti et al., 2020). 

Traditional methods primarily rely on discriminative models that map human emotions to the most likely categories from predefined emotion taxonomies. The most widely used taxonomy is Ekman’s theory (Ekman & Keltner, 1970), which classifies all emotions into six basic categories: _sadness_ , _happiness_ , _fear_ , _anger_ , _surprise_ , and _disgust_ . However, such categorical frameworks exhibit some limitations in modeling human affective states. For example, our emotional expressions are diverse and nuanced due to culture-specific idioms (Matsumoto, 2001), context-dependent metaphors (Kovecses¨ , 2003), and highly personalized behavioral patterns (Izard et al., 1993). Current closed-set classification paradigms fail to capture the rich diversity of emotional expressions in real-world scenarios (Plutchik, 1980). Meanwhile, the rigid emotion taxonomies oversimplify the continuous spectrum of emotional experiences by forcing discrete labels (e.g., _anger_ or _surprise_ ) onto nuanced affective states that often coexist (Cowen & Keltner, 2017). Illustrations are provided in Figure 1, where the diverse and coexisting issues are presented in Figs. (a) and (b). 

Recent advances in multi-modal large language models (MLLMs) enable emotion understanding to move beyond traditional discriminative approaches, embracing a more generative framework (Liang et al., 2024). This shift allows models to describe complex, coexisting emotional states in natural language. With the vast vocabulary, MLLMs can generate diverse, descriptive emotion categories beyond basic emotions, offering new opportunities for emotional understanding. However, recent research highlights 

**==> picture [453 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Diversity<br>Description: The sentence might be the woman's comment or reaction to someone nearby. Given the audio cues of a steady tone with humor<br>and the woman's smile and glance to the left, we can infer that her words carry a mocking or sarcastic tone. Thus, this sentence likely expresses<br>the woman's mockery of the other person's lack of understanding or recognition of something, conveyed in a humorous manner.<br>(b) Coexistence<br>Description: In the video, his eyes are wide open and his mouth is also open, indicating a surprised facial expression. In the audio, the<br>character speaks with a stutter, which usually expresses feelings of nervousness, anxiety, or unease. Combined with the text content, the<br>character seems to be unhappy and angry due to the prejudice of the people around him.<br>**----- End of picture text -----**<br>


Figure 1: **Emotion complexity analysis** . Human emotions are often diverse and coexist simultaneously. Such complex emotional states are difficult to describe using discriminative frameworks. However, MLLMs can generate emotional descriptions, offering new possibilities for complex emotion modeling. Since the original videos contain real people, to address copyright concerns, we first use DemoAI to remove personal information and then proceed with visualization. 

that MLLMs still face limitations in emotion understanding (Lian et al., 2024a;d). To address these challenges, this paper aims to advance emotional understanding from two key perspectives: the dataset and the model. Finally, we establish a unified benchmark tailored to the free-form, natural language output style of MLLMs. 

**Dataset.** The current community still suffers from a lack of large-scale datasets with intensive, descriptive emotion annotations to realize the potential of MLLMs. The annotation strategies for constructing descriptive emotion datasets can be classified into three types: _model-based_ , _humanbased_ , and _human-model collaborative_ strategies. The _human-based_ strategy is the most common way to construct emotion datasets with rich descriptive annotations. However, it’s costly to conduct crowd-sourcing to scale up the dataset size with this purely manual annotation manner. Besides, humans tend to focus on main cues, resulting in brief and incomplete descriptions (Liu et al., 2022a). Thus, researchers propose _model-based_ automatic annotation approaches. However, due to the lack of human proofreading, this approach may result in insufficient label quality (Cheng et al., 2024). Recently, Lian et al. (2024a) propose a _humanmodel collaborative_ strategy, in which models provide prelabeled cues and humans conduct multiple rounds of checks, which can be seen as a _human-led, model-assisted_ strategy. Although this approach offers more comprehensive descriptions, it is costly and difficult to scale the dataset. To balance label quality and dataset size, we introduce a novel annotation strategy that conducts model-based crowd-sourcing labeling with human priors, named _model-led human-assisted_ , to construct a large-scale emotion descriptive dataset with diverse emotional categories. 

**Models.** Existing MLLMs typically consist of three key components: a modality encoder that converts audio and video into low-level hidden features, a connector that transforms these features into a format more suitable for LLMs, and an LLM-based generator that produces responses based on the given instructions. While the results of MLLMs are promising, existing models generally leave everything of multimodal fusion to LLMs, which is insufficient for MER that emphasizes multimodal characteristics. This paper introduces the AffectGPT model, designed with a pre-fusion operation to emphasize multimodal integration. 

**Benchmark.** Although it’s desirable to generate emotional descriptions in a free-form, natural language style (see Appendix D), this poses challenges for quantitative comparison. To address this, we propose metrics specifically designed for this output style. Additionally, to ensure fair and comprehensive evaluation, we introduce MERUniBench, a benchmark that incorporates three typical tasks: fine-grained emotion recognition, basic emotion recognition, and sentiment analysis. We believe this work can enhance the emotion understanding capabilities of MLLMs and open possibilities for complex emotion modeling. The main contributions of this paper are summarized as follows: 

- We construct a large-scale emotional description dataset **MER-Caption** , which adopts a model-led, human-assisted annotation strategy to strike a balance between label quality and dataset size. 

- We develop **AffectGPT** , which uses additional prefusion operations to enhance multimodal integration, thereby improving emotion understanding. 

Table 1: **Dataset comparison.** “I”, “A”, “V”, and “T” stand for image, audio, video, and text, respectively. We observe that descriptive datasets contain more diverse labels, providing the potential for modeling complex emotions. 

||**Dataset**<br>**Modality**<br>**# Samples**<br>**Description**<br>**# Emotions**<br>**Annotation Manner**|
|---|---|
|**Categorical**<br>**Dataset**|RAF-DB (Li et al.,2017)<br>I<br>29,672<br>�<br>7<br>Human<br>AffectNet (Mollahosseini et al.,2017)<br>I<br>450,000<br>�<br>8<br>Human<br>EmoDB (Burkhardt et al.,2005)<br>A<br>535<br>�<br>7<br>Human<br>MSP-Podcast (Lotfan & Busso,2017)<br>A<br>73,042<br>�<br>8<br>Human<br>DFEW (Jiang et al.,2020)<br>V<br>11,697<br>�<br>7<br>Human<br>FERV39k (Wang et al.,2022)<br>V<br>38,935<br>�<br>7<br>Human<br>MER2023 (Lian et al.,2023)<br>A,V,T<br>5,030<br>�<br>6<br>Human<br>MELD(Poria et al.,2019)<br>A,V,T<br>13,708<br>�<br>7<br>Human|
|**Descriptive**<br>**Dataset**|EmoVIT (Xie et al.,2024)<br>I<br>51,200<br>�<br>988<br>Model<br>MERR-Coarse (Cheng et al.,2024)<br>A,V,T<br>28,618<br>�<br>113<br>Model<br>MAFW (Liu et al.,2022a)<br>A,V,T<br>10,045<br>�<br>399<br>Human<br>OV-MERD (Lian et al.,2024a)<br>A,V,T<br>332<br>�<br>236<br>Human-led+Model-assisted<br>MERR-Fine (Cheng et al.,2024)<br>A,V,T<br>4,487<br>�<br>484<br>Human-led+Model-assisted<br>**MER-Caption**<br>A,V,T<br>115,595<br>�<br>2,932<br>Model-led+Human-assisted<br>**MER-Caption+**<br>A,V,T<br>31,327<br>�<br>1,972<br>Model-led+Human-assisted|



- We build **MER-UniBench** , which encompasses typical MER tasks with tailored metrics. This benchmark can offer comprehensive evaluation results for MLLMbased emotion understanding. 

- Extensive experiments demonstrate the effectiveness of AffectGPT, which achieves over a 9% performance improvement compared to existing MLLMs. 

## **2. MER-Caption: Dataset Construction** 

Table 1 summarizes existing emotion datasets, which can be broadly classified into categorical and descriptive datasets. The former directly provides emotion labels (e.g., _happy_ ), while the latter offers textual descriptions related to emotions. We first conduct preliminary experiments to extract emotion labels from descriptive datasets (see Appendix E). As shown in Table 1, descriptive datasets contain more diverse labels, offering the potential to capture complex emotions. Thus, this paper focuses on descriptive datasets. 

Based on the annotation manner, descriptive datasets can be categorized into _model-based_ , _human-based_ , and _humanmodel collaborative_ strategies (see Table 1). Although the model-based approach makes it easy to expand the dataset size, it mainly relies on experience to select models and lacks human intervention, resulting in insufficient label quality (Cheng et al., 2024). To enhance label quality, Liu et al. (2022a) relied on human annotators to generate emotion descriptions. However, humans tend to focus on primary clues, easily leading to incomplete descriptions. To this end, Lian et al. (2024a) proposed a _human-led, model-assisted_ strategy. Specifically, the model first provides pre-labeled descriptions, and then multiple annotators perform multiround checks. Although this strategy produces more comprehensive descriptions, it comes with high annotation costs 

and faces challenges in scaling the dataset. In this paper, we review these annotation methods and introduce a _modelled, human-assisted_ strategy. As shown in Figure 2, we leverage human priors to guide description generation and sample filtering, ultimately achieving automatic annotation for unlabeled data. Using this strategy, we construct the MER-Caption dataset, which includes 115K coarse-labeled samples and 31K fine-labeled samples, making a significant contribution to current descriptive datasets. The raw data in MER-Caption is sourced from the unlabeled portions of MER2024 (Lian et al., 2024b), with explicit permission from the dataset owners. Therefore, this paper does not involve the collection of new data but provides additional annotations for existing datasets. Appendix H provides more comparisons with existing datasets. 

## **2.1. Description Generation** 

The choice of base models is critical for generating accurate descriptions. Unlike previous work that relied solely on experience (Cheng et al., 2024), we guide model selection using _human priors_ . Specifically, we first select a small subset of samples for preliminary experiments. In this phase, we annotate fine-grained labels for each sample, allowing annotators to assign any emotions they deem appropriate, thus providing more diverse and precise labels. Based on the results in preliminary experiments (see Appendix F), we employ SALMONN (Tang et al., 2023) as the audio LLM (ALLM) to generate audio cues, Chat-UniVi (Jin et al., 2024) as the video LLM (VLLM) to extract visual cues, and GPT-3.5 (OpenAI, 2022) (“gpt-3.5-turbo-16k-0613”) to merge the audio and video cues with text content. Then, to further reduce annotation costs, we experimented with replacing GPT-3.5 with other open-source LLMs but observed a drop in performance. The primary reason is that multimodal fusion in MER is inherently complex, often en- 

**==> picture [371 x 212] intentionally omitted <==**

**----- Start of picture text -----**<br>
Text<br>Audio Model-led Human-Assisted  MER-Caption Model-led Human-Assisted  MER-Caption+<br>Description Generation 115k Sample Filtering 31k<br>Video<br>Low-level filtering  High-level filtering<br>Crowdsourcing Sentiment<br>Mismatched Audio and Video Text Sentiment Classifiers Prediction<br>Text Audio Are they consistent?<br>Audio ALLM Merge GPT A: no talking, visible speaker mismatch B: talking, invisible speaker Video Emotion ClassifiersCrowdsourcing PredictionEmotion  Yes<br>Sentiment<br>Video VLLM CaptionMER- LLM Prediction Emotion  Are they consistent?<br>Abnormal Length of MER-Caption Prediction<br>Human Priors<br>in ALLM and VLLM Selection × √ × Model-Based Crowdsourcing for Sample Selection<br>Fine-grained Emotions Sentiment Basic Emotions<br>**----- End of picture text -----**<br>


Figure 2: **Dataset construction pipeline.** To create a large-scale dataset with guaranteed label quality, we propose a _model-led, human-assisted_ annotation strategy. In this approach, we leverage human priors to guide description generation and sample filtering, ultimately achieving automatic annotation for unlabeled data. 

countering issues such as modality conflict, where inconsistencies or contradictions arise between different modalities (see Appendix G). This places high demands on the LLM’s reasoning capabilities. Then, we adopt the above strategy for automatic annotation and create the MER-Caption dataset. 

## **2.2. Sample Filtering** 

Since the descriptions generated by the above process have not been manually verified, MER-Caption inevitably contains some errors. To this end, we implement a two-level filtering process to enhance the label quality. 

**Low-level Filtering.** First, we observe that some samples contain mismatched audio and video. As shown in Figure 2, the visible person is not speaking, while the audio comes from an invisible person. This setup differs from our task, where we aim to analyze a person’s emotions based on their audio, video, and text content. Mismatched data complicates this task, shifting the focus to understanding how the interlocutor’s actions may influence the target person’s emotions. Therefore, we remove such data and plan to address this issue in future work. To automatically determine whether the visible person is speaking, we use TalkNet (Tao et al., 2021). Preliminary experiments indicate that this tool achieves over 90% accuracy in identifying the speaking individual. Then, we remove samples with mismatched audio and video. Second, the length distribution of the generated descriptions roughly follows a Gaussian distribution (see Figure 2). Preliminary experiments reveal that descriptions at both ends of the distribution are more likely to contain 

errors. For instance, when ALLM and VLLM (in Section 2.1) fail to generate responses, the resulting descriptions tend to be short. As a result, we further remove descriptions located at both ends of the distribution. 

**High-level Filtering.** In addition to low-level filtering, we propose a _model-based crowdsourcing_ technique for highlevel filtering. Specifically, we train multiple multimodal emotion and sentiment classifiers using human-annotated categorical datasets. Guided by MERBench (Lian et al., 2024c), we use CLIP ViT-L (Radford et al., 2021) as the visual encoder and HUBERT-L (Hsu et al., 2021) as the acoustic encoder, followed by an attention-based fusion strategy to make final emotion and sentiment predictions. These pre-trained models are then used to predict labels for unlabeled data, generating multiple predictions for each sample. To mitigate potential prediction errors, we apply majority voting to determine the final label, ensuring more reliable results. We refer to this process as _model-based crowdsourcing_ . Alternatively, emotions and sentiments can also be predicted based on the descriptions using the strategy outlined in Appendix E. If the labels extracted from the descriptions differ from those obtained through _modelbased crowdsourcing_ , we consider these descriptions to be of low quality and remove them. Through this process, we can extract knowledge from multiple human-based datasets to guide sample selection. After applying multi-level filtering, we obtain the _MER-Caption+_ dataset. Table 1 presents detailed comparisons between our dataset and existing ones, highlighting that our dataset is the largest multimodal emotion description dataset with diverse emotion categories. 

**==> picture [487 x 87] intentionally omitted <==**

**----- Start of picture text -----**<br>
LLM LLM LLM LLM<br>LoRA LoRA LoRA LoRA<br>Projector Tokenizer Projector Tokenizer Projector Projector Tokenizer Projector Pre-fusion &  Projector Tokenizer<br>Audio  Video  Audio  Video  Audio  Projector Video<br>Encoder Encoder Encoder Encoder Encoder Encoder<br>Audio Prompt Video Prompt Audio Video Prompt Audio Video Prompt<br>Audio LLM (ALLM) Video LLM (VLLM) Audio-Video LLM (AV-LLM) AffectGPT (Ours)<br>**----- End of picture text -----**<br>


Figure 3: **Model comparison.** ALLM and VLLM primarily use modality-specific encoders and align them with the LLM through projection layers. AV-LLM mainly facilitates cross-modal interaction within the language model. In AffectGPT, we move the cross-modal interaction outside the language model and use a pre-fusion operation to enhance multimodal integration. In these figures, **P** can be determined based on the requirement of whether to include **Xt** . 

## **3. AffectGPT: Model Design** 

Our primary goal is to map audio-video-text inputs to emotion-related descriptions. In this section, we first review the current mainstream architectures. We then introduce AffectGPT, a model specifically designed to highlight multimodal characteristics in emotion understanding. 

## **3.1. Mainstream Architecture** 

MLLM aims to understand multimodal input and generate appropriate responses based on the input and user instructions. Unlike pure-text LLMs, the primary challenge for MLLMs lies in enabling the model to perceive multimodal input, i.e., providing the model with “eyes” and “ears”. In existing models, the most common approach is to first extract modality-specific embeddings and then align them with the LLM through projection layers. For audio-video joint tasks, Audio-Video LLMs (AV-LLMs) typically facilitate cross-modal interaction within the language model. Figure 3 illustrates the current mainstream architecture. 

Formally, for each sample **X** , we represent its video, audio, and text content as **Xv** , **Xa** , and **Xt** , respectively. Given an instruction **Q** , the goal is to output the correct response **R** . For the visual input **Xv** , we use a video expert to encode it into a latent space **Zv** , then apply a projector _Gv_ ( _·_ ) to generate visual tokens **Hv** = _Gv_ ( **Zv** ). Similarly, for the acoustics input **Xa** , we use an audio expert and a projector to generate the acoustic embeddings **Za** and tokens **Ha** . For the instruction **Q** and text content **Xt** , we use a template to merge them into a prompt **P** , and then map them to the corresponding tokens through the tokenizer and embedding layer in the language model. After obtaining these tokens, we concatenate them and feed them into the LLM decoder. The primary objective is to maximize the likelihood of the target response **R** , conditioned on multimodal content ( **Xv** , **Xa** , **Xt** ) and user instruction **Q** : 

**==> picture [177 x 11] intentionally omitted <==**

The above formula is optimized in an autoregressive manner, consistent with the objective function of LLMs. We 

represent the response as **R** = _{ri}[L] i_ =1 _[r]_[,][where] _[L][r]_[is][the] number of tokens. Then, Eq. 1 is transformed into: 

**==> picture [195 x 30] intentionally omitted <==**

In this equation, _rl_ is the current token to be predicted, and **R** _<l_ = _{ri}i[l][−]_ =1[1][is the previously generated tokens, which] serve as additional conditioning during training. 

## **3.2. Pre-fusion Operation** 

Mainstream AV-LLMs leave everything of cross-modal interaction to the LLMs (in Figure 3), which is insufficient for handling MER with multimodal characteristics. To address this, we propose a pre-fusion operation that moves the cross-modal interaction outside the LLMs, further enhancing multimodal integration. We refer to this model as AffectGPT. This paper introduces two types of pre-fusion operations: Q-Former-based and attention-based pre-fusion. By default, we apply this operation to **Zv** _∈_ R _[t][v][×][d]_ and **Za** _∈_ R _[t][a][×][d]_ . We also experimented with **Hv** and **Ha** , but this choice led to a decrease in performance. 

**Q-Former.** In this module, we preserve the temporal information in the vision features **Zv** and audio features **Za** , and utilize Q-Former (Li et al., 2023b) for multimodal fusion. Specifically, to compress the multimodal content, we first create _K_ learnable query tokens **Zq** _∈_ R _[K][×][d]_ . Then, we interact **Zq** with the concatenated **Zv** and **Za** through cross-attention, thereby distilling the knowledge from the multimodal content into the query tokens. Formally, this process can be represented as: 

**==> picture [169 x 11] intentionally omitted <==**

**==> picture [200 x 11] intentionally omitted <==**

where **Zav** _∈_ R[(] _[t][a]_[+] _[t][v]_[)] _[×][d]_ , with the concatenation operation applied along the temporal dimension. Here, **Zf** _∈_ R _[K][×][d]_ , and PE( _·_ ) represents the positional encoding. 

**Attention.** Unlike Q-Former which preserves temporal information, we propose a simpler architecture that directly compresses temporal information and applies attention mechanisms for multimodal fusion. This simplified module is inspired by MERBench (Lian et al., 2024c), which proves that in MER tasks, features with temporal information do not always lead to better performance than compressed features. Formally, we first apply average pooling to compress unimodal features. Then, we calculate the attention weights to emphasize important modalities: 

**==> picture [202 x 13] intentionally omitted <==**

**==> picture [187 x 47] intentionally omitted <==**

where **Z[ˆ] a** _∈_ R _[d]_ , **Z[ˆ] v** _∈_ R _[d]_ , **Z[ˆ] av** _∈_ R[2] _[×][d]_ , and **W** _∈_ R[2] _[×]_[2] _[d]_ . Finally, we obtain the fused features **Zf** _∈_ R _[d]_ . 

Regarding computational efficiency, the pre-fusion operation relies on Q-Former or attention mechanisms, which are significantly less computationally intensive than LLMs. Theoretically, the Q-Former enables cross-modal interaction by distilling multimodal content into query tokens, whereas the attention mechanism achieves this by dynamically computing attention weights based on multimodal inputs. 

## **4. MER-UniBench: Evaluation Benchmark** 

We introduce MER-UniBench, a comprehensive evaluation benchmark designed to cover typical MER tasks. Given the free-form, natural language output style of MLLMs (see Appendix D), we also design specialized evaluation metrics. More details can be found in Appendix J. 

**Fine-grained Emotion Recognition.** This task enables the prediction of fine-grained emotions, extending beyond basic categories. OV-MERD (Lian et al., 2024a) is a typical dataset for this task. To improve the reliability of the evaluation results, we expand its dataset size, referring to it as OV-MERD+. For the evaluation metrics, we draw inspiration from previous work (Lian et al., 2024a) and calculate results in two steps: eliminating the impact of synonyms and using set-level metrics. First, we apply a three-level grouping strategy to mitigate the impact of synonyms: 

- **Level 1.** We map different forms of emotion words to their base form. For example, we map _happier_ and _happiness_ to _happy_ . This function is denoted as _Fl_ 1( _·_ ). 

- **Level 2.** We map synonyms to a unified label. For example, we map _happy_ and _joyful_ to _happy_ . This mapping function is represented as _Fl_ 2( _·_ ). 

- **Level 3.** Emotion wheel provides natural grouping information, with core emotions displayed in the inner part and more nuanced labels in the outer part (Plutchik, 1980). Since there is no consensus on the emotion wheel, we use _K_ emotion wheels (see Appendix K). For each sector of the emotion wheel _wk, k ∈_ [1 _, K_ ], we map all outer labels to the corresponding inner labels. This mapping function is denoted as _Fl[w]_ 3 _[k]_[(] _[·]_[)][.] 

The above grouping functions can be summarized as: 

**==> picture [202 x 13] intentionally omitted <==**

For each sample, the number of labels is variable. Therefore, we define a set-based evaluation metric. Specifically, suppose the dataset contains _N_ samples. For sample _xi_ , the true labels are **Y** _i_ = _{yi[j][}][n] j_ =1 _[i]_[, and the predicted labels are] ˆ **ˆY** _i_ = _{yi[j][}][n] j_[ˆ] =1 _[i]_[.][The evaluation metric is defined as follows:] 

**==> picture [211 x 40] intentionally omitted <==**

**==> picture [213 x 70] intentionally omitted <==**

Finally, we compute the average results across different emotion wheels for ranking. Take Fs as an example: 

**==> picture [152 x 30] intentionally omitted <==**

Here, Precisions indicates the number of correctly predicted labels, and Recalls indicates whether the prediction covers all ground truth. Fs is a harmonic mean of two metrics. Since Fs considers both accuracy and completeness, we use it as the primary metric, with Precisions and Recalls serving as secondary metrics. To extract the predicted emotions **Y[ˆ]** _i_ , we employ the strategy mentioned in Appendix E. 

**Basic Emotion Recognition.** This task is a key branch of MER, whose main goal is to select the most likely label from a fixed set of basic emotions. For this task, we select four widely used benchmark datasets: MER2023 (Lian et al., 2023), MER2024 (Lian et al., 2024b), IEMOCAP (Busso et al., 2008), and MELD (Poria et al., 2019). However, ˆ the output of MLLMs, **Y[ˆ]** _i_ = _{yi[j][}][n] j_[ˆ] =1 _[i]_[, contains a variable] number of labels, while the dataset only provides one true label _yi_ . In this case, traditional metrics (such as _accuracy_ ) are not suitable for performance evaluation. To address this, we propose a new metric, _hit rate_ , which is set to 1 

Table 2: **Main results.** This table presents the results for the primary metrics, with Section 4 outlining the primary metrics for each task. The values for other metrics can be found in Appendix L. In this table, “MOSI”, “MOSEI”, “SIMS”, and “SIMS v2” refer to CMU-MOSI, CMU-MOSEI, CH-SIMS, and CH-SIMS v2, respectively. The last column shows the dataset-wise mean score, i.e., the average score across all datasets. 

|**Modality**<br>A<br>V<br>T|**Basic**<br>MER2023<br>MER2024<br>MELD<br>IEMOCAP|**Sentiment**<br>MOSI<br>MOSEI<br>SIMS<br>SIMS v2|**Fine-grained**<br>OV-MERD+|**Mean**|
|---|---|---|---|---|
|OneLLM<br>~~_√_~~<br>_×_<br>~~_√_~~<br>SECap<br>_√_<br>_×_<br>_√_<br>PandaGPT<br>_√_<br>_×_<br>_√_<br>Qwen-Audio<br>_√_<br>_×_<br>_√_<br>SALMONN<br>_√_<br>_×_<br>_√_<br>**AffectGPT**<br>_√_<br>_×_<br>_√_|25.52<br>17.21<br>28.32<br>33.44<br>40.95<br>52.46<br>25.56<br>36.92<br>33.57<br>39.04<br>31.91<br>36.55<br>41.85<br>31.61<br>49.09<br>35.47<br>55.53<br>45.38<br>45.62<br>46.84<br>**72.94**<br>**73.41**<br>**56.63**<br>**55.68**|64.01<br>54.09<br>63.39<br>61.98<br>55.76<br>54.18<br>59.51<br>57.41<br>66.06<br>61.33<br>62.93<br>58.88<br>70.09<br>46.90<br>70.73<br>65.26<br>81.00<br>67.03<br>68.69<br>65.93<br>**83.46**<br>**80.74**<br>**82.99**<br>**83.75**|22.25<br>36.97<br>31.33<br>32.36<br>45.00<br>**59.98**|41.14<br>46.64<br>46.84<br>49.26<br>57.89<br>**72.18**|
|Otter<br>_×_<br>~~_√_~~<br>~~_√_~~<br>Video-LLaVA<br>_×_<br>_√_<br>_√_<br>PandaGPT<br>_×_<br>_√_<br>_√_<br>Video-ChatGPT<br>_×_<br>_√_<br>_√_<br>VideoChat2<br>_×_<br>_√_<br>_√_<br>LLaMA-VID<br>_×_<br>_√_<br>_√_<br>VideoChat<br>_×_<br>_√_<br>_√_<br>Chat-UniVi<br>_×_<br>_√_<br>_√_<br>mPLUG-Owl<br>_×_<br>_√_<br>_√_<br>**AffectGPT**<br>_×_<br>_√_<br>_√_|16.41<br>14.65<br>22.57<br>29.08<br>36.93<br>30.25<br>30.73<br>38.95<br>39.13<br>47.16<br>38.33<br>47.21<br>44.86<br>46.80<br>37.33<br>56.83<br>33.67<br>54.50<br>36.64<br>48.70<br>50.72<br>57.60<br>42.75<br>46.02<br>48.73<br>57.30<br>41.11<br>48.38<br>57.62<br>65.67<br>45.61<br>52.37<br>56.86<br>59.89<br>49.11<br>55.54<br>**74.58**<br>**75.29**<br>**57.63**<br>**62.19**|52.89<br>50.44<br>57.56<br>53.12<br>56.37<br>61.64<br>53.28<br>57.45<br>58.50<br>64.25<br>62.07<br>65.25<br>54.42<br>63.12<br>64.82<br>65.80<br>66.84<br>54.32<br>69.49<br>70.66<br>61.78<br>63.89<br>69.35<br>67.48<br>65.13<br>63.61<br>69.52<br>72.14<br>54.53<br>63.18<br>68.15<br>66.36<br>72.40<br>72.91<br>72.13<br>75.00<br>**82.39**<br>**81.57**<br>**87.20**<br>**86.29**|16.63<br>34.00<br>35.07<br>39.80<br>39.21<br>45.01<br>44.52<br>48.00<br>48.18<br>**61.65**|34.82<br>44.40<br>50.77<br>52.64<br>52.67<br>56.07<br>56.71<br>57.94<br>62.45<br>**74.31**|
|PandaGPT<br>~~_√_~~<br>~~_√_~~<br>~~_√_~~<br>Emotion-LLaMA<br>_√_<br>_√_<br>_√_<br>**AffectGPT**<br>_√_<br>_√_<br>_√_|40.21<br>51.89<br>37.88<br>44.04<br>59.38<br>73.62<br>46.76<br>55.47<br>**78.54**<br>**78.80**<br>**55.65**<br>**60.54**|61.92<br>67.61<br>68.38<br>67.23<br>66.13<br>67.66<br>78.32<br>77.23<br>**81.30**<br>**80.90**<br>**88.49**<br>**86.18**|37.12<br>52.97<br>**62.52**|52.92<br>64.17<br>**74.77**|



when _yi ∈_ **Y[ˆ]** _i_ and 0 otherwise. Considering that **Y[ˆ]** _i_ is in free-form and _yi_ belongs to basic emotions _Y_ , we may encounter cases where _y_ ˆ _i ∈Y/_ . To this end, we use the mapping function _Gwk_ ( _·_ ) and define the metric as follows: 

**==> picture [201 x 31] intentionally omitted <==**

where I[ _·_ ] is an indicator function. The motivation for this metric stems from the fact that basic emotion recognition tasks typically provide majority-voted labels _yi_ , which are generally reliable. However, emotion descriptions produce free-form outputs **Y[ˆ]** _i_ that may contain multiple labels, including fine-grained ones beyond basic emotions. Therefore, we use the _hit rate_ as the metric, ensuring that the basic label _yi_ should be at least in **Y[ˆ]** _i_ . 

During the design of this metric, we also explored the possibility of evaluating potentially incorrect labels in **Y[ˆ]** _i_ . However, the labels in **Y[ˆ]** _i_ that differ from the basic label _yi_ are not necessarily incorrect - they may represent some finegrained emotions not covered by basic categories. Since basic emotion recognition tasks lack fine-grained reference labels, we have not yet established appropriate evaluation metrics for this purpose. This remains an important research direction for our future work. 

**Sentiment Analysis.** This task is more fundamental than the two tasks mentioned above, aiming to predict the sentiment polarity. For this task, we select four benchmark 

datasets: CMU-MOSI (Zadeh et al., 2017), CMU-MOSEI (Zadeh et al., 2018), CH-SIMS (Yu et al., 2020), and CHSIMS v2 (Liu et al., 2022b). For these benchmark datasets, the original labels are floating-point values, ranging from [ _−_ 1 _,_ 1] or [ _−_ 3 _,_ 3]. We map scores of _<_ 0 to negative sentiment and scores of _>_ 0 to positive sentiment. To extract sentiment labels from the MLLM’s output, we employ the strategy outlined in Appendix E. Following previous work (Zadeh et al., 2017; 2018), we evaluate performance using accuracy (ACC) and weighted average F-score (WAF). Due to the inherent label imbalance, we choose WAF as the primary metric and ACC as the secondary metric. 

## **5. Results and Discussion** 

In this section, we present the experimental results and provide an in-depth analysis. Detailed implementation information can be found in Appendix B. 

**Main Results.** We compare the performance of AffectGPT with other MLLMs on MER-UniBench. Since our inputs include audio, video, and text content, we only select MLLMs that support at least audio or video. For models that support both audio and video, we test different modality combinations. Model cards are provided in Appendix C. To ensure a fair comparison, we use their official weights and input corresponding multimodal content, asking them to infer the emotional state. In Table 2, AffectGPT significantly outperforms existing MLLMs. This can be attributed to the 

Table 3: **Dataset comparison** . We only change the training dataset, keeping all other aspects consistent. This table reports the mean score across all datasets in MER-UniBench. 

||**Dataset**<br>**Filtering**|**MER-UniBench**|
|---|---|---|
|**General**<br>**Instruction**|MiniGPT4<br>_×_<br>_√_|31.74<br>35.53|
||VideoChat<br>_×_<br>_√_|37.16<br>37.63|
||LLaVA<br>_×_<br>_√_|46.69<br>46.27|
||WavCaps<br>_×_<br>_√_|21.65<br>37.91|
|**Emotion**<br>**Description**|EmoVIT<br>–<br>MAFW<br>–<br>MERR-Coarse<br>–<br>MERR-Fine<br>–<br>MER-Caption<br>–<br>MER-Caption+<br>–|51.05<br>58.16<br>49.85<br>64.55<br>68.91<br>**74.77**|



fact that current instruction datasets pay little attention to MER tasks. Additionally, existing models place the entire multimodal fusion within the LLM, which is insufficient for MER tasks that require effective multimodal integration. By leveraging our newly proposed dataset and model, we provide a promising approach to enhancing emotion understanding capability in MLLMs. Meanwhile, for different datasets, increasing the input modality does not always improve performance, as it may also introduce irrelevant information that interferes with emotional understanding. 

**Effectiveness of MER-Caption.** Table 3 compares the performance of MER-Caption with existing datasets. For a fair comparison, we use the same model architecture and experimental settings and only change the training data. For general instruction datasets, we further conduct filtering experiments to remove samples without emotion-related content, emphasizing emotion-related subsets. Specifically, we use the prompt in Appendix E and extract emotion labels from each instruction-answer pair. Samples yielding empty emotion outputs are removed. 

In Table 3, the excellent performance of MER-Caption proves the limitations of current datasets in addressing MER. On the one hand, general instruction datasets pay insufficient attention to emotion-related tasks. On the other hand, emotional description datasets often suffer from inadequate dataset scales or insufficient annotation quality. Therefore, our dataset can serve as an important complement to existing datasets. Meanwhile, for the general instruction datasets, the filtering approach is less effective on the LLaVA and VideoChat datasets. We hypothesize that the detailed descriptions in non-emotion subsets may also provide valuable cues for inferring emotional states in some scenarios. 

Furthermore, we would like to acknowledge that MERCaption+ may contain inaccurate descriptions due to the use 

Table 4: **Necessity of filtering.** Besides the results on MERUniBench, we also provide task-level results. “E” and “S” are abbreviations for emotion and sentiment, respectively. 

|**Low**<br>**High**|**Fine-grained**<br>**Basic**<br>**Sentiment**|**MER-UniBench**|
|---|---|---|
|_×_<br>_×_<br>_√_<br>_×_<br>_√_<br>E<br>_√_<br>S<br>_√_<br>E+S|58.42<br>64.36<br>76.09<br>58.61<br>62.78<br>79.04<br>61.72<br>67.83<br>82.74<br>61.00<br>66.33<br>**85.04**<br>**62.52**<br>**68.38**<br>84.22|68.91<br>69.54<br>73.78<br>74.05<br>**74.77**|



Table 5: Role of pre-fusion operation. 

|**Pre-fusion**|**Fine-grained**<br>**Basic**<br>**Sentiment**|**MER-UniBench**|
|---|---|---|
|_×_|61.21<br>66.28<br>82.57|72.95|
|Q-Former<br>Attention|**62.65**<br>66.89<br>**84.30**<br>62.52<br>**68.38**<br>84.22|74.16<br>**74.77**|



of an automatic annotation strategy without manual checks. However, the experimental results in Table 3 show that MER-Caption+ achieves significantly better performance than the manually annotated MAFW dataset. The main reason is that humans tend to focus on major clues, which can easily lead to incomplete descriptions. These results confirm that, despite the lack of manual checks in MERCaption+, we can still ensure the quality of the labels. In the future, we will investigate other post-filtering techniques to further improve MER-Caption+’s annotation quality. 

**Ablation Study on MER-Caption.** As shown in Table 4, compared to the results without filtering or with only lowlevel filtering, our two-level filtering leads to a performance improvement, further verifying the effectiveness of our filtering technique. These findings underscore that dataset quality is as critical as quantity, and fewer training samples do not necessarily lead to worse performance. Please see Appendix M for more details. 

**Ablation Study on Model.** Table 5 compares different architectures and examines the impact of pre-fusion operations. Our results show that pre-fusion operations generally improve performance. This highlights the importance of treating cross-modal interactions as separate modules to more effectively capture multimodal characteristics. 

**Analysis of Input Impact.** Table 6 reveals the impact of different inputs. The distinction between “face” and “frame” lies in whether an additional face extractor is used to extract faces from frames. We observe a general trend: multimodal results outperform unimodal results. These findings suggest that humans express emotions through multiple modalities, and integrating them leads to improved performance. Additionally, face inputs slightly outperform frame inputs, and their combination does not result in further improvement. This suggests that current MER datasets mainly 

Table 6: **Input impact analysis.** The difference between “face” and “frame” is whether an additional face extractor is used to extract faces from frames. 

**==> picture [215 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input<br>MER-UniBench<br>audio face frame text<br>√ × × × 60.08<br>unimodal × √ × × 60.47<br>× × √ × 59.47<br>× × × √ 67.44<br>√ √ × √ 74.77<br>multimodal √ × √ √ 73.39<br>√ √ √ √ 74.60<br>Table 7: User study.<br>**----- End of picture text -----**<br>


|**Dataset**|**MER-Caption+**<br>Wins<br>Losses<br>Ties|
|---|---|
|MERR-Coarse<br>MERR-Fine|0.86<br>0.04<br>0.10<br>0.59<br>0.27<br>0.14|



focus on people, with limited emotional information conveyed through the environment. As a result, in this paper, we default to using audio, face, and text as the inputs. 

**User Study.** We conduct a user study to evaluate the quality of our proposed dataset. Since MERR-Fine and MERRCoarse (Cheng et al., 2024) share some samples with our dataset, we randomly select 20 overlapping samples. We then hire four expert annotators and present them with two descriptions for each sample: one from our dataset and one from the other datasets. The annotators are asked to watch the video and select the more accurate description. As shown in Table 7, our dataset provides more accurate descriptions than both the model-based MERR-Coarse and the human-filtered MERR-Fine, thereby validating the effectiveness of our proposed annotation strategy. 

**Choice of LLMs.** This paper adopts Qwen2.5 as the default LLM. In Figure 4(a), we further explore the impact of different LLMs. Experimental results show that the performance difference brought by LLM is limited. These results verify that the superior performance of AffectGPT over the existing MLLMs does not come from LLM but from our proposed emotion description dataset and model. 

**Choice of Audio and Video Encoders.** In Figures 4(b) and 4(c), the choice of audio and video encoder has a minimal impact on performance. This underscores that AffectGPT’s exceptional performance is primarily driven by our proposed high-quality, large-scale dataset and effective framework, rather than the specific acoustic or visual encoders used. For audio encoders (Figure 4(b)), ImageBind exhibits slightly inferior performance compared to other audio encoders. This may be attributed to the fact that other audio encoders are predominantly utilized in audio 

**==> picture [217 x 63] intentionally omitted <==**

**----- Start of picture text -----**<br>
78.00 78.00 78.00<br>74.40 74.40 75.06 75.12 74.77 74.40 73.81 74.48 74.39 74.77 74.40 73.42 73.12 74.63 74.77<br>70.80 70.80 70.80<br>67.20 67.20 67.20<br>63.60 63.6060.00 63.60<br>60.00 60.00<br>(a) LLM (b) Audio Encoder (c) Video Encoder<br>LLaMA3Baichuan2Qwen2 Qwen2.5 EVA_CLIPDINOv2 SigLIP CLIP_VIT<br>ImageBindData2vec_BaseWavLM_LargeHUBERT_Large<br>MER-UniBench MER-UniBench MER-UniBench<br>**----- End of picture text -----**<br>


Figure 4: Ablation studies on LLMs, audio encoders, and video encoders. 

Table 8: **Impact of rank value in LoRA.** In this table, we count the increase in trainable parameters when using LoRA for the LLM branch. The first row represents the model without the LoRA module. 

|**Rank**|**# Increased Parameters**|**MER-UniBench**|
|---|---|---|
|–<br>8<br>16<br>32|0<br>20,185,088<br>40,370,176<br>80,740,352|73.30<br>74.65<br>74.77<br>74.92|



content understanding tasks (e.g., ASR), where audio content plays a critical role in emotion recognition. Similarly, for video encoders (Figure 4(c)), CLIP ~~V~~ IT marginally outperforms EVA ~~C~~ LIP and DINOv2, aligning with findings from MERBench (Lian et al., 2024c), a unified benchmark for traditional categorical frameworks. These results suggest that insights derived from traditional categorical frameworks, such as encoder selection, may also be applicable to MLLM-based descriptive frameworks. 

**Role of LoRA in LLMs.** In Table 8, we count the increase in trainable parameters when using LoRA for the LLM branch. The first row represents the model without the LoRA module. Experimental results show that fine-tuning the LLM with LoRA improves performance compared to models without LoRA. However, increasing the rank for LoRA-based models does not yield significant performance gains and instead increases computational costs. 

## **6. Conclusion** 

This paper aims to enhance the emotional understanding of MLLMs from three aspects: (1) the dataset MER-Caption, which uses a model-led human-assisted strategy to create a large-scale dataset with guaranteed quality; (2) the model AffectGPT, which enhances multimodal fusion by moving cross-modal interactions outside of the LLM; and (3) the benchmark, which provides comprehensive evaluation metrics tailored to the free-form, natural language output style of MLLMs. Extensive experiments validate the effectiveness of our model and dataset. This work lays the foundation for building MLLMs with emotional understanding, contributing to the advancement of emotion AI. 

## **Submission and Formatting Instructions for ICML 2025** 

## **Acknowledgments** 

This work is supported by the Excellent Youth Program of State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS2024311), the National Natural Science Foundation of China (62201572, 62322120, 61831022, 62276259, U21B2010, 62271083, 62306316, 62176165, 62206136, 62476146), the Stable Support Projects for Shenzhen Higher Education Institutions (20220718110918001), the Young Elite Scientists Sponsorship Program by CAST (2024QNRC001), and the University of Oulu& Research Council of Finland Profi 7 (352788). 

## **Impact Statements** 

**Social Impact.** Emotion plays an important role in human communication, conveying human intentions and deep thoughts. As Minsky (Minsky, 1988) stated: _The question is not whether intelligent machines can have any emotions, but whether machines can be intelligent without any emotions._ The development of MER technology can enhance the human-computer interaction experience. 

**Ethics Statement.** This paper does not involve the collection of new data. The original data comes from the unlabeled part of MER2024 (Lian et al., 2024b), with permission from the dataset owners. The annotation process does not involve hiring external annotators, and no ethical issues are associated with this process. Additionally, we restrict the use of this dataset under the license of CC BY-NC 4.0, requiring researchers to use our dataset responsibly. Therefore, no ethical concerns are raised in this paper. 

## **References** 

- Burkhardt, F., Paeschke, A., Rolfes, M., Sendlmeier, W. F., Weiss, B., et al. A database of german emotional speech. In _Interspeech_ , volume 5, pp. 1517–1520, 2005. 

- Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., and Narayanan, S. S. Iemocap: Interactive emotional dyadic motion capture database. _Language Resources and Evaluation_ , 42:335– 359, 2008. 

- Chen, H., Shi, H., Liu, X., Li, X., and Zhao, G. Smg: A micro-gesture dataset towards spontaneous body gestures for emotional stress state analysis. _International Journal of Computer Vision_ , 131(6):1346–1366, 2023. 

   - Chu, Y., Xu, J., Zhou, X., Yang, Q., Zhang, S., Yan, Z., Zhou, C., and Zhou, J. Qwen-audio: Advancing universal audio understanding via unified large-scale audiolanguage models. _arXiv preprint arXiv:2311.07919_ , 2023. 

   - Cowen, A. S. and Keltner, D. Self-report captures 27 distinct categories of emotion bridged by continuous gradients. _Proceedings of the national academy of sciences_ , 114 (38):E7900–E7909, 2017. 

   - Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., and Ravi, S. Goemotions: A dataset of fine-grained emotions. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 4040–4054, 2020. 

   - Du, S., Tao, Y., and Martinez, A. M. Compound facial expressions of emotion. _Proceedings of the national academy of sciences_ , 111(15):E1454–E1462, 2014. 

   - Ekman, P. and Keltner, D. Universal facial expressions of emotion. _California mental health research digest_ , 8(4): 151–158, 1970. 

   - Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., Cukierski, W., Tang, Y., Thaler, D., Lee, D.-H., et al. Challenges in representation learning: A report on three machine learning contests. In _Neural information processing: 20th international conference, ICONIP 2013, daegu, korea, november 3-7, 2013. Proceedings, Part III 20_ , pp. 117–124. Springer, 2013. 

   - Gu, Y., Yang, K., Fu, S., Chen, S., Li, X., and Marsic, I. Multimodal affective analysis using hierarchical attention strategy with word-level alignment. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics_ , pp. 2225–2235, 2018. 

   - Han, J., Gong, K., Zhang, Y., Wang, J., Zhang, K., Lin, D., Qiao, Y., Gao, P., and Yue, X. Onellm: One framework to align all modalities with language. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 26584–26595, 2024. 

   - Hazarika, D., Zimmermann, R., and Poria, S. Misa: Modality-invariant and-specific representations for multimodal sentiment analysis. In _Proceedings of the 28th ACM International Conference on Multimedia_ , pp. 1122– 1131, 2020. 

- Cheng, Z., Cheng, Z.-Q., He, J.-Y., Wang, K., Lin, Y., Lian, Z., Peng, X., and Hauptmann, A. Emotion-llama: Multimodal emotion recognition and reasoning with instruction tuning. _Advances in Neural Information Processing Systems_ , 37:110805–110853, 2024. 

- Hsu, C.-C., Chen, S.-Y., Kuo, C.-C., K. Huang, T.-H., and Ku, L.-W. Emotionlines: An emotion corpus of multiparty conversations. In _Proceedings of the Eleventh International Conference on Language Resources and Evaluation_ , pp. 1597–1601, 2018. 

- Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., and Mohamed, A. Hubert: Selfsupervised speech representation learning by masked prediction of hidden units. _IEEE/ACM transactions on audio, speech, and language processing_ , 29:3451–3460, 2021. 

- Huang, Z., Zhao, J., and Jin, Q. Ecr-chain: Advancing generative language models to better emotion-cause reasoners through reasoning chains. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI_ , pp. 6288–6296, 2024. 

- Izard, C. E., Libero, D. Z., Putnam, P., and Haynes, O. M. Stability of emotion experiences and their relations to traits of personality. _Journal of personality and social psychology_ , 64(5):847, 1993. 

- Jiang, X., Zong, Y., Zheng, W., Tang, C., Xia, W., Lu, C., and Liu, J. Dfew: A large-scale database for recognizing dynamic facial expressions in the wild. In _Proceedings of the 28th ACM international conference on multimedia_ , pp. 2881–2889, 2020. 

- Jin, P., Takanobu, R., Zhang, W., Cao, X., and Yuan, L. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 13700–13710, 2024. 

- Kovecses, Z.¨ _Metaphor and emotion: Language, culture, and body in human feeling_ . Cambridge University Press, 2003. 

- Li, B., Zhang, Y., Chen, L., Wang, J., Yang, J., and Liu, Z. Otter: A multi-modal model with in-context instruction tuning. _arXiv preprint arXiv:2305.03726_ , 2023a. 

- Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International Conference on Machine Learning_ , pp. 1–13, 2023b. 

- Li, K., He, Y., Wang, Y., Li, Y., Wang, W., Luo, P., Wang, Y., Wang, L., and Qiao, Y. Videochat: Chat-centric video understanding. _arXiv preprint arXiv:2305.06355_ , 2023c. 

- Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., Wang, L., and Qiao, Y. Mvbench: A comprehensive multi-modal video understanding benchmark. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2024a. 

- Li, S., Deng, W., and Du, J. Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 2852– 2861, 2017. 

- Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. In _European Conference on Computer Vision_ , pp. 323–340. Springer, 2024b. 

- Lian, Z., Sun, H., Sun, L., Chen, K., Xu, M., Wang, K., Xu, K., He, Y., Li, Y., Zhao, J., et al. Mer 2023: Multilabel learning, modality robustness, and semi-supervised learning. In _Proceedings of the 31st ACM International Conference on Multimedia_ , pp. 9610–9614, 2023. 

- Lian, Z., Sun, H., Sun, L., Chen, L., Chen, H., Gu, H., Wen, Z., Chen, S., Zhang, S., Yao, H., et al. Open-vocabulary multimodal emotion recognition: Dataset, metric, and benchmark. _arXiv preprint arXiv:2410.01495_ , 2024a. 

- Lian, Z., Sun, H., Sun, L., Wen, Z., Zhang, S., Chen, S., Gu, H., Zhao, J., Ma, Z., Chen, X., et al. Mer 2024: Semi-supervised learning, noise robustness, and openvocabulary multimodal emotion recognition. In _Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing_ , pp. 41–48, 2024b. 

- Lian, Z., Sun, L., Ren, Y., Gu, H., Sun, H., Chen, L., Liu, B., and Tao, J. Merbench: A unified evaluation benchmark for multimodal emotion recognition. _arXiv preprint arXiv:2401.03429_ , 2024c. 

- Lian, Z., Sun, L., Sun, H., Chen, K., Wen, Z., Gu, H., Liu, B., and Tao, J. Gpt-4v with emotion: A zero-shot benchmark for generalized emotion recognition. _Information Fusion_ , 108:102367, 2024d. 

- Liang, Z., Xu, Y., Hong, Y., Shang, P., Wang, Q., Fu, Q., and Liu, K. A survey of multimodel large language models. In _Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering_ , pp. 405–409, 2024. 

- Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , pp. 5971–5984, 2024. 

- Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y., and Huang, M. Towards emotional support dialog systems. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pp. 3469–3483, 2021. 

- Liu, Y., Dai, W., Feng, C., Wang, W., Yin, G., Zeng, J., and Shan, S. Mafw: A large-scale, multi-modal, compound affective database for dynamic facial expression recognition in the wild. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pp. 24–32, 2022a. 

- Liu, Y., Yuan, Z., Mao, H., Liang, Z., Yang, W., Qiu, Y., Cheng, T., Li, X., Xu, H., and Gao, K. Make acoustic and visual cues matter: Ch-sims v2. 0 dataset and av-mixup consistent module. In _Proceedings of the International Conference on Multimodal Interaction_ , pp. 247–258, 2022b. 

- Lotfian, R. and Busso, C. Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings. _IEEE Transactions on Affective Computing_ , 2017. 

- Maaz, M., Rasheed, H., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 12585–12602, 2024. 

- Matsumoto, D. Culture and emotion. _The handbook of culture and psychology_ , pp. 171–194, 2001. 

Minsky, M. _Society of mind_ . Simon and Schuster, 1988. 

- Mollahosseini, A., Hasani, B., and Mahoor, M. H. Affectnet: A database for facial expression, valence, and arousal computing in the wild. _IEEE Transactions on Affective Computing_ , 10(1):18–31, 2017. 

OpenAI. Chatgpt, 2022. URL https://openai.com/ blog/chatgpt. 

- Plutchik, R. A general psychoevolutionary theory of emotion. _Emotion: Theory, research, and experience_ , 1, 1980. 

- Poria, S., Cambria, E., Hazarika, D., Majumder, N., Zadeh, A., and Morency, L.-P. Context-dependent sentiment analysis in user-generated videos. In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics_ , volume 1, pp. 873–883, 2017. 

- Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., and Mihalcea, R. Meld: A multimodal multi-party dataset for emotion recognition in conversations. In _Proceedings of the 57th Conference of the Association for Computational Linguistics_ , pp. 527–536, 2019. 

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In _International conference on machine learning_ , pp. 8748–8763. PMLR, 2021. 

Schutz, P. A. Emotion in education, 2007. 

Spezialetti, M., Placidi, G., and Rossi, S. Emotion recognition for human-robot interaction: Recent advances and future perspectives. _Frontiers in Robotics and AI_ , 7:532279, 2020. 

- Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., and Cai, D. Pandagpt: One model to instruction-follow them all. In _Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the era of Interactive Assistants_ , pp. 11–23, 2023. 

- Tang, C., Yu, W., Sun, G., Chen, X., Tan, T., Li, W., Lu, L., MA, Z., and Zhang, C. Salmonn: Towards generic hearing abilities for large language models. In _The Twelfth International Conference on Learning Representations_ , 2023. 

- Tao, R., Pan, Z., Das, R. K., Qian, X., Shou, M. Z., and Li, H. Is someone speaking? exploring long-term temporal features for audio-visual active speaker detection. In _Proceedings of the 29th ACM international conference on multimedia_ , pp. 3927–3935, 2021. 

- Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., and Salakhutdinov, R. Multimodal transformer for unaligned multimodal language sequences. In _Proceedings of the 57th Conference of the Association for Computational Linguistics_ , pp. 6558–6569, 2019. 

- Wang, Y., Sun, Y., Huang, Y., Liu, Z., Gao, S., Zhang, W., Ge, W., and Zhang, W. Ferv39k: A large-scale multiscene dataset for facial expression recognition in videos. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 20922–20931, 2022. 

- Xie, H., Peng, C.-J., Tseng, Y.-W., Chen, H.-J., Hsu, C.-F., Shuai, H.-H., and Cheng, W.-H. Emovit: Revolutionizing emotion insights with visual instruction tuning. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 26596–26605, 2024. 

- Xu, Y., Chen, H., Yu, J., Huang, Q., Wu, Z., Zhang, S.X., Li, G., Luo, Y., and Gu, R. Secap: Speech emotion captioning with large language model. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , pp. 19323–19331, 2024. 

- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. _arXiv preprint arXiv:2412.15115_ , 2024. 

- Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J., Hu, A., Shi, P., Shi, Y., et al. mplug-owl: Modularization empowers large language models with multimodality. _arXiv preprint arXiv:2304.14178_ , 2023. 

- Yu, W., Xu, H., Meng, F., Zhu, Y., Ma, Y., Wu, J., Zou, J., and Yang, K. Ch-sims: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pp. 3718–3727, 2020. 

- Zadeh, A., Chen, M., Poria, S., Cambria, E., and Morency, L.-P. Tensor fusion network for multimodal sentiment analysis. In _Proceedings of the Conference on Empirical Methods in Natural Language Processing_ , pp. 1103– 1114, 2017. 

- Zadeh, A. B., Liang, P. P., Poria, S., Cambria, E., and Morency, L.-P. Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pp. 2236–2246, 2018. 

- Zhang, H., Li, X., and Bing, L. Video-llama: An instructiontuned audio-visual language model for video understanding. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pp. 543–553, 2023. 

## **A. Related Works** 

This paper focuses on constructing datasets and designing models to enhance the emotional understanding capability of MLLMs. In this section, we mainly review related work in these two aspects. 

## **A.1. Emotion Dataset** 

Emotion datasets are the foundation for building MER systems (Wang et al., 2022; Chen et al., 2023). Most research has focused on building categorical datasets, where basic emotions are first defined, and annotators are asked to select the most likely one (Goodfellow et al., 2013) or multiple (Li et al., 2017) labels from basic emotions. However, emotions are often diverse (Demszky et al., 2020) and can coexist (Du et al., 2014), making it challenging for categorical datasets to fully capture these complex emotions. 

To address this, recent studies have shifted from categorical datasets to descriptive datasets, as emotion descriptions provide greater flexibility and enable the description of complex emotions in natural language. To construct such datasets, Liu et al. (2022a) used a human-based annotation strategy to capture the environment, body movements, facial expressions, and other emotion-related cues. However, the high annotation cost limits the scalability of these datasets. With the development of MLLMs, Cheng et al. (2024) used a more cost-effective automatic annotation method, where MLLMs are used to extract emotion-related descriptions from audio, facial expressions, and visual objects. However, they lacked pre-experimentation on MLLM selection, relying on empirical model choices, leading to insufficient label quality. In this paper, we propose a solution to balance label quality and dataset size. By leveraging high-quality human-based datasets to guide description generation and sample filtering, we achieve a quality-assured automatic annotation process and ultimately construct MER-Caption. 

## **A.2. Emotion Models** 

Emotion models are closely related to the training corpus. For categorical datasets, researchers often build classifiers to map multimodal human information to corresponding emotion labels. Apart from choosing the architecture (such as CNN, RNN, or Transformers), most research focuses on how to align and fuse multimodal information. For example, Hazarika et al. (2020) introduced a decomposition module to split features into modality-specific and modality-invariant representations. Gu et al. (2018) aligned different modalities at the word level and then learned time-dependent cross-modal interactions. Tsai et al. (2019) proposed using cross-attention to align features in the latent space. More recently, Lian et al. (2024c) conducted a fair comparison of various fusion and alignment strategies, showing that temporal-preserving features do not always outperform time-compressed features, suggesting that MER may be more suitable to solve from a global perspective. 

For descriptive datasets, due to their natural language style output, the framework needs to shift from traditional discriminative methods to generative methods. With the development of LLMs and MLLMs, researchers have started to build models based on them. For example, Huang et al. (2024) used Vicuna as the language model, jointly training emotion labels and descriptions. Xie et al. (2024) used the instruction-aware Q-Former module to learn the mapping between input images and emotional descriptions. Cheng et al. (2024) integrated different encoders to understand multimodal inputs and used LLaMA-2 as an LLM decoder. However, current models either only focus on unimodal information (Huang et al., 2024; Xie et al., 2024) or leave all cross-modal interactions to the LLM (Cheng et al., 2024), which is insufficient for solving MER tasks with multimodal characteristics. To this end, we introduce the AffectGPT model in this paper. 

## **B. Implementation Details** 

Our choice of unimodal encoders is guided by previous research (Lian et al., 2024c), using CLIP ViT-L (Radford et al., 2021) as the visual encoder and HUBERT-L (Hsu et al., 2021) as the acoustic encoder. Given the remarkable performance of Qwen-2.5 (Yang et al., 2024), we choose it as the LLM. To ensure training efficiency, we only fine-tune an extra LoRA module (in the LLM), projector, and pre-fusion branch, while freezing the weights of the LLM and unimodal encoders (see Figure 3). We default to setting the rank in the LoRA module to 16. This approach reduces GPU memory usage and speeds up training. Additionally, through preliminary experiments, we found that pre-training on other instruction datasets followed by a second-stage training on MER-Caption did not lead to performance improvements. The primary reason is the large scale of our dataset and the limited focus on MER in current instruction datasets. Therefore, we do not perform multi-stage training in our experiments. All models are implemented in PyTorch and conducted training and inference on 80GB NVIDIA Tesla A100 GPU. During training, we set the maximum number of epochs to 60, each epoch contains 5000 iterations, and the 

batch size of each iteration is 3. To optimize all trainable parameters, we use the AdamW optimizer and set the learning rate to 1e-5. For more implementation details, please refer to the code provided in https://github.com/zeroQiaoba/AffectGPT. 

## **C. Details about MLLMs** 

Table 9 provides model cards for different MLLMs, including reference papers, supported modalities, and links to pre-trained weights. 

Table 9: Model cards for MLLMs. 

||**Supported Modality**|**Link**|
|---|---|---|
|Otter (Li et al.,2023a)<br>VideoChat (Li et al.,2023c)<br>VideoChat2 (Li et al.,2024a)<br>Video-LLaVA (Lin et al.,2024)<br>Video-LLaMA (Zhang et al.,2023)<br>Video-ChatGPT (Maaz et al.,2024)<br>LLaMA-VID (Li et al.,2024b)<br>mPLUG-Owl (Ye et al.,2023)<br>Chat-UniVi (Jin et al.,2024)<br>SALMONN (Tang et al.,2023)<br>Qwen-Audio (Chu et al.,2023)<br>SECap (Xu et al.,2024)<br>OneLLM (Han et al.,2024)<br>PandaGPT (Su et al.,2023)<br>Emotion-LLaMA(Chenget al.,2024)|Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Video, Text<br>Audio, Text<br>Audio, Text<br>Audio, Text<br>Audio, Video, Text<br>Audio, Video, Text<br>Audio,Video,Text|https://github.com/Luodian/Otter<br>https://github.com/OpenGVLab/Ask-Anything/tree/main/video<br>chat<br>https://github.com/OpenGVLab/Ask-Anything/tree/main/video<br>chat2<br>https://github.com/PKU-YuanGroup/Video-LLaVA<br>https://github.com/DAMO-NLP-SG/Video-LLaMA<br>https://github.com/mbzuai-oryx/Video-ChatGPT<br>https://github.com/dvlab-research/LLaMA-VID<br>https://github.com/X-PLUG/mPLUG-Owl<br>https://github.com/PKU-YuanGroup/Chat-UniVi<br>https://github.com/bytedance/SALMONN<br>https://github.com/QwenLM/Qwen-Audio<br>https://github.com/thuhcsi/SECap<br>https://github.com/csuhan/OneLLM<br>https://github.com/yxuansu/PandaGPT<br>https://github.com/ZebangCheng/Emotion-LLaMA|



## **D. Visualization of MLLM Outputs** 

Figure 5 provides an example to visualize the outputs of different MLLMs. These outputs contain varying numbers of emotions, with emotion labels that are open-ended and not restricted to any predefined taxonomy. Therefore, traditional classification metrics, such as accuracy and F1 score, are not suitable for performance evaluation. In this paper, we propose evaluation metrics specifically tailored for the free-form, natural language output style of MLLMs. 

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [78 x 23] intentionally omitted <==**

**==> picture [411 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
Subtitle: Why are you all looking at me like that? So, as long as it's a woman, does she have to have a relationship with me?<br>**----- End of picture text -----**<br>


Chat-UniVi-main: In the video, a man is seen speaking to the camera while making facial expressions and gestures. The man's body language and expressions suggest that he might be trying to convey a message or share his thoughts with the audience. The environment appears to be a simple setting, with no significant background details. The man's emotional state is not entirely clear from the video alone, as it depends on the context of his words and actions. However, based on his facial expressions and body movements, we can infer that he might be feeling passionate, expressive, or even slightly nervous while addressing the camera. The video's content and the man's demeanor invite the viewer to interpret his emotions and message further. LLaMA-VID-main: The man in the video appears to be in a state of confusion or uncertainty, as he looks around and appears to be unsure of what to do. His facial expressions and body movements suggest that he is trying to process information or make a decision. The environment in which he is speaking, which is a laboratory, may also contribute to his emotional state, as it may suggest that he is dealing with a difficult or complex problem. Overall, it is difficult to predict the exact emotional state of the individual based on the information provided, but it is clear that he is experiencing some level of confusion or uncertainty. 

SALMONN: It is difficult to predict the emotional state of the individual in the audio without more context. However, based on the given information, it is possible that the individual is feeling frustrated or misunderstood. They may be feeling like they are being judged or scrutinized by others, and this could be causing them to feel upset or angry. Alternatively, they may be feeling defensive or protective of themselves, and this could be causing them to become aggressive or confrontational. Ultimately, without more information, it is impossible to accurately predict the emotional state of the individual in the audio. Video-LLaVA: The man in the video is sitting in front of a camera and talking. He appears to be in a relaxed state, with his hands resting on his lap. His facial expressions are calm and composed, and his body language suggests that he is comfortable in front of the camera. The environment around him is well-lit, and there are no visible signs of distractions or interruptions. The subtitle content of the video does not provide any specific information about the individual's emotions or the context of the video. However, based on the visual cues, it can be inferred that the man is likely in a positive emotional state. Video-ChatGPT: In the video, a man is seen talking to the camera while a woman looks on. The man is wearing a white shirt and a tie, and he appears to be engaged in a conversation with the woman. The woman's facial expression is not clearly visible, but her body language suggests that she is listening attentively to the man. The environment appears to be an office setting, with a desk and a computer visible in the background. The acoustic information, such as the sound of the man speaking, is not provided. The subtitles in the video indicate that the man is asking the woman if she has a relationship with him, and the woman responds by saying that she is not interested in him. Based on these visual and auditory cues, it is difficult to determine the exact emotional state of the individual. However, it is clear that the man is engaged in a conversation with the woman, and the woman is listening attentively to him. VideoChat2: The video doesn't seem to have any information related to the given subtitle content. It would be best to provide more details or context to help me understand the context and answer the question accurately. SECap: feeling astonished 

Figure 5: Visualization of MLLM outputs. 

## **E. Prompt for Label Extraction** 

To extract emotion labels from MLLM outputs, we use Qwen2.5 and apply the following prompt: 

_Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. If none are identified, please output an empty list._ 

For sentiment analysis, we use the multi-step prediction process. Specifically, we first extract emotion labels using the prompt above, and then apply the following prompt for sentiment analysis: 

_Please act as an expert in the field of emotions. We provide a few words to describe the emotions of a character. Please choose the most likely sentiment from the given candidates: [positive, negative, neutral]._ 

## **F. Choice of Description Generation Strategy** 

This section aims to determine the optimal strategy for generating descriptions. In Table 10, we present the results of preliminary experiments. First, we evaluate the performance of different ALLMs and VLLMs. Then, we investigate whether combining these models leads to improved performance. To do this, we use GPT-3.5 to integrate audio and video cues, extracted by the ALLM and VLLM, with text content. As shown in Table 10, we observe that these combinations generally outperform the use of either ALLM or VLLM alone. Based on these findings, we select SALMONN as the ALLM for generating audio cues, Chat-UniVi as the VLLM for generating visual cues, and GPT-3.5 to combine the audio, video, and text cues, resulting in the final descriptions. 

We would like to clarify that, in this paper, we do not use _combined results_ for model selection. Instead, we rely on the performance of _individual models_ . Specifically, for VLLM, Chat-UniVi outperforms mPLUG-Owl and Video-ChatGPT; for ALLM, SALMONN outperforms SECap. As a result, we employ the combination of Chat-UniVi and SALMONN for description generation. The combination experiments are primarily designed to demonstrate that integrating multimodal cues can enhance performance. In future work, we will conduct additional experiments where _combined results_ are used for model selection. For example, leveraging the combination of SALMONN and Chat-UniVi for description generation. 

Table 10: **Preliminary experiments.** We choose Fs as the primary metric, as this metric considers both accuracy and completeness. 

|Model|Fs(_↑_)<br>Precisions(_↑_)<br>Recalls(_↑_)|
|---|---|
|SECap (Xu et al.,2024)<br>SALMONN (Tang et al.,2023)<br>Video-ChatGPT (Maaz et al.,2024)<br>mPLUG-Owl (Ye et al.,2023)<br>Chat-UniVi(Jin et al.,2024)|45.72_±_0_._09<br>54.52_±_0_._15<br>39.37_±_0_._05<br>47.96_±_0_._04<br>50.20_±_0_._04<br>45.92_±_0_._04<br>50.52_±_0_._06<br>54.03_±_0_._04<br>47.44_±_0_._07<br>52.73_±_0_._13<br>54.54_±_0_._13<br>51.04_±_0_._13<br>53.08_±_0_._01<br>53.68_±_0_._00<br>52.50_±_0_._02|
|SECap + mPLUG-Owl<br>SECap + Video-ChatGPT<br>SECap + Chat-UniVi<br>SALMONN + Video-ChatGPT<br>SALMONN + Chat-UniVi<br>SALMONN + mPLUG-Owl|56.69_±_0_._03<br>50.05_±_0_._23<br>65.38_±_0_._33<br>56.90_±_0_._08<br>52.03_±_0_._04<br>62.79_±_0_._14<br>57.34_±_0_._16<br>48.85_±_0_._29<br>69.41_±_0_._13<br>58.19_±_0_._23<br>53.16_±_0_._17<br>64.26_±_0_._31<br>58.43_±_0_._06<br>51.62_±_0_._00<br>67.31_±_0_._15<br>**58.70**_±_0_._04<br>51.77_±_0_._01<br>67.76_±_0_._11|



## **G. Prompt for Clue Merge** 

To merge multimodal clues, we use GPT-3.5 and apply the following prompt: 

_Please act as an expert in the field of emotions. We provide acoustic and visual clues that may be related to the character’s emotional state, along with the original subtitle of the video. Please analyze which parts can infer the emotional state and explain the reasons. During the analysis, please integrate the textual, audio, and visual clues._ 

Even when modality conflicts exist (i.e., the emotions conveyed by audio, video, and text are not the same, as shown in Figure 6), GPT-3.5 can provide reasonable responses, primarily due to its powerful reasoning ability. 

**==> picture [80 x 23] intentionally omitted <==**

**==> picture [80 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [79 x 23] intentionally omitted <==**

**==> picture [476 x 52] intentionally omitted <==**

**----- Start of picture text -----**<br>
Text: Why are you all looking at me like that? So, as long as it's a woman, does she have to have a relationship with me?<br>Visual Clue: In the video, the screen shows a male character in an indoor setting. At the beginning of the video, his eyes are wide open and his<br>mouth is also open, indicating a surprised facial expression. In the following scenes, he looks around, seemingly explaining or narrating<br>something to the people around him. Overall, his emotions are not positive or optimistic.<br>Acoustic Clue: In the audio, the character speaks with a stutter, which usually expresses feelings of nervousness, anxiety, or unease.<br>**----- End of picture text -----**<br>


**==> picture [475 x 35] intentionally omitted <==**

**----- Start of picture text -----**<br>
Merged Clue: The subtitle expresses the male character's dissatisfaction and anger towards the people around him. Based on the surprised and<br>negative facial expression of the male character in the video clues, as well as the stuttering speech in the audio clues, we can infer that the male<br>character is expressing a feeling of dissatisfaction and anger in this sentence. He may feel troubled by the prejudice of the people around him<br>and is unhappy with this unfair treatment.<br>**----- End of picture text -----**<br>


Figure 6: Example of modality conflict. 

## **H. Dataset Comparison** 

Figure 7 compares the distribution of description lengths and the number of emotions per sample. We observe that our dataset provides detailed descriptions and rich emotion labels for each sample. 

**==> picture [76 x 57] intentionally omitted <==**

**==> picture [77 x 57] intentionally omitted <==**

**==> picture [76 x 57] intentionally omitted <==**

**==> picture [77 x 57] intentionally omitted <==**

**==> picture [76 x 57] intentionally omitted <==**

**==> picture [77 x 57] intentionally omitted <==**

**==> picture [464 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) EmoVIT (b) MERR-Fine (c) MERR-Coarse (d) MAFW (e) OV-MERD (f) MER-Caption<br>(g) EmoVIT (h) MERR-Fine (i) MERR-Coarse (j) MAFW (k) OV-MERD (l) MER-Caption<br>**----- End of picture text -----**<br>


Figure 7: **Dataset comparison** . The first row compares the lengths of the descriptions, while the second row compares the number of labels per sample. 

## **I. Video Duration Distribution** 

Figure 8 presents the video duration distribution of the MER-Caption dataset. We observe that the majority of samples have durations ranging from 2 to 5 seconds. 

**==> picture [176 x 132] intentionally omitted <==**

Figure 8: Video duration distribution. 

## **J. MER-UniBench Details** 

MER-UniBench is a comprehensive evaluation benchmark covering three typical tasks in MER, including fine-grained emotion recognition, basic emotion recognition, and sentiment analysis. Different tasks involve different datasets, and we provide their statistical information in Table 11. In this paper, we intentionally focus on single-person videos, as this allows us to eliminate interference from other speakers and reduce task difficulty. Multi-person MER belongs to another research topic and will be addressed in our future work. 

Table 11: Dataset statistics in MER-UniBench. All datasets in our study focus on single-person videos. 

||**Dataset**<br>**Chosen Set**<br>**# Samples**<br>**Label Description**<br>**Data Source**|
|---|---|
|**Fine-grained Emotion**|OV-MERD+<br>All<br>532<br>unfxed categories and<br>diverse number of labelsper sample<br>movies, TV series|
|**Basic Emotion**|MER2023<br>MER-MULTI<br>411<br>most likely label among six candidates<br>movies, TV series<br>MER2024<br>MER-SEMI<br>1,169<br>most likely label among six candidates<br>movies, TV series<br>IEMOCAP<br>Session5<br>1,241<br>most likely label among four candidates<br>actor’s performance<br>MELD<br>Test<br>2,610<br>most likelylabel amongseven candidates<br>”Friends” TV series|
|**Sentiment Analysis**|CMU-MOSI<br>Test<br>686<br>sentiment intensity, ranging from [-3, 3]<br>opinion videos in YouTube<br>CMU-MOSEI<br>Test<br>4,659<br>sentiment intensity, ranging from [-3, 3]<br>opinion videos in YouTube<br>CH-SIMS<br>Test<br>457<br>sentiment intensity, ranging from [-1, 1]<br>movies, TV series, and shows<br>CH-SIMS v2<br>Test<br>1,034<br>sentiment intensity,rangingfrom[-1,1]<br>movies,TV series,and shows|



**OV-MERD+** is our newly collected dataset, an extended version of the previous OV-MERD (Lian et al., 2024a). Unlike traditional datasets, which select a single label from basic emotions, OV-MERD is a fine-grained emotion dataset that allows each sample to have a variable number of emotions, using any emotion not restricted to predefined taxonomies. OV-MERD initially contains 332 samples, and we further expand its dataset size, obtaining OV-MERD+. 

**MER2023** (Lian et al., 2023) and **MER2024** (Lian et al., 2024b) are widely used in Chinese MER research, with MER2024 being an extended version of MER2023. The original data in both datasets comes from movies and TV shows. They use various techniques to segment video clips, ensuring that each clip has only one person, with their speech content being relatively complete. To ensure annotation quality, they hire multiple annotators, each selecting the most likely label from six candidate emotions: _worry_ , _happy_ , _neutral_ , _angry_ , _surprised_ , and _sad_ . The final label is determined through majority voting. 

**IEMOCAP** (Busso et al., 2008) is one of the most widely used emotion datasets. It contains five sessions, each with a male and a female actor in a laboratory environment. The dataset includes the following emotion labels: _anger_ , _happiness_ , _sadness_ , _neutral_ , _excitement_ , _frustration_ , _fear_ , _surprise_ , and _others_ . Following previous research (Poria et al., 2017), we choose the last session for testing, and use the first four emotions, and merge _surprise_ and _happiness_ into _happiness_ . 

**MELD** (Poria et al., 2019) is an extension of the text-centered EmotionLines dataset (Hsu et al., 2018), adding audio and video content. The raw data is derived from the Friends TV series. The dataset has seven emotion labels, and each sample is assigned to one of the most likely labels: _anger_ , _joy_ , _sadness_ , _neutral_ , _disgust_ , _fear_ , and _surprise_ . 

**CMU-MOSI** (Zadeh et al., 2017) and **CMU-MOSEI** (Zadeh et al., 2018) consist of opinion videos collected from online platforms. CMU-MOSEI is an extended version of CMU-MOSI, with more samples and a wider range of topics. In these 

datasets, each sample is labeled with a sentiment intensity score ranging from -3 to +3, where -3 represents extremely negative emotion and +3 represents extremely positive emotion. 

**CH-SIMS** (Yu et al., 2020) and **CH-SIMS v2** (Liu et al., 2022b) differ from the English-centered CMU-MOSI and CMU-MOSEI by focusing on emotions within the Chinese culture. The original data comes from movies, TV series, and shows. Similar to CMU-MOSI and CMU-MOSEI, these datasets also annotate sentiment intensity, but with a different range [ _−_ 1 _,_ 1], where -1 represents extremely negative emotion and +1 represents extremely positive emotion. 

## **K. Emotion Wheel** 

Since there is no universal definition of the emotion wheel, we follow previous work (Lian et al., 2024a) and use five emotion wheels in this paper. 

**==> picture [127 x 127] intentionally omitted <==**

**==> picture [127 x 128] intentionally omitted <==**

**==> picture [128 x 128] intentionally omitted <==**

**==> picture [297 x 159] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) W1 (b) W2 (c) W3<br>(d) W4 (e) W5<br>**----- End of picture text -----**<br>


Figure 9: **Emotion wheel** . We use five emotion wheels, all of which are derived from previous research (Lian et al., 2024a). 

## **L. Main Results** 

Table 12 reports the complete results, with several metrics for each dataset, and the primary metrics are highlighted in gray. In the last column, we report the average value of the primary metrics. These results verify the effectiveness of our AffectGPT in multimodal emotion understanding. 

## **M. Ablation Study on MER-Caption** 

Table 13 compares the performance across different datasets. To ensure a fair comparison, we keep the model architecture and experimental setup unchanged, only altering the training dataset. Experimental results in Table 13 demonstrate the effectiveness of our MER-Caption dataset for emotion understanding. It addresses the issue of existing datasets either giving insufficient attention to emotion tasks or lacking high-quality emotion descriptions. 

Table 12: **Main results** . In this table, “A”, “V”, and “T” represent audio, video, and text, indicating the input information used by each MLLM during inference. The gray-highlighted columns represent the primary metric for each dataset, while the “Mean” column reports the average score of the primary metrics across all datasets. 

|A<br>V<br>T|**MER2023**|**MER2023**|**MER2024**|**MER2024**|**MELD**|**IEMOCAP**|**IEMOCAP**|**CMU-MOSI**|**CMU-MOSEI**<br>WAF(_↑_)<br>ACC(_↑_)|**CMU-MOSEI**<br>WAF(_↑_)<br>ACC(_↑_)|
|---|---|---|---|---|---|---|---|---|---|---|
||HIT(_↑_)||HIT(_↑_)||HIT(_↑_)|HIT(_↑_)||WAF(_↑_)<br>ACC(_↑_)|||
|Otter<br>_×_<br>~~_√_~~<br>~~_√_~~<br>OneLLM<br>_√_<br>_×_<br>_√_<br>Video-LLaVA<br>_×_<br>_√_<br>_√_<br>SECap<br>_√_<br>_×_<br>_√_<br>PandaGPT<br>_√_<br>_×_<br>_√_<br>Qwen-Audio<br>_√_<br>_×_<br>_√_<br>PandaGPT<br>_×_<br>_√_<br>_√_<br>Video-ChatGPT<br>_×_<br>_√_<br>_√_<br>VideoChat2<br>_×_<br>_√_<br>_√_<br>PandaGPT<br>_√_<br>_√_<br>_√_<br>LLaMA-VID<br>_×_<br>_√_<br>_√_<br>VideoChat<br>_×_<br>_√_<br>_√_<br>SALMONN<br>_√_<br>_×_<br>_√_<br>Chat-UniVi<br>_×_<br>_√_<br>_√_<br>mPLUG-Owl<br>_×_<br>_√_<br>_√_<br>Emotion-LLaMA<br>_√_<br>_√_<br>_√_<br>**AffectGPT**<br>_√_<br>_√_<br>_√_|16.41||14.65||22.57|29.08||52.89<br>54.27|50.44<br>50.77<br>54.09<br>54.18<br>61.64<br>64.20<br>54.18<br>53.85<br>61.33<br>60.73<br>46.90<br>51.16<br>64.25<br>65.55<br>63.12<br>65.66<br>54.32<br>54.82<br>67.61<br>**68.82**<br>63.89<br>66.21<br>63.61<br>63.02<br>67.03<br>66.90<br>63.18<br>67.47<br>**72.91**<br>**73.17**<br>**67.66**<br>67.25<br>**80.90**<br>**80.68**||
||25.52||17.21||28.32|33.44||64.01<br>64.48|||
||36.93||30.25||30.73|38.95||56.37<br>57.62|||
||40.95||52.46||25.56|36.92||55.76<br>56.71|||
||33.57||39.04||31.91|36.55||66.06<br>65.85|||
||41.85||31.61||**49.09**|35.47||70.09<br>71.49|||
||39.13||47.16||38.33|47.21||58.50<br>60.21|||
||44.86||46.80||37.33|**56.83**||54.42<br>57.77|||
||33.67||54.50||36.64|48.70||66.84<br>67.23|||
||40.21||51.89||37.88|44.04||61.92<br>62.80|||
||50.72||57.60||42.75|46.02||61.78<br>62.65|||
||48.73||57.30||41.11|48.38||65.13<br>65.09|||
||55.53||45.38||45.62|46.84||**81.00**<br>**81.25**|||
||**57.62**||**65.67**||45.61|52.37||54.53<br>57.62|||
||56.86||59.89||**49.11**|**55.54**||**72.40**<br>**72.26**|||
||**59.38**||**73.62**||46.76|55.47||66.13<br>66.31|||
||**78.54**||**78.80**||**55.65**|**60.54**||**81.30**<br>**81.25**|||
|A<br>V<br>T||**CH-SIMS**||**CH-SIMS v2**|||**OV-MERD+**<br>Fs(_↑_)<br>Precisions(_↑_)<br>Recalls(_↑_)|||**Mean**|
|||WAF(_↑_)<br>ACC(_↑_)||WAF(_↑_)<br>ACC(_↑_)|||||||
|Otter<br>_×_<br>~~_√_~~<br>~~_√_~~<br>OneLLM<br>_√_<br>_×_<br>_√_<br>Video-LLaVA<br>_×_<br>_√_<br>_√_<br>SECap<br>_√_<br>_×_<br>_√_<br>PandaGPT<br>_√_<br>_×_<br>_√_<br>Qwen-Audio<br>_√_<br>_×_<br>_√_<br>PandaGPT<br>_×_<br>_√_<br>_√_<br>Video-ChatGPT<br>_×_<br>_√_<br>_√_<br>VideoChat2<br>_×_<br>_√_<br>_√_<br>PandaGPT<br>_√_<br>_√_<br>_√_<br>LLaMA-VID<br>_×_<br>_√_<br>_√_<br>VideoChat<br>_×_<br>_√_<br>_√_<br>SALMONN<br>_√_<br>_×_<br>_√_<br>Chat-UniVi<br>_×_<br>_√_<br>_√_<br>mPLUG-Owl<br>_×_<br>_√_<br>_√_<br>Emotion-LLaMA<br>_√_<br>_√_<br>_√_<br>**AffectGPT**<br>_√_<br>_√_<br>_√_||57.56<br>60.57||53.12<br>56.20|||16.63<br>17.67<br>15.74<br>22.25<br>24.49<br>20.41<br>34.00<br>36.48<br>31.86<br>36.97<br>43.51<br>32.17<br>31.33<br>33.08<br>29.77<br>32.36<br>38.52<br>27.91<br>35.07<br>37.86<br>32.67<br>39.80<br>43.12<br>36.97<br>39.21<br>42.85<br>36.16<br>37.12<br>39.64<br>34.91<br>45.01<br>46.83<br>43.32<br>44.52<br>44.55<br>44.49<br>45.00<br>43.57<br>46.61<br>48.00<br>**48.20**<br>47.81<br>**48.18**<br>47.91<br>**48.47**<br>**52.97**<br>**54.85**<br>**51.22**<br>**62.52**<br>**62.21**<br>**63.00**|||34.82<br>41.14<br>44.40<br>46.64<br>46.84<br>49.26<br>50.77<br>52.64<br>52.67<br>52.92<br>56.07<br>56.71<br>57.89<br>57.94<br>**62.45**<br>**64.17**<br>**74.77**|
|||63.39<br>63.92||61.98<br>62.46|||||||
|||53.28<br>54.64||57.45<br>59.28|||||||
|||59.51<br>62.89||57.41<br>60.92|||||||
|||62.93<br>62.37||58.88<br>58.84|||||||
|||70.73<br>**73.45**||65.26<br>68.17|||||||
|||62.07<br>61.60||65.25<br>65.31|||||||
|||64.82<br>64.43||65.80<br>66.85|||||||
|||69.49<br>69.59||70.66<br>71.13|||||||
|||68.38<br>67.78||67.23<br>67.40|||||||
|||69.35<br>68.81||67.48<br>67.73|||||||
|||69.52<br>69.33||72.14<br>72.12|||||||
|||68.69<br>69.85||65.93<br>67.07|||||||
|||68.15<br>67.78||66.36<br>67.18|||||||
|||**72.13**<br>71.65||**75.00**<br>**74.97**|||||||
|||**78.32**<br>**78.61**||**77.23**<br>**77.39**|||||||
|||**88.49**<br>**88.40**||**86.18**<br>**86.17**|||||||



Table 13: Dataset comparison. 

||**Dataset**<br>**Filtering**|**MER2023 MER2024 MELD IEMOCAP CMU-**<br>**MOSI**<br>**CMU-**<br>**MOSEI**<br>**CH-**<br>**SIMS**<br>**CH-**<br>**SIMS v2**<br>**OV-**<br>**MERD+**|**Mean**|
|---|---|---|---|
|**General**<br>**Instruction**|MiniGPT4<br>_×_<br>_√_|11.56<br>12.91<br>18.89<br>16.06<br>53.57<br>45.98<br>57.66<br>55.16<br>13.86<br>17.57<br>16.65<br>22.60<br>30.18<br>52.58<br>56.50<br>52.36<br>51.19<br>20.16|31.74<br>35.53|
||VideoChat<br>_×_<br>_√_|24.87<br>22.42<br>21.56<br>32.91<br>50.13<br>56.17<br>50.07<br>51.71<br>24.56<br>27.70<br>24.73<br>27.66<br>39.46<br>45.45<br>56.86<br>43.68<br>47.05<br>26.09|37.16<br>37.63|
||LLaVA<br>_×_<br>_√_|42.21<br>41.54<br>32.97<br>49.96<br>54.48<br>56.42<br>52.04<br>54.80<br>35.75<br>41.56<br>42.30<br>32.61<br>46.21<br>52.82<br>57.72<br>52.78<br>53.44<br>36.96|46.69<br>46.27|
||WavCaps<br>_×_<br>_√_|5.75<br>7.71<br>4.35<br>4.99<br>45.59<br>22.76<br>53.04<br>45.68<br>4.95<br>23.72<br>26.97<br>23.39<br>27.30<br>54.67<br>49.54<br>58.12<br>55.93<br>21.57|21.65<br>37.91|
|**Emotion**<br>**Description**|EmoVIT<br>–<br>MAFW<br>–<br>MERR-Coarse<br>–<br>MERR-Fine<br>–<br>MER-Caption<br>–<br>MER-Caption+<br>–|39.31<br>50.24<br>32.36<br>48.24<br>53.40<br>61.53<br>69.72<br>66.53<br>38.09<br>52.67<br>55.99<br>40.85<br>57.60<br>66.11<br>62.27<br>75.20<br>70.02<br>42.75<br>35.34<br>36.60<br>29.37<br>36.94<br>65.10<br>63.27<br>75.12<br>73.76<br>33.18<br>69.00<br>72.84<br>47.38<br>54.49<br>66.21<br>60.03<br>79.90<br>78.54<br>52.56<br>72.12<br>74.21<br>54.69<br>56.41<br>75.10<br>70.97<br>80.21<br>78.06<br>58.42<br>**78.54**<br>**78.80**<br>**55.65**<br>**60.54**<br>**81.30**<br>**80.90**<br>**88.49**<br>**86.18**<br>**62.52**|51.05<br>58.16<br>49.85<br>64.55<br>68.91<br>**74.77**|



## **N. Impact of Sampling Frames in Video Branch** 

This paper defaults to sampling 8 frames per video. But if we change the number of sampled frames, will it significantly impact performance? To answer this, we conducted additional experiments in this section. Specifically, we compare two types of inputs: (1) face-only and (2) face-text combinations, and evaluate model performance across different sampling frame counts, ranging from 2 to 64. In Figure 10, we observe that using too few frames (e.g., fewer than 2) results in a noticeable decline in performance, indicating that insufficient frames lead to information loss. However, further increasing the number of sampling frames (e.g., more than 16) does not yield significant performance improvements. This can be attributed to the fact that MER tasks typically use short-duration videos with relatively stable facial expressions. 

**==> picture [361 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
61.49 61.49 74.31 74.31 74.10  [74.23]<br>61.26 73.54 73.51<br>61.10 72.94<br>61.03 72.94<br>60.76<br>60.56 60.47 71.56<br>60.10 70.19<br>59.63 68.81<br>59.17 67.44<br>59.17 67.44<br>58.71 66.07<br>0 2 4 8 16 32 64 0 2 4 8 16 32 64<br># of frames # of frames<br>(a) Face-only Input (b) Face-text Input<br>MER-UniBench MER-UniBench<br>**----- End of picture text -----**<br>


Figure 10: Impact of sampling frames. 

