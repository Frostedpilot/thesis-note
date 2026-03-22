**==> picture [558 x 45] intentionally omitted <==**

**==> picture [526 x 81] intentionally omitted <==**

Dong Thap University 

National Cheng Kung University 

University of Greenwich 

**==> picture [301 x 38] intentionally omitted <==**

International University 

Emotion Analysis, Large Language Models, BERT, Voting Mechanisms, Multi-label Classi�cation 

November 6th, 2025 

https://doi.org/10.21203/rs.3.rs-7501928/v1 

  This work is licensed under a Creative Commons Attribution 4.0 International License. Read Full License 

No competing interests reported. 

# Enhancing Multi-label Emotion Prediction through Rule-based Voting with LLM and BERT Variants 

Minh Hieu Le[1][†] , Cong-Phuoc Phan[2][†] , Thanh Tuan Nguyen[3,4*][†] , Thi Thanh Sang Nguyen[4,5*][†] 

- 1School of Technology and Engineering, Dong Thap University, Cao Lanh, Vietnam. 

- 2Computer Science & Information Engineering Department, National Cheng-Kung University, Tainan, Taiwan. 

- 3*School of Computing and Mathematical Sciences, University of Greenwich, London, UK. 

- 4School of Computer Science and Engineering, International University, Ho Chi Minh, Vietnam. 

   - 5Vietnam National University, Ho Chi Minh, Vietnam. 

- *Corresponding author(s). E-mail(s): tuan.nguyen@greenwich.ac.uk; nttsang@hcmiu.edu.vn; 

Contributing authors: lmhieu@dthu.edu.vn; peter.phancong@iir.csie.ncku.edu.tw; 

> †These authors contributed equally to this work. 

## Abstract 

Emotion analysis in text has become increasingly crucial for applications ranging from social media monitoring to mental health assessment. While advancements in natural language processing (NLP) have improved capabilities in this area, accurately identifying and categorizing complex emotional expressions remains a challenging task. This difficulty arises from the contextual implications and the complexity inherent in human emotions. This paper presents a novel framework that combines Large Language Models (LLMs) and BERT variants through an adaptive rule-based voting mechanism for robust multi-label emotion analysis. Our approach introduces three key components: (1) an adaptive weighted voting strategy that dynamically adjusts model contributions based on confidence scores, (2) a sophisticated prompt engineering technique that enables LLMs to better understand emotional context through template-based approaches, and 

(3) a hybrid decision-making mechanism that effectively integrates the complementary strengths of both LLM and BERT architectures through rule-based aggregation. Experimental results on the SemEval-2025 Task 11 (Track A) test set demonstrate that our proposed method achieves a macro F1 of 80.42% and micro F1 of 82.33%, outperforming the strongest individual transformer architecture (DeBERTa) by 9.8% and 7.4% respectively, and highest-performing LLM method (SFT Data-Augmented) by 2.1% and 1.7% respectively. Notably, our system shows particular strength in handling complex emotional expressions and ambiguous contexts, with consistent improvements across all five emotion categories, particularly excelling in fear detection (86.97% F1-score) and demonstrating robust performance on challenging low-frequency emotions like anger (74.62% F1-score). 

Keywords: Emotion Analysis, Large Language Models, BERT, Voting Mechanisms, Multi-label Classification 

## 1 Introduction 

Emotion analysis in text has emerged as a fundamental component of modern natural language processing, with applications spanning social media monitoring, customer feedback analysis, mental health assessment, and human-computer interaction systems. The ability to automatically detect and classify emotions from textual content has become increasingly important as digital communication continues to proliferate across various platforms and domains. 

Despite significant advancements in deep learning and transformer-based architectures, accurately identifying and categorizing complex emotional expressions remains a formidable challenge. This difficulty stems from several factors: the contextual nature of emotional expression, the subtlety of linguistic cues that convey emotion, the frequent co-occurrence of multiple emotions within a single text, and the inherent subjectivity in emotional interpretation. Traditional approaches often struggle with these complexities, particularly in multi-label scenarios where multiple emotions may be present simultaneously. 

Recent developments in Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding nuanced textual content, while BERT and its variants have established themselves as powerful tools for fine-grained text classification tasks. However, each approach has distinct strengths and limitations: BERT variants excel at capturing local contextual relationships and can be fine-tuned for specific tasks, while LLMs demonstrate superior performance in understanding global context and handling diverse linguistic patterns through their extensive pre-training. 

This work bridges these gaps with a rule-based voting framework that adaptively integrates BERT variants and LLMs, achieving state-of-the-art multi-label emotion detection. It addresses three fundamental research questions: 

RQ1: How can BERT variants be optimally utilized for multi-label emotion detection? 

As well known, compared to traditional methods for word embedding, BERT is the best one so far. With the bidirectional architecture, BERT can take into account the preceding and following context of a word. Therefore, it can understand the context of words through identifying emotional nuances in text, and detect precisely sentiment implied in text. 

RQ2: Why do Rule-based voting aggregator outperform traditional ensemble approaches? 

Normal voting classifiers are not good as voting with rules. This will be demonstrated experimentally. Moreover, specifying reasonable rules would help handle classification efficiently in a specific domain rather than using default or normal voting rules. Traditional ensemble approaches process classifiers and vote for the highest output in a similar way but lack of weighting, thus encountering the problem of data imbalance. This study will conduct some experiments to evaluate the performance of rule-based voting classification method compared with normal voting classification method and traditional ensemble methods. 

RQ3: How can Large Language Models enhance emotion detection capabilities? In recent years, Large Language Models (LLMs) have demonstrated significant capabilities in detecting emotions within text; however, systematic and comprehensive evaluations of their performance remain limited. The integration of sentiment analysis into LLMs enhances their ability to generate contextually appropriate, empathetic, and accurate responses, thereby improving user engagement and interaction quality. This reciprocal relationship allows LLMs not only to contribute to emotion recognition systems but also to leverage sentiment analysis feedback to refine their conversational abilities. Modern LLMs, such as ChatGPT [1] and Gemini [2], now embed dedicated sentiment analysis modules, leading to more precise and effective emotion detection in textual data. 

This paper makes the following key contributions: 

1. We propose a unified framework that effectively combines traditional transformerbased approaches (BERT variants) with modern generative models (LLMs) through an adaptive voting mechanism. 

2. Unlike traditional ensemble methods that apply uniform model combinations, our approach selects optimal models independently for each emotion category based on performance metrics. 

3. We introduce a sophisticated voting system that goes beyond simple majority voting by incorporating confidence scores, probability thresholds, and hierarchical decision rules. 

We provide extensive experimental validation across multiple BERT variants and LLMs, demonstrating consistent improvements over individual model approaches. Our method achieves superior performance on the SemEval-2025 Task 11 benchmark, establishing new performance benchmarks for multi-label emotion detection. 

The remainder of this paper is structured as follows: Section 2 presents a comprehensive literature review covering traditional emotion analysis approaches, deep learning methods, and ensemble techniques. Section 3 details our proposed methodology, including the system architecture, model selection mechanisms, and voting 

strategies. Section 4 describes data pre-processing, the experimental setup, and implementation details. Section 5 presents comprehensive experimental results and ablation studies. Section 6 provides detailed discussion of findings, limitations, and comparative analysis. Finally, Section 7 concludes the paper with key findings and future research directions. 

## 2 Literature Review 

## 2.1 Traditional Emotion Analysis and Detection Approaches 

Emotion analysis plays a crucial role in understanding user sentiments across social networks and e-commerce platforms [3]. Traditional approaches employed machine learning methods including Support Vector Machines, Naive Bayes, Decision Trees, and ensemble methods such as Random Forest [4] and Gradient Boosting [5] for classification tasks [6]. 

While these methods established foundational principles for emotion detection, they encountered limitations in capturing complex contextual relationships and semantic nuances. The emergence of deep learning architectures addressed these constraints by enabling sophisticated feature learning and enhanced contextual understanding capabilities. 

## 2.2 Deep Learning in Emotion Analysis 

## 2.2.1 Transformer-based models 

Transformer architecture [7] has become foundational in modern NLP, offering significant improvements in emotion detection through self-attention mechanisms that model contextual relationships without sequential processing limitations [8]. This capability proves valuable for recognizing implicit emotions by capturing dependencies between distant words, particularly in social media contexts where emotional expressions are often indirect [9]. 

BERT (Bidirectional Encoder Representations from Transformers) [10] represents a significant advancement in transformer-based language understanding, employing bidirectional representation learning through masked language modeling to capture contextual relationships between words. The BERT framework involves pre-training on unlabeled data using masked language modeling objectives, followed by fine-tuning on specific downstream tasks with labeled data. 

Numerous BERT variants have been developed to address specific limitations and improve performance across different applications. These variants encompass different architectural innovations: standard pretraining models (BERT, RoBERTa [11]), discriminator-based approaches (ELECTRA [12]), autoregressive models (XLNet [13]), parameter-efficient variants (ALBERT [14], DistilBERT [15], TinyBERT [16]), multilingual models (mBERT [17], XLM-RoBERTa [18]), and specialized architectures (SpanBERT [19], BART [20]). 

Research confirms that different BERT variants demonstrate varying strengths across emotion categories, with no single variant achieving optimal performance for all emotions in multi-label classification tasks [21]. This observation motivates the 

exploration of ensemble approaches that can leverage the complementary strengths of different transformer architectures for comprehensive emotion detection systems. 

## 2.2.2 Large Language Models (LLMs) 

Sentiment analysis automates emotion detection, helping businesses align offerings with consumer expectations. Integrating sentiment analysis into Large Language Models (LLMs) enhances their ability to generate context-aware, empathetic, and accurate responses, improving customer engagement [22]. Modern LLMs, such as ChatGPT [1] and Gemini [2], now incorporate sentiment analysis modules, enabling more effective emotion detection in text. 

Supervised Fine-Tuning (SFT) adapts pre-trained language models to specific downstream tasks through parameter optimization using labeled datasets. The theoretical foundation builds upon transfer learning principles, where knowledge acquired during pre-training is transferred and refined for task-specific objectives [23, 24]. The SFT process operates through gradient-based optimization to minimize taskspecific loss functions, with effectiveness stemming from pre-trained representations that capture semantic relationships and contextual understanding [25]. 

In-Context Learning (ICL) enables large language models to perform tasks through demonstration examples and instructions provided within the input context, without parameter updates. The theoretical basis relates to meta-learning and few-shot learning principles, where models learn to recognize patterns from limited examples presented during inference [22, 26]. ICL operates through attention-based architecture’s capacity to establish relationships between demonstration examples and target instances within the same context window. Research indicates that ICL performance correlates with model scale, with larger models demonstrating superior few-shot learning capabilities [27]. 

Both SFT and ICL approaches offer distinct advantages for emotion detection tasks. SFT provides task-specific optimization through parameter adaptation, typically yielding superior performance when sufficient labeled data is available. ICL offers flexibility and rapid deployment capabilities without requiring model training infrastructure. Recent advances in prompt engineering and parameter-efficient fine-tuning methods have enhanced both paradigms’ effectiveness [28, 29], establishing both as viable approaches for integrating LLMs into emotion detection systems. 

## 2.3 Ensemble and Voting Methods 

Ensemble methods have long been recognized for their ability to improve predictive performance by combining multiple models. Traditional ensemble approaches, such as bagging, boosting, and stacking, aggregate the outputs of base learners to reduce variance, mitigate overfitting, and improve generalisation. 

In multi-class and multi-label classification, voting mechanisms are among the simplest yet effective ensemble techniques. Majority voting, soft voting (averaging probabilities), and weighted voting are frequently employed to consolidate predictions from multiple classifiers [30]. 

Recent work has explored hybrid voting strategies that go beyond uniform aggregation. For instance, emotion-specific model selection and rule-based weighting schemes have shown promise in multi-label emotion detection, where different models may excel at different emotional categories. Such selective ensemble frameworks introduce additional flexibility and have been demonstrated to improve both macro and micro performance metrics. 

## 3 Proposed Methodology 

## 3.1 System Architecture 

The proposed system combines Transformer-based models and Large Language Models through a rule-based voting aggregator for multi-label emotion detection, as shown in Figure 1. The architecture comprises three main components: (1) transformer-based models, (2) large language models, and (3) rule-based voting aggregator. 

**==> picture [372 x 207] intentionally omitted <==**

Fig. 1: The proposed rule-based voting system architecture for multi-label emotion detection. 

The system architecture operates through a systematic three-stage pipeline for multi-label emotion detection. Initially, input text undergoes parallel processing through multiple model pathways: transformer-based models (BERT variants) perform fine-tuned classification with sigmoid activation to generate emotion probabilities, while LLMs execute either SFT or ICL approaches to produce binary emotion predictions. Subsequently, the Weight assignment mechanism converts these diverse model outputs into standardized probability estimates through sigmoid outputs for BERT variants and F1-based calibration for LLM binary predictions. Finally, the Rule-based 

voting aggregator processes these calibrated probabilities through emotion-specific model selection and hierarchical decision rules to generate final multi-label emotion classifications. This comprehensive workflow ensures systematic integration of complementary model strengths while maintaining consistent probability-based aggregation across diverse architectural approaches. 

## 3.2 Transformer-based Models 

To systematically evaluate Transformer-based architectures for multi-label emotion detection, we selected twelve widely-adopted BERT variants based on their availability and established usage within the research community. Our methodology employs a comprehensive empirical evaluation approach wherein we assess all model variants to determine which architectures exhibit optimal performance for individual emotion categories, rather than relying on predetermined model selection criteria. 

The selected models encompass significant advancements in Transformer architecture development: 

Standard pretraining models: BERT [10] serves as the foundational model, trained via masked language modeling. RoBERTa [11] extends BERT with dynamic masking, longer training durations, and larger batch sizes, enhancing performance consistency. 

Autoregressive and Permutation-based models: XLNet [13] employs a permutation-based objective in combination with Transformer-XL recurrence, enabling modeling of bidirectional contexts without relying on masking strategies. Discriminator-based pretraining: ELECTRA [12] replaces the masked token prediction task with a discriminator trained to distinguish real input tokens from synthetically generated ones, significantly improving training sample efficiency. Lightweight and compressed variants: DistilBERT [15], a distilled version of BERT, offers reduced size and inference time while retaining performance. ALBERT [14] achieves compression through cross-layer parameter sharing and embedding factorization. TinyBERT [16] is trained via knowledge distillation from a larger BERT model. 

Multilingual and cross-lingual Models: mBERT [17] supports 104 languages using a shared WordPiece vocabulary. XLM-RoBERTa [18] is pretrained on an extended multilingual corpus to enhance cross-lingual generalization. 

Domain-specific and span-based variants: SpanBERT [19] improves spanlevel understanding by masking and predicting continuous text spans. BART [20] combines bidirectional encoding and autoregressive decoding within a denoising sequence-to-sequence framework. 

Our system architecture supports unlimited model variants without constraints, enabling continuous integration of new Transformer models as they become available. This scalability ensures that the approach remains adaptable to evolving NLP architectures while maintaining empirical performance assessment for emotion-specific model selection. 

Each transformer-based model underwent supervised fine-tuning using the training and development partitions from Track A of SemEval 2025 Task 11, an established 

benchmark for multi-label emotion detection. We augmented each pretrained backbone with a task-specific classification head comprising a linear layer that generates independent binary predictions for the five target emotion categories (anger, fear, joy, sadness, surprise). The fine-tuning process followed standard transfer learning protocols, enabling end-to-end optimization of both the pretrained representations and the classification layer on the emotion-annotated corpus. 

Prediction Generation, F1 Performance, and Probability Procedures 

The transformer-based models generate binary predictions for each emotion category across all test data instances, accompanied by corresponding probability estimates for each emotion class. Each model produces sigmoid-activated probability scores within the range [0, 1], representing the confidence level for the presence of each target emotion in the input text. These probability estimates serve as direct inputs to the Weight Assignment Mechanism described in Section 3.4. The binary predictions are subsequently processed by the Voting Rules Mechanism outlined in Section 3.4 for emotion decisions. 

Performance evaluation is conducted through systematic comparison of binary predictions against ground truth annotations in the development dataset, yielding F1 metrics computed at both individual emotion levels and aggregate levels (macro and micro F1 scores). These F1 performance metrics function as selection criteria for the emotion-specific model selection mechanism within the Rule-based Voting Aggregator described in Section 3.4, ensuring that only the most effective models contribute to each emotion category’s final classification. 

## 3.3 Large Language Models (LLMs) 

We employ Large Language Models through two distinct methodologies: Supervised Fine-Tuning (SFT) and In-Context Learning (ICL). This dual approach combines the adaptability of fine-tuned models with the zero-shot capabilities of pre-trained LLMs for multi-label emotion detection. 

To ensure consistent prompt formatting across all LLM approaches, we design standardized prompt templates that facilitate reliable model training and inference. Table 1 presents the two primary template configurations employed throughout our methodology: Template 1 for unified multi-label classification and Template 2 for binary emotion-specific detection. These templates are strategically designed to optimize model performance while maintaining consistency across different LLM architectures and fine-tuning strategies. 

## 3.3.1 SFT Approaches 

Supervised fine-tuning adapts pre-trained models using labeled data through promptlabel pairs. We implement three strategies using Gemini-1.5-flash-001-tuning, each targeting different aspects of multi-label emotion classification (Figure 2). 

Our SFT methodology follows a two-phase process: fine-tuning with labeled data using specific prompt templates (Table 1), then inference using identical templates for 

Table 1: Prompt templates for supervised fine-tuning approaches 

||Table 1: Prompt templates for supervised fne-tuning approaches|
|---|---|
|Template|Prompt Structure|
|||
|Template 1|”Analyze the sentiment of the following Tweets {input text} and classify them as ’ANGER’,|
||’FEAR’, ’JOY’, ’SADNESS’, ’SURPRISE’. Choose all possible emotions in the list that|
||match the sentence without any explanation.”|
|||
|Template 2|”Analyze the sentiment of the following Tweets {input text} Do you fnd any {emotion} in|
||that sentence? Use the schema: Yes or No. Make sure you do not add any explanation nor|
||other detail.”|



**==> picture [297 x 293] intentionally omitted <==**

Fig. 2: Overview of the three SFT approaches for multi-label emotion detection 

prediction consistency. During training, models receive input text formatted with templates and corresponding emotion labels as targets. For prediction, the same template structure queries the fine-tuned model for emotion classifications. 

Approach 1: Vanilla Multi-Label SFT 

Uses Template 1 for unified multi-label detection. A single model trains on the complete dataset to predict all emotions simultaneously, capturing inter-emotion relationships within one inference pass. 

Approach 2: Task-Decomposed SFT 

Employs Template 2 to create five specialized binary classifiers. Each model focuses on one emotion category, reducing interference between different emotional signals and enabling targeted optimization. 

## Approach 3: Data-Augmented SFT 

Builds on Approach 2 using Template 2 but incorporates augmented training data with positive-filtered external instances. This addresses class imbalance while maintaining the focused benefits of binary decomposition. 

## 3.3.2 ICL Prompt Approaches 

ICL performs tasks through carefully designed prompts without parameter updates. We apply Template 1 from Table 1 across multiple LLM variants for comprehensive multi-label 

Our ICL methodology employs Template 1 to enable simultaneous detection of all five emotion categories within one inference pass. The template structure is: 

”Analyze the sentiment of the following Tweets {input text} and classify them as ’ANGER’, ’FEAR’, ’JOY’, ’SADNESS’, ’SURPRISE’. Choose all possible emotions in the list that match the sentence without any explanation.” 

For each test instance, we format the input text according to Template 1 and send the prompt to the respective LLM through API calls. The models process the structured prompt and return predictions indicating which emotions are present in the input text. We implement this approach across multiple LLM variants without any fine-tuning, relying solely on the models’ pre-trained capabilities to respond to the formatted prompts and provide emotion classifications for the given text instances. 

## Prediction Generation and F1 Performance Assessment 

The LLM approaches generate binary predictions for each emotion category across all test dataset instances through two distinct pathways: SFT models produce direct binary outputs via fine-tuned classification heads, while ICL models generate classifications through prompt-response mechanisms.These binary predictions undergo two-stage processing within the rule-based voting framework: conversion to calibrate probability estimates for weight assignment (Section 3.4.2) and direct integration into the hierarchical voting rules mechanism (Section 3.4.3). 

Performance evaluation employs systematic comparison of binary predictions against ground truth annotations in the development dataset, computing F1 metrics at both individual emotion and aggregate levels (macro and micro F1 scores). These F1 performance metrics serve dual functions within the rule-based voting framework: determining which LLM models participate in voting for each emotion category through emotion-specific model selection and enabling probability estimation by converting binary predictions into calibrated confidence scores through the Weight Assignment Mechanism outlined in Section 3.4.2. 

## 3.4 Rule-based Voting Aggregator 

The core component of our proposed methodology is the Rule-based Voting Aggregator, which combines predictions from multiple different models to produce final emotion classifications. Figure 3 shows the detailed structure and decision flow 

of this aggregation process. In this framework, each participating model—whether transformer-based or LLMs—contributes binary predictions, F1 performance metrics, and probability estimates for individual emotion categories. These outputs are systematically integrated through our hierarchical rule-based decision mechanism to produce optimal multi-label emotion classifications. 

**==> picture [316 x 322] intentionally omitted <==**

Fig. 3: Details of rule-based voting aggregator predictions 

## 3.4.1 Model Selection Mechanism 

Traditional ensemble methods apply uniform model combinations across all prediction tasks. However, selecting the most appropriate subset of models to participate in the voting process is a critical step for optimizing classification performance. Our approach recognizes that different models exhibit varying strengths for different emotions, necessitating adaptive selection strategies. In this study, model selection is based on the F1 score — a balanced metric that accounts for both precision and recall in multi-label classification tasks [30, 31]. 

Specifically, for each individual emotion category (e.g., joy, sadness, fear, anger, and surprise), models are independently evaluated on the development set. An F1 score is calculated per emotion, and only the top-performing models for each category are selected to contribute predictions. This per-emotion selection strategy enables the ensemble to leverage the unique strengths of different architectures for specific emotional labels. 

By relying on F1-based selection, the ensemble voting process effectively filters out underperforming or noisy models, thereby reducing the risk of deteriorating the overall accuracy of the emotion classification system. 

For each emotion category e ∈{anger, fear, joy, sadness, surprise}, we select the top-N performing models based on development set F1 scores. The selection process follows: 

Algorithm 1 Emotion-Specific Model Selection 

Input: Model set M , emotion e, development set Ddev Output: Selected model subset Me for each model mi ∈ M do Evaluate F 1[e] i[on][D][dev][for][emotion][e] end for Sort models by F 1[e] in descending order Select top-N models: Me = {m1, m2, ..., mN } return Me 

## 3.4.2 Weight Assignment Mechanism 

The weight assignment mechanism transforms diverse model outputs into standardized voting weights through a systematic three-stage process that enables effective integration of heterogeneous architectures within the rule-based voting framework. Step 1: Class Probability Generation 

The first stage generates probability estimates from heterogeneous model architectures through architecture-specific approaches. 

Transformer-based Models (BERT Variants): Class probabilities are generated directly through sigmoid activation functions applied to the final classification layer outputs, producing probability values in the range [0,1] that reflect the model’s confidence in emotion presence. 

Large Language Models: Class probabilities are estimated through F1-based calibration since LLMs produce discrete binary predictions. The calibration process maps binary outputs to probability estimates using the model’s empirical performance on the development set: 

**==> picture [372 x 56] intentionally omitted <==**

## Step 2: Probability-to-Weight Conversion 

The second stage transforms the generated class probabilities into discrete voting weights using the systematic mapping scheme presented in Table 2. This conversion process applies uniformly to all probability estimates, ensuring consistent weight assignment across diverse model architectures. 

Table 2: Probability-to-weight mapping scheme with confidence interpretations 

||||
|---|---|---|
||||
|Class Probability|Weight|Confdence Interpretation|
||||
|0.8 – 1.0|+2|Strong model confdence that the emotion is present|
|0.6 – 0.8|+1|Moderate confdence levels indicating emotion presence|
|0.4 – 0.6|0|Model uncertainty about emotion presence or absence|
|0.2 – 0.4|-1|Moderate confdence suggesting emotion absence|
|0.0 – 0.2|-2|Strong confdence in emotion absence|



## Step 3: Weight Aggregation 

The final stage integrates the assigned weights into the voting rules mechanism through systematic aggregation of individual model contributions. For each emotion category e, the total weight score is computed as: 

**==> picture [214 x 23] intentionally omitted <==**

i=1 

where wi[e][represents][the][weight][assigned][to][model][i][for][emotion][e][,][and][k][denotes] the number of selected models participating in the voting process. 

These aggregated weight scores serve as primary inputs to the hierarchical voting rules mechanism described in Section 3.4.3, enabling effective integration of diverse architectural strengths through standardized probability-based aggregation. 

## 3.4.3 Voting Rules Mechanism 

The voting rules mechanism implements a hierarchical decision-making process that systematically aggregates weighted predictions from selected models to determine final emotion classifications. As illustrated in Figure 3, this mechanism employs a threetier rule structure that addresses various prediction scenarios, from clear consensus to highly ambiguous cases. 

## Rule 1: Total Weight Aggregation 

The primary decision criterion evaluates the total weight score computed from all selected models for each emotion category. For each emotion e, the total weight We is calculated as: 

**==> picture [214 x 31] intentionally omitted <==**

where wi[e][represents][the][weight][assigned][to][model][i][for][emotion][e][based][on][the] probability-to-weight mapping scheme (Table 2). The decision logic follows: 

- If We > 0: Assign positive value (1) - the emotion is present 

- If We < 0 : Assign negative value (0) - the emotion is absent 

- If We = 0: Proceed to Rule 2 

This primary rule handles the majority of classification decisions where model consensus provides a clear directional preference. 

Rule 2: Positive vs Negative Weight Count 

When the total weight equals zero, indicating balanced positive and negative evidence, the system evaluates the distribution of model predictions by counting the number of models contributing positive versus negative weights: 

- Count positive weights: Npos = |{i : wi[e][>][ 0][}|] 

- `•` Count negative weights: Nneg = |{i : wi[e][<][ 0][}|] 

The decision logic applies: 

- If Npos > Nneg: Assign positive value (1) 

- If Npos < Nneg: Assign negative value (0) 

- If Npos = Nneg: Proceed to Rule 3 

This secondary rule leverages model agreement patterns when weighted aggregation fails to provide decisive evidence. 

Rule 3: Average Probability Threshold 

In the most ambiguous scenario where both total weight and vote counts are balanced, the system employs a probability-based tie-breaking mechanism. The average class probability across all selected models is computed as: 

**==> picture [216 x 30] intentionally omitted <==**

where p[e] i[represents][the][probability][output][of][model][i][for][emotion][e][.][The][final] decision criterion applies: 

- If P[¯] e > 0.5: Assign positive value (1) 

- If P[¯] e < 0.5: Assign negative value (0) 

This tertiary rule ensures that even in highly uncertain cases, the system can make informed decisions based on the collective confidence of all participating models. 

## 4 Pre-processing and Experimental Results 

## 4.1 Datasets 

Datasets used in this study come from SemEval 2025 Task 11 [32] for multi-label emotion detection. We focus on the problem on Track A (Multi-label Emotion Detection), classifying binary emotions. The datasets include the following attributes: id, text, joy, sadness, fear, anger, and surprise (five emotion classes). The values of the emotion label are binary (0 or 1), indicating the presence or absence of each emotion. There are three datasets: train, dev and test sets in the same structure with the size of 2,768, 

116 and 2767 rows, respectively. The distribution of each emotion class in the data sets is shown in Figures 4. 

**==> picture [119 x 86] intentionally omitted <==**

**==> picture [119 x 86] intentionally omitted <==**

**==> picture [119 x 86] intentionally omitted <==**

**==> picture [326 x 10] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Training dataset (b) Development dataset (c) Test dataset<br>**----- End of picture text -----**<br>


Fig. 4: Distribution of emotion classes across all datasets showing class imbalance 

As shown, the datasets are imbalanced in emotion classes except for the fear class, for example, there are 2435 values of 0 (negative) and only 333 values of 1 (positive) for the anger class in the train dataset, that is, only around 12% rows belong to ’anger’. This ratio is also similar in the other datasets. 

## 4.2 Data pre-processing 

## 4.2.1 Data Pre-Processing for BERT 

The preprocessing pipeline for BERT variants ensures consistent input formatting across all twelve Transformer models while maintaining compatibility with multi-label classification requirements. We implemented a standardized approach that addresses both text tokenization and label preprocessing systematically. 

Raw input texts undergo tokenization using model-specific tokenizers corresponding to each BERT variant (e.g., BertTokenizer, RobertaTokenizer, DebertaTokenizer). The tokenization process applies uniform parameters across all models: maximum sequence length of 512 tokens, truncation for longer texts, padding to ensure consistent sequence lengths, and automatic attention mask generation to distinguish actual tokens from padding tokens. 

Given the multi-label nature of emotion detection, labels require careful preprocessing to ensure compatibility with Binary Cross-Entropy with Logits loss function. The preprocessing pipeline converts all emotion labels to float32 tensors with values normalized between 0 and 1, maintaining a tensor shape of (batch ~~s~~ ize, 5) corresponding to the five target emotions. Multi-class labels, when detected, are automatically converted to one-hot encoded vectors to ensure independent binary classification for each emotion category. Each BERT variant undergoes standardized initialization using AutoConfig framework with multi-label classification settings: problem type set to multi ~~l~~ abel ~~c~~ lassification, number of labels fixed at 5, and precision handling standardized to float32 for numerical consistency across all models. 

This preprocessing approach ensures fair comparison across different BERT architectures while maintaining the flexibility necessary for effective integration within the Rule-based Voting Aggregator. 

## 4.2.2 Data Pre-processing for LLM (SFT Only) 

Data preprocessing for Large Language Models in Supervised Fine-Tuning (SFT) approaches requires careful consideration of dataset composition and label format conversion to ensure optimal performance across the three distinct fine-tuning strategies. We implement a comprehensive preprocessing pipeline that addresses the unique requirements of each SFT approach while maintaining consistency in prompt template design. 

Our LLM preprocessing pipeline utilizes two primary datasets: SemEval-2025 Task 11 (2,768 training instances) and SemEval-2018 Task 1 (17,465 instances), resulting in a combined dataset of 20,233 annotated instances. This augmentation strategy addresses the limited size of the primary SemEval-2025 dataset while maintaining consistency in emotion taxonomy across both sources. 

For SFT approaches, we implement three distinct dataset preparation strategies corresponding to the different fine-tuning methodologies described in Section 3. The complete dataset strategy utilizes all 20,233 instances without filtering, preserving all multi-label combinations including instances where no emotions are present. The emotion-specific decomposition strategy partitions the combined dataset into five emotion-specific subsets: anger-specific training employs 3,160 instances, fearspecific training utilizes 11,536 instances, joy-specific training incorporates 6,578 instances, sadness-specific training uses 5,262 instances, and surprise-specific training employs 5,744 instances. The positive-enhanced strategy filters SemEval-2018 data to retain only instances with positive emotion annotations, which are then merged with corresponding SemEval-2025 emotion-specific subsets to create balanced training distributions. 

These emotion-specific subset sizes demonstrate the varying availability of positive instances across emotion categories, with fear showing the highest representation (11,536 instances) and anger exhibiting the lowest availability (3,160 instances), reflecting the inherent class imbalance characteristics of the combined training corpus. 

For SFT approaches, binary emotion vectors undergo systematic conversion into natural language labels compatible with LLM training protocols. The conversion process varies according to the specific SFT strategy: 

For Approach 1 (single model), multi-label instances are transformed into comma-separated emotion lists enclosed in brackets (e.g., ”[Anger, Fear]”), while instances with no detected emotions receive empty labels ”[]”. This approach enables the model to capture inter-dependencies among the five predefined emotion categories (anger, fear, joy, sadness, surprise) within a single inference pass. 

For Approach 2 and 3 (specialized models), binary emotion vectors are converted into simple ”Yes” or ”No” responses corresponding to the presence or absence of each specific emotion category. This binary format aligns with the decomposed classification approach and enables focused learning for individual emotions. 

## 4.3 Experimental settings 

## 4.3.1 BERT settings 

Model training was conducted using PyTorch framework [33] on GPU-enabled environments with systematic hyperparameter optimization. We implemented early stopping with patience of 5 epochs to prevent overfitting, triggered when no improvement in training loss was observed for consecutive epochs. All models were trained using consistent preprocessing parameters: input texts were tokenized with truncation and padding to a maximum sequence length of 512 tokens, ensuring uniform input dimensions across all twelve BERT variants. 

Our optimization strategy employed the AdamW optimizer [34] with a learning rate of 1 × 10[−][5] and weight decay regularization of 0.02 to mitigate overfitting risks. We selected Binary Cross-Entropy with Logits (BCEWithLogitsLoss) as the objective function due to its computational efficiency and appropriateness for multi-label classification tasks. Training stability was enhanced through implementation of a StepLR scheduler that reduces the learning rate by a factor of 0.5 every 3 epochs, facilitating convergence in later training phases. 

The training process utilized gradient clipping and model checkpointing strategies. Model states corresponding to the lowest training loss were preserved alongside their corresponding optimizer states for subsequent evaluation and ensemble integration. Training was conducted with a batch size of 8 samples per iteration to balance computational efficiency with gradient estimation quality. The maximum training duration was set to 80 epochs, with early stopping typically occurring between epochs 20-40 depending on model convergence characteristics. 

## 4.3.2 LLM Settings 

Large Language Model experiments were conducted using two distinct configuration paradigms corresponding to the SFT and ICL approaches described in Section 3.3. 

SFT Configuration Parameters: Supervised fine-tuning experiments utilized Gemini-1.5-flash-001 as the base architecture through Google AI Studio platform. Training hyperparameters were optimized for multi-label emotion detection performance: 

- Learning Rate: 2×10[−][5] for all approaches, selected through preliminary validation experiments 

- Training Epochs: Variable by approach - 90 epochs (Approach 1), 15 epochs (Approach 2), 45 epochs (Approach 3) 

- Batch Processing: Automatic batching handled by AI Studio infrastructure 

- Early Stopping: Implemented based on validation loss plateaus to prevent overfitting 

Training data preparation followed the three-strategy framework: complete dataset utilization (20,233 instances), emotion-specific decomposition (variable subset sizes), and positive-enhanced filtering (balanced distributions). Each approach employed corresponding prompt templates from Table 1 for consistent input formatting. 

ICL Configuration Parameters: ICL experiments were conducted across four LLM variants through their respective API interfaces: 

- ChatGPT-4o: OpenAI API with standard temperature settings (0.0) for reproducible outputs 

- DeepSeek V3: DeepSeek API with consistent prompt formatting and response parsing 

- Grok: X.AI platform access with standardized query parameters 

All ICL experiments employed the binary classification template outlined in Section 3.3 (ICL Prompt Approaches): ”Does this tweet express {emotion}? {input text} Answer: Yes/No only.” Temperature parameters were set to 0.0 across all models to ensure deterministic responses for evaluation consistency. 

This experimental configuration ensures reproducible results while maintaining consistency with the methodological framework established in Section 3.3. 

## 5 Experimental Results 

## 5.1 Evaluation Metrics 

We employ standard multi-label classification metrics following SemEval 2025 Task 11 — Track A official guidelines. Model performance is assessed using precision, recall, and F1 score computed at both individual emotion and aggregate levels. 

Precision measures the ratio of correctly predicted positive instances to all predicted positive instances for each emotion category. Recall quantifies the ratio of correctly predicted positive instances to all actual positive instances. F1 score represents the harmonic mean of precision and recall, providing balanced performance assessment. 

We compute two aggregate performance measures: 

- Macro-averaged metrics: unweighted mean across all emotion classes, treating each emotion equally regardless of frequency. 

- Micro-averaged metrics: metrics calculated by aggregating contributions across all classes, weighting performance by emotion frequency. 

Evaluations were conducted using sklearn.metrics functions (precision ~~s~~ core, recall ~~s~~ core, f1 ~~s~~ core) with appropriate averaging settings to generate comprehensive per-class and global performance measures. 

## 5.2 Performance Analysis 

Table 3 presents comprehensive performance evaluation results across three distinct methodological categories: voting and ensemble methods, transformer-based models, and large language models. The experimental evaluation demonstrates that our proposed rule-based voting aggregator achieves superior performance compared to all individual models and alternative ensemble approaches across multiple evaluation metrics. 

Table 3: Comprehensive F1 evaluation results of proposed voting rules, transformer models, and large language models on test dataset 

|Models / Methods|Macro||Micro||Anger||Fear||Joy|Sadness<br>Surprise|Sadness<br>Surprise|Sadness<br>Surprise|Sadness<br>Surprise|Sadness<br>Surprise|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||
|Voting & Ensemble|||||||||||||||
|Rule-based Voting (our proposal)|0.8042||0.8233||0.7462||0.8697||0.8229||0.8208||0.7727||
|Majority Voting - BERT Variants|0.7462||0.7770||0.6394||0.8375||0.7663||0.7599||0.6728||
|Majority Voting - LLM Models|0.7889||0.8088||0.7220||0.8604||0.8073||0.7967||0.7583||
|Majority Voting - Hybrid (LLM+BERT)|0.7834||0.8107||0.6886||0.8609||0.8033||0.8076||0.7566||
|Traditional Ensemble - BERT Variants|0.6495||0.7419||0.4553||0.8278||0.7367||0.7354||0.6921||
|Transformer-based Models|||||||||||||||
|BERT|0.7049||0.7430||0.5755||0.8209||0.7198||0.7230||0.6854||
|RoBERTa|0.7346||0.7654||0.6342||0.8332||0.7370||0.7592||0.7095||
|DeBERTa|0.7324||0.7665||0.6007||0.8302||0.7567||0.7564||0.7179||
|ELECTRA|0.7152||0.7541||0.5764||0.8316||0.7388||0.7353||0.6940||
|DistilBERT|0.6762||0.7182||0.5352||0.8032||0.6759||0.6811||0.6854||
|ALBERT|0.6542||0.6980||0.5091||0.7867||0.6587||0.6735||0.6428||
|BART|0.7333||0.7662||0.6183||0.8355||0.7500||0.7467||0.7161||
|mBERT|0.6139||0.6632||0.4710||0.7662||0.5890||0.6206||0.6225||
|SpanBERT|0.6988||0.7334||0.5850||0.8080||0.7001||0.7053||0.6954||
|TinyBERT|0.5557||0.6371||0.3625||0.7741||0.6255||0.6069||0.4092||
|XLM-R|0.6656||0.7070||0.5258||0.7917||0.6661||0.6682||0.6762||
|XLNet|0.7065||0.7461||0.5611||0.8221||0.7185||0.7435||0.6877||
|Large Language Models|||||||||||||||
|SFT - Vanilla Multi-Label|0.7871||0.8071||0.7097||0.8572||0.8027||0.7958||0.7699||
|SFT - Task-Decomposed|0.7761||0.7987||0.7003||0.8509||0.7796||0.7771||0.7615||
|SFT - Data-Augmented|0.7875||0.8096||0.7051||0.8626||0.8080||0.7998||0.7620||
|ICL - ChatGPT-4o|0.6874||0.6909||0.6030||0.6667||0.7893||0.7575||0.6204||
|ICL - DeepSeek V3|0.7084||0.7094||0.6760||0.7126||0.7919||0.7406||0.6208||
|ICL - Grok|0.6631||0.6538||0.6353||0.6103||0.7519||0.7471||0.5707||



## 5.2.1 Rule-based voting aggregator performance 

Our proposed Rule-based Voting Aggregator demonstrates exceptional performance, achieving the highest scores across six out of seven evaluation metrics. The system attains a macro F1 score of 0.8042 and micro F1 score of 0.8233, representing substantial improvements over the best-performing individual models. Specifically, our approach outperforms the top transformer-based model (DeBERTa: macro F1 = 0.7324, micro F1 = 0.7665) by 9.8% and 7.4% respectively, while surpassing the highest-performing LLM approach (SFT Data-Augmented: macro F1 = 0.7875, micro F1 = 0.8096) by 2.1% and 1.7% respectively. 

The rule-based voting aggregator demonstrates consistent superiority across individual emotion categories, achieving the highest F1 scores for anger (0.7462), fear (0.8697), joy (0.8229), and sadness (0.8208). Notably, the system exhibits particularly strong performance in fear detection (0.8697), which represents the highest individual emotion score achieved across all evaluated models and methods. For surprise detection, while our approach achieves competitive performance (0.7615), it ranks as the second-highest score, marginally trailing the SFT Task-Decomposed approach (0.7727) by 1.5%. 

The effectiveness of our rule-based voting aggregator stems from its comprehensive integration of diverse model architectures. The system incorporates predictions from all twelve transformer-based models (BERT, RoBERTa, DeBERTa, ELECTRA, DistilBERT, ALBERT, BART, mBERT, SpanBERT, TinyBERT, XLM-R, XLNet) alongside all six large language model approaches (three SFT variants and three ICL 

variants). Through the emotion-specific model selection mechanism, the voting aggregator dynamically identifies optimal model combinations for each emotion category, ensuring that the most effective predictors contribute to final classifications. 

## 5.2.2 Comparative analysis with alternative methods 

Our experimental evaluation includes systematic comparison with four alternative ensemble strategies to validate the effectiveness of the proposed rule-based approach. Traditional ensemble methods using BERT variants achieve substantially lower performance (macro F1 = 0.6495, micro F1 = 0.7419), demonstrating a performance gap of 23.9% and 11.0% respectively compared to our approach. This significant difference highlights the limitations of conventional stacking approaches when applied to multi-label emotion detection tasks. 

Majority voting strategies consistently underperform compared to our rule-based approach: majority voting with BERT variants (macro F1 = 0.7462, micro F1 = 0.7770) shows performance deficits of 7.8% and 6.0%, while hybrid majority voting combining LLM and BERT predictions (macro F1 = 0.7834, micro F1 = 0.8107) falls short by 2.7% and 1.6% respectively. Even majority voting using only LLM models (macro F1 = 0.7889, micro F1 = 0.8088) achieves lower performance than our proposed method by 1.9% and 1.8%. 

Among transformer-based models, DeBERTa achieves the highest individual performance (macro F1 = 0.7324, micro F1 = 0.7665), followed by RoBERTa and BART. Within the LLM category, SFT Data-Augmented demonstrates the strongest overall performance (macro F1 = 0.7875, micro F1 = 0.8096), while ICL approaches show more variable results. Our rule-based voting aggregator systematically outperforms all individual models by leveraging complementary strengths through emotion-specific model selection and weighted aggregation mechanisms. 

The balanced performance distribution across all five emotion categories, with F1 scores ranging from 0.7462 (anger) to 0.8697 (fear), indicates successful mitigation of class imbalance effects and effective handling of emotion-specific classification challenges. These comparative results demonstrate that uniform aggregation strategies fail to capture the emotion-specific strengths of individual models, whereas our adaptive rule-based mechanism effectively exploits model diversity to achieve superior performance across all evaluation metrics, establishing our rule-based voting aggregator as a highly effective approach for multi-label emotion detection. 

## 5.3 Ablation Studies 

## 5.3.1 Individual Model Performance Analysis 

This section analyzes individual model performance across transformer-based and large language model architectures, examining the relationship between training characteristics and final performance metrics. 

BERT Variants: Training Efficiency and Performance Correlation 

Training convergence analysis (Table 4) reveals interesting patterns between optimization efficiency and final performance. ALBERT achieved the lowest training loss (0.0246) but required 36 epochs, while DeBERTa converged faster at epoch 25 with 

Table 4: Training convergence for BERT variants 

|Model|Best|Epoch|Min Loss|Final LR|
|---|---|---|---|---|
||||||
|ALBERT||36|0.0246|2.44×10−9|
|DEBERTA||25|0.0308|3.91×10−8|
|BERT||26|0.0458|3.91×10−8|
|ROBERTA||31|0.0525|9.77×10−9|
|MBERT||28|0.0664|1.95×10−8|
|DISTILBERT||36|0.0794|2.44×10−9|
|XLNET||25|0.0814|3.91×10−8|
|ELECTRA||30|0.0885|9.77×10−9|
|SPANBERT||33|0.1213|4.88×10−9|
|BART||21|0.1216|7.81×10−8|
|XLM-R||24|0.1418|3.91×10−8|
|TINYBERT||22|0.3807|7.81×10−8|



competitive loss (0.0308) and achieved the highest individual performance (macro F1 = 0.7324, micro F1 = 0.7665). This suggests that rapid convergence with moderate loss values indicates superior architectural design rather than simply achieving the lowest training loss. 

RoBERTa (macro F1 = 0.7346, micro F1 = 0.7654) and BART (macro F1 = 0.7333, micro F1 = 0.7662) demonstrated balanced training-performance relationships, converging at epochs 31 and 21 respectively. Notably, compressed models showed predictable degradation: TinyBERT achieved the lowest scores among compressed variants (macro F1 = 0.5557, micro F1 = 0.6371), while DistilBERT (macro F1 = 0.6762, micro F1 = 0.7182) demonstrated better compression-performance trade-offs. The key finding indicates that models with efficient convergence patterns (2030 epochs) and moderate training losses (0.03-0.06) consistently outperformed those requiring extensive training or achieving extremely low losses, suggesting optimal capacity-complexity balance for emotion detection tasks. 

LLM Performance: Fine-tuning Strategy Impact 

SFT approaches demonstrated clear performance advantages over ICL methods, with fine-tuning strategy directly influencing results. SFT Data-Augmented achieved the highest individual performance (macro F1 = 0.7875, micro F1 = 0.8096), benefiting from expanded training data that addressed class imbalance issues, particularly excelling in fear detection (F1 = 0.8626) and joy recognition (F1 = 0.8080). 

SFT Task-Decomposed (macro F1 = 0.7761, micro F1 = 0.7987) showed the most efficient training, requiring only 15 epochs due to simplified binary classification objectives, while achieving the highest surprise detection performance (F1 = 0.7727). SFT Vanilla Multi-Label (macro F1 = 0.7871, micro F1 = 0.8071) required the longest training time (90 epochs) but demonstrated balanced performance across all emotions. 

The performance hierarchy directly correlates with data strategy effectiveness: augmented data > vanilla multi-label > task decomposition, indicating that addressing class imbalance through data augmentation provides greater benefits than architectural simplification. 

ICL Performance Analysis: Among ICL approaches, DeepSeek V3 achieved the highest performance (macro F1 = 0.7084, micro F1 = 0.7094) with balanced 

emotion recognition capabilities, followed by ChatGPT-4o (macro F1 = 0.6874, micro F1 = 0.6909), which excelled in joy detection (F1 = 0.7893) but struggled with fear recognition (F1 = 0.6667). Grok demonstrated the weakest ICL performance (macro F1 = 0.6631, micro F1 = 0.6538). 

The performance gap between SFT and ICL approaches (approximately 10-15% macro F1 difference) demonstrates that task-specific fine-tuning provides substantial advantages over prompt-based learning for complex multi-label emotion detection, with DeepSeek V3’s superior ICL performance suggesting better foundational capabilities for zero-shot emotional understanding. 

These findings establish that architectural sophistication (DeBERTa), data augmentation strategies (SFT Data-Augmented), and foundational model capabilities (DeepSeek V3) represent the key factors driving superior performance within each model category. 

## 5.3.2 Rule-based Voting Aggregator Effectiveness 

To validate the effectiveness of our proposed Rule-based Voting Aggregator, we conducted systematic ablation studies comparing different aggregation strategies and examining the impact of key components within our framework. 

Voting Strategy Comparison: As demonstrated in Table 3, our rule-based approach (macro F1 = 0.8042) substantially outperforms uniform majority voting strategies across all model combinations. Compared to majority voting with BERT variants (macro F1 = 0.7462), our method achieves 7.8% improvement, while outperforming majority voting with LLM models (macro F1 = 0.7889) by 1.9% and hybrid majority voting (macro F1 = 0.7834) by 2.7%. 

The superior performance stems from our hierarchical decision structure that systematically resolves ambiguous predictions where traditional voting methods fail, particularly benefiting challenging emotions like anger where class imbalance affects traditional approaches. 

Model Selection Strategy Impact: We compared two selection strategies: emotion-specific selection (our proposal) versus all-models participation. Emotionspecific selection achieves macro F1 = 0.8042, significantly outperforming the all-models approach (macro F1 = 0.7834) by 2.7%. 

This performance gap demonstrates the critical importance of selective participation. Different models exhibit varying strengths across emotion categories. For example, DeBERTa excels in fear detection (F1 = 0.8302) but performs moderately in anger classification (F1 = 0.6007), while SFT Data-Augmented shows superior anger detection capabilities (F1 = 0.7051). 

The all-models participation strategy suffers from including poorly performing models such as TinyBERT (macro F1 = 0.5557) and other underperforming variants, which introduce systematic noise across all emotion categories. For anger detection specifically, our emotion-specific selection improves performance from 0.6886 (all-models) to 0.7462 (emotion-specific), representing a 8.4% enhancement. 

The emotion-specific approach validates our hypothesis that optimal ensemble performance requires adaptive model selection rather than uniform aggregation, particularly in multi-label scenarios where individual models demonstrate heterogeneous strengths across different classification tasks. 

Comparison with Traditional Ensemble Methods: Table 3 shows our Rulebased Voting Aggregator achieves substantial improvements over traditional ensemble approaches. Compared to stacking ensemble (macro F1 = 0.6495), our method demonstrates 23.9% improvement, with particularly pronounced gains for challenging emotions like anger where traditional ensemble methods struggle with class imbalance 

These ablation results establish that the combination of emotion-specific model selection, hierarchical rule-based aggregation, and fine-grained weight assignment collectively contributes to the superior performance of our proposed voting mechanism, with each component providing measurable improvements over traditional ensemble approaches. 

## 6 Discussions 

## 6.1 Ensemble Integration of LLM and Transformer Models 

Our research demonstrates that combining transformer-based models and Large Language Models through rule-based voting achieves superior performance compared to individual models or traditional voting approaches within homogeneous model groups. 

Our rule-based voting aggregator (macro F1 = 0.8042) significantly outperforms all individual approaches: 9.8% improvement over the best transformer model (DeBERTa: macro F1 = 0.7324) and 2.1% enhancement over the highest-performing LLM (SFT Data-Augmented: macro F1 = 0.7875). This demonstrates that combining diverse architectures produces synergistic effects exceeding any single model type. 

Traditional voting with homogeneous models shows significant limitations. Majority voting with BERT variants achieves macro F1 = 0.7462 (7.8% lower than our approach), while LLM-only voting reaches macro F1 = 0.7889 (1.9% lower). Even hybrid majority voting combining both model types (macro F1 = 0.7834) underperforms our method by 2.7%. 

These findings establish that combining different model types through our voting system effectively uses their individual strengths while reducing their weaknesses, achieving better performance than traditional approaches. 

## 6.2 A preliminary study of BERT-based classifiers 

We have a preliminary study of BERT-based classifiers for observing other relevant experimental cases. In particular, the BERTopic model [35] is used to encode text data before learning by popular classification algorithms. Therefore, a framework for classifying emotions in text is proposed. The framework is depicted in Figure 5. BERTopic plays a role as a transformer encoding the input text data to vectors, in that each vector is the embedding of an input document and represents a tuple of probabilities 

of the document with respect to topics discovered by the BERTopic model. The document embeddings are used to train multi-classifiers using the algorithms of random forest, voting, and multi-layer perceptrons (MLP). In the voting classifier, the classification algorithms of AdaBoost, Random Forest, Gradient Boosting, and Gaussian Naive Bayes are used. 

**==> picture [372 x 109] intentionally omitted <==**

Fig. 5: BERTopic-based emotion classification framework 

As a result, the experimental performance evaluations of the three BERT-based classifiers are shown in Table 5, comparing the F1-score values obtained in each classifier. In those, the F1-scores of the ’fear’ class are always the highest in the three classifiers. It has been shown that imbalanced classes lead to lower classification performance. MLP outperforms the others with the highest Macro F1 and Micro F1 scores (0.367 and 0.5135, respectively), however these scores are still lower than the ones of the above proposed models. 

Table 5: Performance evaluation of BERTopic-based emotion classification approaches 

|Models / Methods|Macro|Micro|Anger|Fear|Joy|Sadness|Surprise|
|---|---|---|---|---|---|---|---|
|||||||||
|MLP|0.3637|0.5135|0.1082|0.7224|0.2768|0.3352|0.3757|
|Voting Classifer|0.3373|0.4635|0.1000|0.6652|0.2682|0.3179|0.3351|
|Random Forest|0.3053|0.4959|0.0950|0.7266|0.1516|0.2421|0.3113|



## 6.3 The Problems of Data Class Imbalance 

The SemEval-2025 Task 11 dataset exhibits significant class imbalance, with fear comprising 58.2% of instances while anger represents only 12.0% (Figure 4a). This imbalance causes models to overfit toward frequent emotions while underperforming on rare categories, particularly affecting multi-label scenarios where less frequent emotions may be overlooked. 

External data augmentation attempts using GoEmotions dataset resulted in performance degradation due to differences in emotion taxonomies and annotation 

standards. These findings highlight the critical need for balanced datasets and specialized handling strategies beyond traditional augmentation methods for effective emotion recognition systems. 

## 6.4 Challenges in emotion representation and data integration 

The SemEval-2025 Task 11 dataset focuses on perceived emotions, which may differ from actual emotions due to factors such as cultural context, individual differences in emotional expression, and the inherent limitations of text-based communication . This distinction introduces ambiguity, as the dataset captures the emotions as interpreted by annotators rather than the true emotional state of the speaker. 

Furthermore, our experiments revealed that augmenting the training data with external emotion datasets, such as GoEmotions, led to a decline in prediction performance. This suggests potential discrepancies in emotion taxonomies, annotation guidelines, or domain-specific language usage between datasets, which can hinder the generalizability and effectiveness of models trained on combined data sources. 

## 7 Conclusions 

In summary, this study has contributed to sentiment prediction with the integration of BERT and LLM into the Voting model. For the first research question (RQ1), BERT models were shown to nearly outperform traditional machine learning models, such as Random Forest, Voting classifier, or MLP. For the second question (RQ2), the modified Voting classifier with voting rules improved the original model. For the third question (RQ3), LLM enhanced BERT in cases where BERT could not predict sentiment with high probability. 

Not all BERT variants are good at predicting emotions. Therefore, several BERT variants were examined. The experimental results showed that some BERT variants perform well with some emotion classes. DeBERTa outperforms other BERT variants. On the other hand, LLMs can achieve better performance than BERT models. This is not surprising because LLMs have large linguistic resources. By taking the advantages of LLMs and BERT models, the proposed voting model has improved the performance of emotion prediction, even better than LLMs. This shows that LLMs fail in some emotion prediction cases, but the selected BERT models outperform it. 

As it is known, the used dataset is limited in scope, more datasets need to be observed and examined in the future. The imbalance problem of the dataset needs to be handled by using Exploratory Data Analysis methods or augmentation methods. Handling these issues can make emotion prediction better. 

## References 

- [1] OpenAI: GPT-4 Technical Report (2023) 

- [2] Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A.M., Hauth, A., et al.: Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023) 

- [3] Zhang, L., Wang, S., Liu, B.: Sentiment analysis and opinion mining. Synthesis lectures on human language technologies 5(1), 1–167 (2012) 

- [4] Breiman, L.: Random forests. Machine learning 45(1), 5–32 (2001) 

- [5] Friedman, J.H.: Greedy function approximation: a gradient boosting machine. Annals of statistics, 1189–1232 (2001) 

- [6] Calvo, R.A., D’Mello, S.: Affect Detection: An Interdisciplinary Review of Models, Methods, and Their Applications vol. 1, pp. 18–37 (2010) 

- [7] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30, pp. 5998–6008 (2017) 

- [8] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al.: Transformers: State-of-the-art natural language processing. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45 (2020) 

- [9] Sun, C., Huang, L., Qiu, X.: Emobert: Learning emotion representations using bert for emotion detection. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 4498–4508 (2020) 

- [10] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, vol. 1, pp. 4171–4186 

- [11] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V.: Roberta: A robustly optimized bert pretraining approach (2019) 

- [12] Clark, K., Luong, M.-T., Le, Q.V., Manning, C.D.: Electra: Pre-training text encoders as discriminators rather than generators. arXiv:2003.10555 [cs.CL] (2020) 

- [13] Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., Le, Q.V.: Xlnet: Generalized autoregressive pretraining for language understanding. arXiv:1906.08237 [cs.CL] (2020) 

- [14] Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R.: Albert: A lite bert for self-supervised learning of language representations. In: International Conference on Learning Representations (2020) 

- [15] Sanh, V., Debut, L., Chaumond, J., Wolf, T.: Distilbert, a distilled version of 

bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019) 

- [16] Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., Liu, Q.: Tinybert: Distilling bert for natural language understanding. In: Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4163–4174 (2020) 

- [17] Pires, T., Schlinger, E., Garrette, D.: How multilingual is multilingual bert? In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4996–5001 (2019) 

- [18] Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzm´an, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V.: Unsupervised cross-lingual representation learning at scale. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 8440–8451 (2020) 

- [19] Joshi, M., Chen, D., Liu, Y., Weld, D.S., Zettlemoyer, L., Levy, O.: Spanbert: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics 8, 64–77 (2020) 

- [20] Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., Zettlemoyer, L.: Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7871–7880 (2020) 

- [21] Demszky, D., Mahowald, K., Kohli, P., Zhao, J., Gibson, E., Sachan, M., Jurafsky, D.: Goemotions: A dataset of fine-grained emotions. arXiv preprint arXiv:2005.00547 (2020) 

- [22] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Girish, S., Askell, A., et al.: Language models are fewshot learners. Advances in Neural Information Processing Systems 33, 1877–1901 (2020) 

- [23] Howard, J., Ruder, S.: Universal language model fine-tuning for text classification. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 328–339 (2018) 

- [24] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2019) 

- [25] Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21(140), 1–67 (2020) 

- [26] Dong, Q., Li, L., Dai, D., Zheng, C., Wu, Z., Chang, B., Sun, X., Xu, J., Sui, Z.: 

A survey for in-context learning. arXiv preprint arXiv:2301.00234 (2023) 

- [27] Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al.: Emergent abilities of large language models. Transactions on Machine Learning Research (2022) 

- [28] Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., Tang, J.: Gpt understands, too. AI Open 4, 40–47 (2023) 

- [29] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021) 

- [30] Zhang, M.-L., Zhou, Z.-H.: A review on multi-label learning algorithms. IEEE Transactions on Knowledge and Data Engineering 26(8), 1819–1837 (2014) 

- [31] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al.: Scikit-learn: Machine learning in python. Journal of machine learning research 12, 2825–2830 (2011) 

- [32] Organizers, S.-.: SemEval-2025 Task 11: Multi-label Emotion Detection in Social Media Posts. Shared Task Competition. Available at: https://github.com/ emotion-analysis-project/SemEval2025-task11 (2025) 

- [33] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019) 

- [34] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017) 

- [35] Grootendorst, M.: Bertopic: Neural topic modeling with a class-based tf-idf procedure. arXiv preprint arXiv:2203.05794 (2022) 

