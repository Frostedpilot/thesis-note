# **CUSTOMISING GENERAL LARGE LANGUAGE MODELS FOR SPECIALISED EMOTION RECOGNITION TASKS** 

_Liyizhe Peng_[1] _, Zixing Zhang_[1] _*, Tao Pang_[1] _, Jing Han_[2] _*, Huan Zhao_[1] _*, Hao Chen_[1] _, Bj¨orn W. Schuller_[3] 

1 College of Computer Science and Electronic Engineering, Hunan University, China 2 Department of Computer Science and Technology, University of Cambridge, UK 3 GLAM, Department of Computing, Imperial College London, UK 

## **ABSTRACT** 

The advent of large language models (LLMs) has gained tremendous attention over the past year. Previous studies have shown the astonishing performance of LLMs not only in other tasks but also in emotion recognition in terms of accuracy, universality, explanation, robustness, few/zero-shot learning, and others. Leveraging the capability of LLMs inevitably becomes an essential solution for emotion recognition. To this end, we further comprehensively investigate how LLMs perform in linguistic emotion recognition if we concentrate on this specific task. Specifically, we exemplify a publicly available and widely used LLM – Chat General Language Model, and customise it for our target by using two different modal adaptation techniques, i. e., deep prompt tuning and low-rank adaptation. The experimental results obtained on six widely used datasets present that the adapted LLM can easily outperform other state-ofthe-art but specialised deep models. This indicates the strong transferability and feasibility of LLMs in the field of emotion recognition. 

_**Index Terms**_ **—** Emotion Recognition, Large Language Model, Prompt Tuning, Low-Rank Adaptation. 

## **1. INTRODUCTION** 

Emotion recognition, a highly interdisciplinary research field spanning psychology, cognitive, and computer science, plays an increasingly important role in research related to human-computer interaction [1]. Over the past decades, the domain of emotion recognition has undergone a profound transformation, thanks to the growing wealth of emotion datasets, enhanced computational capabilities, and continuous advancements in deep learning algorithms. 

Recently, the emergence of large language models (LLMs), exemplified by ChatGPT and Claude, has ushered in a new era in the domain of emotion recognition. LLMs are typically pretrained on vast text corpora, showcasing their robust capabilities in various domains, including text generation and natural language understanding (NLU). Prior research has illuminated the remarkable capability of LLMs in the realm of emotion recognition, attaining commendable benchmarks in accuracy, universality, explanation, robustness, and few/zero-shot learning, among others [2]. However, it is imperative to address certain challenges. While few-shot learning can enhance model performance by providing limited demonstration examples within prompts, extending prompt length results in a quadratic escalation in inference computational costs. Furthermore, overly lengthy prompts risk truncation, as they may exceed maximum input limits, leading to diminished LLM output quality. 

> _∗_ Corresponding authors. The work was funded by the National Science Foundation of China under Grant Number 62076092. 

As such, researchers and engineers are tasked with the endeavour to devise efficient methodologies for fine-tuning LLMs on domainspecific datasets. The modal adaptation technique represents a training strategy meticulously crafted to further refine a pretrained model, with the principal objective of aligning the model’s capabilities with specific tasks or domains. This methodology can facilitate the tailoring of pretrained LLMs to cater to particular downstream tasks, all while preserving their formidable language comprehension prowess. 

To this end, we aim to shed some light on how LLMs perform if they are customised to the emotion recognition domain, and to find out whether they are competitive or better than a conventional deep model specifically designed for emotion recognition. For this purpose, we select a specific open-source LLM, i. e., the Chat General Language Model, and employ two distinct model adaptation techniques: deep prompt tuning (P-Tuning v2) and low-rank adaptation (LoRA). Subsequently, we conduct a comprehensive comparative analysis, evaluating the performance of LLMs both pre- and post-adaptation on six emotional datasets. Furthermore, to provide a holistic perspective on the effectiveness and advancements achieved through these model adaptation approaches, we conducted comparative assessments with other state-of-the-art (SOTA) non-LLM-based studies. These comparative evaluations enable us to gauge the relative merits and contributions of different model adaptation strategies in the context of emotion recognition. It is hoped that this work will bring more discussions in the field of emotion recognition, as the new era of general large models is coming. 

## **2. RELATED WORK** 

A substantial body of research has been dedicated to the domain of model adaptation for pre-trained LLMs. Within the spectrum of contemporary approaches, a widely-used and the most basic technique is known as full fine-tuning (FFT), necessitating the retraining of all model parameters. While FFT has proven effective in enhancing LLM performance, it demands substantial computational resources during training, incurring significant costs and rendering it increasingly impractical. In response to the need for reducing the computational burden, the research community has introduced numerous parameter-efficient fine-tuning (PEFT) methods [3]. These innovative methodologies entail the selective training of a limited subset of model parameters, either by modifying existing parameters or introducing novel ones into the model architecture. 

The strategies for training partial model parameters involve adapting the characteristics of layer types or internal architecture within a network [3]. Methods involving the introduction of additional parameters can be broadly categorised into two groups: Adapter-like methods and Soft Prompts methods. Adapters in- 

**==> picture [245 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
h<br>Back Propagation<br>Prompt Encoder<br>x �<br>EmbeddingInput  � PretrainedWeights<br>�<br>Transformers<br>. . . . . . x<br>�<br>(a) P-Tuning v2 (b) Low-Rank Adaptation<br>**----- End of picture text -----**<br>


**Fig. 1** : Schematic representation of two model adaptation methods: P-Tuning v1 (a) and Low-Rank Adaption (b). Red blocks refer to the additional trainable parameters, while the yellow blocks represent the frozen parameters of the pre-trained model. 

troduce small fully-connected networks after Transformer sublayers [4]. Soft prompts can be trained for the input layer exclusively or for all layers within the model. Furthermore, reparametrisationbased PEFT methods leverage low-rank representations to minimise the number of trainable parameters. 

In this paper, we choose two widely recognised model adaptation methods, namely P-Tuning v2 and LoRA. These two methods involve a limited number of trainable parameters, which ensures their computational resource demands remain within reasonable limits. Consequently, the adaptation training process can be feasibly conducted on consumer-grade graphics processing units (GPUs), rendering both P-Tuning v2 and LoRA highly practical choices. 

## **3. ADAPTATION OF LARGE LANGUAGE MODELS** 

In this section, we present an introduction to the selected LLM – Chat General Language Model, alongside an explanation of the core principles underlying the two model adaptation techniques, namely P-Tuning and LoRA. 

## **3.1. General Language Model** 

General Language Model (GLM) is a general pre-training framework based on a novel autoregressive blank infilling objective and can be adapted to various NLU and natural language generation tasks [5]. GLM formulates NLU tasks as ‘cloze’ questions that contain task descriptions, which can be answered by autoregressive generation. Remarkably, when operating with equivalent parameter counts and computational resources, GLM consistently outperforms BERT on the SuperGLUE benchmark, and excels beyond RoBERTa and BART when pretrained on corpora of comparable sizes [5]. Furthermore, GLM demonstrates superior performance to T5 in NLU and generation tasks, while utilising fewer parameters and data [5]. 

Chat General Language Model (ChatGLM), launched in March 2023 by the Tsinghua University KEG Laboratory and the Zhipu AI Company, is a GLM-based AI Chatbot. ChatGLM draws inspiration from ChatGPT to integrate code pre-training into the trillionparameter base model GLM-130B [6], achieving human intent alignment through techniques like supervised fine-tuning. It is worth mentioning that ChatGLM was pre-trained on both Chinese and English corpora, thus, it possesses bilingual capabilities. ChatGLM2, a second-generation model, is open-sourced in June 2023. It boasts enhanced performance, extended context capabilities, improved inference efficiency, and a more permissive open-source license. In our study, we opted to utilise the more lightweight model ChatGLM26B, which employs the same technology as ChatGLM2 but with a reduced parameter count of 6.2 billion. It requires a minimum of 

13GB of GPU memory for inference when using FP16 precision, enabling the deployment of ChatGLM-6B on consumer-grade graphics cards. 

## **3.2. P-Tuning** 

P-Tuning ( _aka_ Prompt Tuning) is a cost-effective model adaptation methodology, which freezes all model parameters and introduces additional parameters [7]. It can be viewed as an optimised and tailored implementation of deep prompt tuning, specifically designed for generation and knowledge probing tasks. Deep prompt tuning expands the capacity of continuous prompts, bridging the adaptation gap across various settings, with particular efficacy for smaller models and challenging tasks. Nevertheless, the P-Tuning method is subject to certain constraints due to its exclusive utilisation of continuous prompts within the input embedding sequence. This results in a limited number of trainable parameters, and the input embeddings have a relatively indirect impact on model predictions. 

To overcome these challenges, the P-Tuning v2 technique incorporates continuous prompts into every layer of the model (shown in Fig. 1 (a)), rather than solely in the input embedding sequence [7]. Such adjustment in P-tuning v2 introduces a greater number of tunable task-specific parameters (from 0.01% to 0.1%-3%) to enhance task-specific capacity while maintaining parameter efficiency. Additionally, prompts added to deeper layers have a more direct impact on model predictions. Currently, P-Tuning v2 consistently achieves comparable performance to fine-tuning across a broad spectrum of model scales, ranging from 300M to 10B parameters [7]. It was frequently demonstrated to perform particularly well on challenging sequence tagging tasks, including extractive question answering and named entity recognition. 

## **3.3. Low-Rank Adaptation** 

Low-Rank Adaptation (LoRA) is one of the reparametrisation-based model adaptation methods [8] and is illustrated in Fig. 1 (b). It freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer of the Transformer architecture, considerably reducing the number of trainable parameters for downstream tasks. Its inspiration comes from a statement that pretrained language models have a lower “intrinsic dimension” and can still effectively learn despite being randomly projected into smaller subspaces [8]. Therefore, researchers hypothesised the updates to the weights also have a low “intrinsic rank” during adaptation and proposed LoRA methods. Compared to GPT-3’s 175B parameters fine-tuned with Adam, LoRA can achieve a 10,000-fold reduction in the number of trainable parameters and a 3-fold decrease in GPU memory requirements [8]. In terms of effectiveness, LoRA matches or surpasses full fine-tuning for RoBERTa, DeBERTa, GPT-2, and GPT-3, even though it has fewer trainable parameters and a higher training throughput [8]. More importantly, it does not introduce any additional inference latency. 

## **4. EXPERIMENTS AND RESULTS** 

## **4.1. Selected Datasets** 

In this part, we present the six datasets utilised in our research. More detailed information is placed in Table 1, ranging from English to Chinese languages, from binary/ternary sentiment anlaysis to multiclass emotion classification. Since these datasets are publicly accessible, we can use them to verify the effectiveness of different model adaptation methods. 

**SST** : The Stanford Sentiment Treebank is an English corpus containing fine-grained sentiment annotations for 11,855 individual 

**Table 1** : Detailed information of the selected six datasets. #sp., #dia., #total (test) utt., #words/utt., #classes denotes the number of distinct speakers, dialogues, utterances of the whole dataset and its test subset, words per utterance, and emotional classes. 

|**dataset**|**language**|**modality**|**dialogue**|**data**|**#sp.**|**#dia.**|**#utt. total**|**#words/utt.**<br>**#classes**|**#words/utt.**<br>**#classes**|
|---|---|---|---|---|---|---|---|---|---|
|||||**source**|||**(test)**|||
|SST|English|t|no|movie|-|-|11 855 (2 210)|-|5 (negative, somewhat|
|||||review|||||negative, neutral, positive,|
||||||||||somewhat positive)|
|Friends|English|t|yes|Friends TV|-|1 000|14 503 (2 764)|10.7|7 (neutral, joy, sadness,|
|||||shows|||||fear, anger, surprise,|
||||||||||disgust)|
|Mastodon|English|t|yes|Mastodon|-|505|2 217 (1 142)|-|3 (positive, neutral,|
||||||||||negative)|
|MOSI|English|a, v, t|no|YouTube|89|-|2 199 (686)|12.0|7_{_-3, -2, -1, 0, 1, 2, 3_}_|
|CH-SIMS|Mandarin|a, v, t|no|movies,|474|-|2 281 (457)|15.0|5_{_-1.0, -0.8_}{_-0.6, -0.4,|
|||||TVs, &|||||-0.2_} {_0.0_} {_0.2, 0.4,|
|||||shows|||||0.6_}{_0.8, 1.0_}_|
|M3ED|Mandarin|a, v, t|yes|TV series|626|990|24 449 (4 201)|7.4|7 (happy, surp., sad,|
||||||||||disgust, anger, fear, neut.)|



sentences sourced from movie review data [9]. For the fine-grained task, each sentence is categorised into one of five sentiment classes. For the binary task, each sentence is simply classified as either positive or negative, with the neutral category excluded. 

**Friends** : Friends is an English corpus based on the TV show Friends, containing 1,000 dialogues from seasons one to nine [10]. The 14,503 utterances from the 1,000 dialogues are categorised into seven classes. The annotators considered the context of the dialogue when labelling sentiments. 

**Mastodon** : The Mastodon dataset [11] consists of English posts from the Mastodon social media platform. While the dataset was initially designed for both sentiment recognition and dialogue act recognition, we only focus on the former. 

**MOSI** : The Multimodal Opinion-level Sentiment Intensity (MOSI) dataset [12] is a multimodal sentiment analysis dataset, including 2,199 opinion segments extracted from 93 videos. Each opinion segment received annotations on a sentiment spectrum ranging from highly negative to highly positive within the interval [-3, 3]. 

**CH-SIMS** : CH-SIMS is a Chinese single- and multi-modal sentiment analysis dataset [13]. It collected 2,281 video segments from movies, TV series, and a variety of shows. The sentiment annotation is divided into five categories. 

**M**[3] **ED** : Multi-modal Multi-scene Multi-label Emotional Dialogue (M[3] ED) is the first multimodal emotional dialogue dataset in Chinese [14]. The dataset contains 990 dyadic emotional dialogues from 56 different TV series, including 9,082 turns and 24,449 utterances. 

## **4.2. Implementation Details** 

We conducted emotion recognition tasks on six selected datasets to evaluate the effectiveness of two model adaptation methods on LLMs, i. e., P-Tuning v2 and LoRA. For the SST dataset, we conducted both binary and five-class classification tasks. Three-class sentiment classification tasks, distinguishing among positive, neutral, and negative sentiments, were performed on the MOSI and the Mastodon dataset. The CH-SIMS dataset and the MOSI were used for a binary classification task: positive and negative. Finally, we implemented a seven-class emotion classification task on the Friends and M[3] ED datasets. Moreover, for the purpose of performance comparison with specialised emotion recognition models, we have chosen recently published SOTA works that exhibit competitive perfor- 

mance on each selected datasets, separately, under strictly comparable conditions. 

For each individual dataset, we designed three sets of comparative experiments: ChatGLM2 without adaptation, ChatGLM2 adapted with P-Tuning v2, and ChatGLM2 adapted with LoRA. During the inference with ChatGLM2, we utilise a “prompt” to acquire a response from it. The prompt should encompass both task-guiding sentences that necessitate emotion recognition. Our prompt is structured as follows: Classify the sentiment of the sentence to Emotion 1, Emotion 2, ... or Emotion k: _<_ provide only one sentence from a test set _>_ . The value of _k_ here is determined by the number of sentiment/emotion categories specific to the dataset. For example, the prompt is “Classify the sentiment of the sentence to Positive, Negative or Neutral” for MOSI as _k_ = 3. When adapting the model, we add a task-guiding sentence before each training sample to construct a complete prompt, and then present it into the model for learning. Note that, although the Mastodon, Friends, and M[3] ED datasets are context-based, we treat them like other datasets, regardless of the context. 

Our experiments were conducted on an NVIDIA GeForce RTX 3090 with 24GB of RAM, and the adaptation training and inference tasks were performed only on one single GPU. For the adaptation training, we set the training batch size to 16 due to the constraints of GPU memory. Additionally, we set the prompt length to 32 for P- Tuning v2, and configured the rank of 8 for LoRA. We employed the accuracy and macro F1 score as the primary metrics for performance evaluation. For the M[3] ED dataset, we employed the weighted average F1 score to provide equitable comparisons with other research. 

## **4.3. Results and Discussion** 

To evaluate the effectiveness and the transferability of generalised LLMs in emotion recognition, we conducted extensive experiments on six publicly available datasets (see Section 4.1). Table 2 to Table 5 present the results obtained from the ChatGLM2 with or without adaptation technologies on these six datasets, respectively. Besides, for each selected dataset, we offer SOTA performance from specialised models in the latest studies for comparison. 

First of all, we can see that ChatGLM2 performs competitively with these specialised models in many datasets, such as MOSI, CHSIMS, and Mastodon. This finding is consistent with the one shown in our previous work but evaluated with other LLMs [2]. For the datasets of Friends and M[3] ED, there is an obvious performance gap, 

**Table 2** : Performance comparison between adapted models and SOTA works on the **MOSI** datasets measured by accuracy (Acc) and macro-F1 (F1). 

||MOSI-2|MOSI-2|MOSI-3|
|---|---|---|---|
|Model [%]|Acc<br>F1||Acc<br>F1|
|||||
|TFR-Net (2021) [15]<br>CHFN (2022) [16]<br>SeqSeq2Sent (2018) [17]<br>CTFN (2021) [18]|83.49<br>-<br>85.20<br>-<br>-<br>-<br>-<br>-||-<br>-<br>-<br>-<br>77.00<br>-<br>80.79<br>-|
|||||
|ChatGLM2<br>ChatGLM2 (P-Tuning)<br>ChatGLM2 (LoRA)|84.12<br>84.12<br>84.60<br>84.04<br>**87.02**<br>**86.56**||77.26<br>58.19<br>81.78<br>**61.03**<br>**83.82**<br>57.04|



**Table 3** : Performance comparison between adapted models and SOTA works on the **SST** datasets measured by accuracy (Acc) and macro-F1 (F1). 

||SST-2<br>SST-5|SST-2<br>SST-5|
|---|---|---|
|Model [%]|Acc<br>F1|Acc<br>F1|
||||
|BT-TAPT (2021) [19]<br>SEMGraph-P (2022) [20]<br>SentiLARE (2020) [21]<br>SentiWSP (2022) [22]|92.40<br>-<br>94.23<br>-<br>-<br>-<br>-<br>-|-<br>-<br>-<br>-<br>58.59<br>-<br>**59.32**<br>-|
||||
|ChatGLM2<br>ChatGLM2 (P-Tuning)<br>ChatGLM2 (LoRA)|82.33<br>82.33<br>95.20<br>95.20<br>**95.69**<br>**95.69**|30.09<br>25.82<br>57.59<br>**56.45**<br>54.45<br>52.51|



which might be attributed to the lack of context information provided for ChatGLM2 for inference. 

Then, when comparing the performance of ChatGLM2 with or without adaptation, we can generally observe that the adapted large models, either by P-Tuning v2 or by LoRA, considerably outperform the non-adapted ones, both in binary and multi-class classification tasks. For instance, on the SST-5 dataset (cf. Table 3), the P- Tuning v2 method performs the most substantial improvement. The accuracy and macro F1 scores increase from 30.09 % and 25.82 % to 57.59 % and 56.45 %, nearly doubling the performance before adaptation. This suggests that both P-Tuning v2 and LoRA algorithms work efficiently for the adaptation of LLMs in emotion recognition. 

Moreover, it can be seen that the adapted ChatGLM2s outperform other SOTA-specialised model in most cases, but vary depending on the complexity of the classification tasks. In simpler tasks like binary or three-class classification, adapted models often outperform SOTA-specialised models. Conversely, for tasks involving five or more categories (e. g., Friends and M3ED), models with adaptation remain a substantial performance gap compared to SOTA works. This is largely due to the missing of context information for training and inference as aforementioned. Surprisingly, for the context-rich Mastodon dataset, even without considering context during adaptation, the adapted model exhibits superior performance compared to the SOTA works. This could be attributed to the relative simplicity of the three-class classification task or the dataset may have a less pronounced dependency on contextual information. Generally speaking, these observations indicate that the pretrained and generalised ChatGLM2 can efficiently transfer their knowledge to a specific domain without much training data and computation resources. 

Finally, in comparison with the two selected model adaptation 

**Table 4** : Performance comparison between adapted models and SOTA works on the **CH-SIMS** and **Mastodon** datasets measured by accuracy (Acc) and macro-F1 (F1). 

||CH-SIMS|CH-SIMS|Mastodon|
|---|---|---|---|
|Model [%]|Acc<br>F1||Acc<br>F1|
|||||
|MLF-DNN (2020) [13]<br>DARER (2022) [23]|80.26<br>-<br>-<br>-||-<br>-<br>-<br>59.59|
|||||
|ChatGLM2<br>ChatGLM2 (P-Tuning)<br>ChatGLM2 (LoRA)|77.58<br>75.95<br>82.47<br>81.12<br>**82.73**<br>**81.25**||55.43<br>55.45<br>**67.25**<br>**67.23**<br>67.08<br>66.81|



**Table 5** : Performance comparison on **Friends** (first half) and **M**[3] **ED** (second half) in terms of accuracy (Acc), F1, and unweighted accuracy (UA). Note that, F1 indicates macro-F1 and weighted average F1 for Friends and M[3] ED, respectively, for a fair performance comparison. 

|Model [%]|Friends|
|---|---|
||Acc<br>F1<br>UA|
|BERT+SRL-GNN-8 (2020) [24]<br>XLNet+SRL-GNN-8 (2020) [24]<br>PRE-CODE (2020) [25]|72.10<br>-<br>53.71<br>72.82<br>-<br>53.41<br>**81.30**<br>**65.90**<br>-|
|ChatGLM2<br>ChatGPT (P-Tuning)<br>ChatGPT (LoRA)|63.79<br>29.48<br>26.03<br>54.92<br>51.92<br>**55.06**<br>72.83<br>52.97<br>51.93|
|Model [%]|M3ED|
||Acc<br>F1<br>UA|
|DialogueGCN (2019) [26]<br>DialogueRNN (2019) [27]<br>MDI (2022) [14]|-<br>46.09<br>-<br>-<br>48.80<br>-<br>-<br>**49.42**<br>-|
|ChatGLM2<br>ChatGLM2 (P-Tuning)<br>ChatGLM2 (LoRA)|45.68<br>30.52<br>16.82<br>**45.75**<br>37.31<br>**28.64**<br>42.54<br>33.31<br>23.59|



methods, i. e., P-Tuning v2 and LoRA, the latter outperforms in binary tasks, while the former demonstrates superior performance in ternary and multi-class tasks. Consequently, there is no consistent observation to definitively favour one method over the other, as the optimal adaptation approach varies across different datasets. 

## **5. CONCLUSION** 

In this paper, we focused on the capability of different model adaptation methods for Large Language Models (LLMs) in the field of emotion recognition. We investigate this by assessing the performance of the Chat General Language Model on six datasets using two adaptation techniques, i. e., deep prompt tuning and lowrank adaptation. The experimental result shows that both adaptation methods perform exceptionally well in emotion recognition tasks, particularly for simple classification tasks that are without context. Compared to traditional specialised models, utilising the adapted LLMs for emotion recognition considerably reduces the modelling efforts for researchers, and the computational resources required for adaptation are also accessible. This opens up brand-new possibilities for future emotion recognition systems. 

## **6. REFERENCES** 

- [1] J. Han, Z. Zhang, and B. W. Schuller, “Adversarial training in affective computing and sentiment analysis: Recent advances and perspectives,” _IEEE Computational Intelligence Magazine_ , vol. 14, no. 2, pp. 68–81, Sep. 2019. 

- [2] Z. Zhang, L. Peng, T. Pang, J. Han, H. Zhao, and B. W. Schuller, “Refashioning emotion recognition modelling: The advent of generalised large models,” _arXiv preprint arXiv: 2308.11578_ , Aug. 2023. 

- [3] V. Lialin, V. Deshpande, and A. Rumshisky, “Scaling down to scale up: A guide to parameter-efficient fine-tuning,” _arXiv preprint arXiv: 2303.15647_ , Apr. 2023. 

- [4] J. Pfeiffer, A. R¨uckl´e, C. Poth, and et al., “Adapterhub: A framework for adapting transformers,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2020, pp. 46–54. 

- [5] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, “GLM: general language model pretraining with autoregressive blank infilling,” in _Proc. the 60th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022, pp. 320–335. 

- [6] A. Zeng, X. Liu, Z. Du, and etc., “GLM-130B: an open bilingual pre-trained model,” in _Proc. International Conference on Learning Representations (ICLR)_ , 2023. 

- [7] X. Liu, K. Ji, Y. Fu, Z. Du, Z. Yang, and J. Tang, “P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks,” _arXiv preprint arXiv: 2110.07602_ , Sep. 2021. 

- [8] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in _Proc. The Tenth International Conference on Learning Representations (ICLR)_ , 2022. 

- [9] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts, “Recursive deep models for semantic compositionality over a sentiment treebank,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2013, pp. 1631–1642. 

- [10] C.-C. Hsu, S.-Y. Chen, C.-C. Kuo, T.-H. Huang, and L.-W. Ku, “EmotionLines: An emotion corpus of multi-party conversations,” in _Proc. the 11th International Conference on Language Resources and Evaluation (LREC)_ , 2018, pp. 1597–1601. 

- [11] C. Cerisara, S. Jafaritazehjani, A. Oluokun, and H. T. Le, “Multi-task dialog act and sentiment recognition on mastodon,” in _Proc. the 27th International Conference on Computational Linguistics (COLING)_ , 2018, pp. 745–754. 

- [12] A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency, “Multimodal sentiment intensity analysis in videos: Facial gestures and verbal messages,” _IEEE Intelligent Systems_ , vol. 31, no. 6, pp. 82–88, Nov. 2016. 

- [13] W. Yu, H. Xu, F. Meng, Y. Zhu, Y. Ma, J. Wu, J. Zou, and K. Yang, “CH-SIMS: A Chinese multimodal sentiment analysis dataset with fine-grained annotation of modality,” in _Proc. the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020, pp. 3718–3727. 

- [14] J. Zhao, T. Zhang, J. Hu, Y. Liu, Q. Jin, X. Wang, and H. Li, “M3ED: multi-modal multi-scene multi-label emotional dialogue database,” in _Proc. the 60th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022, pp. 5699– 5710. 

- [15] Z. Yuan, W. Li, H. Xu, and W. Yu, “Transformer-based feature reconstruction network for robust multimodal sentiment analysis,” in _Proc. the 29th ACM International Conference on Multimedia (MM)_ , 2021, pp. 4400–4407. 

- [16] J. Guo, J. Tang, W. Dai, Y. Ding, and W. Kong, “Dynamically adjust word representations using unaligned multimodal information,” in _Proc. the 30th ACM International Conference on Multimedia (MM)_ , 2022, pp. 3394–3402. 

- [17] H. Pham, T. Manzini, P. P. Liang, and B. Pocz´os, “Seq2Seq2Sentiment: Multimodal sequence to sequence models for sentiment analysis,” in _Proc. Grand Challenge and Workshop on Human Multimodal Language (Challenge-HML)_ , 2018, pp. 53–63. 

- [18] J. Tang, K. Li, X. Jin, A. Cichocki, Q. Zhao, and W. Kong, “CTFN: hierarchical learning for multimodal sentiment analysis using coupled-translation fusion network,” in _Proc. the 59th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2021, pp. 5301–5311. 

- [19] J. Lee, J. Kim, and P. Kang, “Back-translated task adaptive pretraining: Improving accuracy and robustness on text classification,” _arXiv preprint arXiv:2107.10474_ , Aug. 2021. 

- [20] B. Wang, B. Liang, J. Du, M. Yang, and R. Xu, “SEMGraph: Incorporating sentiment knowledge and eye movement into graph model for sentiment analysis,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2022, pp. 7521–7531. 

- [21] P. Ke, H. Ji, S. Liu, X. Zhu, and M. Huang, “SentiLARE: Sentiment-aware language representation learning with linguistic knowledge,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2020, pp. 6975– 6988. 

- [22] S. Fan, C. Lin, H. Li, Z. Lin, J. Su, H. Zhang, Y. Gong, J. Guo, and N. Duan, “Sentiment-aware word and sentence level pretraining for sentiment analysis,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2022, pp. 4984–4994. 

- [23] B. Xing and I. W. Tsang, “DARER: dual-task temporal relational recurrent reasoning network for joint dialog sentiment classification and act recognition,” in _Proc. the 60th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2022, pp. 3611–3621. 

- [24] C. T. Heaton and D. M. Schwartz, “Language models as emotional classifiers for textual conversation,” in _Proc. the 28th ACM International Conference on Multimedia (MM)_ , 2020, pp. 2918–2926. 

- [25] W. Jiao, M. R. Lyu, and I. King, “Exploiting unsupervised data for emotion recognition in conversations,” in _Proc. the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ , 2020, pp. 4839–4846. 

- [26] D. Ghosal, N. Majumder, S. Poria, N. Chhaya, and A. F. Gelbukh, “DialogueGCN: A Graph Convolutional Neural Network for emotion recognition in conversation,” in _Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , 2019, pp. 154–164. 

- [27] N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, A. F. Gelbukh, and E. Cambria, “DialogueRNN: An attentive RNN for emotion detection in conversations,” in _Proc. AAAI Conference on Artificial Intelligence_ , 2019, pp. 6818–6825. 

