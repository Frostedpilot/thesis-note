# **DialogueLLM: Context and Emotion Knowledge-Tuned Large Language Models for Emotion Recognition in Conversations** 

**Yazhou Zhang**[1] _[,]_[2] **, Mengyao Wang**[2] **, Youxi Wu**[3] _[∗]_ **, Prayag Tiwari**[4] **, Qiuchi Li**[5] **, Benyou Wang**[6] **, Jing Qin**[1] _[∗]_ 1The Hong Kong Polytechnic University 

2Zhengzhou University of Light Industry 3Hebei University of Technology 

4Halmstad University 

5Copenhagen University 

6The Chinese University of Hong Kong, Shenzhen 

## **Abstract** 

Large language models (LLMs) and their variants have shown extraordinary efficacy across numerous downstream natural language processing (NLP) tasks, which has presented a new vision for the development of NLP. Despite their remarkable performance in natural language generating, LLMs lack a distinct focus on the emotion understanding domain. As a result, using LLMs for emotion recognition may lead to suboptimal and inadequate precision. Another limitation of the current emotion LLMs is that they are typical trained without leveraging multi-modal information. To overcome these limitations, we propose DialogueLLM, a context and emotion knowledge tuned LLM that is obtained by fine-tuning large language models with benchmarking multimodal (i.e., texts and videos) emotional dialogues. The visual information is considered as the supplementary knowledge to construct high-quality instructions. We offer a comprehensive evaluation of our proposed model on three benchmarking emotion recognition in conversations (ERC) datasets and compare the results against the state-of-the-art baselines and other state-of-the-art LLMs. Additionally, DialogueLLM-7B can be easily trained using LoRA on a 40GB A100 GPU in 5 hours, facilitating reproducibility for other researchers. 

## **1 Introduction** 

Scaling up language models has been proved to be an effective way to improve the performance and sample efficiency in downstream NLP tasks. The rise of instruction-following LLMs has garnered considerable attention from academy and industry, due to their outstanding performance in human instruction understanding and responsing. Language modeling has evolved from small language models (SLMs), e.g., GPT [30], BERT [4], 

**==> picture [224 x 226] intentionally omitted <==**

Figure 1: Sample utterances in a multi-modal conversation from the MELD dataset. 

RoBERTa [20], etc., to LLMs, e.g., ChatGPT[1] GPT4 [23], Claude[2] , etc. 

Compared with SLMs, LLMs are characterized by their enormous parameter size, typically reaching tens of billions or even more. They often have stronger generalization across various downstream tasks and unique emergent ability to tackle complex tasks. Despite that LLMs possess numerous commendable qualities, they also present a couple of limitations that deserve careful consideration and in-depth exploration: (1) the nonopen source status may restrict the development of LLMs community and (2) they are not specifically designed for emotion understanding task. Their broad domain knowledge frequently proves insufficient when tackling such specialized domains. For example, Zhang et al. [41] showed LLMs’ unsatisfactory performance in many emotion recognition 

> 1https://chat.openai.com/ 

> _∗_ Corresponding authors 

> 2https://www.anthropic.com/product 

tasks without fine-tunning on emotional knowledge. Let et al. [12] presented a retreival based framework to improve the adaptability of LLMs to emotion recognition. Hence, the potential of LLMs in understanding emotional communication needs to be explored further. 

Human communication is the process of exchanging information, thoughts, ideas, and feelings between individuals, which is naturally filled with the participant’s subjective attitudes or emotions. Emotion recognition in conversations (ERC) aims to accurately detect the feelings and emotions expressed in the utterances. It has immense potential in dialogue understanding and intent analysis, and has been an active task in the recent literature [19, 42, 45]. In general, there are two key factors that contribute the classification performance, i.e., multi-modal fusion and context dependency (also known as intra- and inter-speaker dependency) [21]. Multi-modal fusion involves combining information from different sources or modalities, such as text, visual cues, to obtain a more comprehensive and accurate understanding of the emotional utterance. In view that emotions are influenced by the surrounding environment, the relationship between the participants, etc., context is a critical factor in accurately classifying emotions in conversations. The same utterance in different contexts might express different emotions. Fig. 1 illustrates an example to introduce the presence of both challenges. 

To overcome the above-mentioned limitations, it’s crucial to develop emotion-tailored LLMs that can better understand human-human conversation and take a further step towards emotion intelligence. In this paper, we present DialogueLLM, an emotion and context knowledge enhanced language model, which is specifically designed for ERC based on the open-source base models, namely LLaMA 2 [35]. By collecting diverse instruction conversational data based on emotional knowledge from five open-source benchmarking datasets (i.e., MELD [27], IEMOCAP [1], EmoryNLP [39].), we obtain 2411 multi-party dialogues, over 24,304 utterances. Meanwhile, the visual information (i.e., videos) will be forward into ERNIE Bot[3] to automatically generate the text descriptions, which will be considered as the supplementary knowledge to construct high-quality instructions. We adopt an end-to-end supervised instruction-finetuning ap- 

proach on the open-source LLaMA 2-7B base models. Additionally, DialogueLLM-7B can be easily trained using LoRA on a 40GB A100 GPU in 5 hours, facilitating reproducibility for other researchers. 

We offer a comprehensive evaluation of our proposed DialogueLLM model across three ERC tasks and compare the results against 15 state-ofthe-art ERC baselines, including bc-LSTM [46], MTL [15], ICON [7], DialogXL [32], TODKAT [49], CoG-BART [16], DialogueGCN [5], RGAT [11], DAG-ERC [33], DialogueRNN [22], DialogueCRN [10], CauAIN [48], COIN [40], GraphCFC [14], SACL-LSTM [9] and three SOTA LLMs, i.e., LLaMA, Alpaca[4] and LLaMA 2. The experimental results show the effectiveness of DialogueLLM with the margin of 5.36%, 1.03% and 1.5% for three benchmarking ERC tasks. The study reveals that DialogueLLM significantly outperforms the SOTA baselines on ERC tasks requiring deeper understanding or conversational emotion information. A series of sub-experiments underscore how emotion and context knowledge enhanced LLMs deal with ERC tasks. The main innovations of the work are concluded as follows: 

- To the best of our knowledge, DialogueLLM is the first open source emotional LLM that is specifically designed for ERC tasks. 

- The visual information is proposed to construct high-quality instructions. 

- We show a comprehensive dataset of over 24K utterances to serve as a knowledge corpus, supporting the training and testing of emotional LLMs with accurate and domainspecific knowledge. 

- Our model achieves state-of-the-art performance on ERC tasks. We show that an open-sourced model finetuned with emotional knowledge has the potential to achieve even higher accuracy than SOTA. 

The rest of this paper is organized as follows. Section 2 briefly outlines the related work. In Section 3, we describe the proposed DialogueLLM in detail. In Section 4, we report the empirical experiments and analyze the results. Section 5 concludes the paper and points out future research directions. 

## **2 Related Work** 

We depict two lines of research that form the basis of this work: large language models and emotion recognition in conversations models. 

## **2.1 Large Language Models** 

In recent years, significant advancements in natural language processing (NLP) have been attributed to the emergence of large language models. These models have showcased remarkable capabilities such as in-context learning, few-shot prompting, instruction following, etc. These dynamic abilities have greatly contributed to boosting the effectiveness of language models, thus enabling AI algorithms to achieve unparalleled levels of effectiveness and productivity. Typically, models like the transformer architecture-based LLMs are first pretrained using extensive datasets comprising diverse languages and domains [47]. 

OpenAI has achieved significant milestones with the creation of two groundbreaking models: ChatGPT and GPT-4. These models herald a new era in language processing. However, due to their proprietary nature, there has been a proliferation of LLM variants featuring tens or even hundreds of billions of parameters. Our aim is to categorize these LLMs into two distinct groups based on their specialization: general LLMs and specialized LLMs. General LLMs are designed for versatility across a wide spectrum of NLP tasks, including machine translation, language comprehension, and dialogue generation. Prominent examples of these models are GPT-4, Claude, ChatGPT, LLaMA, PanGu-Σ [31], Bard[5] , Falcon [24], etc. Such LLMs are not specifically optimized for any particular task. While they can perform well across a range of tasks, but their potentials in specific scenarios await further explore. 

In contrast, specialized LLMs also known as task-specific LLMs, are fine-tuned for specific tasks via task-specific architectures and knowledge, allowing them to achieve higher or comparable performance against general LLMs with fewer parameters. For example, Wang et al. [2] released a large language model ‘Phoenix’ to meet the needs of multiple languages. Liu and Low [18] finetuned a Goat model based on LLaMA model to deal with arithmetic tasks. In view that LLMs have not yet performed optimally in medical domain tasks, a few Chinese and English medical 

knowledge enhanced LLMs have been proposed, such as HuaTuo [36], PMC-LLaMA [38], Dr. LLaMA [6], ChatDoctor [17]. Different from the above-mentioned works, we aim to explore the potential of LLMs in emotion understanding domain and take a further step towards emotional intelligence. 

## **2.2 Emotion Recognition in Conversations** 

Emotion recognition in conversation (ERC) has become a popular research topic. In this task, the conversational context dependency and multi-modal fusion have been considered through deep learning approaches. These efforts can be broadly categorized into methods based on sequences and those based on the Transformer architecture. 

Sequence based approaches often use the sequential information in a dialogue to capture the contextual and emotional features. For example, Poria et al. [26] introduced an LSTM-based model that effectively captured conversational context from surrounding videos, thereby enhancing the classification process. Building upon this idea, Hazarika et al. [8] presented the conversational memory network (CMN), which harnessed contextual information from the conversation history to improve ERC. Another approach by Majumder et al. [22] introduced the DialogueRNN model, which meticulously monitored the states of individual participants throughout the conversation, utilizing this information for ERC. In terms of multimodal advancements, Poria et al. [27] played a pivotal role by crafting the first-ever multimodal conversational dataset named the multimodal emotionlines dataset (MELD). This dataset was instrumental in propelling the field of conversational sentiment analysis. Further innovation came from Zhang et al. [43], who devised the quantum-inspired interactive network (QIN) model for conversational emotion recognition, showcasing its effectiveness. Moreover, their research extended to the realm of multitask learning. Zhang et al. [44] devised a quantuminspired multi-task learning framework catering to both sarcasm detection and emotion recognition in conversations. 

Transformer based approaches often adopt the “fine-tuning” paradigm. They build the models upon the foundation of Transformer based pretrained language models. Then, such models are supervised-fine-tuned with labeled samples and are adapted to the specific task. For instance, Li et 

**==> picture [447 x 203] intentionally omitted <==**

Figure 2: Overview of DialogueLLM fine-tuning and classification pipeline. 

al. [16] used a supervised contrastive term and a response generation task to enhance BART’s ability for ERC. Zhang et al. [42] proposed a multi-modal multi-task network based on BERT and graph attention network (GAT) to detect sentiment and emotion. They also proposed a quantum inspired multitask interactive Transformer to model sentiment and emotion [19]. Chudasama et al. [3] presented a multi-modal fusion network (M2FNet) to learn emotion-relevant multi-modal features by revising the Transformer encoder. Qiao et al. [29] built a mutual-enhanced incongruity learning network upon RoBERTa and graph convolution networks to identify sarcasm. Pramanick [28] combined selfattention with BERT to model intra-modal correspondence and optimal transport for cross-modal correspondence, aiming to discover sarcasm and humor. Lei et al. [12] replaced the ERC task from a traditional discriminative model to a generative model, and proposed a simple but effective retrieval template modulem, named InstructERC to help the model explicitly integrate multi-granular dialog supervision information. 

Compared with them, DialogueLLM possesses the abilty to understand complex emotions without introducing any other components. In addition, our model would also benefit the development of task-specific LLMs. 

## **3 Methodology** 

In this section, we detail the comprehensive pipeline for training DialogueLLM models, as shown in Fig. 2. 

## **3.1 Problem Formulation** 

Assume that there are _N_ conversation instances in the instruction dataset, the _i[th]_ conversation _Di_ contains _K_ multi-modal utterances, which is represented as _Di_ = _{_ ( _Cz, Mk_ ) _, Yk}_ , where _Cz_ denotes previous _z_ contextual utterances, _Mk_ represents the _k[th]_ target utterance to be classified, _Yk_ means the emotion label of the _k[th]_ target utterance, where _i ∈_ [1 _,_ 2 _, ..., N_ ], _k ∈_ [1 _,_ 2 _, ..., K_ ], _z ≥_ 0. The target utterance consists of textual ( _T_ ) and visual ( _V_ ) modalities, i.e., _Mk_ = ( _Tk, Vk_ ), where _Tk ∈R[l][Tk][×][d][Tk]_ , _Vk ∈R[l][Ik][×][d][Ik]_ . Here, _lTk_ and _lVk_ denote the sequence length of textual and visual utterances, _dTk_ and _dVk_ represents the dimensions of the textual and visual features. 

Now, we summarize our research problem as: _Given one multi-speaker conversation including K multi-modal utterances, how to detect their emotions?_ It could be written as: 

**==> picture [168 x 25] intentionally omitted <==**

where Θ denotes the parameter set. 

## **3.2 Base Model** 

The first key component is to select open-source and strong foundation language models. LLaMA is a collection of open source foundation language models ranging from 7B to 65B parameters, which is trained on trillions of tokens using publicly available datasets. It achieves state-of-the-art performance on numerous benchmarks, which has greatly promoted the research progress of LLMs. A considerable number of researchers choose to expand the 

**==> picture [196 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
32.14%<br>42.68%<br>25.18%<br>Meld<br>IEMOCAP<br>Emorynlp<br>**----- End of picture text -----**<br>


which involves training on supervised instruction/input/output data. Instruction tuning helps align DialogueLLM with human prompts, enabling precise customization for emotional domains. This allows DialogueLLM to become adaptable and proficient at generating accurate emotional responses. In this paper, we create a high-quality instruction dataset by leveraging three widely used benchmarking ERC datasets. Since many potential shortcomings exist in automatic generation of samples using strong language models (e.g., ChatGPT), such as low quality, repetition, and lack of diversity, etc., different from the existing works [18, 25], we do not use ChatGPT to generate instances. The benchmarking ERC datasets have provided clean samples with precise annotations, which will be an optimal choice for creating instruction dataset. The training sets of three benchmarking datasets (i.e., MELD, IEMOCAP and EmoryNLP) are treated as the data source, altogether 2,411 multi-party dialogues, over 24,304 utterances are collected. In view that the labels are from different datasets, we first pre-process the lables. For example, “joy”, “happy” and “happiness” will be normalized to be “happiness”. The instructions are constructed based on the task definition and label space, e.g., _“Given the Video Description and Context, detect the emotion of the input, and assign an accuracy label from [‘happiness’, ‘anger’, ‘fear’, ‘sadness’, ‘disgust’, ‘surprise’, ‘neutral’].”_ . The textual raw samples and the counterpart labels are normalized to the input/output pairs. 

Figure 3: The distribution of three ERC datasets. 

capabilities of LLaMA models though instruction tuning, due to the lower computational costs. 

Furthermore, Meta AI has just developed and released LLaMA 2, which is an updated version of LLaMA 1. Compared with LLaMA 1, the training data used for LLaMA 2 was increased by 40% and the context length was doubled. LLaMA 2 also incorporated grouped query attention mechanisms. LLaMA 2 shows many behaviors similar to ChatGPT, but is also surprisingly small and easy to reproduce. Hence, we adopt LLaMA 2-7B model as our base model. Furthermore, LLaMA-7B have also been attempted and evaluated in the experiments. We use low-rank adaptation (LoRA) to finetune them with only 2.1 million trainable parameters. 

In view that LLaMA 2 possess a powerful generative capability, we treat ERC as a conditional generative task, where the output _Yk_ will be an emotion label. We first propose a general zero-shot prompt template, namely _Prompt[k] erc_[that consists] of the contextual, multi-modal utterances and the instruction _Ierc_ by merging them together: 

In view of the importance of the conversational context and multi-modal knowledge, the contextual utterances and the visual information are incorporated into instruction instances. Assume that there are _z_ contextual utterances before the target utterance, we would list them before the input content. In this work, the default size is _z_ = 1 (where the impact of varying size will be discussed in Sec. 4.9). In addition, the corresponding video is split into frames and forward them through the ERNIE Bot, to generate the descriptions of this video. Then, such descriptions are considered as the supplementary knowledge. More statistics of this dataset are presented in Fig. 3 and Fig. 4. Notably, “Neutral” and “Happiness” accounted for the largest percentage of the total instances, about 31.1% and 15.4%, respectively. In contrast, “Fear”, “Powerful”, and “Peaceful” are represented at lower proportions. “Fear" comprises around 6.1% of the 

**==> picture [238 x 13] intentionally omitted <==**

**==> picture [13 x 10] intentionally omitted <==**

where _TextDescription_ ( _Vk_ ) denotes the text description of the _k[th]_ video produced by the ERNIE Bot. Then, we ask LLMs to generate an emotion label by providing the above-mentioned prompts: 

**==> picture [174 x 21] intentionally omitted <==**

## **3.3 Emotion and Context Knowledge Based Instruction Dataset** 

Human conversation is filled with different emotions, such as neutral, anger, happiness, surprise, etc. To satisfy complex emotion recognition needs, DialogueLLM undergos an instruction-tuning step 

**==> picture [225 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
7000<br>6000<br>5000<br>4000<br>3000<br>2000<br>1000<br>0<br>Pea Pow Dis Fru Neu Hap Sad Ang Fea Exi Sur<br>Emotion<br>No. of Conversations<br>**----- End of picture text -----**<br>


Figure 4: The distribution of seven basic emotions across three datasets. 

## **Algorithm 1** The details of DialogueLLM 

1: **Require:** Task description _Ierc_ , Dataset _D_ (containing _N_ dialogues, _K_ utterances and the emotion label _Yk_ for each utterance), Base model _LLM_ 

- 2: **Parameter:** _θ_ 

- 3: **Ensure:** the emotion label _Yk_ 

- 4: /* Step1: Generate _Prompt_ */ 

- 5: _Ierc_ , _D → Prompt {_ ( _Ierc, Cz, Tk, Vk_ ) _}_ 

- 6: /* Step 2: Input the Prompt of the _t[th]_ sample */ 7: _LLM_ ( _Prompt_ ) _∼ E_ 1 _, ..., Ek_ 

- 8: /* Step 3: Call LoRA function for fine-tuning */ 9: _LoRA_ ( _E_ 1 _, ..., EK_ ) _→ DialogueLLM_ 

10: /* Step 4: Generate output */ 11: _Output_ = _DialogueLLM_ ( _Ierc, Cz, Tk, Vk_ ) **Output:** _Yk[p]_ 

## **4 Experiments** 

dataset, and “Powerful" represents about 3.5% of the dataset. Similarly, “Peaceful" constitutes approximately 6.3% of the dataset, indicating a notable but still comparatively moderate occurrence of this particular emotion. “Anger", “Sadness", “Frustration", “Surprise", “Excitement" and “Disgust" collectively account for around 37.6% of the dataset. Specifically, “Anger" accounts for about 9.7%, “Sadness" for about 6.4%, “Frustration" for about 5.6%, “Surprise" is about 9.4%, “Excitement" is about 4.2%, and “Disgust" is about 2.3%. 

Finally, this instruction dataset is used for supervised fine-tuning. Notably, all the instances in the dataset are normalize as “instruction/video descriptions/context/input/output” pairs (see Fig. 2). 

## **4.1 Research Question** 

**RQ1:** Is it effective to propose an emotion-tailored LLM? 

**RQ2:** Does modeling of the contextual dependency and multi-modal information help improve performance? 

**RQ3:** Does DialogueLLM has powerful incontext learning abilities? 

To answer RQ1, we compare the proposed DialogueLLM with a wide range of state-of-the-art baselines and other LLMs on three benchmark datasets in Sec. 4.4. To answer RQ2, we conduct a ablation test by removing one component at one time in Sec. 4.5. To answer RQ3, we consider zeroshot and few-shot prompting setups, and report their results in Sec. 4.6. 

## **4.2 Experimental Settings** 

## **3.4 Training and Implementation** 

The DialogueLLM-7B model is fine-tuned LLaMA 2-7B with the emotional knowledge based instruction data to acquire emotion recognition skills. Training a DialogueLLM-7B model will cost about 5 hours on a 40GB A100 GPU. The total approximate tokens seen during pre-training is approximately 22 billion tokens. We optimize our model with the AdamW optimizer with the following hyper-parameters: _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 95. We use a cosine learning rate schedule, such that the final learning rate is equal to 10% of the maximal learning rate. The activation function is set to SwiGLU to improve performance. The target utterances are forward through DialogueLLM models to generate the emotion labels. The details of the algorithm are outlined in Algorithm 1. 

**Datasets.** Three benchmark ERC datasets which include the textual and visual utterances with high quality emotion annotations, are selected as the experimental beds, _viz._ MELD[6] [27], IEMOCAP[7] , and EmoryNLP[8] . 

**MELD.** It consists of 13,708 multi-modal utterances from 1,433 multi-party dialogues of Friends TV series. The utterances in each dialogue are annotated with one of three sentiments (positive, negative or neutral) and one of seven emotions (anger, disgust, fear, joy, neutral, sadness or surprise). The overall Fleiss’ kappa score reaches 0.43. In this work, we only use textual and visual information. **IEMOCAP.** It is comprised of 151 recorded dialogue videos, encompassing a total of 302 videos 

> 6https://github.com/declare-lab/MELD. 

> 7https://sail.usc.edu/iemocap/. 

> 8https://github.com/emorynlp 

across the entire dataset, each involving two speakers per session. The annotations for this dataset encompass 9 distinct emotions (anger, excitement, fear, sadness, surprise, frustration, happiness, disappointment, and neutrality). The recordings are distributed across five sessions, with each session featuring five pairs of speakers. 

**EmoryNLP.** consists of 97 episodes, 897 scenes, and 12,606 utterances, which is a textual corpus that comprises multi-party dialogue transcripts of the Friends TV show. Each utterance is annotated with one of seven emotions, i.e., sad, mad, scared, powerful, peaceful, joyful, and neutral. The detailed statistics are shown in Table 1. 

**Evaluation metrics.** In line with the previous approaches, _accuracy_ (Acc) and _weighted-F1_ (w-F1) are used as evaluation metrics. For each method, we run five random seeds and report the average result of the test sets. 

**Hyper-parameter.** We report the detailed hyperparameter settings of DialogueLLM on three datasets in Table 2. The maximum context length is set to 4,096. We use a weight decay of 0.1 and gradient clipping of 1.0. The batch size is set to 128. 

## **4.3 Compared Baselines** 

A wide range of SOTA baselines are included for comparison including pre-trained language model (PLM) based and LLM based approaches. They are: 

## • _**PLM based approaches:**_ 

**(1) bc-LSTM [46]** implements an utterancelevel LSTM to capture contextual features. 

**(2) ICON [7]** hierarchically models the selfand inter-speaker emotional influences into global memories, and generates contextual summaries. 

**(3) MTL [15]** exploits speaker identification (SI) as an auxiliary task to enhance the utterance representation in conversations. 

**(4) DialogXL [32]** modifies the recurrence mechanism of XLNet to store longer historical context and dialog-aware self-attention to deal with the multi-party structures. 

**(5) TODKAT [49]** designs a transformerbased encoder-decoder architecture fuses the topical and commonsense information, and performs the emotion label sequence prediction. 

**(6) CoG-BART [16]** uses the pre-trained encoder-decoder model BART as the backbone model and utilizes an auxiliary response generation task to enhance the model’s ability of handling context information. 

**(7) DialogueRNN [22]** designs a method based on recurrent neural networks (RNN) that keeps track of the individual party states throughout the conversation and uses this information for emotion classification. 

**(8) DialogueGCN [5]** leverages self and interspeaker dependency of the interlocutors to model conversational context for emotion recognition. 

**(9) DialogueCRN [10]** designs multi-turn reasoning modules to extract and integrate emotional clues. 

**(10) RGAT [11]** proposes relational position encodings to capture both the speaker dependency and the sequential information. 

**(11) DAG-ERC [5]** regards each conversation as a directed acyclic graph to model the conversation context. 

**(12) CauAIN [48]** retrieves causal clues provided by commonsense knowledge to guide the process of causal utterance traceback. 

**(13) COIN [40]** is a conversational interactive model to mitigate the problem of overlooking the immediate mutual interaction between different speakers by applying state mutual interaction within history contexts. 

**(14) GraphCFC [14]** is a module adept at modeling context and interaction information in ERC tasks with high efficiency. It leverages multiple extractors and PairCC strategies to effectively tackle the heterogeneity present in multimodal fusion. 

**(15) SACL-LSTM [9]** applies contrast-aware adversarial training to generate worst-case samples and uses a joint class-spread contrastive learning objective on both original and adversarial samples. 

## • _**LLMs based approaches:**_ 

**(1) LLaMA [34]** takes a sequence of words as an input and predicts a next word to recursively generate text. 

**(2) Alpaca** is a state-of-the-art finedtuning version of LLaMA, by using supervised learning 

Table 1: Testing dataset statistics. 

|**Type**<br>**Dataset**|**Dialogue**<br>**Utterance**<br>**Class**<br>**Metric**<br>**Train**<br>**Validation**<br>**Test**<br>**Train**<br>**Validation**<br>**Test**|
|---|---|
|Main Datasets<br>MELD<br>IEMOCAP<br>EmoryNLP|1,039<br>114<br>280<br>9,989<br>1,109<br>2,610<br>7<br>Weighted-F1<br>120<br>31<br>5,810<br>1,623<br>8<br>Weighted-F1<br>659<br>89<br>79<br>7,551<br>954<br>984<br>7<br>Weighted-F1|



Table 2: Hyperparameters for fine-tuning DialogueLLM. 

|**Hyperparameter**|**Value**|
|---|---|
|Batch size|128|
|Micro batch size|8|
|Epoch|10|
|Learning rate|3e-4|
|Lora r|4|
|Lora alpha|16|
|Lora dropout|0.05|
|Cutoff length|256|



from a LLaMA 7B model on 52K instructionfollowing demonstrations. 

**(3) LLaMA 2 [35]** is trained on 2 trillion tokens, and have double the context length than Llama 1, and outperforms other open source language models on many external benchmarks. 

## **4.4 Results and Anlysis** 

The experimental performance of all baselines is shown in Table 3. We divide these baselines into two categories, i.e., pre-trained language models and large language models without fine-tuning. We will conduct a detailed analysis of their classification performance. 

For the PLM based baselines, we can observe that MTL demonstrates very poor performance compared to the other baseline models, with the worst classification accuracy on the MELD and EmoryNLP datasets. One possible reason is that MTL disregards modeling conversation-level interaction information. Without capturing contextual information, the model struggles to learn effectively, resulting in inaccurate classification outcomes. In contrast, DialogueCNN, DialogueGCN, and DialogueCRN aim to model the contextual information derived from the speaker. Compared to MTL, their performance substantially improves. This further verifies that incorporating contextual information is vital for ERC. 

The SACL-LSTM model performs very well, being the second-best in terms of the average scores 

on three datasets. It may benefit from an architecture that combines the strengths of LSTM with self-attention mechanisms, enabling it to capture both long-term dependencies and subtle contextual cues within dialogues. Other strong models like TODKAT, CoG-BART, and DialogXL show competitive performance but do not reach the same level as SACL-LSTM across MELD and IEMOCAP datasets. They do not achieve top scores, which could be attributed to the complexity of emotion recognition tasks that may not be entirely captured by the models’ pre-training data or architecture. With 6.1M parameters, CauAIN demonstrates solid performance, especially on the MELD and IEMOCAP datasets. This model’s architecture likely includes mechanisms that aid in capturing causal relationships within dialogues, which is a critical factor for understanding emotions. The COIN model, despite its smaller size of 0.6M parameters, still achieves competitive accuracy on the IEMOCAP dataset. But it was not evaluated on other datasets. 

In addition, Table 3 shows that when LLMs are fine-tuned without using the proposed emotional knowledge, all of LLaMA-7B, Alpaca, and LLaMA2-7B perform very poor on three emotion recognition tasks. This proves that the general priori knowledge of LLMs are not sufficient to handle complex and subjective emotion understanding tasks. The emotion-specific knowledge is needed to further deepen their potential. In contrast, The proposed DialogueLLM model achieves the stateof-the-art performance across three datasets, which proves the effectiveness of fine-tuning LLMs with task specific knowledge. 

**MELD.** DialogueLLM achieves remarkable results on the MELD dataset, demonstrating its robustness with a leading F1 score of 71.90% and an accuracy of 71.96%. This is a significant uptick from the other strong contender on this dataset, SACL-LSTM, which registers an F1 score of 66.45% and an accuracy of 67.51%. The MELD dataset, known for its realistic conversational scenarios from a popular TV show, poses a challeng- 

Table 3: Comparison results (%) on different methods. The best scores are in bold. 

|**Methods**|**# Param.**|**MELD**|**MELD**|**IEMOCAP**|**IEMOCAP**|**EmoryNLP**|**EmoryNLP**|**Avgerage**|**Avgerage**|
|---|---|---|---|---|---|---|---|---|---|
|||Acc<br>w-F1||Acc<br>w-F1||Acc<br>w-F1||Acc<br>w-F1||
|||||||||||
|bc-LSTM<br>ICON<br>MTL<br>DialogXL<br>TODKAT<br>CoG-BART<br>DialogueRNN<br>DialogueGCN<br>DialogueCRN<br>RGAT<br>DAG-ERC<br>CauAIN<br>COIN<br>GraphCFC<br>SACL-LSTM|1.2M<br>0.5M<br>1.2M<br>510M<br>330M<br>415.1M<br>9.9M<br>2.1M<br>3.3M<br>13M<br>9.5M<br>6.1M<br>0.6M<br>0.6M<br>2.6M|65.87<br>-<br>62.45<br>-<br>67.24<br>64.95<br>65.96<br>63.62<br>66.93<br>-<br>63.75<br>65.85<br>-<br>-<br>**67.51**|64.87<br>-<br>61.90<br>62.41<br>65.47<br>63.82<br>65.30<br>62.68<br>65.77<br>60.91<br>63.36<br>64.89<br>-<br>58.86<br>**66.45**|63.08<br>64.00<br>-<br>-<br>61.11<br>65.02<br>64.85<br>62.49<br>67.39<br>-<br>66.54<br>65.08<br>66.05<br>-<br>**69.08**|62.84<br>63.50<br>-<br>65.94<br>61.33<br>64.87<br>64.65<br>62.11<br>67.53<br>65.22<br>66.53<br>65.01<br>65.37<br>68.91<br>**69.22**|40.85<br>-<br>36.36<br>-<br>42.38<br>40.94<br>43.66<br>36.87<br>41.04<br>-<br>39.64<br>43.13<br>-<br>-<br>**42.21**|36.84<br>-<br>35.92<br>34.73<br>38.69<br>37.33<br>37.54<br>36.43<br>38.79<br>34.42<br>38.29<br>37.87<br>-<br>-<br>**39.65**|56.60<br>-<br>49.40<br>-<br>56.91<br>56.97<br>58.16<br>54.33<br>58.45<br>-<br>56.64<br>58.02<br>-<br>-<br>**59.60**|54.85<br>-<br>48.91<br>54.36<br>55.16<br>55.34<br>55.83<br>53.14<br>57.36<br>53.52<br>56.06<br>55.92<br>-<br>-<br>**58.44**|
|||||||||||
|LLaMA-7B<br>Alpaca<br>LLaMA 2-7B|2.1M<br>2.1M<br>4.2M|15.09<br>19.22<br>23.71|16.02<br>18.37<br>24.12|19.32<br>20.35<br>26.73|18.24<br>19.16<br>24.35|17.78<br>17.95<br>25.50|17.40<br>17.33<br>17.27|17.40<br>19.17<br>25.31|17.22<br>18.29<br>21.91|
|||||||||||
|DialogueLLM<br>Improve_△_|4.2M|**71.96**<br>_↑_6_._59%|**71.90**<br>_↑_8_._20%|**70.62**<br>_↑_2_._22%|**69.93**<br>_↑_1_._03%|**41.88**<br>_↓_0_._78%|**40.05**<br>_↑_1_._00%|**61.49**<br>_↑_3_._17%|**60.52**<br>_↑_3_._56%|



ing benchmark due to its diverse emotional expressions and informal dialogue. DialogueLLM’s performance here suggests its superior ability to decode nuanced emotional cues within a naturalistic dialogue setting. 

**IEMOCAP.** DialogueLLM showcases its prowess with an accuracy of 70.62% and an F1 score of 69.93%, outstripping the previously leading SACL-LSTM model, which had an accuracy of 69.08% and an F1 score of 69.22%. The IEMOCAP dataset is unique due to its focus on dyadic conversations with a rich set of emotional annotations, ranging from anger to happiness. The high performance of DialogueLLM on this dataset underscores its effectiveness in understanding and interpreting complex emotional dynamics in close-ended conversations. 

**Emorynlp.** DialogueLLM maintains a competitive edge, securing an F1 score of 40.05% and an accuracy of 41.88%. DialogueLLM consistently surpasses all of the 15 baselines. This performance is indicative of DialogueLLM’s versatile capacity to capture emotional nuances across varied conversational contexts 

The experimental results demonstrate the effectiveness of the DialogueLLM model in emotion recognition tasks across different datasets. It consistently achieves the highest F1 scores and accuracy, outperforming 15 state-of-the-art models. Notably, DiaologueLLM’s performance is robust 

across various datasets, which underscores its versatility and reliability in handling different data sources and domains. This also suggests a trend where specialized pre-training on tasks closely related to the downstream application can yield significant benefits. 

_Training loss._ The training loss is shown in Figure 5. The loss result on MELD dataset shows a rapid initial decrease with some early fluctuations, eventually stabilizing at around a 0.2 loss value. The IEMOCAP dataset’s training loss drops more abruptly than MELD, suggesting a faster learning rate or easier dataset for the model to learn, and levels off at a lower value near 0.1, indicating a more successful training outcome. Lastly, the EmoryNLP dataset’s loss decreases smoothly without the volatility observed in the MELD graph, also stabilizing at a loss value just under 0.2. This smooth decrease may point to a stable learning process. 

## **4.5 Ablation Test** 

The ablation study is conducted across three datasets, which will provide a structured insight into the contribution of different components to the model’s performance. The term “w/o” indicates the model’s performance without a specific feature, where the term “w” indicates the model’s performance with a specific feature. 

From Table 4, we have four observations: (1) 

**==> picture [433 x 159] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.8 Meld-Train-Loss 1.0 IEMOCAP-Train-Loss Emorynlp-Train-Loss<br>0.7 0.8<br>0.8<br>0.6<br>0.6<br>0.5 0.6<br>0.4<br>0.4<br>0.4<br>0.3<br>0.2 0.2 0.2<br>0.1<br>0 2000 4000 6000 8000 0 500 1000 1500 2000 2500 3000 0 1000 2000 3000 4000 5000<br>Step Step Step<br>Figure 5: The training loss of DialogueLLM.<br>Loss<br>**----- End of picture text -----**<br>


the DialogueLLM’s performance decreases across all datasets when any component is removed, underscoring the integral role each part plays in the model’s design for emotion recognition; (2) the performance drops with the removal of context on all datasets suggests that contextual information is important for emotion recognition, aligning with the premise that conversational emotion understanding is heavily reliant on context; (3) the removal of LoRA leads to a significant decrease in model performance, because the small training size leads to underfitting; (4) removing the visual information leads to a noticeable decrease in performance, suggesting that multimodal information may be beneficial for emotion recognition in dialogues. Notably, we do not use the visual information from IEMOCAP, because the actors in this corpus are sitting on chairs for face-to-face conversations, and the descriptions of the image information are too similar, e.g., “a man and a woman sitting on chairs for face-to-face exchanges”. Here, we have given the answer to RQ2. 

## **4.6 Zero-shot v/s Few-shot Prompting** 

This paper also performs zero-shot and few-shot experiments to evaluate whether DialogueLLM can perform better when a limited number of cases are available for emotion recognition tasks. The results are shown in Table 5. We design four _H_ -shot settings: zero-shot, one-shot, five-shot, ten-shot. For each setting, we sample _H_ = _{_ 0 _,_ 1 _,_ 5 _,_ 10 _}_ examples for emotion classification. These sampling examples serve as the learning samples for DialogueLLM. 

The impact of adding shots varies with the number of shots. The performance gains are not significant or even decreased when adding too many shots. The change from zero-shot to one-shot results in a 

Table 4: Ablation experiment results across three ERC tasks in a zero-shot setting. 

|**Dataset**|**Models**|**Acc**|**w-F1**|
|---|---|---|---|
||_w/o_Context|70.91|67.94|
|MELD|_w/o_Lora<br>_w/o_Video Description|66.17<br>60.80|64.42<br>59.75|
||DialogueLLM|**71.91**|**71.81**|
||_w/o_Context|68.14|68.01|
|IEMOCAP|_w/o_Lora<br>_w/o_Video Description|65.23<br>-|63.78<br>-|
||DialogueLLM|**70.48**|**69.40**|
||_w/o_Context|39.41|36.25|
|Emorynlp|_w/o_Lora<br>_w/o_Video Description|35.66<br>-|33.83<br>-|
||DialogueLLM|**41.76**|**38.47**|



slight improvement in classification performance. With the gradual increase in the number of shots, the performance drops down. 

This could be attributed to misclassifications made by DialogueLLM, potentially arising from the model learning excessive redundant information when handling too long contextual data. This suggests that roughly increasing the number of extra shots does not necessaryly result in a stable performance improvement. 

Table 5: Few shot performance of emotion recognition task. 

|**Meld**<br>**IE**|**MOCAP**<br>**Emorynlp**|
|---|---|
|Acc<br>w-F1<br>Acc|w-F1<br>Acc<br>w-F1|
|71.91<br>71.81<br>70.4<br>71.96<br>71.90<br>70.6<br>71.92<br>71.63<br>70.5<br>70.65<br>71.04<br>69.9|8<br>69.40<br>41.76<br>38.47<br>2<br>69.93<br>41.88<br>40.05<br>4<br>69.37<br>41.62<br>38.61<br>4<br>68.82<br>41.25<br>38.19|



## **4.7 Error Analysis** 

The detailed error analysis is also conducted via the confusion matrices that are shown in Figure 6. Each cell ( _i, j_ ) represents the percentage of class _i_ is classified to be class _j_ . Upon reviewing the classification results produced by DialogueLLM on the three datasets, we discover that imbalanced emotion categories and the similarity across different emotions are the key factors contributing to misclassification. 

By examining the diagonal elements of the matrices, DialogueLLM demonstrates effective truepositive categorization for most fine-grained emotions. However, it exhibits a tendency to misclassify the utterances to be “neutral”, particularly in the EmoryNLP dataset. This misclassification is influenced by the dataset’s imbalance, where the percentages of “neutral” utterances in EmoryNLP, MELD and IEMOCAP are 32.48%, 47.5%, and 22.23%, respectively. This highly unbalanced data distribution leads to the model’s excessive preference for “neutral” emotions. 

Additional, we show a few typical misclassification examples in Figure 7. It is evident that DialogueLLM encounters challenges in distinguishing closely related pairs of emotions. In the confusion matrices, we observe a consistent misclassification of “anger” to be “disgust” on the MELD dataset, see the example _B_ in Figure 7. In this case, when Phoebe makes a negative comment to another person, it is challenging to discover whether the expressed emotion is “disgust” or “anger”. A few pairs of emotions such as “surprise” vs “excitement”, “anger” vs “frustration” and “peaceful” vs “happiness”. The slight similarity across such emotions poses a challenge for the model to accurately distinguish them. This difficulty in discerning emotions may result in errors during emotion categorization. 

## **4.8 The Impact of Epoch** 

In this section, the impact of the number of epoch on the classification performance is shown in Figure 8. We can notice that the performance increases with iterative epoch on all three datasets. 

In particular, there is a sharp leap in performance when epoch ranges from 1 to 3, and a slow increase in performance when epoch ranges from 3 to 10. However, the performance slightly decreases when epoch varies from 7 to 10 on the EmoryNLP dataset. One possible explanation is overfitting on the train- 

**==> picture [167 x 145] intentionally omitted <==**

**==> picture [174 x 145] intentionally omitted <==**

**==> picture [167 x 142] intentionally omitted <==**

Figure 6: The normalized confusion matrices for DialogueLLM. The rows represent the truth label, where the columns represent the predicted labels. 

ing set due to the small size of this dataset. This result shows the importance of selecting an appropriate number of iteration rounds. Adding the number of epoch often leads to longer training times. Due to the limitation of our GPUs, we refrained from testing with a higher number of epochs. 

## **4.9 Effect of Context Range** 

This subsection aims to explore the impact of context length on the performance of DialogueLLM. We select and evaluate the context length from the pool _{_ 0 _,_ 1 _,_ 2 _,_ 3 _, All}_ , where _All_ represents using all the contexts. The experimental results are shown in Figure 9. 

In general, longer context length will allow the 

**==> picture [462 x 253] intentionally omitted <==**

Figure 7: A few typical misclassified examples on three datasets. 

**==> picture [440 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
71.81<br>70 66.85 67.94 69.02 66.62 67.18 68.11 69.4 1-E3-Epochpoch 7-E10-Epochpoch<br>5-Epoch<br>60 60.03 59.47<br>50<br>39.68<br>40 37.66 38.47<br>35.24<br>30.69<br>30<br>Meld IEMOCAP Emorynlp<br>Datasets<br>w-F1 (%)<br>**----- End of picture text -----**<br>


Figure 8: The impact of epoch on the classification performance in a zero-shot setting. 

model to access more information, thus making accurate prediction. We can notice that there is a slight improvement when the context length ranges from 0 to 2, and the performance slightly decreases when the context length ranges from 3 to All. This shows that too short can not provide supplementary knowledge where too long will introduce excessive amounts of noise. Hence, taking the previous two utterances before the target utterance into consideration may be a optimal choice. Additionally, processing long contextual information demands increased computational resources, thereby constraining the model’s utility. 

## **4.10 Analysis of DialogueLLM’s Emotional Intelligence** 

Emotional intelligence (EQ) is the ability to manage both human emotions and understand the emotions of other people. A people’s EQ affects his/her daily behavior and decision making. Since LLMs (including DialogueLLM) have shown strong emotion understanding ability, then evaluating their emotion intelligence will be the new target. We will answer the question: _can DialogueLLM be as emotionally intelligent as humans?_ 

In particular, we evaluate 12 LLMs’ EQ via a benchmarking testing bed, namely SECEU [37]. This test requires evaluating complex emotions (e.g., surprised, joyful, puzzled, proud) in realistic scenarios (e.g., despite feeling underperformed, 

**==> picture [383 x 145] intentionally omitted <==**

**----- Start of picture text -----**<br>
85<br>Meld Meld<br>EmorynlpIEMOCAP 11 EmorynlpIEMOCAP<br>75<br>10<br>65<br>9<br>55 8<br>7<br>45<br>6<br>35<br>0 1 2 3 All 0 1 2 3 All<br>Context Range Context Range<br>Accuracy(%)<br>Train and Test Times(hours)<br>**----- End of picture text -----**<br>


Figure 9: The experiments of the varying context lengths. 

**==> picture [216 x 196] intentionally omitted <==**

Figure 10: The proposed prompt for the emotional intelligence test[9] . 

John surprisingly achieved a top score). According SECEU’s requirement, we design a standard prompt to ask LLMs to do multi-choice questions in SECEU, where the prompt template is illustrated in Figure 10. 

The model is tasked with scoring the extent to which the protagonist experiences a given emotion on a scale from 0 to 10, with 10 indicating the highest possible level of that emotion. The cumulative score for the last four emotions must be constrained to 10. To standardize the performance of LLMs, we calculate the Euclidean distance between the LLMs’ responses for the _i[th]_ item (denoted as _Li_ ) and the standard human scores (denoted as _SSi_ ). We then average all 40 distances and generate the SECEU score. We then normalize the SECEU scores to derive an EQ score designed to conform to a normal distribution with a mean of 100 and a standard deviation of 15. The calculating process- 

ing of EQ is written as: 

**==> picture [204 x 121] intentionally omitted <==**

where _M_ = 2 _._ 79 and _SD_ = 15 represents the mean value and the standard deviation. 

The results of 12 LLMs’ EQ scores are shown in Table 6 and Figure 11. We can notice that the LLaMA base model cannot complete the EQ test, where our DialogueLLM achieves the second highest scores across 12 LLMs. GPT-4 achieves the highest EQ scores against ours (117 _v/s_ 109). However, our DialogueLLM model has only 7 billion parameters, which is 1/257 of GPT-4 (1.8 trillion parameters). The training time required for DialogueLLM is considerably shorter than that of GPT4. DialogueLLM performs the best among all the LLaMA series, and exhibits human-like response patterns, demonstrating a balanced mechanism for high emotion understanding proficiency. 

## **4.11 Effects of Emotional Stimuli** 

Psychological studies have shown that adding emotional stimuli related to expectations, confidence and social influence can have an impact on individual behavior. For example, real-life students who are taught using encouraging and positive words have a higher success rate than those who are not taught using these words. Therefore, we use the EmotionPrompt approach [13] to explore the performance of DialogueLLM in the face of emotional 

Table 6: LLMs’ EQ results. % indicates the percentage of LLM who outperforms humans in this EQ test. 

|**Based**|**Models**|**SECEU**|**EQ**|**%**|**Size**|
|---|---|---|---|---|---|
|**OpenAI**|**GPT series**|||||
||text-davinci-001|2.4|107|64%|<175B|
||text-davinci-002|3.3|91|23%|<175B|
||GPT-3.5-turbo|2.63|103|52%|175B|
||GPT-4|1.89|117|89%|1800B|
|**LLaMA**||||||
||LLaMA|Failed|–|–|7B|
||Alpaca|2.56|104|56%|13B|
||Koala|3.72|83|13%|13B|
||Vicuna|2.5|105|59%|13B|
|**Flan-t5**||||||
||Fastchat|Failed|–|–|3B|
|**GLM**||||||
||ChatGLM|3.12|94|28%|6B|
|**Claude**||||||
||Claude|2.46|106|61%|10B|
|**Ours**||||||
||DialogueLLM|2.31|109|72%|7B|



**==> picture [221 x 145] intentionally omitted <==**

Figure 11: LLMs’ EQ. The y-axis indicates the EQ score and the x-axis shows the percentage of total participants. 

stimuli. The prompt template for adding emotional stimuli is shown in Figure 12, with no subsequent prompt included for brevity. The DialogueLLM using the proposed emotion prompt is called StimuliLLM. 

Table 7 shows the comparison between StimuliLLM and the original DialogueLLM model on three ERC datasets. We can notice that the EmotionPrompt approach significantly boosts the performance of DialogueLLM (1.06% and 1.14% average improvement in terms of performance in zeroshot and one-shot settings). 

## **4.12 Limitations** 

Although DialogueLLM tries to accurately perform emotion classification by considering both conversational contexts and video descriptions of the utterances, this takes more computing power and train- 

**==> picture [188 x 181] intentionally omitted <==**

Figure 12: Prompt template with emotional stimuli 

Table 7: The comparison between StimuliLLM and DialogueLLM. 

|Setting|Model|MELD|IEMOCAP|EmoryNLP|
|---|---|---|---|---|
|0-shot|DialogueLLM<br>StimuliLLM|71.81<br>72.19|69.40<br>69.82|38.47<br>39.25|
|1-shot|DialogueLLM<br>StimuliLLM|71.90<br>72.67|69.93<br>70.42|40.05<br>40.71|



ing time. Additionally, the speaker information is also important for improving the performance since different speakers have their own characters. But DialogueLLM does not take it into consideration, due to the limit of the dataset. 

Also, technology for generating accurate video descriptions automatically still has room for improvement, and inaccurate descriptions can mislead the model’s prediction. The issue of using multiple data sources like images and video to improve emotion classification in large language models isn’t fully solved yet. Lastly, we train DialogueLLM using a specific approach that focuses on identifying emotions, rather than a general-purpose training method for all affects. Hence, our future work will collect a large scale knowledge corpus that contains over 1M subjective examples covering differnt types of affects, e.g., sentiment, emotion, sarcasm, humor, enthusiasm, etc. 

## **5 Conclusion and Future Work** 

ERC presents an intriguing and challenging natural language processing endeavor. In this paper, inspired by the remarkable performance of LLMs and their variants in NLP tasks, we propose DialogueLLM, a context and emotion knowledge 

tuned LLM that is obtained by fine-tuning large language models with benchmarking multi-modal (i.e., texts and videos) emotional dialogues. We offer a comprehensive evaluation of our proposed model on three benchmarking ERC datasets and achieves the state-of-the-art results. This proves that fine-tuning LLMs with task-specific knowledge will yield significant improvement over other PLM based approaches. In future work, we plan to design and generate more precise video descriptions, incorporating multimodal information to further explore the potential of LLMs in the NLP domain. Additionally, considering the close connections between emotions, sarcasm, passion, and depression, we aim to design a multi-affect learning framework based on LLMs. 

## **References** 

- [1] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S Narayanan. 2008. Iemocap: Interactive emotional dyadic motion capture database. _Language resources and evaluation_ , 42:335–359. 

- [2] Zhihong Chen, Feng Jiang, Junying Chen, Tiannan Wang, Fei Yu, Guiming Chen, Hongbo Zhang, Juhao Liang, Chen Zhang, Zhiyi Zhang, et al. 2023. Phoenix: Democratizing chatgpt across languages. _arXiv preprint arXiv:2304.10453_ . 

- [3] Vishal Chudasama, Purbayan Kar, Ashish Gudmalwar, Nirmesh Shah, Pankaj Wasnik, and Naoyuki Onoe. 2022. M2fnet: Multi-modal fusion network for emotion recognition in conversation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 4652–4661. 

- [4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 

- [5] Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya, and Alexander Gelbukh. 2019. Dialoguegcn: A graph convolutional neural network for emotion recognition in conversation. _arXiv preprint arXiv:1908.11540_ . 

- [6] Zhen Guo, Peiqi Wang, Yanwei Wang, and Shangdi Yu. 2023. Dr. llama: Improving small language models in domain-specific qa via generative data augmentation. _arXiv preprint arXiv:2305.07804_ . 

- [7] Devamanyu Hazarika, Soujanya Poria, Rada Mihalcea, Erik Cambria, and Roger Zimmermann. 2018. Icon: Interactive conversational memory network for multimodal emotion detection. In _Proceedings of the 2018 conference on empirical methods in natural language processing_ , pages 2594–2604. 

- [8] Devamanyu Hazarika, Soujanya Poria, Amir Zadeh, Erik Cambria, Louis-Philippe Morency, and Roger Zimmermann. 2018. Conversational memory network for emotion recognition in dyadic dialogue videos. In _Proceedings of the conference. Association for Computational Linguistics. North American Chapter. Meeting_ , volume 2018, page 2122. NIH Public Access. 

- [9] Dou Hu, Yinan Bao, Lingwei Wei, Wei Zhou, and Songlin Hu. 2023. Supervised adversarial contrastive learning for emotion recognition in conversations. _arXiv preprint arXiv:2306.01505_ . 

- [10] Dou Hu, Lingwei Wei, and Xiaoyong Huai. 2021. Dialoguecrn: Contextual reasoning networks for emotion recognition in conversations. _arXiv preprint arXiv:2106.01978_ . 

- [11] Taichi Ishiwatari, Yuki Yasuda, Taro Miyazaki, and Jun Goto. 2020. Relation-aware graph attention networks with relational position encodings for emotion recognition in conversations. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7360–7370. 

- [12] Shanglin Lei, Guanting Dong, Xiaoping Wang, Keheng Wang, and Sirui Wang. 2023. Instructerc: Reforming emotion recognition in conversation with a retrieval multi-task llms framework. _CoRR_ , abs/2309.11911. 

- [13] Cheng Li, Jindong Wang, Yixuan Zhang, Kaijie Zhu, Wenxin Hou, Jianxun Lian, Fang Luo, Qiang Yang, and Xing Xie. 2023. Large language models understand and can be enhanced by emotional stimuli. 

- [14] Jiang Li, Xiaoping Wang, Guoqing Lv, and Zhigang Zeng. 2022. Graphcfc: A directed graph based crossmodal feature complementation approach for multimodal conversational emotion recognition. _CoRR_ , abs/2207.12261. 

- [15] Jingye Li, Meishan Zhang, Donghong Ji, and Yijiang Liu. 2020. Multi-task learning with auxiliary speaker identification for conversational emotion recognition. _ArXiv_ , abs/2003.01478. 

- [16] Shimin Li, Hang Yan, and Xipeng Qiu. 2022. Contrast and generation make bart a good dialogue emotion recognizer. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 36, pages 11002–11010. 

- [17] Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, Steve Jiang, and You Zhang. 2023. Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge. _Cureus_ , 15(6). 

- [18] Tiedong Liu and Bryan Kian Hsiang Low. 2023. Goat: Fine-tuned llama outperforms gpt-4 on arithmetic tasks. _arXiv preprint arXiv:2305.14201_ . 

- [19] Yaochen Liu, Yazhou Zhang, and Dawei Song. 2023. A quantum probability driven framework for joint multi-modal sarcasm, sentiment and emotion analysis. _IEEE Transactions on Affective Computing_ , pages 1–15. 

- [20] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. _arXiv preprint arXiv:1907.11692_ . 

- [21] Junxia Ma, Lu Rong, Yazhou Zhang, and Prayag Tiwari. 2023. Moving from narrative to interactive multi-modal sentiment analysis: A survey. _ACM Transactions on Asian and Low-Resource Language Information Processing_ . 

- [22] Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander Gelbukh, and Erik Cambria. 2019. Dialoguernn: An attentive rnn for emotion detection in conversations. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 33, pages 6818–6825. 

- [23] OpenAI. 2023. Gpt-4 technical report. 

- [24] Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only. 

- [25] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4. _arXiv preprint arXiv:2304.03277_ . 

- [26] Soujanya Poria, Erik Cambria, Devamanyu Hazarika, Navonil Majumder, Amir Zadeh, and LouisPhilippe Morency. 2017. Context-dependent sentiment analysis in user-generated videos. In _Proceedings of the 55th annual meeting of the association for computational linguistics (volume 1: Long papers)_ , pages 873–883. 

- [27] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2018. Meld: A multimodal multi-party dataset for emotion recognition in conversations. _arXiv preprint arXiv:1810.02508_ . 

- [28] Shraman Pramanick, Aniket Roy, and Vishal M Patel. 2022. Multimodal learning using optimal transport for sarcasm and humor detection. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 3930–3940. 

- [29] Yang Qiao, Liqiang Jing, Xuemeng Song, Xiaolin Chen, Lei Zhu, and Liqiang Nie. 2023. Mutualenhanced incongruity learning network for multimodal sarcasm detection. In _Proceedings of the AAAI_ 

_Conference on Artificial Intelligence_ , volume 37, pages 9507–9515. 

- [30] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. 

- [31] Xiaozhe Ren, Pingyi Zhou, Xinfan Meng, Xinjing Huang, Yadao Wang, Weichao Wang, Pengfei Li, Xiaoda Zhang, Alexander Podolskiy, Grigory Arshinov, et al. 2023. Pangu- _{\_ Sigma _}_ : Towards trillion parameter language model with sparse heterogeneous computing. _arXiv preprint arXiv:2303.10845_ . 

- [32] Weizhou Shen, Junqing Chen, Xiaojun Quan, and Zhixian Xie. 2021. Dialogxl: All-in-one xlnet for multi-party conversation emotion recognition. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 13789–13797. 

- [33] Weizhou Shen, Siyue Wu, Yunyi Yang, and Xiaojun Quan. 2021. Directed acyclic graph network for conversational emotion recognition. _arXiv preprint arXiv:2105.12907_ . 

- [34] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. _arXiv preprint arXiv:2302.13971_ . 

- [35] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. 

- [36] Haochun Wang, Chi Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, and Ting Liu. 2023. Huatuo: Tuning llama model with chinese medical knowledge. _arXiv preprint arXiv:2304.06975_ . 

- [37] Xuena Wang, Xueting Li, Zi Yin, Yue Wu, and Liu Jia. 2023. Emotional intelligence of large language models. _CoRR_ , abs/2307.09042. 

- [38] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. 2023. Pmc-llama: Further finetuning llama on medical papers. _arXiv preprint arXiv:2304.14454_ . 

- [39] Sayyed M Zahiri and Jinho D Choi. 2017. Emotion detection on tv show transcripts with sequencebased convolutional neural networks. _arXiv preprint arXiv:1708.04299_ . 

- [40] Haidong Zhang and Yekun Chai. 2021. Coin: Conversational interactive networks for emotion recognition in conversation. In _Proceedings of the Third Workshop on Multimodal Artificial Intelligence_ , pages 12–18. 

   - [48] Weixiang Zhao, Yanyan Zhao, and Xin Lu. 2022. Cauain: Causal aware interaction network for emotion recognition in conversations. In _Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI_ , pages 4524–4530. 

   - [49] Lixing Zhu, Gabriele Pergola, Lin Gui, Deyu Zhou, and Yulan He. 2021. Topic-driven and knowledgeaware transformer for dialogue emotion detection. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)_ , pages 1571–1582, Online. Association for Computational Linguistics. 

- [41] Wenxuan Zhang, Yue Deng, Bing Liu, Sinno Jialin Pan, and Lidong Bing. 2023. Sentiment analysis in the era of large language models: A reality check. _arXiv preprint arXiv:2305.15005_ . 

- [42] Yazhou Zhang, Ao Jia, Bo Wang, Peng Zhang, Dongming Zhao, Pu Li, Yuexian Hou, Xiaojia Jin, Dawei Song, and Jing Qin. 2023. M3gat: A multimodal multi-task interactive graph attention network for conversational sentiment analysis and emotion recognition. _ACM Transactions on Information Systems_ . 

- [43] Yazhou Zhang, Qiuchi Li, Dawei Song, Peng Zhang, and Panpan Wang. 2019. Quantum-inspired interactive networks for conversational sentiment analysis. In _Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence IJCAI-19_ , pages 5436–5442. International Joint Conferences on Artificial Intelligence Organization. 

- [44] Yazhou Zhang, Yaochen Liu, Qiuchi Li, Prayag Tiwari, Benyou Wang, Yuhua Li, Hari Mohan Pandey, Peng Zhang, and Dawei Song. 2021. Cfn: a complexvalued fuzzy network for sarcasm detection in conversations. _IEEE Transactions on Fuzzy Systems_ , 29(12):3696–3710. 

- [45] Yazhou Zhang, Jinglin Wang, Yaochen Liu, Lu Rong, Qian Zheng, Dawei Song, Prayag Tiwari, and Jing Qin. 2023. A multitask learning model for multimodal sarcasm, sentiment and emotion recognition in conversations. _Information Fusion_ , 93:282– 301. 

- [46] Yazhou Zhang, Yang Yu, Dongming Zhao, Zuhe Li, Bo Wang, Yuexian Hou, Prayag Tiwari, and Jing Qin. 2023. Learning multi-task commonness and uniqueness for multi-modal sarcasm detection and sentiment analysis in conversation. _IEEE Transactions on Artificial Intelligence_ . 

- [47] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. _arXiv preprint arXiv:2303.18223_ . 

