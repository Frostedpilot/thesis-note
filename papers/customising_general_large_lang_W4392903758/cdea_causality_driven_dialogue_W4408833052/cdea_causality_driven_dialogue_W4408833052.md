**==> picture [35 x 35] intentionally omitted <==**

## _Article_ 

## **CDEA: Causality-Driven Dialogue Emotion Analysis via LLM** 

**Xue Zhang[1,2,] * , Mingjiang Wang[1,2] , Xuyi Zhuang[1,2] , Xiao Zeng[1,2] and Qiang Li[3]** 

- 1 Key Laboratory for Key Technologies of IoT Terminals, Harbin Institute of Technology, Shenzhen 518055, China; mjwang@hit.edu.cn (M.W.); zhuangxuyi@stu.hit.edu.cn (X.Z.); zengxiao0106@163.com (X.Z.) 

- 2 School of Electronics and Information Engineering, Harbin Institute of Technology, Shenzhen 518055, China 3 Shenzhen Zhili Middle School, Shenzhen 518055, China; 18934382665@163.com 

- Correspondence: zhangxue888000@163.com 

**Abstract:** With the rapid advancement of human–machine dialogue technology, sentiment analysis has become increasingly crucial. However, deep learning-based methods struggle with interpretability and reliability due to the subjectivity of emotions and the challenge of capturing emotion–cause relationships. To address these issues, we propose a novel sentiment analysis framework that integrates structured commonsense knowledge to explicitly infer emotional causes, enabling causal reasoning between historical and target sentences. Additionally, we enhance sentiment classification by leveraging large language models (LLMs) with dynamic example retrieval, constructing an experience database to guide the model using contextually relevant instances. To further improve adaptability, we design a semantic interpretation task for refining emotion category representations and fine-tune the LLM accordingly. Experiments on three benchmark datasets show that our approach significantly improves accuracy and reliability, surpassing traditional deeplearning methods. These findings underscore the effectiveness of structured reasoning, knowledge retrieval, and LLM-driven sentiment adaptation in advancing emotion–causebased sentiment analysis. 

**Keywords:** dialogue sentiment analysis; emotion causes; reasoning; commonsense knowledge; LLM; prompt engineering 

Academic Editors: Quanxin Zhu and Zhengqiu Zhang 

Received: 14 February 2025 Revised: 13 March 2025 Accepted: 20 March 2025 Published: 25 March 2025 

**Citation:** Zhang, X.; Wang, M.; Zhuang, X.; Zeng, X.; Li, Q. CDEA: Causality-Driven Dialogue Emotion Analysis via LLM. _Symmetry_ **2025** , _17_ , 489. https://doi.org/10.3390/ sym17040489 

**Copyright:** © 2025 by the authors. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (https://creativecommons.org/ licenses/by/4.0/). 

## **1. Introduction** 

Dialogue sentiment analysis is a crucial branch of natural language processing, focused on identifying and understanding emotional information expressed in dialogue content. Unlike general text-based sentiment analysis, dialogue sentiment analysis requires not only understanding the sentiment of individual sentences but also considering context, interactions between dialogue participants, and dynamic emotional shifts to accurately determine sentiment categories. Sentiment analysis has a wide range of applications, including emotional chatbots [1], social sentiment mining [2], healthcare [3], legal trials [4], and intelligent assistants [5]. 

Early sentiment analysis methods [6,7] relied on lexicon-based keyword matching, which struggled with sentences lacking explicit emotional cues. Later, feature engineering approaches improved performance but were complex and lacked generalization ability. Recently, deep learning models (e.g., CNNs, RNNs, Transformers) have enabled automatic feature extraction, but these approaches require accurately labeled datasets, and relying solely on context can lead to inconsistent annotations [8]. As shown in Figure 1, one of the most advanced strategies for enhancing the objectivity of dialogue sentiment analysis is incorporating the underlying causes of sentiment generation. If a model can identify the 

root causes of sentiment within a dialogue and, based on this information, accurately infer the sentiment category of a given statement, its reliability and interpretability would be significantly improved. 

**==> picture [256 x 189] intentionally omitted <==**

**Figure 1.** During dataset annotation, annotators use commonsense to identify emotional triggers in the dialogue and then integrate these cues to infer an objective sentiment category, minimizing subjective bias. 

However, there has been limited work [9,10] explicitly considering emotional causes to identify sentiment categories, and this field still faces several challenges. 

Lack of Explicit Reasoning Paths. Most sentiment analysis methods rely on semantic similarity or implicit feature extraction [8], capturing contextual associations without revealing the causal logic behind sentiment shifts. DialogueCRN [9] introduces Contextual Reasoning Networks (CRNs) to model contextual dependencies in conversations. While it effectively captures the sequential influence of emotions, its approach to emotional cause recognition remains implicit, as it does not explicitly construct reasoning paths. Consequently, its ability to establish causal relationships between dialogue utterances is limited, making it susceptible to errors in complex multi-turn interactions. Causal reasoning in sentiment analysis presents bi-directional symmetry, meaning that emotions arise not only from past contexts (forward causality) but are also influenced by the way speakers respond to them (backward causality) [11,12]. Current models struggle with this causal symmetry, leading to limited interpretability and reasoning accuracy. CauAIN [10] attempts to address this by introducing a causal-aware interaction network, which explicitly models inter-utterance causal dependencies. However, it primarily focuses on local causal inference, failing to capture global sentiment reasoning patterns across entire conversations. 

Insufficient Commonsense Knowledge. While some studies incorporate sentiment causes, their reasoning remains constrained by incomplete commonsense knowledge bases and limited contextual understanding [13,14]. Human sentiment cognition relies not only on linguistic cues but also on background knowledge and cultural norms, such as typical emotional responses to specific events. DialogueCRN and CauAIN, despite their advancements in causal reasoning, rely solely on dialogue context, lacking external commonsense knowledge support. This weakens their ability to establish causal links between sentiment triggers and emotional expressions, ultimately reducing classification accuracy and robustness [15,16]. 

To address these limitations, we designed a dual-module framework. First, we extract causal relationship information from a structured machine commonsense knowledge graph to detect emotional triggers between historical and target utterances in a dialogue. 

Next, we employ an attention mechanism to transform the extracted sentiment causes into prompts, leveraging a pre-trained language model for integrated reasoning to accurately predict the sentiment category of the target utterance. This process mimics human intuition and deliberate reasoning, significantly enhancing the model’s ability to capture sentiment causality. To better reflect speaker interactions, our sentiment cause detection module differentiates between “other-induced reasoning paths” and “self-induced reasoning paths”, enabling a more precise analysis of the contributions of different roles to emotional expression. With the advancement of large language models (LLMs) such as ChatGPT (https://chat.openai.com/, accessed on 14 March 2023), GPT (version 4.0) [17], and Claude 2 (https://www.anthropic.com/product/, accessed on 11 July 2023), their 

powerful contextual understanding and commonsense reasoning capabilities offer new opportunities for complex sentiment analysis tasks. However, leveraging LLMs’ contextual learning for dialogue sentiment analysis remains challenging [18], as identifying sentiment causes requires both deep contextual understanding and commonsense reasoning to establish causal relationships [19]. 

To tackle this challenge, we propose a prompt engineering approach that guides LLMs with high-quality instructions, integrating dialogue context and sentiment knowledge to enhance sentiment cause reasoning and classification, while overcoming the commonsense limitations of traditional methods. 

In summary, the main contributions of this paper are threefold: 

- We propose a dialogue emotion analysis method based on explicit reasoning for emotional causes. This method solves the problem of the lack of explicit reasoning path information in current methods by providing a clear reasoning path, allowing for accurate identification of emotional causes and establishing causal symmetry between emotional causes and emotion categories. 

- In addition, we leverage the rich knowledge embedded in GPT-4 and its powerful generalization ability to enhance the effectiveness of the emotion analysis method by constructing instructions that include historical content, emotional causes, and empirical examples. This approach not only effectively compensates for the lack of common knowledge support in current methods that consider emotional causes but also strengthens the accuracy and flexibility of emotional reasoning through a symmetry mechanism. 

- We conducted extensive experiments on three benchmark datasets to validate the model’s effectiveness and advantages in constructing explicit reasoning paths and LLM commonsense reasoning. The experimental results show that our model outperforms existing baseline methods in terms of accuracy and reliability, further highlighting the close connection between the emotion causal reasoning model and the concept of symmetry. 

## **2. Related Work** 

In this section, we provide a comprehensive overview of existing techniques for sentiment analysis and conversational sentiment analysis, followed by a detailed description of current work related to explicitly considering emotional reasons in conversational sentiment analysis. 

## _2.1. Sentiment Analysis Techniques_ 

Early sentiment analysis relied on rule-based methods using sentiment lexicons [20]. While straightforward, these methods were highly dependent on lexicon quality and struggled with complex sentence structures. The introduction of machine learning improved sentiment classification by learning patterns from labeled data [21,22]. However, these ap- 

proaches relied on manually extracted features and lacked strong contextual understanding, particularly in dialogues [23,24]. 

Bengio et al large corpora and can be fine-tuned for specific applications. Language models, which predict word sequences probabilistically [25], improve natural language processing (NLP) tasks by capturing contextual semantics. Pre-trained models primarily fall into two categories: LSTM-based and Transformer-based. LSTM-based models, such as ELMo [26], use multi-layer bi-directional LSTMs to create contextual word embeddings, effectively addressing polysemy issues. In contrast, Transformer-based models, including GPT [27] and BERT [28], leverage self-attention mechanisms to capture long-range dependencies, significantly improving sentiment classification. Further extensions of BERT have been optimized specifically for sentiment analysis in social media contexts, such as tweets [29–31]. 

Despite these advancements, effectively designing prompts to guide pre-trained models in sentiment reasoning remains challenging. This paper proposes a structured prompt engineering approach that integrates command statements, dialogue history, affective reasoning, task descriptions, and empirical examples. By incorporating conversation context with commonsense knowledge, our method enhances the model’s ability to accurately infer sentiment categories. 

## _2.2. Conversational Sentiment Analysis Techniques_ 

Unlike general text sentiment analysis, dialogue sentiment analysis must account for diverse factors such as dialogue context, speaker interactions, and external information. Existing approaches typically employ Recurrent Neural Networks (RNNs), Graph Neural Networks (GNNs), and attention mechanisms to model speaker interactions and extract context-relevant sentiment representations. For example, ref. [32] introduced DialogueGCN to capture intra- and inter-speaker dependencies, though it struggles with sentence-level relationships. Similarly, ref. [33] proposed R-GAT, a location-aware graph attention network, while [34] modeled dialogue frames as directed acyclic graphs (DAG-ERCs) to represent speaker relationships and sentence positions. While GNN-based methods effectively model conversational context, they often fall short in capturing speaker-specific nuances and precise sequence information. Furthermore, emotion subjectivity leads to inconsistent annotations, undermining the reliability of existing methods. A promising strategy to address this issue is to focus on the objective causes of subjective emotions. For instance, ref. [9] designed a multi-round inference module using LSTMs to retrieve and integrate emotional causes based on sentence similarity, while [10] distinguished self- and other-induced causes by concatenating their respective features before implicitly integrating them via a fully connected network. However, these methods face two key challenges: (1) a lack of explicit inference paths for capturing accurate causal relationships and (2) insufficient commonsense support. To mitigate these limitations, ref. [35] developed KET, a Transformer-based model that integrates commonsense knowledge through a context-based emotion graph attention mechanism, though its limited external relationship modeling may cause it to miss certain semantic details. In contrast, ref. [36] introduced COSMIC, a commonsense-based architecture that better captures the complex interplay between different commonsense knowledge categories and emotions. 

Building on these insights, we propose a dialogue sentiment analysis method based on explicit reasoning of sentiment causes, which addresses the lack of clear inference paths and insufficient commonsense support. Our approach leverages LLMs to enhance the model’s capacity for commonsense reasoning and improve the accuracy of sentiment cause identification. 

## **3. Method** 

## _3.1. Mission Definition and Model Overview_ 

Sentiment Analysis in Dialogue is a classification task where, given a continuous dialogue and the corresponding speaker information for each sentence as input, the goal is to identify and output the sentiment category of the target speaker’s utterance from a set of predefined sentiment categories. Specifically, assume that each dialogue consists of _N_ consecutive sentence _C_ = _{u_ 1, _u_ 2, . . . , _un}_ and their corresponding sentiment labels _YC_ = _{y_ 1, _y_ 2, _· · ·_ , _yN} ∈ E_ , where E denotes the sentiment category. For a sentence, it consists of _M_ tokens _ui_ = _{wi_ ,1, _wi_ ,2, _· · ·_ , _wi_ , _M}_ . Each sentence in a conversation C is uttered by a speaker and can be represented as _s_ ( _C_ ) = [ _s_ ( _u_ 1), _· · ·_ , _s_ ( _ui_ ), _· · ·_ , _s_ ( _uN_ )], where _s_ ( _ui_ ) _∈ S_ . The function _s_ maps the index of a sentence to its corresponding speaker, and _S_ denotes the category of the speaker. Thus, the whole problem can be formulated as follows: to obtain the sentiment label of each sentence based on the context and the corresponding speaker information in a conversation: _YC_ = _f_ ( _C_ , _s_ ( _C_ )). 

The LLM dialogue sentiment analysis model based on explicit reasoning for emotional reasons proposed in this study is shown in Figure 2. 

**==> picture [354 x 218] intentionally omitted <==**

**Figure 2.** This model operates in three phases: (1) Sentiment cause sentence acquisition identifies emotional causes from dialogue history, distinguishing between self-reasons and other-reasons. (2) Dynamic retrieval of experience examples—retrieves relevant examples from an experience database, refined by BART to enhance contextual alignment. (3) Prompt instruction construction model fine-tuning—constructs structured prompts integrating sentiment causes, historical context, and retrieved examples to optimize LLM-based sentiment classification. 

## _3.2. Sentiment Cause Sentence Acquisition_ 

Token representation. First, the context-independent sentence feature representation of the sentence is obtained. Here again in this section, the widely used pre-trained language model RoBERTa is used to extract the context-independent feature vector representations of sentences. Specifically, for each sentence _ui_ = _{wi_ ,1, _wi_ ,2, _· · ·_ , _wi_ , _M}_ , a special token [ _CLS_ ] is attached to the beginning of the sentence. After that, the sequence _{_ [ _CLS_ ], _wi_ ,1, _wi_ ,2, _· · ·_ , _wi_ , _M}_ is sent as input to the pre-trained language model RoBERTa, which is trained for the sentence-level sentiment classification task, i.e., the fine-tuning of the context-independent sentiment recognition task, and the feature vectors corresponding to the last layer of the [ _CLS_ ] tokens are sent to the pooling layer to be finally classified into the corresponding sentiment categories. After fine-tuning the RoBERTa model, each 

sentence is evaluated in the same format, i.e., _{_ [ _CLS_ ], _wi_ ,1, _wi_ ,2, _· · ·_ , _wi_ , _M}_ to obtain the context-independent feature vectors of the sentences corresponding to [ _CLS_ ] markers of the sentences _ci_ : 

**==> picture [288 x 12] intentionally omitted <==**

where _ci ∈_ R _[d][m]_ and _dm_ is the dimension of the labeled hidden state in RoBERTa. This section aligns with previous work [36] by also averaging the [ _CLS_ ] markers of the last four layers to finally obtain a context-independent feature vector representation of each sentence. Based on the above acquisition of context-independent representations of sentences, the next step is to acquire context-related representations of sentences. In a setting such as a dialog, the sentiment expressed by a sentence usually depends on the context of the whole dialog. Therefore, based on the context-independent feature representation of the sentence _ci_ , LSTM is used here to model the sequential dependencies between the sentences, and the reason for not using bi-directional long and short-term memory networks is that the scenario of the dialogue sentiment analysis task involved in this paper is biased towards real-time dialogue sentiment analysis in human–computer dialogue systems, where the information from the future context is not visible to the current sentence. Finally, the context-dependent feature representation _hi_ of the sentence is computed as follows: 

**==> picture [242 x 11] intentionally omitted <==**

where _hi ∈_ R _[d][h]_ denotes the hidden state vector at the ith time step and dh is the vector dimension of the output of each cell of LSTM. 

Inference path information acquisition. We use event-related reasoning from ATOMIC to address the lack of explicit reasoning paths in dialogues. Each sentence is treated as an event, and reasoning from historical sentences fills the gap in causal association with the target sentence. The emotional cause of the target sentence cannot stem from its own or future context. As shown in Table 1, we explore six relationship types: xReact, xEffect, and xWant (own reasoning paths, reflecting the speaker’s impact on themselves), and oReact, oEffect, and oWant (others’ reasoning paths, showing the effect on others). By incorporating these, we can more accurately identify emotional causes in conversation. 

**Table 1.** Examples of different types of inference path information. 

|**Sentence (Event)**|**X Pays Y a Compliment**|**X Pays Y a Compliment**|
|---|---|---|
||xEffect|be acknowledged|
|Self-Reasoning Path|xReact|feel good|
||xWant|chat with Y|
||oEffect|smile|
|Other-Reasoning Path|oReact|feel fattered|
||oWant|compliment X back|



In this section, we use the Common Sense Transformer model COMET [37] to extract inferential commonsense information from a structured machine commonsense graph, ATOMIC (Action and Temporal Commonsense Knowledge Graph) [38]. COMET is an encoder–decoder model that uses a pre-trained autoregressive linguistic model GPT as a base generative model that is trained on multiple commonsense knowledge graphs to automatically construct a knowledge graph. ATOMIC is a large-scale commonsense knowledge graph designed to model if-then reasoning about everyday events. ATOMIC focuses on social and inferential knowledge, allowing models to predict possible causes, effects, and intent of human actions. It contains over 880 K tuples structured as ( _s_ , _r_ , _o_ ) triplets, 

where _s_ (subject) represents an event, _r_ (relation) denotes a commonsense relationship, and _o_ (object) refers to the inferred commonsense consequence. COMET obtains a ternary _s_ , _r_ , _o_ from the graph and is trained to generate an object phrase _o_ based on a subject phrase _s_ and a relation phrase _r_ . In order to accomplish the task of automatically generating structured general knowledge, COMET is trained on a structured machine general knowledge graph, ATOMIC. The given event (i.e., sentence _ui_ in the dialogue scenario) and the selected relation type are first stitched together with mask tags [ _mask_ ] as inputs to the COMET, referring to previous work [36], where the representation of the hidden state of the last encoder layer of the COMET is also used as inference commonsense for the sentence _ui_ : 

**==> picture [284 x 13] intentionally omitted <==**

Then, the generated three self-referential constants are spliced and mapped to a feature vector of dimension size _dh_ , which is used as the self-referential path information _in f path[inter] i_ of the sentence _ui_ , where [; ] denotes feature splicing: 

**==> picture [292 x 17] intentionally omitted <==**

Affective Cause Detection. In this stage, we assume all historical sentences before the target sentence in the dialogue are potential emotional causes, aiming to measure their causal correlation with the target sentence. The output is a causal correlation matrix for the dialogue. To explicitly model emotional interactions between speakers, we categorize causal sentences into self-caused and other-caused sentences. 

For self-caused sentences, we focus on the causal influence of historical sentences from the same speaker on the target sentence’s emotion. We combine features from the historical sentence and its inference path with the target sentence for similarity calculation, using _α_ as the causal correlation value between the self-caused sentence and the target sentence, calculated as follows: 

**==> picture [310 x 29] intentionally omitted <==**

where _lq_ ( _x_ ), _lk_ ( _x_ ) and _lv_ ( _x_ ) are all linear transformations, and _dh_ is the dimension size of the key vector. _mask_ serves two purposes, one is to ensure that the detected historical causal sentences _hj_ and the target sentences _hi_ are all the same speaker, i.e., to ensure that the causal sentences _hj_ detected by the process are self-causal sentences. The second is to ensure that the detected self-causal sentences are from the dialogue context before the target sentence, and there will be no self-causal sentences from the future dialogue context, which is in line with the nature of causality, and the _mask_ is expressed as follows: 

**==> picture [294 x 37] intentionally omitted <==**

where _s_ is used to map the index of the sentence to the corresponding speaker index. For other people’s cause sentence, focusing on the degree of causal influence of historical sentences from different speakers on the emotion expressed in the target sentence, firstly, the historical sentence features and other people’s reasoning path information features are spliced with the target sentence for the similarity calculation, and the score is taken as the value of the degree of causal correlation between the other people’s cause sentence and the target sentence _score_ . The specific calculation method is as follows: 

**==> picture [309 x 29] intentionally omitted <==**

where _lq_ ( _x_ ), _lk_ ( _x_ ) , _lv_ ( _x_ ), and _dh_ are as described earlier. Here again, the mask serves two purposes: first, it ensures that the detected historical causative sentence _hj_ comes from a different speaker than the target sentence _hi_ , i.e., it ensures that the causative sentence _hj_ detected by the process is an other person’s causative sentence. Second, it also ensures that the detected others are from the dialogue context before the target sentence, which is represented by the _mask_ as follows: 

**==> picture [290 x 37] intentionally omitted <==**

After obtaining the values of causal influence of own cause statements as well as other people’s cause statements on the target statement, it is necessary to assess the degree of causal influence of the cause statements on the target statement under the same criterion, and the values in the matrix of the degree of causal correlation that are finally obtained are calculated in the following way: 

**==> picture [267 x 17] intentionally omitted <==**

Ultimately, we obtain the _m_ other-cause statements and _n_ self-cause statements in the dialogue context that have the highest degree of causal association with the target statement. 

## _3.3. Dynamic Retrieval of Experience Examples_ 

For LLMs, we use GPT-4, which has demonstrated a strong ability to learn with fewer examples, excelling at adapting to new tasks with minimal context. However, their performance depends on the selection of demo examples [39]. To leverage GPT-4’s powerful generalization capabilities while mitigating biases introduced by manual example selection, this study dynamically retrieves examples tailored to each input query, improving learning efficiency and contextual adaptation. 

We begin by constructing an empirical database, _DBexp_ , for conversational sentiment analysis, based on the EmoryNLP dataset [40]. The EmoryNLP dataset is a widely used benchmark for conversational sentiment analysis, originally derived from the TV show _Friends_ . It contains 897 dialogues and 12,606 utterances, where each utterance is labeled with one of seven sentiment categories: Neutral, Joyful, Peaceful, Powerful, Sad, Mad, and Scared. These labels provide a fine-grained emotional understanding of conversational exchanges. In this study, we preprocess the dataset in the following ways to ensure effective example retrieval: 

- Speaker information removal: To prevent speaker identity bias, all speaker metadata are removed, ensuring that the retrieved examples are selected purely based on textual content rather than specific speakers’ emotional tendencies. 

- Sentiment category balancing: Since some emotion categories (e.g., Neutral) are more frequent than others, we apply category balancing techniques to ensure that all sentiment classes have a uniform distribution within _DBexp_ . This prevents the model from over-relying on dominant categories during retrieval. 

- Text normalization: To reduce variability in sentence structure, we perform basic text preprocessing, such as lowercasing, punctuation normalization, and stop-word removal. 

For a target sentence _ui_ , the most relevant sentiment analysis examples are retrieved from _DBexp_ in two steps, based on semantic similarity, to serve as empirical examples _dexp_ for context learning in the LLMs. 

First, to ensure the semantic similarity between the experience example _dexp_ and the target statement _ui_ , we use the text searcher BERTScore [41] to compute the semantic similarity between the target statement _ui_ and the statement _udb_ in the experience database _DBexp_ . The _k_ most similar statements are selected as the candidate experience examples set _Dexp[cand]_[.][The similarity calculation is defined as follows:] 

**==> picture [272 x 11] intentionally omitted <==**

BERTScore is used here because it provides fine-grained semantic similarity by comparing token-level contextual embeddings of sentences. This allows it to capture synonymy and paraphrasing effects, making it well suited for identifying semantically similar sentences regardless of word choice. 

Then, since the same sentence can express different emotions in different dialogue contexts, we use cosine similarity to calculate the contextual semantic similarity between the target sentence _ui_ and the candidate experience examples _u[cand] exp_[from] _[ D] exp[cand]_[.][The statement] with the highest similarity score is selected as the final empirical example _dexp_ for the target sentence _ui_ , as determined by the following formula: 

**==> picture [265 x 25] intentionally omitted <==**

## _3.4. Prompt Instruction Construction_ 

To better leverage the rich commonsense knowledge in the LLM for dialogue sentiment analysis based on sentiment reasons, this section reconstructs the task in a generative framework by fine-tuning the LLM. As shown in Figure 3, we design a five-part prompt template—comprising an instruction statement, dialogue history, sentiment reasons, task statement, and experience example—to guide the LLM in analyzing the sentiment category of a sentence based on sentiment reasons, contextual dialogue, and relevant commonsense knowledge. The prompt components, except for the instruction statement, are tailored around the target sentence _ui_ . 

Instruction Statement: Defines the model’s role, details the dialogue sentiment analysis task, and standardizes the input format. 

Dialogue History: Accurate emotion detection relies on context. Unlike studies that use future dialogue [41,42], only prior dialogue is used here, limited by a hyperparameter (i.e., the history window, denoted as _w_ ) that is used in this section to denote the number of dialogue history sentences to be considered. For the target sentence _ui_ , the details of its dialogue history _u_ ( _i_ , _H_ ) are shown in Figure 3. 

Emotional Reasons: This study focuses on sentiment analysis based on emotional reasons. Previous methods lacked commonsense support, making it challenging to analyze sentiment categories based on emotional cues. To address this, the LLM is guided to better recognize the emotion of _ui_ by including _m_ relevant other’s reasons and _n_ self-reasons in the prompt, totaling _m_ + _n_ reason statements. However, retrieved reasons may be incomplete or lack explicit causal connections between historical utterances and the target sentence. To refine and enhance these reasons, BART is employed to generate an augmented explanation for each retrieved reason, leveraging its pre-trained generative capability to fill in missing causal links and improve reasoning coherence. The enhanced reason is formulated as follows: 

**==> picture [252 x 16] intentionally omitted <==**

where _rj_ is the original retrieved reason, and _hj_ represents its corresponding historical context. This augmentation ensures that the provided reasons are more interpretable, logically structured, and contextually aligned, improving sentiment classification. Details are shown in Figure 3. 

**==> picture [390 x 301] intentionally omitted <==**

**Figure 3.** The prompt consists of six components: (1) target sentence _ui_ , (2) instruction statement defining the model’s role and input format, (3) dialogue history providing contextual information within a history window _w_ , (4) emotional reasons incorporating _m_ other- and _n_ self-reasons, refined by BART, (5) task statement defining sentiment classification with predefined categories _L_ , and (6) an experience example retrieved and refined by BART for contextual alignment. 

Task Statement: The task statement reconstructs the sentiment analysis task by combining emotional reasons with the LLM generative. It confines the LLM’s output to a predefined set of sentiment categories _L_ = _{l_ 1, _l_ 2, . . . , _lλ}_ , facilitating statistical analysis. The task statement _u_ ( _i_ , _T_ ) focuses the LLM on categorizing the target sentence’s sentiment. Details are shown in Figure 3. 

Example of Experience: To enhance the LLM’s emotional understanding through context learning, the prompt provides a dynamically retrieved experience example _u_ ( _i_ , _E_ ) from the experience database. This example, similar in contextual semantics to the target sentence _ui_ , enables the LLM to analyze emotion more accurately using commonsense knowledge. Additionally, BART is leveraged to reconstruct and refine the retrieved experience example, ensuring that it aligns more effectively with the specific contextual and emotional aspects of _ui_ . This further enhances the LLM’s ability to perform nuanced sentiment classification. Details are shown in Figure 3. 

When analyzing the sentiment category of the target sentence _ui_ using the method based on combining the sentiment reasons with the LLM, we construct the input _xi_ of the target sentence _xi_ according to the prompt instruction template by splicing the instruction statement text _u_ ( _i_ , _I_ ), the dialogue history text _u_ ( _i_ , _H_ ), the BART-refined sentiment reason 

text _r[BART] j_ , the task statement text _u_ ( _i_ , _T_ ), and the experience example text _u_ ( _i_ , _E_ ) with the symbols [; ]. The input _xi_ of the target sentence _xi_ is constructed as follows: 

**==> picture [296 x 15] intentionally omitted <==**

To help the LLM better understand and adapt to dialogue sentiment analysis, we introduce an auxiliary task during fine-tuning: semantic interpretation of sentiment categories. This allows the LLM to deepen its understanding of sentiment categories and distinguish between similar ones, such as “happy” vs. “excited” or “sad” vs. “frustrated”. First, the most common semantic interpretations corresponding to the sentiment categories are obtained. For the set of sentiment categories in the dataset _L_ = _{l_ 1, _l_ 2, . . . , _lλ}_ , retrieve the set of the most common semantic interpretations _SI_ = _{si_ 1, _si_ 2, . . . , _siλ}_ corresponding to the sentiment categories from the sentiment lexicon SentiWordNet3.0 [43]: 

**==> picture [250 x 11] intentionally omitted <==**

During fine-tuning, after the LLM generates the sentiment category _e_ for the target sentence _ui_ , it generates the most common semantic interpretation _siK_ ( _ei_ ) for that sentiment category. This step helps the LLM differentiate sentiment categories. For example, if _ei_ : frustrated, the LLM generates the corresponding semantic interpretation _siK_ ( _ei_ ): disappointedly unsuccessful. To incorporate this semantic task in fine-tuning, the task statement _ui_ , _T_ in the prompt instruction is replaced with _ui[train]_ , _T_[, instructing the LLM: “Please first select] the sentiment category of the sentence _ui_ from _< l_ 1, _l_ 2, _· · ·_ , _lλ >_ , followed by the semantic interpretation corresponding to that sentiment category from _< si_ 1, _si_ 2, _· · ·_ , _siλ >_ ”. Therefore, during fine-tuning, the input _xi[train]_ for the target sentence _ui_ is constructed as follows: 

**==> picture [273 x 15] intentionally omitted <==**

_3.5. Training and Loss Functions_ 

In the LLM fine-tuning training phase, given an input sentence _ui_ , the instruction _xi[train]_ constructed according to the prompted instruction template is taken as input, and the LLM inference framework for dialogue sentiment analysis returns the logits value _gi_ of the entire complete sentence and the corresponding generated text _yi_ : 

**==> picture [246 x 13] intentionally omitted <==**

where _θ_ represents all trainable parameters of the LLM, including the Transformer layers, word embeddings, and output layers, controlling how the model processes input and generates output. The LLM optimizes _θ_ to learn language patterns and improve the quality and accuracy of generated text. _gi ∈_ R _[L][×][V]_ , _L_ and _V_ denote the length of the entire sentence and the size of the vocabulary used by the LLM, respectively. The LLM predicts the conditional probability _p_ ( _ti|xi[train]_ , _θ_ ) of each token _ti_ in the generated text _yi_ until the output of the end token _< eos >_ . Consistent with the original training goal of the LLM, we use the next labeled predictive loss function to train the model. Thus, the loss function for the main task of dialogue sentiment analysis is defined as follows: 

**==> picture [270 x 28] intentionally omitted <==**

where _ei_ denotes the sentiment category labeling corresponding to the target sentence _ui_ generated by the LLM, and _N_ denotes the number of all sentences contained in the dataset. 

The loss function used for the task of generating semantic interpretation of sentiment categories is defined as follows: 

**==> picture [283 x 29] intentionally omitted <==**

where _sir_ denotes the r-th token of the semantic interpretation corresponding to the sentiment category token _ei_ . Therefore, the overall loss function of the model training process is defined as follows: 

**==> picture [243 x 10] intentionally omitted <==**

where the hyperparameter _α_ is used to adjust the weight of the sentiment category semantic interpretation generation task loss in the overall fine-tuning training loss of the LLM. 

## **4. Experiments** 

_4.1. Setup_ 

4.1.1. Models and Datasets 

This study uses the datasets IEMOCAP [44], MELD [45], and DailyDialog [46] for the experiments, and the following is a detailed description of these three benchmark datasets. (1) The IEMOCAP dataset includes 151 conversations, 7433 sentences, 10 conversational roles, and six emotion categories, with 77% non-neutral emotions. Created by the SAIL Lab at USC, it contains two-person conversations between 10 professional actors, spanning five sections and 12 h of multimodal audio and video data. The dialogues consist of both fixed scripts and improvised scenarios. IEMOCAP is widely used in dialogue sentiment analysis due to its rich multimodal data and high-quality items. 

(2) The MELD dataset includes 1433 conversations, 13,708 sentences, and seven sentiment categories, with 53% non-neutral sentiment. This dataset, a multimodal extension of the EmotionLines dataset based on parts of the show _Friends_ , contains both text and video. MELD is commonly used for conversation sentiment analysis due to its high-quality data and multimodal content. 

(3) The DailyDialog dataset comprises 13,118 conversations across seven sentiment categories, four dialogue behavior types, and 10 topics, representing various daily life scenarios without fixed speaker roles. It is suitable for sentiment analysis, dialogue behavior analysis, and sentiment dialogue generation. The dataset’s main strength is its large volume and low noise, but it has a significant drawback—83% of the data is neutral sentiment. Only textual information was used in the experiments in this section. Detailed statistics are shown in Tables 2 and 3. 

**Table 2.** Statistical information on datasets. 

|**Dataset**|**Number of Dialogs**<br>**Number of Sentences**<br>**Train**<br>**Dev**<br>**Test**<br>**Train**<br>**Dev**<br>**Test**|
|---|---|
|IEMOCAP<br>MELD<br>DailyDialog|108<br>12<br>31<br>5163<br>647<br>1623<br>1039<br>114<br>280<br>9989<br>1109<br>2610<br>11,118<br>1000<br>1000<br>87,170<br>8069<br>7740|



**Table 3.** Dataset sentiment category information. 

|**Dataset**|**Classes**|**Sentiment Category**|
|---|---|---|
|IEMOCAP|6|happy, sad, neutral, angry, excited, frustrated|
|MELD|7|anger, disgust, fear, joy, neutral, sadness, surprise|
|DailyDialog|7|anger, disgust, fear, joy, neutral, sadness, surprise|



Sentiment analysis of conversations based on emotional reasons is a novel and advanced research area with limited related work. To demonstrate the effectiveness of the large-model-based explicit inference of emotional reasons for conversational sentiment analysis, this section compares it with traditional deep network methods, small pre-trained language model-based methods, and emotional reason-based methods. The LLM used in this study is Llama2-7B [47], fine-tuned with the LoRA approach [48]. LoRA is a parameter-efficient fine-tuning method. Instead of updating all model parameters, LoRA injects low-rank adaptation matrices into the Transformer layers while keeping the pretrained model weights frozen. This significantly reduces computational and memory costs while maintaining the model’s performance. By using LoRA, Llama2-7B can be effectively adapted for conversational sentiment analysis while requiring fewer trainable parameters compared to full fine-tuning. The baseline models for comparison are as follows: 

- COSMIC [36]: the first model that takes into account different categories of commonsense knowledge in a conversational sentiment analysis task and utilizes them to update conversational states. 

- DAG-ERC [34]: models the conversation structure as a directed acyclic graph, modeling both distant and proximate information interactions in a conversation. 

- DialogueCRN [9]: attempts to model intuitive retrieval and conscious reasoning processes by designing a multi-round reasoning module that iteratively performs the process of extracting and integrating emotional cues. 

- SKAIG [49]: uses the structure of a connectivity graph to enrich the representation of edges in the graph with commonsense knowledge, and enriches the representation of target utterances with past and future contextual information in the context. 

- CauAIN [10]: takes commonsense knowledge as the cause of emotion generation in dialogs and utilizes attentional mechanisms to update deeper representations of the target utterance in relation to emotion. 

- ERCMC [50]: uses the generated pseudo-future contexts in combination with historical contexts to improve emotion recognition in conversation. 

- UniMSE [51]: A multi-task learning framework for information extraction that leverages multiple data sources to generate structured outputs. It enhances extraction efficiency and accuracy by integrating a structured extraction language with a pretrained text-to-structure model. 

- InstructERC [52]: is a model for dialogue emotion recognition that uses large-scale language models to improve the accuracy of emotion recognition. The model enhances its understanding of emotions with two auxiliary tasks—speaker identification and emotion prediction. 

- Ref. [53]: uses commonsense knowledge to complement the contextual information contained in utterances and enrich the extracted conversation information. 

- CKERC [54]: is a novel emotion recognition in conversation (ERC) model that improves the accuracy of emotion recognition by combining large-scale language models (LLMs) and commonsense knowledge. 

## 4.1.2. Implementation Details 

In this paper, we use the Llama-2-7b macromodel3 from the model library provided by Hugging Face and fine-tune it using the LoRA method, with the learning rate set to 2 _×_ 10 _[−]_[2] and the hyperparameter _α_ set to 0.2. The size of the history window, _w_ , is set to 1, 5, 10, 15, and 20, respectively. _k_ is set to 5 to retrieve the number of candidate examples from the experience database, and _k_ is set to 2 to retrieve the number of other reasons statements, while the number of own reason statements ( _n_ ) varies depending on the dataset. Specifically, for the IEMOCAP dataset, both the other’s reasons and the own reasons are set 

to 2; for the MELD dataset, the other’s reasons are set to 1, and the own reasons are set to 3; for the DailyDialog dataset, other’s reasons remain 1, while the own reasons are set to 3. These settings are determined based on the characteristics of each dataset to optimize sentiment reasoning. 

As shown in Table 4, referring to previous work [34], this section also selects the weighted average F1 scores as the evaluation metrics for the datasets IEMOCAP and MELD, and for the dataset DailyDialog, the micro-averaged F1 scores are used as the evaluation metrics, but statements labeled neutral are excluded from the calculation of the results. 

**Table 4.** Information on indicators for the assessment of dataset indicators. 

|**Dataset**|**Metric**|**Addition**|
|---|---|---|
|**COSMIC**|weighted F1|-|
|**DAG-ERC**|weighted F1|-|
|**DialogueCRN**|weighted F1|w/o neutral category sentences|



## _4.2. Results and Analysis_ 

4.2.1. Overall Results 

The sentiment classification performance of the different models on the three publicly available benchmark test datasets is shown in Table 5. From the data in the table, it can be observed that the sentiment classification performance of the proposed dialogue sentiment analysis method based on combining sentiment reasons with the LLM is better than that of the comparative baseline models on all three datasets. 

**Table 5.** Comparison of sentiment classification performance of different models on the benchmark dataset. 

|**Dataset**|**IEMOCAP**<br>**MELD**<br>**DailyDialog**<br>**Weighted-F1**<br>**Acc**<br>**Weighted-F1**<br>**Micro-F1**<br>**Macro-F1**<br>**Micro-F1**|
|---|---|
|**COSMIC**[36]<br>**DAG-ERC**[34]<br>**DialogueCRN**[9]<br>**SKAIG**[49]<br>**CauAIN**[10]<br>**ERCMC**[50]<br>**UniMSE**[51]<br>**InstructERC**[52]<br>[53]<br>**CKERC**[54]|65.28<br>64.25<br>65.21<br>65.13<br>51.05<br>58.48<br>67.10<br>66.47<br>63.37<br>-<br>-<br>58.25<br>66.20<br>67.01<br>58.39<br>58.26<br>-<br>55.46<br>66.96<br>-<br>65.18<br>-<br>51.95<br>59.75<br>64.29<br>63.84<br>65.15<br>64.85<br>53.85<br>58.21<br>66.07<br>65.58<br>65.64<br>-<br>52.11<br>59.92<br>70.66<br>70.56<br>65.51<br>65.09<br>-<br>-<br>71.39<br>71.43<br>69.15<br>68.96<br>-<br>-<br>68.31<br>-<br>66.25<br>-<br>-<br>60.21<br>72.40<br>-<br>69.27<br>-<br>-<br>-|
|**CDEA**<br>**CDEA + llama**|66.92<br>66.44<br>65.73<br>66.59<br>54.29<br>60.44<br>**73.26**<br>**72.25**<br>**69.34**<br>**69.61**<br>**63.25**<br>**64.59**|



In the IEMOCAP dataset, dialogues contain numerous turns, rich contextual information, and frequent emotional interactions. Graph Neural Network (GNN)-based approaches perform well due to their ability to model these interactions. DAG-ERC uses a directed acyclic graph, closely matching the conversation patterns and achieving strong classification results. However, SKAIG, which incorporates external knowledge, performs worse due to the introduction of noisy information. Small pre-trained models struggle on this dataset due to limited input windows, while the method in this section mitigates this by leveraging structured external knowledge to identify emotional reasons. However, its 

performance is slightly behind GNN-based methods, though using an LLM with richer knowledge and generalization improves classification performance. 

In the MELD dataset, conversations are shorter with many speakers, so methods using small-scale pre-trained models or external knowledge perform better. LLMs handle the dataset’s complexity well, achieving top results. DialogueCRN underperforms, while CauAIN models speaker interactions use external commonsense, improving results. CDEA, building on this, uses the BART model to enhance classification performance further. 

In the DailyDialog dataset, real-world conversations make emotional category analysis more challenging, requiring precise speaker interaction modeling. Previous methods struggle here due to insufficient modeling of emotional interactions and lack of knowledge. SKAIG performs better by using a graph structure with external knowledge, but CDEA improves slightly by using the BART model for explicit reasoning. CDEA+Llama, leveraging the rich knowledge and generalization capabilities of LLMs, significantly boosts classification performance, demonstrating better reliability and generalization. 

4.2.2. Ablation Study 

- w/o Inter-Path: In the emotion cause detection module, we do not use the other’s reasoning path information provided by the structured machine commonsense map, and only recognize the other’s cause statements based on the semantic similarity, but we still use the own reasoning path information provided by the structured machine commonsense map to recognize the self cause statements that are consistent with the causality. 

- w/o Intra-Path: In the emotion cause detection module, instead of using the own reasoning path information provided by the structured machine commonsense map, it only recognizes the own cause statements based on the semantic similarity, but it still uses the other’s reasoning path information provided by the structured machine commonsense map to recognize the other’s cause statements that are consistent with the causal relationship. 

- w/o Inf-Path: Instead of using the inference path information provided by the structured machine commonsense map in the emotion cause detection module, the emotion cause statements are identified only by the semantic similarity between the historical statements and the target statements. 

In order to study the effect of using different inference path information in the structured machine general knowledge map to detect different emotional cause statements on recognizing emotional categories, the parts of the emotional cause detection module that recognize other people’s cause statements and own cause statements by other people’s inference path information and own inference path information, respectively, are removed in turn. Specifically, the corresponding inference path information in ATOMIC is discarded, and emotional reason statements are detected only by the semantic similarity between historical and target statements. The corresponding part of Table 6 shows that the results on all three datasets are somewhat reduced. This suggests that reasoning path information of others and self is crucial for recognizing causal reason statements, and further illustrates the importance of improving the model’s sentiment categorization performance by taking into account the sentiment reason statements. 

At the same time, the importance of explicitly and comprehensively modeling the speaker’s own and inter-speaker dependencies is also demonstrated. In addition, on the MELD dataset, the performance degradation of the model is particularly noticeable in the case of removing information about one’s own reasoning paths compared to removing information about others’ reasoning paths, which corresponds to the fact that the MELD dataset contains fewer utterances per conversation and more speakers, and on the other 

hand, demonstrates the generalization performance of the methodology proposed in this section on different datasets. The removal of the sentiment cause detection module means that the inference path information generated by COMET from ATOMIC is not introduced, and the identification of sentiment cause utterances is based only on the semantic similarity between the historical utterances and the target utterances. The decrease in model results demonstrates the importance of improving the performance of the dialogue sentiment analysis task by identifying sentiment reasons that are more consistent with causality. 

Further, in order to verify the validity of the LLM, this paper further develops the ablation experimental study, the results of which are shown in Table 7. 

- w/o Exper Demonstration: removing empirical examples from the input of the LLM, i.e., removing empirical examples dynamically selected from the empirical database based on the contextual semantics of a particular target utterance at the time of constructing the command. 

- w/o Label Paraphrasing: removing the auxiliary task of generating semantic interpretations of sentiment categories and only fine-tuning the LLM with the main task of dialogue sentiment analysis. 

- w/o LoRA: do not use LoRA to fine-tune the LLM, use the full-parameter finetuning approach. 

**Table 6.** CDEA ablation experiment results. 

||**IEMOCAP**|**MELD**|**DailyDialog**|
|---|---|---|---|
|CDEA|66.92|65.73|60.44|
|w/o Inter-Path|65.91|65.24|59.69|
|w/o Intra-Path|65.96|63.53|59.33|
|w/o Inf-Path|65.17|65.38|59.04|



**Table 7.** CDEA+llama ablation experiment results. 

||**IEMOCAP**|**MELD**|**DailyDialog**|
|---|---|---|---|
|CDEA+llama|73.26|69.34|64.59|
|w/o exper demonstration|70.65|67.29|63.68|
|w/o label paraphrasing|70.55|67.41|63.13|
|w/o LoRA|70.23|63.88|63.52|



From the results of the ablation studies, several conclusions can be drawn from this section: first, each module in the dialogue sentiment analysis approach based on combining sentiment factors with the LLM is an enhancement to the final performance of the LLM, i.e., by removing any module in the framework, the LLM’s ability to analyze the sentiments will be affected. After removing the empirical example retrieval module, the performance of the LLM drops dramatically on all datasets, which demonstrates the important role of retrieval examples in stimulating the LLM’s sentiment understanding. Second, after removing the auxiliary generation task of semantic interpretation of emotion categories, the performance decreases significantly, which is consistent with the conjecture of this paper, because the auxiliary generation task of semantic interpretation of emotion categories not only makes the LLM understand various emotion categories more deeply, but also enhances the ability to differentiate between various types of emotions; without this auxiliary task, the LLM’s understanding of emotion categories will be more vague, and the effect of emotion recognition will deteriorate. After fine-tuning the LLM without LoRA, the performance of the model also decreases, which indicates that LoRA can effectively prevent the LLM from overfitting. 

## 4.2.3. Hyperparametric Study 

Due to the limitation of the input size of the pre-trained language model, the number of historical statements, i.e., the window _w_ , cannot be infinitely large when constructing the input for the LLM. To explore its impact, we examine different history window sizes. The Llama2-7B LLM supports 20 rounds of conversation, whereas small-scale models are limited to five rounds. As shown in Figure 4, on the IEMOCAP dataset, the best performance is achieved with 15 historical utterances, while MELD and DailyDialog perform optimally with 10 utterances. Performance improves as the history window expands, particularly in IEMOCAP, where dialogues are longer. However, beyond a certain point, excessive context introduces noise, reducing classification accuracy—most notably in MELD and DailyDialog. 

**==> picture [386 x 196] intentionally omitted <==**

**Figure 4.** History window exploration experiment chart. 

Another key factor in conversational sentiment analysis is whether the distance of historical sentences within the window impacts sentiment classification. The results suggest that while earlier utterances in a conversation contribute to understanding the evolving emotional trajectory, their impact weakens as their distance from the target utterance increases. Specifically, recent utterances tend to have a stronger influence on sentiment prediction, while distant historical sentences may have diminished relevance. This is particularly evident in datasets such as MELD and DailyDialog, where shorter conversational structures mean that distant utterances are often less contextually relevant. In contrast, IEMOCAP, which features longer and more contextually connected conversations, benefits more from a longer history window before reaching its optimal performance. 

Beyond historical context, the speaker’s own previous utterances may also play a crucial role in determining sentiment. Sentiment is not only shaped by contextual interactions but also by the internal emotional consistency of a speaker. If a speaker maintains a consistent emotional tone over multiple turns, the model can leverage this self-consistency to make more accurate predictions. However, in conversations where the speaker’s emotions shift frequently—such as emotionally intense discussions or conflict-driven dialogues—relying on self-referential utterances may introduce ambiguity. Datasets like IEMOCAP, which contain expressive dialogues with emotional transitions, highlight cases where both the speaker’s and the interlocutor’s utterances must be jointly considered for optimal classification. 

Overall, these findings suggest that both the recency and relevance of historical sentences, as well as the emotional consistency of a speaker, impact the performance of conversational sentiment analysis models. While increasing the history window generally 

improves sentiment classification, careful balance is needed to prevent excessive noise and irrelevant information from degrading model performance. 

4.2.4. Comparative Experiments with Different LLM in Different Supervised Scenarios 

In order to gain a deeper understanding of the performance of different macromodels on the three benchmark datasets under different supervised scenarios, this section conducts experiments on the proposed method on the mainstream macromodels ChatGLM-6B and ChatGLM2-6B [55], and Llama-7B and Llama2-7B, respectively, under the settings of zero samples and LoRA. The experimental results are shown in Figure 5. 

The classification of the different large models in the zero-sample and the method setup of this section is shown in Figure 5. Even with the instructions designed in this paper that include the sentiment reason sentences and experience examples, the LLM still performs mediocrely in the zero-sample scenario, which further confirms that the large model cannot take advantage of the LLM’s rich commonsense knowledge and powerful generalization ability when directly applied to the task of dialogue sentiment analysis. 

**==> picture [511 x 117] intentionally omitted <==**

**Figure 5.** Here, ( **a** ) represents the comparison of LLM classification performance with zero samples and the method setup of this study; ( **b** ) represents the comparison of LLM classification performance with the LoRA setup; and ( **c** ) represents the comparison of LLM classification performance with the LoRA and the method setup of this study. 

Compared to the zero-sample context learning strategy, fine-tuning the LLM using LoRA not only preserves the rich common sense knowledge inherent in the LLM, but also significantly improves the model’s performance on the task of dialogue sentiment analysis. This demonstrates the effectiveness of the LoRA fine-tuning approach in enhancing the adaptability of large pre-trained models for specific tasks. 

Finally, by applying the methodology proposed in this study to the LLMs under the LoRA setting, the performance of all four LLMs is significantly improved, especially on the IEMOCAP dataset. This demonstrates the effectiveness and generalization of the dialogue sentiment analysis framework based on the combination of sentiment reasons and LLMs in this section, which greatly enhances the ability of LLMs to understand the sentiments in long texts. 

## _4.3. Case Study_ 

In Figure 6, a case from the IEMOCAP test set is used as an example to illustrate the important role of accurately identifying cause sentences that are consistent with causality when detecting the sentiment of a target statement. The situation is the process of a man who fails to make a credit card payment and seeks help from the official human customer service. It is easy to notice that there is no direct sentiment descriptor in the target sentence #16. Therefore, the emotion “happy” should be inferred from the dialogue context. Through the emotion cause detection module, the emotion cause statements are detected. Self-caused statement #1 expresses the initial emotional state of anger, while other-caused statement #13 asks the man to call back if there is any problem with the bill after checking 

it, which causes the man’s worry. Self-caused statement #14 expresses that the man does not want to be tortured by the intelligent customer service anymore. However, other-cause statement #15 completely allays the man’s concerns, and the human customer service agent will give the man their number so that he can call back directly next time. The self-causal and other-causal statements that are consistent with the causal relationship provide an important discriminatory basis for the model. 

**==> picture [390 x 262] intentionally omitted <==**

**Figure 6.** Case study 1: real examples of successful model predictions. 

Table 8 presents the results of our dialogue sentiment analysis method, which integrates affective reasons with the LLM, compared to three baseline models. The results demonstrate how LLM’s commonsense knowledge and generalization ability enhance sentiment classification reliability. Bolded sentences indicate target sentences whose sentiment categories are to be identified. 

**Table 8.** Case Study 2: The table below provides a dialogue sample in which Chandler’s emotions are predicted, illustrating the impact of LLM-driven reasoning. Incorrect predictions are marked ✕, while correct predictions are marked ✓. Target sentences (i.e., Chandler’s utterances that require sentiment classification) are highlighted in bold. 

## **Contents of the Dialog** 

Joey: Oh, yeah, yeah, sure. We live in the building by the uh sidewalk. (neutral) Chandler: You know it? (surprise) Joey: Hey, look, since we are neighbors and all, what do you say we uh, get together for a drink? (neutral) **Chandler: Oh, sure, they love us over there. (neutral)** Joey: Ben! Ben! Ben! (neutral) 

|**Chandler: Oh, sure, they love us over there. (neutral)**<br>Joey: Ben! Ben! Ben! (neutral)||
|---|---|
|Model|Prediction|
|DialogueCRN|surprise✕|
|CKERC|joy✕|
|ECERN|joy✕|
|**ECERN + llama**|**neutral**✓|



DialogueCRN struggles with causal reasoning, focusing too much on surrounding statements. CKERC and ECERN improve in this aspect but still lack sufficient commonsense support. In contrast, our method effectively leverages LLM’s commonsense knowledge to accurately identify emotion causes, leading to more precise sentiment classification. 

## _4.4. Module Time Consumption Analysis_ 

To evaluate the efficiency of our proposed model, we conducted a module-level time consumption analysis on the DailyDialog dataset, which consists of 13,118 conversations covering seven sentiment categories, four dialogue behavior types, and 10 topics. This dataset represents diverse real-life conversations is widely used in sentiment analysis, dialogue behavior analysis, and sentiment dialogue generation. 

For this experiment, we randomly sampled 5000 sentences from the validation set of DailyDialog and measured the execution time of each system module, including sentiment cause sentence acquisition, dynamic retrieval of experience examples, and prompt instruction construction with model fine-tuning. The average processing time per sentence (ms/sentence) for each module is reported in Table 9. 

**Table 9.** Module-level time consumption analysis on DailyDialog. 

|**Module**|**Phase**|**Average Time (ms/sentence)**|
|---|---|---|
|Dialogue History Preprocessing|Sentiment Cause Sentence Acquisition|42|
|Sentiment Cause Detection (Self and Other)|Sentiment Cause Sentence Acquisition|185|
|Experience Example Retrieval|Dynamic Retrieval of Experience Examples|160|
|BART-Based Experience Refnement|Dynamic Retrieval of Experience Examples|225|
|Prompt Construction|Prompt Instruction Construction and Fine-tuning|105|
|LLM Inference|Prompt Instruction Construction and Fine-tuning|3492|
|Total System Runtime|-|4209|



The total system runtime averages 4209 ms/sentence, with the overall system efficiency meeting practical application requirements. Future optimizations in model inference speed could further enhance real-time performance, making the system more suitable for largescale deployment. 

## **5. Conclusions** 

In this paper, we propose a method for analyzing the sentiment of dialogues based on emotional reasons to improve the accuracy and reliability of sentiment classification in dialogue systems. First, a model based on explicit inference of sentiment reasons is proposed, which integrates and infers sentiment reasons by extracting inference path information from structured commonsense maps and combining causal associations between historical and target sentences. Secondly, the method of combining large-scale pre-trained language models is further proposed to accurately analyze the emotional causes by constructing an experience database and prompt engineering and utilizing the common sense knowledge and generalization ability of an LLM. Experimental results show that this paper’s method outperforms existing methods on multiple benchmark datasets, significantly improves the performance of the dialogue sentiment analysis task, and validates its effectiveness. 

Although the sentiment analysis method for dialogue based on emotional reasons proposed in this paper achieves significant performance improvement on several benchmark datasets, it still has some limitations. First, the method in this paper is limited to textual data, whereas human emotions are expressed in a variety of forms, and in addition to text, multimodal information such as voice tones, facial expressions, and body movements are also important carriers of emotion transfer. Therefore, purely text-based sentiment analysis cannot fully capture the richness of human emotional expression. Future work 

should focus on developing a dialogue sentiment analysis model based on multimodal data, combining multiple data sources such as speech, image, and text. This will not only improve the machine’s ability to understand emotions, but also be more applicable to complex application scenarios in real life, such as emotional chatbots and intelligent assistants, to realize a higher level of emotional intelligence. 

**Author Contributions:** Conceptualization, X.Z. (Xue Zhang) and M.W.; methodology, X.Z. (Xue Zhang); software, X.Z. (Xue Zhang), X.Z. (Xuyi Zhuang) and Q.L.; validation, X.Z. (Xue Zhang), X.Z. (Xue Zhang) and X.Z. (Xiao Zeng); formal analysis, Q.L.; investigation, M.W.; resources, M.W.; data curation, X.Z. (Xue Zhang); writing—original draft preparation, X.Z. (Xue Zhang); writing—review and editing, X.Z. (Xue Zhang) and M.W.; visualization, X.Z. (Xue Zhang); supervision, M.W.; project administration, M.W.; funding acquisition, M.W. All authors have read and agreed to the published version of the manuscript. 

**Funding:** This research received no external funding. 

**Data Availability Statement:** Data will be made available on request. 

**Acknowledgments:** During the course of this research and the preparation of the manuscript, the open-source large language model LLaMA2-7B was primarily employed to support emotion cause identification, empirical example retrieval, prompt template construction, and model fine-tuning for sentiment classification tasks. Additionally, by combining instruction tuning with structured prompt engineering, the large language model significantly enhanced the causal reasoning capability and classification accuracy in dialogue sentiment analysis. All generated content was reviewed and revised by the authors, who assume full responsibility for the content of this publication. 

**Conflicts of Interest:** The authors declare no conflicts of interest. 

## **References** 

1. Zhou, H.; Huang, M.; Zhang, T.; Zhu, X.; Liu, B. Emotional chatting machine: Emotional conversation generation with internal and external memory. In Proceedings of the AAAI Conference on Artificial Intelligence, New Orleans, LA, USA, 2–7 February 2018; Volume 32. 

2. Kumar, A.; Dogra, P.; Dabas, V. Emotion analysis of Twitter using opinion mining. In Proceedings of the 2015 Eighth International Conference on Contemporary Computing (IC3), Noida, India, 20–22 August 2015; IEEE: New York, NY, USA, 2015; pp. 285–290. 

3. Pujol, F.A.; Mora, H.; Martínez, A. Emotion Recognition to Improve E-Healthcare Systems in Smart Cities. In Proceedings of the Research & Innovation Forum 2019: Technology, Innovation, Education, and Their Social Impact, Athens, Greece, 10–12 April 2019; Springer: Berlin/Heidelberg, Germany, 2019; pp. 245–254. 

4. Poria, S.; Majumder, N.; Mihalcea, R.; Hovy, E. Emotion recognition in conversation: Research challenges, datasets, and recent advances. _IEEE Access_ **2019** , _7_ , 100943–100953. 

5. König, A.; Francis, L.E.; Malhotra, A.; Hoey, J. Defining affective identities in elderly nursing home residents for the design of an emotionally intelligent cognitive assistant. In Proceedings of the 10th EAI International Conference on Pervasive Computing Technologies for Healthcare, Cancun, Mexico, 16–19 May 2016; pp. 206–210. 

6. Strapparava, C. WordNet-Affect: An affective extension of WordNet. In Proceedings of the 4th International Conference on Language Resources and Evaluation (LREC 2004), Lisbon, Portugal, 26–28 May 2004. 

7. Mohammad, S.M.; Turney, P.D. Crowdsourcing a word—Emotion association lexicon. _Comput. Intell._ **2013** , _29_ , 436–465. 

8. Lian, Z.; Sun, L.; Xu, M.; Sun, H.; Xu, K.; Wen, Z.; Chen, S.; Liu, B.; Tao, J. Explainable multimodal emotion reasoning. _arXiv_ **2023** , arXiv:2306.15401. 

9. Hu, D.; Wei, L.; Huai, X. Dialoguecrn: Contextual reasoning networks for emotion recognition in conversations. _arXiv_ **2021** , arXiv:2106.01978. 

10. Zhao, W.; Zhao, Y.; Lu, X. CauAIN: Causal Aware Interaction Network for Emotion Recognition in Conversations. In Proceedings of the IJCAI, Vienna, Austria, 23–29 July 2022; pp. 4524–4530. 

11. Schachter, S.; Singer, J. Cognitive, Social, and Physiological Determinants of Emotional State. _Psychol. Rev._ **1962** , _69_ , 379–399. [CrossRef] 

12. Scherer, K.R. _Appraisal Processes in Emotion: Theory, Methods, Research_ ; Oxford University Press: New York, NY, USA, 2001. 

13. Majumder, N.; Poria, S.; Hazarika, D.; Mihalcea, R.; Gelbukh, A.; Cambria, E. Dialoguernn: An attentive rnn for emotion detection in conversations. In Proceedings of the AAAI Conference on Artificial Intelligence, Honolulu, HI, USA, 27 January–1 February 2019; Volume 33, pp. 6818–6825. 

14. Zhang, D.; Chen, X.; Xu, S.; Xu, B. Knowledge aware emotion recognition in textual conversations via multi-task incremental transformer. In Proceedings of the 28th International Conference on Computational Linguistics, Barcelona, Spain, 13–18 September 2020; pp. 4429–4440. 

15. Jiao, W.; Yang, H.; King, I.; Lyu, M.R. Higru: Hierarchical gated recurrent units for utterance-level emotion recognition. _arXiv_ **2019** , arXiv:1904.04446. 

16. Ma, H.; Wang, J.; Qian, L.; Lin, H. HAN-ReGRU: Hierarchical attention network with residual gated recurrent unit for emotion recognition in conversation. _Neural Comput. Appl._ **2021** , _33_ , 2685–2703. 

17. Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F.L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. Gpt-4 technical report. _arXiv_ **2023** , arXiv:2303.08774. 

18. Bhaumik, A.; Strzalkowski, T. Towards a Generative Approach for Emotion Detection and Reasoning. _arXiv_ **2024** , arXiv:2408.04906. 19. Xie, S.M.; Raghunathan, A.; Liang, P.; Ma, T. An explanation of in-context learning as implicit bayesian inference. _arXiv_ **2021** , arXiv:2111.02080. 

20. Reddy, G.R.; Reddy, M.S.; Stanlywit, M.; Khaleel, S. Emotion detection from text and analysis of future work: A survey. _Riv. Ital. Filos. Anal. Jr._ **2023** , _14_ , 59–73. 

21. Zhou, Z.H. A brief introduction to weakly supervised learning. _Natl. Sci. Rev._ **2018** , _5_ , 44–53. [CrossRef] 

22. Sujadi, C.C.; Sibaroni, Y.; Ihsan, A.F. Analysis content type and emotion of the presidential election users tweets using agglomerative hierarchical clustering. _Sink. J. Dan Penelit. Tek. Inform._ **2023** , _7_ , 1230–1237. [CrossRef] 

23. Mahesh, B. Machine Learning Algorithms—A Review. _Int. J. Sci. Res. (IJSR)_ **2020** , _9_ , 381–386. [CrossRef] 

24. Rafath, M.A.H.; Mim, F.T.Z.; Rahman, M.S. An analytical study on music listener emotion through logistic regression. _World Acad. J. Eng. Sci._ **2021** , _8_ , 15–20. 

25. Bengio, Y.; Ducharme, R.; Vincent, P.; Janvin, C. A Neural Probabilistic Language Model. _J. Mach. Learn. Res._ **2003** , _3_ , 1137–1155. 26. Sarzy´nska-Wawer, J.; Wawer, A.; Pawlak, A.; Szymanowska, J.; Stefaniak, I.; Jarkiewicz, M.; Okruszek, L. Detecting formal thought disorder by deep contextualized word representations. _Psychiatry Res._ **2021** , _304_ , 114135. [CrossRef] 

27. Radford, A.; Narasimhan, K. Improving Language Understanding by Generative Pre-Training. OpenAI Technical Report. 2018. Available online: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf (accessed on 19 January 2025). 

28. Kenton, J.D.M.W.C.; Toutanova, L.K. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Minneapolis, MN, USA, 2–7 June 2019; Volume 1, pp. 4171–4186. 

29. Wan, B.; Wu, P.; Yeo, C.K.; Li, G. Emotion-cognitive reasoning integrated BERT for sentiment analysis of online public opinions on emergencies. _Inf. Process. Manag._ **2024** , _61_ , 103609. [CrossRef] 

30. Abu Farha, I.; Magdy, W. A Comparative Study of Effective Approaches for Arabic Sentiment Analysis. _Inf. Process. Manag._ **2021** , _58_ , 102438. [CrossRef] 

31. Bello, A.; Ng, S.C.; Leung, M.F. A BERT framework to sentiment analysis of tweets. _Sensors_ **2023** , _23_ , 506. [CrossRef] 

32. Ghosal, D.; Majumder, N.; Poria, S.; Chhaya, N.; Gelbukh, A. Dialoguegcn: A graph convolutional neural network for emotion recognition in conversation. _arXiv_ **2019** , arXiv:1908.11540. 

33. Ishiwatari, T.; Yasuda, Y.; Miyazaki, T.; Goto, J. Relation-Aware Graph Attention Networks with Relational Position Encodings for Emotion Recognition in Conversations. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Association for Computational Linguistics, Online, Originally Scheduled in Punta Cana, Dominican Republic, 16–20 November 2020; pp. 7360–7370. 

34. Shen, W.; Wu, S.; Yang, Y.; Quan, X. Directed acyclic graph network for conversational emotion recognition. _arXiv_ **2021** , arXiv:2105.12907. 

35. Zhong, P.; Wang, D.; Miao, C. Knowledge-enriched transformer for emotion detection in textual conversations. _arXiv_ **2019** , arXiv:1909.10681. 

36. Ghosal, D.; Majumder, N.; Gelbukh, A.; Mihalcea, R.; Poria, S. Cosmic: Commonsense knowledge for emotion identification in conversations. _arXiv_ **2020** , arXiv:2010.02795. 

37. Bosselut, A.; Rashkin, H.; Sap, M.; Malaviya, C.; Celikyilmaz, A.; Choi, Y. COMET: Commonsense transformers for automatic knowledge graph construction. _arXiv_ **2019** , arXiv:1906.05317. 

38. Sap, M.; Le Bras, R.; Allaway, E.; Bhagavatula, C.; Lourie, N.; Rashkin, H.; Roof, B.; Smith, N.A.; Choi, Y. ATOMIC: An Atlas of Machine Commonsense for If-Then Reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, Honolulu, HI, USA, 27 January–1 February 2019; AAAI Press: Honolulu, HI, USA, 2019; Volume 33, pp. 3027–3035. 

39. Luo, M.; Xu, X.; Liu, Y.; Pasupat, P.; Kazemi, M. In-context learning with retrieved demonstrations for language models: A survey. _arXiv_ **2024** , arXiv:2401.11624. 

40. Zahiri, S.M.; Choi, J.D. Emotion detection on TV show transcripts with sequence-based convolutional neural networks. In Proceedings of the AAAI Workshops, New Orleans, LA, USA, 2–7 February 2018; Volume 18, pp. 44–52. 

41. Zhang, T.; Kishore, V.; Wu, F.; Weinberger, K.Q.; Artzi, Y. Bertscore: Evaluating text generation with bert. _arXiv_ **2019** , arXiv:1904.09675. 42. Lian, Z.; Liu, B.; Tao, J. CTNet: Conversational Transformer Network for Emotion Recognition. _IEEE/ACM Trans. Audio Speech Lang. Process._ **2021** , _29_ , 985–1000. 

43. Baccianella, S.; Esuli, A.; Sebastiani, F. SentiWordNet 3.0: An Enhanced Lexical Resource for Sentiment Analysis and Opinion Mining. In Proceedings of the 7th International Conference on Language Resources and Evaluation (LREC 2010), Valletta, Malta, 17–23 May 2010; European Language Resources Association (ELRA): Valletta, Malta, 2010; pp. 2200–2204. 

44. Busso, C.; Bulut, M.; Lee, C.C.; Kazemzadeh, A.; Mower, E.; Kim, S.; Chang, J.N.; Lee, S.; Narayanan, S.S. IEMOCAP: Interactive Emotional Dyadic Motion Capture Database. _Lang. Resour. Eval._ **2008** , _42_ , 335–359. 

45. Poria, S.; Hazarika, D.; Majumder, N.; Naik, G.; Cambria, E.; Mihalcea, R. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. _arXiv_ **2018** , arXiv:1810.02508. 

46. Li, Y.; Su, H.; Shen, X.; Li, W.; Cao, Z.; Niu, S. DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset. _arXiv_ **2017** , arXiv:1710.03957. 

47. Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. LLaMA: Open and Efficient Foundation Language Models. _arXiv_ **2023** , arXiv:2302.13971. 

48. Hu, E.J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W. LoRA: Low-Rank Adaptation of Large Language Models. _arXiv_ **2021** , arXiv:2106.09685. 

49. Li, J.; Lin, Z.; Fu, P.; Wang, W. Past, Present, and Future: Conversational Emotion Recognition through Structural Modeling of Psychological Knowledge. In Proceedings of the Findings of the Association for Computational Linguistics: EMNLP 2021, Punta Cana, Dominican Republic, 16–20 November 2021; pp. 1204–1214. 

50. Wei, Y.; Liu, S.; Yan, H.; Ye, W.; Mo, T.; Wan, G. Exploiting Pseudo Future Contexts for Emotion Recognition in Conversations. _arXiv_ **2023** , arXiv:2306.15376. 

51. Lei, S.; Dong, G.; Wang, X.; Wang, K.; Wang, S. InstructERC: Reforming Emotion Recognition in Conversation with a Retrieval Multi-Task LLMs Framework. _arXiv_ **2023** , arXiv:2309.11911. 

52. Hu, D.; Bao, Y.; Wei, L.; Zhou, W.; Hu, S. Supervised Adversarial Contrastive Learning for Emotion Recognition in Conversations. _arXiv_ **2023** , arXiv:2306.01505. 

53. Yang, Z.; Li, X.; Cheng, Y.; Zhang, T.; Wang, X. Emotion Recognition in Conversation Based on a Dynamic Complementary Graph Convolutional Network. _IEEE Trans. Affect. Comput._ **2024** , _15_ , 1567–1579. 

54. Fu, Y. CKERC: Joint Large Language Models with Commonsense Knowledge for Emotion Recognition in Conversation. _arXiv_ **2024** , arXiv:2403.07260. 

55. Du, Z.; Qian, Y.; Liu, X.; Ding, M.; Qiu, J.; Yang, Z.; Tang, J. GLM: General Language Model Pretraining with Autoregressive Blank Infilling. _arXiv_ **2021** , arXiv:2103.10360. 

**Disclaimer/Publisher’s Note:** The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content. 

