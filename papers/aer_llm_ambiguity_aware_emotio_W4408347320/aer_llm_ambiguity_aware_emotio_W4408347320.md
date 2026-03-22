# AER-LLM: Ambiguity-aware Emotion Recognition Leveraging Large Language Models 

## Xin Hong 

_School of Computing and Information Systems University of Melbourne_ Melbourne, Australia honxh1@student.unimelb.edu.au 

Yuan Gong _[†]_ 

_CSAIL SLS Massachusetts Institute of Technology_ Boston, US yuangong@mit.edu 

## Vidhyasaharan Sethu 

_School of Electrical Engineering and Telecommunications University of New South Wales_ 

Sydney, Australia v.sethu@unsw.edu.au 

## Ting Dang 

_School of Computing and Information Systems University of Melbourne_ 

Melbourne, Australia ting.dang@unimelb.edu.au 

_**Abstract**_ **—Recent advancements in Large Language Models (LLMs) have demonstrated great success in many Natural Language Processing (NLP) tasks. In addition to their cognitive intelligence, exploring their capabilities in emotional intelligence is also crucial, as it enables more natural and empathetic conversational AI. Recent studies have shown LLMs’ capability in recognizing emotions, but they often focus on single emotion labels and overlook the complex and ambiguous nature of human emotions. This study is the first to address this gap by exploring the potential of LLMs in recognizing ambiguous emotions, leveraging their strong generalization capabilities and in-context learning. We design zero-shot and few-shot prompting and incorporate past dialogue as context information for ambiguous emotion recognition. Experiments conducted using three datasets indicate significant potential for LLMs in recognizing ambiguous emotions, and highlight the substantial benefits of including context information. Furthermore, our findings indicate that LLMs demonstrate a high degree of effectiveness in recognizing less ambiguous emotions and exhibit potential for identifying more ambiguous emotions, paralleling human perceptual capabilities.** 

_**Index Terms**_ **—emotion recognition, ambiguous emotion, large language models, prompt design, multimodal.** 

## I. INTRODUCTION 

Recent advancements in large language models (LLMs) [1] have shown remarkable abilities in comprehending, interpreting, and generating human-like text. This cognitive intelligence facilitates effective human-AI interactions of conversational AI. Equally important is the emotional and social intelligence, which enables it to understand human emotions and adapt its communication accordingly. 

Text-based emotion recognition has shown considerable potential, with various feature engineering techniques and advanced deep learning models [2]–[4]. With the emerging capabilities of LLMs, particularly their proficiency in in-context learning and robust generalization without extensive training, research on exploring their potential for emotion recognition and further use as annotation tools have attracted increasing attentions [5]–[8]. Nonetheless, these investigations predominantly focus on recognizing single emotions, thereby overlooking the complex nature of human emotions. Typically, a single emotion label is obtained from the majority vote of multiple annotators labeling the same stimuli. This approach disregards discrepancies among annotators, which indicates the inherent ambiguity of emotions. High ambiguity, i.e., high disagreement, indicates more complex emotions, and such ambiguity impacts conversations, leading to modified communication strategies and affecting 

_†_ Yuan Gong completed this work at MIT and is now with xAI Corp. 

relationship dynamics [9]. For example, the listener might use more cautious language and tones or avoid sensitive topics to prevent misunderstandings if high emotional ambiguity is perceived. Future LLMs need to understand the complexity of emotions, recognize emotional ambiguity, and adjust their responses dynamically. 

LLMs, trained on diverse and large-scale datasets, enable semantic richness and offer significant potential in comprehending the complexity of emotions. Additionally, their long-range contextual understanding allows them to decode emotions through in-context learning by analyzing conversational history, which is particularly noteworthy. This study aims to explore the potential of LLMs in recognizing inherently ambiguous emotions and the contributions are summarized below: 

- This is the first study to analyze LLMs in recognizing ambiguityaware emotions, demonstrating their potential for human-like emotional intelligence. 

- We proposed zero-shot and few-shot prompt designs and incorporated in-context learning capabilities to enhance recognition, demonstrating an average 35% relative improvements in terms of Bhattacharyya coefficient. 

- We further included speech features as textual prompt to enhance learning, further enhancing the ambiguous emotion recognition. 

- Further analysis concerning different levels of ambiguity revealed that LLMs are more effective at recognizing less ambiguous emotions and less effective with more ambiguous ones. 

This investigation offers valuable insights into emotional intelligence in LLMs and could potentially advance the development of more natural and empathetic conversational AI systems through dynamic emotional responses. 

## II. RELATED WORK 

A range of feature sets and modeling frameworks have been developed to advance text-based emotion recognition. TF-IDF, which highlights frequent keywords to indicate emotional cues, has been effectively used in emotion classification [10]. Subsequently, word embeddings have demonstrated greater effectiveness in capturing semantic relationships, such as Word2Vec, GloVe, and BERT [2]–[4]. Deep learning models have also advanced, with Bi-Gated Recurrent Units (GRUs), Convolutional Neural Networks (CNNs) [4], and hierarchical Long Short-Term Memory (LSTM) networks [2] all showing 

**==> picture [253 x 122] intentionally omitted <==**

Fig. 1: System overview. _Ai, i ∈_ [1 _, N_ ] represents the _i[th]_ annotator, _L_ represent the evaluation metrics, including both ambiguity-centric and accuracy-centric metrics. 

effectiveness. However, their effectiveness is highly dependent on a large annotated dataset for training [4], [11]. 

Recent advances in LLMs, such as LLAMA3 [12] and GPT4 [13], have opened up new possibilities for text understanding and analysis. Their strong generalization capabilities enable effective text comprehension to recognize emotions without the need for extensive retraining [5], [6], [8], [14], [15]. One recent study [6] compared the emotion labels recognized by GPT-4 to human annotators and found that labels generated by GPT-4 were preferred by human evaluators. Another work proposed InstructERC [8], which treats emotion recognition in conversations as a retrieval-based Seq2Seq paradigm utilizing LLMs to infer emotions from conversation history. 

Despite the promises, existing studies all focus on the single emotion label obtained from majority vote and have not studied the inherent ambiguity of emotions. Previous studies on non-LLM models accounting for ambiguity in emotions either use the ambiguity in the loss function as an enhancement for the majority vote recognition [16] or treat ambiguous emotions as an out-of-distribution (OOD) separate class [17]. The recent study based on LLMs [6] includes data with multi-label classes, but the focus is still on single label recognition, and majority of the data is also single-labeled. None of these studies carefully represent emotions with ambiguity or explore the potential of LLMs in recognizing ambiguity-aware emotions, especially their strong capabilities in in-context learning. 

## III. AMBIGUOUS EMOTION RECOGNITION VIA LLMS 

## _A. System overview_ 

Fig. 1 illustrates our framework for recognizing ambiguous emotions using LLMs. For each target utterance, we construct detailed prompts for ambiguous emotion recognition. To evaluate the performance of the LLMs, we compare the predicted emotion distribution _p_ ˆ( _x_ ) with the ground truth distribution _p_ ( _x_ ), which is inferred from _N_ different human annotators _A_ 1 to _AN_ . 

## _B. Prompt design_ 

_1) Zero-shot and few-shot prompting:_ Carefully designed zeroshot (ZS) and few-shot (FS) prompting are key for LLMs to generalize across various domains [18]. Zero-shot prompting evaluates the capability of pre-trained knowledge in LLMs. Few-shot prompting includes a limited set of demonstration examples within the prompts, facilitating the adaptation of pre-trained knowledge to specific new tasks. We design zero-shot prompt as outlined in Table I and Eq. (1): 

**==> picture [211 x 9] intentionally omitted <==**

**Background** (BG) provides information about the conversation scenario, while **Context** (C) incorporates the retrospective dialogue 

TABLE I: Zero-shot prompt template with context = M 

||**Prompt Template**|
|---|---|
|**Background (BG)**|Two speakers are talking.|
|**Context (M=3)**|The conversation is:<br>_•_ Ses01<br>F: ”We could hide away.”<br>_•_ Ses01<br>M: ”Run away?”<br>_•_ Ses01<br>F: ”Mm hmm. We’ll build a bunker<br>and never come out.”|
|**Target**<br>**utterance**<br>**(TU)**|Now Ses01<br>M says: ”I really don’t want to go, I<br>don’t want to go...”|
|**Task**|Predict the probability of the emotion of the sen-<br>tence from the options [neutral, happy, angry, sad],<br>consider the conversation context.|
|**Output**<br>**constraints (OC)**|Output satisfes the following rules.<br>_•_ Rule 1: Generate a dictionary of emo-<br>tion probabilities in format of _{_’neutral’:0.1,<br>’happy’:0.0, ’angry’:0.1, ’sad’:0.8_}_. If you<br>think there is only one emotion in the sen-<br>tence, then give the probability to 1.<br>_•_ Rule 2: Ensure the sum of probability equal<br>to 1.<br>_•_ Rule 3: Do not explain, only need the dictio-<br>nary.<br>Please check again whether your output follows the<br>three rules.|



TABLE II: Examples in few-shot prompt 

||**Prompt Template**|
|---|---|
|**Examples (Exps)**|Examples:<br>_•_ Sentence<br>1:<br>Ses01<br>F:<br>”We<br>could<br>hide<br>away.”. Emotion probabilities: _{_’Sadness’:<br>0.33, ’Happiness’: 0.33, ’Neutral state’: 0.33_}_<br>_•_ Sentence 2: Ses01<br>M: ”Run away?” Emo-<br>tion probabilities: _{_”Sadness’: 0.67, ’Neutral<br>state’: 0.33_}_<br>_•_ Sentence 3: Ses01<br>F: ”Mm hmm. We’ll build<br>a bunker and never come out.” Emotion<br>probabilities: _{_’Sadness’: 0.67, ’Happiness’:<br>0.33_}_|
|with _M_ past consecutive utterances. The **Target Utterance** (TU)<br>indicates the sentence that requires prediction, and the **Task** presents<br>the question of probability distribution prediction. The fnal **Output**<br>**Constraints** (OC) offers specifc instructions to generate the correct<br>form of the distributions.||



In terms of few-shot, we additionally included a few **Examples (Exps)** in the prompt design, as shown in Table II. The corresponding ambiguous emotion labels in terms of probabilities for these examples are provided to the LLMs for learning. The FS prompting with additional examples is shown in Eq. (2). 

**==> picture [228 x 9] intentionally omitted <==**

_2) Prompt with speech features:_ As humans express emotions through multiple cues, we further included speech in addition to text for ambiguity-emotion recognition. We transformed speech features into text format and incorporated them into the prompt design. Since LLMs have been trained extensively on text data, they are expected to understand speech information in text format, e.g., high pitch values: 4. Specifically, we extracted 88-dimensional eGeMAPS features [19], a standard acoustic parameter set. It described the speech features of the target utterance in textual format, as illustrated in Table III, with 

TABLE III: Speech features in textual format in prompt 

||**Prompt Template**|
|---|---|
|**Speech**<br>**Features**<br>Here are 88 speech features of the current speaker’s<br>sentence. The features are: Average Fundamental Fre-<br>quency in Semitones from 27.5 Hz: 37.039505 ...||



the full prompts shown in Eq. (3) and (4). 

**==> picture [223 x 9] intentionally omitted <==**

**==> picture [243 x 10] intentionally omitted <==**

## _C. Context-aware recognition_ 

A major advantage of LLMs lies in their capability for in-context learning, offering the potential to analyze long past conversation histories for emotion recognition. Since emotions generally evolve smoothly within dynamic conversations, considering past conversations provides a more comprehensive understanding of the emotional state over time. A longer context window enables LLMs to effectively decode information over extended ranges, and we formally study this by increasing the number _M_ of context windows in the prompt design, i.e., including the corresponding text information from the past _M_ utterances, and evaluated the performance accordingly. 

## IV. EXPERIMENTAL SETUP AND RESULTS 

## _A. Experimental setup_ 

_1) Dataset:_ Three datasets are used, MSP-Podcast [20], IEMOCAP [21] and GoEmotions [22]. MSP-Podcast contains recordings cover a wide range of subjects in Podcasts. We selected four emotional labels: neutral, angry, happy, and sad. Any utterances annotated by annotators outside of these four categories will be excluded from the analysis, leading to 4114 utterances. We manually reorganized sentences into the original full podcast to enable dynamic information. For IEMOCAP, we also selected the same four emotional labels, resulting in a total of 4370 utterances. Examples in few-shot prompting are from the same session with the target utterance. 

GoEmotions is a text-based dataset sourced from Reddit. Given the long-tail distribution of emotion labels, we selected the 4 most common labels (admiration, gratitude, approval, and amusement) along with Neutral. To ensure adequate representation of ambiguous emotions (more than one label per utterance), we applied log inverse frequency weighting and selected 210 instances, with 33% being multi-labeled.[1] 

_2) Models:_ Gemini-1.5-Flash was chosen as the LLM backbone due to its capability for processing long-range contexts, with a context window of up to one million tokens. The experiments were conducted using the Gemini API [23]. For few-shot prompts, we tested 5 and 10 examples in GoEmotions. In MSP-Podcast and IEMOCAP, we matched the number of few-shot examples to the context window, varying the context window within [0,30][2] . 

_3) Evaluations:_ We include both uncertainty-centric and accuracycentric metrics to evaluate our model’s performance. For uncertaintycentric metrics, we use Jensen-Shannon Divergence (JS), Bhattacharyya coefficient (BC), and _R_[2] . JS divergence measures the difference between predicted and ground truth distributions, with lower values indicating better predictions. BC measures the similarity between these distributions, and _R_[2] assesses the goodness-of-fit, with higher values indicating better performance. Additionally, we 

> 1GoEmotions contains only single utterances without context information. We randomly sampled specific sentences as examples for few-shot prompting. 

> 2Code link: https://github.com/mHealthUnimelb/AER-LLM. 

TABLE IV: Performance on ambiguity-aware emotion prediction. T represents text and S represents speech. 

||Modality<br>Prompt|JS_↓_<br>BC_↑_<br>R2_↑_|ECE_↓_|
|---|---|---|---|
|MSP-Podcast|T<br>ZS<br>FS|0.56<br>0.39<br>0.41<br>0.42<br>0.58<br>0.54|0.62<br>0.42|
||T+S<br>ZS<br>0.45<br>0.54<br>0.52<br>0.47<br>FS<br>**0.40**<br>**0.61**<br>**0.56**<br>**0.40**|||
|IEMOCAP|T<br>ZS<br>FS|0.51<br>0.46<br>0.49<br>0.37<br>0.67<br>0.58|0.56<br>0.30|
||T+S<br>ZS<br>FS|0.47<br>0.51<br>0.51<br>**0.35**<br>**0.69**<br>**0.59**|0.51<br>**0.29**|
|GoEmotions|T<br>ZS<br>FS|0.49<br>0.54<br>0.43<br>**0.44**<br>**0.60**<br>**0.48**|0.47<br>**0.39**|



estimate Expected Calibration Error (ECE), where smaller values denote better calibration of probabilistic predictions. All four metrics range within [0,1]. For accuracy-based metrics, we obtain a single label from the predicted distribution by selecting the maximum probability and compared to the majority vote of the labels to estimate accuracy, F1-score and unweighted average recall (UAR). 

_4) Baseline descriptions:_ Baseline [24] utilizes pretrained embeddings for single emotion recognition, while [25] treats ambiguous emotion sentences as additional out-of-distribution class for ambiguous emotion recognition. The other three studies [5], [6], [8] focus on LLMs for single emotion recognition tasks, with [8] building a two-step system. 

## _B. Performance on ambiguity-aware prediction_ 

Table IV demonstrates the performance on three datasets. For the text modality, the zero-shot prompt achieves relatively acceptable performance, e.g., _R_[2] of 0.41 for MSP-Podcast, 0.49 for IEMOCAP, and 0.43 for GoEmotions. With few-shot prompting, it demonstrates significant improvement, with approximately 25% reduction in JS, 49% increase in BC, and 31% increase in _R_[2] for MSP-Podcast. Similar increasing trend is also observed in IEMOCAP and GoEmotions. This suggests that LLMs are strong in learning from few examples for ambiguous emotion recognition, and leveraging their in-context learning capabilities by looking at past examples is highly beneficial. 

For the joint modeling of text and speech modality, both zeroshot and few-shot demonstrate significant improvement compared to the corresponding performance using text only, suggesting the LLMs’ capabilities in recognizing speech information despite being in textual format. Furthermore, the few-shot prompting consistently outperforms the zero-shot prompting, exhibiting superior performance for all three datasets. The consistent trend in ECE also implies a similar effect on probability calibrations, suggesting an improved interpretation of ambiguity. The disparities are stable because LLM typically provide consistent responses. 

## _C. Impact of context window_ 

Fig. 2 shows the few-shot performance using text and speech with the increasing context windows from 0 to 30 in MSP-Podcast. Including context information proves significantly beneficial compared to that without contextual information. When the window size increases from 0 to 5, we observe a 16%, 28%, 22% and 19% improvements in terms of JS, BC, _R_[2] and ECE, respectively. As the context window expands beyond 10 to 30, the observed improvements become marginal, indicating that further increasing the context window size beyond 10 may not substantially enhance ambiguous emotion recognition. In the IEMOCAP dataset, the most benefit was obtained when the context window increases to 20. These 

**==> picture [227 x 129] intentionally omitted <==**

Fig. 2: Performance comparison with increasing context windows using text and speech in MSP-Podcast. 

**==> picture [253 x 96] intentionally omitted <==**

Fig. 3: Performance comparison among different levels of ambiguity in MSP-Podcast. A small entropy indicates less ambiguous emotion. 

findings collectively suggest that incorporating context information is crucial for ambiguous emotion recognition in LLMs, and a context window of 10 to 20 is likely to be adequate. This is reasonable, as humans do not need infinite context to recognize emotion. 

## _D. Performance with different ambiguity levels_ 

To gain deeper insights into how LLMs recognize varying levels of ambiguity in emotions, we evaluated their performance with respect to different ambiguity levels. We used entropy, inferred from the ground truth distributions, as an indicator of the ambiguity levels. As entropy increases, the utterance exhibits greater ambiguity in emotion. An entropy of 0 indicates unanimous agreement on one emotion class. As shown in Fig. 3, six majority entropy with each more than 100 utterances are shown. 

As entropy increases, the medians (black lines) of JS rise, while BC and R[2] decrease in general, except when entropy is 0.7219. The trend indicates that LLMs are more effective at recognizing less ambiguous emotions. This is likely due to the inherent complexity of predicting a high entropy emotional distribution. This observation aligns with the human difficulty in recognizing more ambiguous emotions [26]. Better performance is observed with an entropy of 0.7219, as we found that 92% of the utterances are annotated with at least a neutral class, more than in neighbouring entropy groups, and LLMs recognize neutral with a 75% true positive rate, leading to higher performance. 

## _E. Prediction on majority vote_ 

We further estimated the single emotion from the distribution by selecting the one with the highest probability and compared it with the majority vote. Noted that the prompt design for LLMs is not optimized for recognizing the majority vote, this analysis is designed to provide insights into LLMs capabilities in understanding the most likely emotion. It is not a direct comparison due to the slightly different data and tasks. 

In Table V, the best performances on MSP-Podcast, IEMOCAP and GoEmotions achieve 56.15%, 59.06% and 51.05% in terms of W-F1, respectively, demonstrating emotional understanding of LLMs. Note 

TABLE V: Performance on majority vote prediction, selected as the class with the maximum probability in the predicted distribution. 

||Modality|Methods|ACC<br>W-F1<br>UAR|
|---|---|---|---|
|MSP-Podcast|T|ZS<br>FS|35.24<br>42.78<br>46.46<br>**50**<br>**53.23**<br>**48.31**|
||T+S|Pretrained [24]<br>MLLMs [5]|-<br>-<br>50.0<br>-<br>-<br>**52.59**|
|||ZS<br>FS|50.55<br>50.4<br>44.21<br>**55.58**<br>**56.15**<br>46.88|
|IEMOCAP|T|InstructERC [8]|-<br>53.38<br>-|
|||ZS<br>FS|43.36<br>42.92<br>52.25<br>**57.87**<br>**58.43**<br>**65.39**|
||T+S|MLLMs [5]<br>Pretrained [25]|-<br>-<br>50.36<br>**78.12**<br>-<br>-|
|||ZS<br>FS|48.12<br>49.18<br>55.20<br>58.75<br>**59.06**<br>**65.68**|
|GoEmotion|T|GPT-4 [6]|-<br>-<br>48.5|
|||ZS<br>FS|37.14<br>35.68<br>44.74<br>**50.48**<br>**51.05**<br>**52.98**|



**==> picture [215 x 131] intentionally omitted <==**

Fig. 4: Comparison of W-F1 across five entropy groups with context window = 0 and 30 using MSP-Podcast. 

that the prompt is not designed for single-label emotion recognition and LLMs is not trained, but it still achieves comparable performance to the models specifically trained for single-label emotion recognition. The accuracy with respect to the context window and the entropy in MSP-Podcast is further demonstrated in Fig. 4. Notably, there is no majority label when entropy is 1.5219 as two classes share the same probabilities. Compared to a context window of 0, a context window of 30 achieved higher accuracy across all entropy groups. Additionally, better performance is observed when entropy is 0, indicating no ambiguity, compared to high entropy of 1.371, which corresponds to high ambiguity. We also observe a similar pattern in IEMOCAP, with a more consistent increasing trend as the context window increases and a decreasing trend as the entropy increases. 

## V. CONCLUSION 

We investigated LLMs potential in recognizing ambiguous emotions and discovered that it can comprehend such emotions to a certain extent without additional training. Incorporating previous dialogues by leveraging LLMs in-context learning capabilities significantly enhances its emotional intelligence, with a context window of 10 to 20 being adequate. Moreover, these models demonstrate greater proficiency in recognizing less ambiguous emotions compared to highly ambiguous ones, similar to human perception. These findings highlight the potential of LLMs for application in emotional conversational AI. 

- [1] J. Santoso, K. Ishizuka, and T. Hashimoto, “Large language modelbased emotional speech annotation using context and acoustic feature for speech emotion recognition,” in _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 11026–11030, IEEE, 2024. 

- [2] C. Huang, A. Trabelsi, and O. R. Za¨ıane, “Ana at semeval-2019 task 3: Contextual emotion detection in conversations through hierarchical lstms and bert,” _arXiv preprint arXiv:1904.00132_ , 2019. 

- [3] P. Agrawal and A. Suri, “Nelec at semeval-2019 task 3: think twice before going deep,” _arXiv preprint arXiv:1904.03223_ , 2019. 

- [4] S. K. Bharti, S. Varadhaganapathy, R. K. Gupta, P. K. Shukla, M. Bouye, S. K. Hingaa, and A. Mahmoud, “Text-based emotion recognition using deep learning approach,” _Computational Intelligence and Neuroscience_ , vol. 2022, no. 1, p. 2645381, 2022. 

   - [22] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi, “Goemotions: A dataset of fine-grained emotions,” _arXiv preprint arXiv:2005.00547_ , 2020. 

   - [23] M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou, O. Firat, J. Schrittwieser, _et al._ , “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,” _arXiv preprint arXiv:2403.05530_ , 2024. 

   - [24] Z. Aldeneh and E. M. Provost, “You’re not you when you’re angry: Robust emotion features emerge by recognizing speakers,” _IEEE Transactions on Affective Computing_ , vol. 14, no. 2, pp. 1351–1362, 2021. 

   - [25] W. Wu, C. Zhang, and P. C. Woodland, “Distribution-based emotion recognition in conversation,” in _2022 IEEE Spoken Language Technology Workshop (SLT)_ , pp. 860–867, IEEE, 2023. 

   - [26] T.-H. Lee, M. T. Perino, N. L. McElwain, and E. H. Telzer, “Perceiving facial affective ambiguity: A behavioral and neural comparison of adolescents and adults.,” _Emotion_ , vol. 20, no. 3, p. 501, 2020. 

- [5] T. Feng and S. Narayanan, “Foundation model assisted automatic speech emotion recognition: Transcribing, annotating, and augmenting,” in _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pp. 12116–12120, IEEE, 2024. 

- [6] M. Niu, M. Jaiswal, and E. M. Provost, “From text to emotion: Unveiling the emotion annotation capabilities of llms,” _INTERSPEECH_ , 2024. 

- [7] A. N. Tak and J. Gratch, “Is gpt a computational model of emotion?,” in _2023 11th International Conference on Affective Computing and Intelligent Interaction (ACII)_ , pp. 1–8, IEEE, 2023. 

- [8] S. Lei, G. Dong, X. Wang, K. Wang, and S. Wang, “Instructerc: Reforming emotion recognition in conversation with a retrieval multitask llms framework,” _arXiv preprint arXiv:2309.11911_ , 2023. 

- [9] L. K. Knobloch, “Relational uncertainty and interpersonal communication,” _New directions in interpersonal communication research_ , pp. 69– 93, 2010. 

- [10] D. E. Cahyani and I. Patasik, “Performance comparison of tf-idf and word2vec models for emotion text classification,” _Bulletin of Electrical Engineering and Informatics_ , vol. 10, no. 5, pp. 2780–2788, 2021. 

- [11] N. Alswaidan and M. E. B. Menai, “A survey of state-of-the-art approaches for emotion recognition in text,” _Knowledge and Information Systems_ , vol. 62, no. 8, pp. 2937–2987, 2020. 

- [12] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozi`ere, N. Goyal, E. Hambro, F. Azhar, _et al._ , “Llama: Open and efficient foundation language models,” _arXiv preprint arXiv:2302.13971_ , 2023. 

- [13] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, _et al._ , “Gpt-4 technical report,” _arXiv preprint arXiv:2303.08774_ , 2023. 

- [14] Y. Fu, “Ckerc: Joint large language models with commonsense knowledge for emotion recognition in conversation,” _arXiv preprint arXiv:2403.07260_ , 2024. 

- [15] Y. Hu, S. Zhang, T. Dang, H. Jia, F. D. Salim, W. Hu, and A. J. Quigley, “Exploring large-scale language models to evaluate eeg-based multimodal data for mental health,” _WellComp co-located with UbiComp 2024_ , 2024. 

- [16] Y. Zhou, X. Liang, Y. Gu, Y. Yin, and L. Yao, “Multi-classifier interactive learning for ambiguous speech emotion recognition,” _IEEE/ACM transactions on audio, speech, and language processing_ , vol. 30, pp. 695–705, 2022. 

- [17] W. Wu, B. Li, C. Zhang, C.-C. Chiu, Q. Li, J. Bai, T. N. Sainath, and P. C. Woodland, “Handling ambiguity in emotion: From out-of-domain detection to distribution estimation,” _arXiv preprint arXiv:2402.12862_ , 2024. 

- [18] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, _et al._ , “Gemini: a family of highly capable multimodal models,” _arXiv preprint arXiv:2312.11805_ , 2023. 

- [19] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. Andr´e, C. Busso, L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan, _et al._ , “The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing,” _IEEE transactions on affective computing_ , vol. 7, no. 2, pp. 190–202, 2015. 

- [20] L. Martinez-Lucas, M. Abdelwahab, and C. Busso, “The mspconversation corpus,” _Interspeech 2020_ , 2020. 

- [21] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “Iemocap: Interactive emotional dyadic motion capture database,” _Language resources and evaluation_ , vol. 42, pp. 335–359, 2008. 

