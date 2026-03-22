**==> picture [26 x 26] intentionally omitted <==**

**==> picture [81 x 21] intentionally omitted <==**

**==> picture [56 x 21] intentionally omitted <==**

**==> picture [53 x 21] intentionally omitted <==**

**==> picture [13 x 13] intentionally omitted <==**

Latest updates: hps://dl.acm.org/doi/10.1145/3746277.3760412 

## RESEARCH-ARTICLE 

## **Multimodal Trait and Emotion Recognition via Agentic AI: An End-toEnd Pipeline** 

**OM DABRAL** , Manipal University Jaipur, Jaipur, RJ, India **SWAYAM BANSAL** , Manipal University Jaipur, Jaipur, RJ, India **MRIDUL MAHESHWARI** , Manipal University Jaipur, Jaipur, RJ, India **HARDIK SHARMA** , Manipal University Jaipur, Jaipur, RJ, India **JASPREET SINGH** , Manipal University Jaipur, Jaipur, RJ, India **BAGESH KUMAR** , Manipal University Jaipur, Jaipur, RJ, India 

**PDF Download 3746277.3760412.pdf 22 March 2026 Total Citations:** 1 **Total Downloads:** 275 

**==> picture [21 x 20] intentionally omitted <==**

**Published:** 26 October 2025 

**Citation in BibTeX format** 

MM '25:The 33rd ACM International Conference on Multimedia _October 31, 2025 Dublin, Ireland_ 

**Conference Sponsors:** SIGMM 

**Open Access Support** provided by: 

**Manipal University Jaipur** 

# **Multimodal Trait and Emotion Recognition via Agentic AI: An End-to-End Pipeline** 

Om Dabral om.229301177@muj.manipal.edu Manipal University Jaipur Jaipur, India 

Swayam Bansal bansalswayam2005@gmail.com Manipal University Jaipur Jaipur, India 

## Mridul Maheshwari 

mridul.23fe10ite00041@muj.manipal.edu Manipal University Jaipur Jaipur, India 

Hardik Sharma 

## Jaspreet Singh 

Bagesh Kumar bagesh.kumar@jaipur.manipal.edu Manipal University Jaipur Jaipur, India 

hardik.23fe10ite00064@muj.manipal.edu Manipal University Jaipur Jaipur, India 

jaspreet.23fe10ite00297@muj.manipal.edu Manipal University Jaipur 

Jaipur, India 

## **Abstract** 

_learning approaches_ ; **Learning paradigms** ; • **Human-centered computing** → **Human computer interaction (HCI)** . 

This paper describes an end-to-end, modular, agentic system for inferring personality traits and emotional states based on multimodal interview-style data, when behavioral metadata (response time, body language, speech features, etc.) are made available. The system architecture consisted of a Perception Agent for classifying the emotions, an Inference Agent for estimating Big Five personality traits, and a Dialogue Agent for producing psychologically informed responses. A retrieval-augmented memory module connected the three agents to maintain context and continuity in the dialogue. The agents used two different language model backbones, LLaMA 3.2 1B and Falcon-RW-1B, learning and reasoning within the same processing pipeline while evaluating their performance on a benchmark of 19 annotated queries across a range of emotional states, conversation scenarios, and context. The evaluation assessed processing latency, lexical diversity, and consistency of trait estimation. The key finding for this study was that integrating metadata and modular reasoning leads to more contextually relevant and empathic responses than just prompting the agent. LLaMA produced longer, richer, and more diverse output while Falcon scores a much lower latency with shorter, predictable, and consistent responses. Overall, this study demonstrates that the combination of multimodal affective signals and agentic reasoning can lead to better human–AI interaction. While the work presented here is a proof-of-concept that is constrained by a benchmark experiment, it lays the grounds for scaling response models beyond the types of responses evaluated in this study, introduces more evaluation metrics, and enables more robust personalization features in future dialogue systems. 

## **Keywords** 

Multimodal Emotion Recognition, Personality Trait Inference, Agentic AI, Multi-Agent Systems, Human-AI Interaction, Affective Computing, Dialogue Generation, Few-Shot Prompting, Chain-of-Thought Reasoning, Behavioral Metadata, Emotionally Adaptive Agents, Self-Auditing AI Systems 

**ACM Reference Format:** 

Om Dabral, Swayam Bansal, Mridul Maheshwari, Hardik Sharma, Jaspreet Singh, and Bagesh Kumar. 2025. Multimodal Trait and Emotion Recognition via Agentic AI: An End-to-End Pipeline. In _Proceedings of the 1st International Workshop on Cognition-oriented Multimodal Affective and Empathetic Computing (CogMAEC ’25), October 27–28, 2025, Dublin, Ireland._ ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3746277.3760412 

## **1 Introduction** 

Emotionally adaptive AI systems are becoming necessary for many applications like interview coaching, mental health chatbots, and personalized tutoring, where detecting and understanding human affect and intent is important to the dialogue. Most dialogue systems utilize only text data for their inputs and respond using static generation processes, because the subtle behavioral signals associated with emotion and other affect variables are unattainable with static generation processes and data processed exclusively in text. 

To assist the emotional dimensions of the interaction, the emphasis should be on the user to, at the very least, initiate and sustain the emotional dialogue or states. 

Multimodal affective computing has made progress in incorporating verbal, prosodic, and visual modalities together [1, 7]. Research investigating the agentic nature of AI has shown how large language models (LLMs) can be thought of as modular agents with perception, reasoning, and reflection abilities [3, 11]. However, not many efforts have merged these ideas into a single coherent end-to-end pipeline that interprets multimodal data and outputs psychologically grounded conversational responses. 

## **CCS Concepts** 

• **Computing methodologies** → **Natural language processing** ; **Natural language generation** ; _Information extraction_ ; _Machine_ 

In this paper, we demonstrate a new pipeline for Multimodal Trait and Emotion Recognition via Agentic AI, from interview-style user input that has behavioral metadata (e.g., response time, body posture, and speech features). There are three agents in the system: a Perception Agent that classifies emotional valence, an Inference 

This work is licensed under a Creative Commons Attribution 4.0 International License. _CogMAEC ’25, Dublin, Ireland_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2059-8/2025/10 https://doi.org/10.1145/3746277.3760412 

**==> picture [242 x 161] intentionally omitted <==**

**Figure 1: A metadata-enriched user input example showing how behavioral cues like response time, body posture, and speech features are integrated alongside the user’s verbal query to support affective inference.** 

Agent that generates estimates of Big Five personality traits through chain-of-thought prompting, and a Dialogue Agent that generates contextually aware responses based on emotional resonation. The agents operate in the following agentic loop: Observe → Reflect → Act → Self-Audit, which allows for adaptively refining both the inference and dialogue over multiple turns. 

To assess our methodology, we carry out a series of benchmarks of dialogue generation performance across two base LLMs—LLaMA 3.2 1B and tiiuae/falcon-rw-1b—tracking models across emotional alignment, trait consistency, and human-rated empathy and coherence. The results show that our multi-agent pipeline consistently outperforms flat prompting methods and agentic baselines on multiple evaluated artifacts, including MAE (trait estimation) and human-rated overall narrative dialogue satisfaction. Coherence varies by model as a result of embedding and model size/content selection, but the agentic control framework on LLaMA demonstrates flexibility at a lower computational cost. Overall, the agent’s confident performance across emotional-linguistic alignment, trait estimation, and triadic dialogue satisfaction highlights the potential for future affective dialogue systems that leverage multi-agent reasoning with trait-aware response generation [15]. 

Ultimately, this work presents a modular and extensible architecture for affective sensitivity in AI, combining multimodal systems of sensing, trait-aware reasoning, and agentic management of dialogue into a singular architecture with real-time processing controls. 

## **2 Related Work** 

## **2.1 Multimodal Personality and Emotion Inference** 

Early work on personality and emotion recognition focused primarily on unimodal data, especially text or speech signals. However, most unimodal models struggle to capture the true richness and complexity of human affect, which is inherently multimodal. More recent research has explored multimodal learning by integrating 

text, visual, and audio data. For instance, Agrawal et al. [1] developed a cross-attention transformer model that fused transcript, audio, and behavioral features, including handcrafted behavioral embeddings. Their model improved recognition accuracy for the Big Five personality traits on the First Impressions V2 dataset and demonstrated the advantages of jointly modeling temporal and behavioral data. 

In their analysis of multimodal models, de Boer [7] explored interpretability using Self-Supervised Embedding Feature Transformers (SSE-FT). The study showed that even state-of-the-art multimodal models often rely predominantly on a single modality to make predictions, a phenomenon known as unimodal collapse. This underscores the need for improved multimodal alignment and fusion strategies and highlights the untapped potential of nonverbal and multi-agreement data streams. 

Additionally, recent models such as Emo-FuSense and H-MMER have demonstrated impressive performance on emotion recognition tasks involving physiological and audiovisual modalities [5]. These models leverage Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and attention mechanisms to capture both temporal and static features, achieving accuracy scores above 90% on benchmark datasets. 

## **2.2 Agentic AI and Multi-Agent Architectures** 

While LLMs have disrupted many approaches in real-world tasks in NLP, and spawned new ones like prompt engineering, the use of LLMs in agentic systems, where many system components reason carefully and unfold over time, is still an untapped potential. Jiang et al. [11] provide an uptodate tutorial on how we evolve from large AI models (LAMs) to Agentic AI. They introduce (slightly) differentiated frameworks in which agents have planners, memory, and tool integration for long-horizon decision-making. Agent-X [3], a more recent benchmark that builds on this idea, benchmarks multi-step tool-augmented reasoning on vision-based tasks. The authors find that all modern LLM-based agents (GPT, Gemini, etc.) have weaknesses in multi-hop reasoning across modalities, with hard visual tasks yielding success rates under 50%. This begs the question, how can we build modular agents that collaborate across periods of processing, reflecting, and acting? This problem is what our proposed multi-agent system is attempting to solve. 

## **2.3 Emotionally Adaptive Dialogue Systems** 

Most research in the context of dialogue has centered on generating semantically well-formed responses; few works integrate inferring personality and emotion profiles of users into dialogue generation. The PEEGM framework [13] describes emotion model- and personality-aware conversational agents that tailor responses based not only on inferred personality or emotional traits, but also on conversational context. Bravo et al. [5] provide a thorough review of AI-based multimodal dialogue systems that recognize emotion, noting the spectrum of systems and that few systems integrate affective understanding with response generation at deep levels. 

Our proposed system expands on these developments with a traitaware, emotion-sensitive dialogue agent that uses agentic loops to improve the understanding of users and response generation continuously. 

Multimodal Trait and Emotion Recognition via Agentic AI: An End-to-End Pipeline 

## **3 System Archetecture** 

The model we propose is an end-to-end pipeline for emotion and personality trait inference from enhanced textual interaction data, followed by response generation based on psychological knowledge. We have taken an agentic multi-stage vision where each agent is paired with a supportive task inside a larger reasoning/generating loop, because we wish the architecture to actuate an agentic structure that has modularity, interpretability, and adaptive response across dialogue turns because we want to focus on emotionally sensitive tasks like interviewing, coaching and assistance interactions. 

As a general process, the system moves through stages: the system receives a query from the user along with contextual metadata the emotional states is inferred to predict; personality traits are inferred; relevant examples from the past are retrieved; and the answer is generated based on the context that has built up across the process. Each of these stages is performed by an agent (or skilled cognitive task), and the agents are part of a coordinated Agent Work Flow Loop, also based on feedback information . This architecture is illustrated in Figure 3 [3, 12]. 

**==> picture [242 x 161] intentionally omitted <==**

**Figure 2: Flowchart of metadata-enriched input processed by the Perception Agent for emotion inference and the Inference Agent for Big Five personality trait estimation using LLM prompting and chain-of-thought reasoning.** 

## **3.1 Input Layer and Metadata Enrichment** 

The pipeline commences with the enriched text input, representing a user message in a quasi-interview-like situation. Unlike traditional dialogue systems that simply view input as text, we leverage behavioral metadata: 

- Response Time: whether the user answered fast, slow, or normally. 

- Body Language: inferred states, for example “relaxed,” “fidgeting,” and “arms crossed,” taken from emotion labels. 

- Speech Features: features we extract, e.g, loudness, hesitancy, excitement, based on punctuation and structure. 

This multimodal contextualization has the effect of producing text that is psychologically and affectively more grounded, even while the main source/modal is text [4, 8]. 

## **3.2 Perception Agent** 

The Perception Agent acts as the system’s emotional classifier. It is responsible for measuring the user’s state-of-mind categorically (e.g., Happy, Sad, Angry, Anxious, Neutral) and using both the user’s utterance and the associated metadata, applies a prompt to LLaMA 3.2 1B model with a custom prompt format that incorporates emotional reasoning and even context awareness. 

The Perception Agent allows the system to function with emotionally tagged dialogue states, which facilitates trait inference and dynamic response modulation. 

## **3.3 Inference Agent** 

The Inference Agent determines the user’s personality profile according to the Big Five (OCEAN) trait model. It takes into account the user’s text and inferred emotion as well as behavior context to generate trait estimates (values between 0.0 and 1.0) for: 

- Openness to Experience 

- Conscientiousness 

- Extraversion 

- Agreeableness 

## • Neuroticism 

Trait estimates are delivered using chain-of-thought prompting using a hybrid approach that assumes some combination of modelgenerated outputs and a rule-based heuristics system. This approach is preferred in retaining interpretability and robustness, especially in conditions when the LLM output is noisy or underspecified. The traits will serve as a psychological signature to be used for recognizing the style of response delivery of the Dialogue Agent. 

## **3.4 Retrieval Memory Module** 

The system’s external memory store is known as the Retrieval Memory Module, and it serves as an external store of memory, holding a continuously growing history of all past user-agent interactions, including queries, inferred emotions and traits, and responses generated by the agent. The retrieval system employs a composite of TF-IDF vectorizing for content and one-hot embedding for emotions and traits to return the top-k most contextually similar past cases based on the current query context [15]. 

The previous cases are included to enrich the prompt that a Dialogue Agent receives, to provide additional contextual grounding to the user, and to introduce retrieval-augmented generation that responds to a user’s unique behavioral tendencies. 

## **3.5 Dialogue Agent** 

The last stage is supervised by the Dialogue Agent, which generates the response for the system. The Dialogue Agent takes as input: 

- the current user query 

- inferred emotion and trait profile 

- similar past interaction records (if applicable) 

The Dialogue Agent is instantiated using different LLM backbones, i.e., LLaMA 3.2 1B and Falcon-RW-1B, so we can make empirical comparisons of response quality across architectures. Each model is prompted using a structured format that is rich in metadata and includes attribution of emotions and personality, which allows the agent to generate contextually rich emotive replies. 

In addition to mood and tone tracking, the interface has the capabilities to adjust the tone of the language to that of the presumed personality of the user. An example might be high neuroticism, which may lead to calmer language use, or high openness, which may prompt sillier or more abstract language use. 

## **3.6 Agentic Workflow Loop** 

**==> picture [242 x 161] intentionally omitted <==**

**Figure 3: Architecture of the multimodal trait and emotion recognition pipeline, showing the Perception Agent, Inference Agent, Dialogue Agent, and their interlinked Agentic Workflow Loop.** 

At the heart of the system is the Agentic Workflow Loop as motivated by cognitive theories of reflective reasoning. The loop comprises four steps: 

- Observe: metadata accumulation from the input as well as environmental context 

- Reflect: forms the reasoning associated with the current and historical state correlated with signals of emotions and personality 

- Act: generates a behavior (or response) based on the output of reasoning 

Self-Audit: review interaction drift or whether user feedback was present over time, permitting the agent to revise its inferences or strategies. 

Overall, it allows for multi-turn adaptability where the system supports the user’s needs as well as improves the agent’s behavior during future interactions. Each turn of the dialogue simultaneously updates the scope of the context, with regard to what is adoptable. 

- Body Language: annotated descriptors (e.g., relaxed, fidgeting, arms crossed) 

- Speech Features: Loudness, Hesitancy, or Excitement levels, noted by observers 

Emotion categories (e,g, happy, sad, angry, anxious, neutral) and Big Five personality trait scores were heuristically estimated using keyword- and pattern-based scripts over this static corpus. This dataset served as a retrieval memory module and as the baseline for initial evaluations. 

**Dynamic Dataset:** During operation, the agentic system continuously collected dynamic data from ongoing interactions. Each new user message, along with the agent’s inferences and response, was recorded in a structured JSON format: 

[ node distance=1.5cm and 2.2cm, every node/.style=draw, rounded corners, align=center, font=, ] (root) JSON Object; (transcript) [below left=of root] transcript _string_ ; (emotion) [below right=of root] emotion _string_ ; (traits) [below=of root] traits {}; (response) [below=of traits] response _string_ ; (timestamp) [below=of response] timestamp _float_ ; (processing) [below=of timestamp] processing_time _float_ ; 

(open) [below left=of traits] openness _float_ ; (cons) [below=of open] conscientiousness _float_ ; (extra) [below right=of traits] extraversion _float_ ; (agree) [below=of extra] agreeableness _float_ ; (neuro) [below=of agree] neuroticism _float_ ; 

(root) – (transcript); (root) – (emotion); (root) – (traits); (traits) – (response); (response) – (timestamp); (timestamp) – (processing); (traits) – (open); (traits) – (cons); (traits) – (extra); (traits) – (agree); (traits) – (neuro); 

**Figure 4: Compact schematic of the JSON data structure for conversation metadata, traits, and responses.** 

This dynamic dataset was continuously appended as the agent ran in real-time, implementing a “self-audit” framework whereby each new turn could become part of retrieval for future queries, and where agent performance metrics could be monitored over time. In this way, the dynamic dataset supported continual learning, adaptation, and empirical measurement of the agent’s evolving behavior. 

## **4.2 Prompt Engineering** 

## **4 Methodology** 

## **4.1 Data Preparation** 

Our data resources were separated into two datasets to support both static retrieval and dynamic adaptation. 

**Static Dataset:** We prepared an interview-style dataset with 31 participants, each answering 61 questions, producing 1891 questionresponse pairs. Each entry was enriched with behavioral metadata including: 

- Response Time: Fast, Slow, or Moderate, recorded manually 

For consistent evaluation, we applied prompt-based instructions to both the LLaMA 3.2 1B and Falcon-RW-1B model backbones. The same functional architecture was preserved across models, but differences arose in specific prompting details due to Falcon’s smaller size and practical performance [13]. 

_4.2.1 Emotion Classification Prompts._ For LLaMA, the Perception Agent used a single-label deterministic prompt with chain-of-thought style instructions, embedding metadata about body language, speech, and response time to guide the classification. The language model 

was constrained to respond with a single word from a predefined emotion list, with rule-based fallbacks for ambiguous cases [6]. For Falcon, emotion classification was handled with a few-shot style prompt, including 4-5 labeled examples before the user query. This improved consistency on smaller models. If the Falcon model response was ambiguous or failed, a fallback keyword-based heuristic classifier was applied, matching emotion keywords to categories. 

_4.2.2 Trait Inference Prompts._ For the LLaMA implementation, the Inference Agent was prompted with a chain-of-thought JSON generation request, which allowed the model to reason through the request step-by-step before making the final structured output. Each Big Five personality variable, Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism, was scored on a continuous 0.0-1.0 scale to support more nuanced personality profiling, as opposed to coarse categorical labeling. We passed some known structured JSON into the prompt directly, in order to direct the model towards a consistent output scheme, and limit hallucinations that may develop from free-form generation. This also ensured that later modules could rely upon a consistent data structure for analysis or visualization [2]. 

Conversely, the Falcon-based implementation employed a lightweight heuristic, rule-based scoring method instead of prompting the model for JSON output. This design decision was influenced by the 1B parameter limitation of Falcon by not overloading the prompt, the additional prompt format simplified memory use and improved response latency. Estimating trait values using the heuristic approach was accomplished by combining emotion-adjusted scores—developed from the previous sentiment and affect analyses—with the relevant lexical keywords that were mapped to respective personality dimensions. Since the approach was not generative, the rule-based strategy provided much more in the way of interpretability and control making it easier for an investigator to see how the derived trait score was calculated from the input text. 

**Listing 1: Example of Big Five personality trait scoring prompt for the Inference Agent.** 

|<|system|>||
|---|---|
|You are a personality assessment expert. Analyze||
|the given text and context||
|to determine Big Five personality trait scores.||
|Provide scores from 0.0 (very low)||
|to 1.0 (very high) for each trait.|Return ONLY a|
|valid JSON object with no additional text.||
|<|user|>||
|Text: "{text}"||
|Emotion: {emo}||
|Context: Response time: {rt}, Body|language: {bl},|
|Speech: {sa}||
|Scoring guidelines:||
|- Openness (0.0<br>1<br>.0): Creativity ,|curiosity ,|
|openness to new experiences||
|* Simple greetings: 0.4<br>0<br>.6 (neutral)||
|* Creative/artistic content: 0.7|0<br>.9|
|* Routine/conventional: 0.2<br>0<br>.4||



**==> picture [248 x 304] intentionally omitted <==**

**----- Start of picture text -----**<br>
- Conscientiousness (0.0 1 .0): Organization ,<br>self -discipline , reliability<br>* Well -structured responses: 0.6 0 .8<br>* Casual/informal: 0.3 0 .5<br>* Detailed/methodical: 0.7 0 .9<br>- Extraversion (0.0 1 .0): Sociability ,<br>assertiveness , energy<br>* Friendly greetings: 0.6 0 .8<br>* Confident/assertive tone: 0.7 0 .9<br>* Withdrawn/quiet: 0.2 0 .4<br>- Agreeableness (0.0 1 .0): Cooperation , trust ,<br>empathy<br>* Polite greetings: 0.6 0 .8<br>* Helpful/considerate: 0.7 0 .9<br>* Confrontational: 0.1 0 .3<br>- Neuroticism (0.0 1 .0): Emotional instability ,<br>anxiety , stress<br>* Calm/confident: 0.1 0 .3<br>* Anxious/worried: 0.7 0 .9<br>* Neutral/stable: 0.3 0 .5<br>Example JSON format:<br>{" openness ":0.5, "conscientiousness ":0.6 , "<br>extraversion ": :0. 7 ,"agreeableness ":0 . 8,"<br>neuroticism ":0.2}<br><|assistant|><br>**----- End of picture text -----**<br>


_4.2.3 Dialogue Agent Prompts._ For LLaMA, the Dialogue Agent used an empathic system prompt with emotion, trait, and retrieved examples, then sampled responses with moderate creativity (temperature 0.7, top-p 0.9). 

For Falcon, the Dialogue Agent used a template-based empathetic generation scheme, selecting from predefined response structures conditioned on the inferred emotion and traits. This approach produced consistent responses with lower latency, more suited to Falcon’s constrained decoding resources [9]. 

In all cases, sampling parameters and prompt templates were kept stable across multiple runs to enable reproducibility and fair comparison of response diversity, empathy, and alignment. 

## **4.3 Behavioral Metadata Inference** 

In addition to purely textual features, the system incorporated behavioral metadata to provide a richer context for affective reasoning. These features included: 

- **Response Time:** measured manually or by timestamp logs and categorized into “fast,” “slow,” or “moderate.” These categories were used as proxies for user cognitive load or emotional urgency. 

- **Body Language:** approximated from manual annotations during interviews, such as “relaxed,” “fidgeting,” or “arms crossed.” For live interactions, the Perception Agent mapped inferred emotions to body-language categories via a static dictionary. 

- **Speech Attributes:** inferred from lexical cues in the user text, including punctuation (e.g., multiple exclamation marks 

for excitement), capitalization (for loudness), or ellipsis/question marks (for hesitancy). A rule-based parser identified these features at runtime. 

These behavioral cues were combined with the user’s text and served as auxiliary inputs to the Perception Agent and Inference Agent. In the case of the Falcon implementation, these features were parsed with simple keyword rules, while in the LLaMA-based agent, they were explicitly inserted into the prompt to condition the language model’s reasoning. This allowed the system to treat the behavioral signals as additional modalities, partially compensating for the lack of audiovisual sensor data. 

Altogether, this metadata-supported approach improved affect recognition and personalization of responses by providing a more holistic user state beyond text alone. 

## **4.4 Retrieval-Augmented Memory** 

The system integrates a retrieval-augmented memory module to supply the Dialogue Agent with relevant prior cases, thereby improving contextual awareness and response coherence. The retrieval module operates over two separate memory stores: 

- **Static Memory:** built from the original interview dataset, with pre-annotated question-response pairs, emotions, and trait labels. This static data formed a stable baseline for context retrieval. 

- **Dynamic Memory:** incrementally updated as the agent interacted with users during live or batch testing. Each new interaction, including user query, inferred emotion, traits, and generated response, was appended to the dynamic memory store in JSON format. This supported continual learning and self-auditing by the agent. 

To represent queries and memory entries, the system applied a hybrid embedding strategy: 

- A TF-IDF vectorizer on the textual transcripts, trained on all available data. 

- One-hot encodings for emotion categories. 

- Raw trait values as a numerical feature vector. 

These feature vectors were concatenated to form a unified embedding for each interaction, which was then indexed for similarity search using cosine similarity. For each new user query, the retrieval module computed its embedding and retrieved the top- _𝑘_ most similar past interactions, subject to a minimum similarity threshold to avoid irrelevant examples. 

These retrieved interactions were then incorporated into the prompt for the Dialogue Agent, providing contextual continuity and exposure to previously successful conversational patterns. This retrieval-augmented approach enabled the system to align with a user’s emotional and personality signals over time, and to adapt responses based on observed behavioral consistency [15]. 

## **4.5 Dialogue Generation** 

The final stage of the pipeline is handled by the Dialogue Agent, responsible for producing empathetic, personality-aligned responses to user inputs. Its prompt included: the current user query, inferred emotion and trait profile, and any top- _𝑘_ retrieved similar examples from memory. This combination created a richly conditioned 

context for generating socially appropriate and psychologically resonant dialogue. 

For the LLaMA 3.2 1B model, the Dialogue Agent prompt was framed with a system role emphasizing empathy, emotional adaptation, and reference to the user’s trait profile. Sampling parameters (temperature 0.7, top-p 0.9) were tuned to balance creativity and consistency. Generated outputs were filtered to maintain the expected tone and to preserve a natural flow of conversation. 

For Falcon-RW-1B, a template-based response mechanism was employed. Here, the Dialogue Agent used predefined empathetic starter phrases mapped to the detected emotion, with follow-up questions drawn from a rule-based template bank. This reduced decoding complexity and ensured stable responses even in the presence of Falcon’s smaller capacity. The template-based responses were enriched with trait-based adjustments, such as using calmer tones for highly neurotic users or more imaginative language for highly open users. 

In both cases, the Dialogue Agent adapted its reply style according to the inferred Big Five trait profile. For example, a user with high agreeableness and high extraversion would receive affirming, outgoing responses, whereas a user with low extraversion and high neuroticism would receive calmer, more supportive messages. 

Ultimately, the Dialogue Agent integrated the outputs from perception, inference, and retrieval, acting as the final stage of the agentic Observe → Reflect → Act → Self-Audit loop. 

## **4.6 Evaluation Protocol** 

To validate the system, we established a multi-layer evaluation protocol targeting both functional and affective performance criteria. The evaluation proceeded as follows: 

- **Benchmark Queries:** We prepared a benchmark set of 19 representative queries derived from interview-style scenarios. These included a variety of emotional states and reflective topics, each labeled with response time, body language, and speech attributes to simulate real-world conversational metadata. 

- **Comparison of Models:** Both the LLaMA 3.2 1B and FalconRW-1B pipelines were tested on exactly the same 19 benchmark queries to ensure a fair and controlled comparison. Prompts, metadata features, and retrieval configurations were kept identical across both models. 

- **Quantitative Metrics:** The following evaluation metrics were computed: 

- _Average Latency_ (seconds): measuring processing time per response. 

- _Average Response Length_ (words): to capture elaboration vs. brevity. 

- _Distinct-2 Diversity_ : proportion of distinct bigrams to measure lexical diversity. 

- _Mean Absolute Difference (MAE)_ in Big Five trait estimates between the models, to assess trait consistency. 

- **Qualitative Observations:** In addition to the quantitative metrics, we manually reviewed sample responses for empathy, coherence, and suitability to the emotional states described in the benchmark queries. 

This evaluation protocol ensured a rigorous assessment of the system’s ability to maintain affective and psychological alignment while supporting a comparative analysis between the agentic LLaMA pipeline and the template-driven Falcon pipeline [3, 5]. 

## **5 Evaluation** 

Table 1 summarizes the comparative evaluation metrics between the LLaMA 3.2 1B and Falcon-RW-1B models on the 19 benchmark queries. The metrics highlight trade-offs between generation richness and efficiency: 

**Table 1: Comparative performance of the two models across benchmark queries** 

|**Metric**|**LLaMA 3.2**|**1B**|**Falcon-RW-1B**|
|---|---|---|---|
|Average Latency (s)|2.07||0.08|
|Average Response Length (words)|88.7||11.6|
|Distinct-2 Diversity|0.958||1.000|
|Mean Absolute Diference (Traits)||0.164||



As shown, LLaMA produced significantly longer and more lexically diverse responses, while Falcon responded far faster due to its template-driven design. The mean absolute difference in Big Five trait estimates between the two models was modest (0.164), indicating similar psychological profiles despite differing generation approaches. These results demonstrate the viability of both generative and template-based architectures within the same agentic pipeline, depending on computational resources and latency requirements [5]. 

## **6 Results and Comparision** 

The evaluation experiments on 19 benchmark queries highlighted key trade-offs between the agentic LLaMA and the template-driven Falcon pipelines. Quantitatively, LLaMA produced significantly longer and more elaborated responses, averaging approximately 88.7 words per reply, while Falcon averaged 11.6 words per reply, showing a concise but less expressive style [3]. 

In terms of lexical diversity, Falcon achieved a distinct-2 score of 1.000, reflecting its deterministic, short, template-driven phrasing. However, this perfect diversity result should be interpreted with caution, because Falcon’s responses were on average very short (mean length 11.6 words), the likelihood of bigram repetition was minimal, trivially inflating its distinct-2 metric. In contrast, the LLaMA 3.2 1B model reached a distinct-2 score of 0.958 while producing much longer responses (average 88.7 words). Maintaining high lexical diversity over extended outputs is considerably more challenging and signals richer, more varied conversational capabilities. Therefore, the LLaMA pipeline demonstrates more meaningful diversity, supporting extended and naturalistic dialogue with consistent language variation across turns. 

Regarding processing latency, LLaMA averaged 2.07 seconds per response due to longer decoding steps, while Falcon was highly efficient, with an average latency of 0.08 seconds. This demonstrates the trade-off between richer, contextually adaptive generation (LLaMA) versus faster, template-based responses (Falcon). 

Trait estimation consistency between the two pipelines was assessed using the mean absolute difference across the Big Five dimensions, yielding a value of 0.164. This suggests that, despite their contrasting response styles, both systems maintained broadly consistent personality trait inferences on identical user inputs. 

Qualitative observations revealed that LLaMA responses were more empathetic, incorporating user traits, retrieved examples, and emotional signals more fluidly. Falcon, in contrast, maintained a stable tone but with less adaptability to nuanced contexts. Human raters informally noted that LLaMA’s responses often felt more natural and socially appropriate in emotionally sensitive scenarios. 

These findings support the feasibility of a modular, agentic architecture combining multimodal metadata, retrieval-augmented memory, and trait-aware generation. They also demonstrate the flexibility of deploying either a highly generative model or a simpler template model under the same agentic framework, depending on latency and resource constraints. 

**==> picture [218 x 215] intentionally omitted <==**

**Figure 5: Bar chart comparing LLaMA 3.2 1B and FalconRW-1B across latency, response length, distinct-2 diversity (rescaled), and Big Five MAE (rescaled). Diversity and MAE values are shown as percentages for visualization clarity.** 

## **7 Limitations and Future Work** 

While the proposed agentic multi-stage pipeline shows promising results, several limitations remain. The evaluation used a small benchmark of 19 queries, which, though diverse, does not capture the full range of real-world user behaviors. Scaling to larger, more varied datasets, including spontaneous conversations, is needed to validate generalizability [14]. 

The static–dynamic dataset strategy relies on heuristic trait estimates, which may introduce bias. Using human-annotated profiles or validated personality surveys could improve accuracy. The retrieval memory’s reliance on TF-IDF may miss deeper semantic or 

emotional connections; dense embeddings or graph-based memory could enhance relevance [10]. 

The system also lacks continual learning for the generative models. Techniques like incremental fine-tuning or RLHF could improve long-term personalization. Current trait-conditioned responses are rule-based and not deeply adaptive over multiple sessions; tracking evolving emotional patterns could build trust and relational depth. 

Future work should scale evaluation, refine memory and learning mechanisms, and include a case study showing real-world, multi-turn interactions. Such a study would demonstrate emotion perception, personality-shaped dialogue, and improved conversational flow over a baseline system, providing both qualitative and quantitative insights into performance. 

## **8 Conclusion** 

We presented a novel agentic, multi-agent dialogue pipeline for inferring emotional states and personality traits from multimodal, interview-style dialog interaction through the integration of verbal content, behavioral metadata, and retrieval-augmented agent perception, inference, dialogue agent capabilities for emotionally adaptive dialog. Evaluating our results with the LLaMA 3.2 1B and Falcon-RW-1B demonstrated that our framework supports both rich generative responses as well as fast template-based reports that achieve consistent trait inference and reasonable classification of emotion. This work directly demonstrates how modular agentic frameworks can connect multimodal affective computing with conversational AI that might operate from a psychological lens while supporting the creation of real-time shared emotionally intelligent experiences at scale. 

## **References** 

- [1] Tanay Agrawal, Dhruv Agarwal, Michal Balazia, Neelabh Sinha, and François Bremond. 2023. Multimodal Personality Recognition using Cross-Attention Transformer and Behaviour Encoding. _arXiv preprint arXiv:2112.12180_ (2023). 

- [2] Venkata Narasareddy Annapareddy, Jeevani Singireddy, Botlagunta Preethish Nanan, Jai Kiran Reddy Burugulla, et al. 2025. Emotional Intelligence in Artificial Agents: Leveraging Deep Multimodal Big Data for Contextual Social Interaction and Adaptive Behavioral Modelling. _Jai Kiran Reddy, Emotional Intelligence in Artificial Agents: Leveraging Deep Multimodal Big_ 

   - _Data for Contextual Social Interaction and Adaptive Behavioral Modelling (April 14, 2025)_ (2025). 

- [3] Tajamul Ashraf, Amal Saqib, Hanan Gani, Muhra AlMahri, Yuhao Li, Noor Ahsan, Umair Nawaz, Jean Lahoud, Hisham Cholakkal, Mubarak Shah, et al. 2025. Agent- 

   - X: Evaluating Deep Multimodal Reasoning in Vision-Centric Agentic Tasks. _arXiv preprint arXiv:2505.24876_ (2025). 

- [4] Alby Babu, T Dharshini, Gayathry Krishnan VS, Ummu Haiman VP, Annie Julie Joseph, and KR Rajesh. 2024. Multimodal Emotion Analysis Using Integrating NLP, AI, and Facial Expression Recognition for Enhanced Emotion Detection. In _2024 IEEE International Conference on Signal Processing, Informatics, Communication and Energy Systems (SPICES)_ . IEEE, 1–6. 

- [5] Luis Bravo, Ciro Rodriguez, Pedro Hidalgo, and Cesar Angulo. 2025. A Systematic Review on Artificial Intelligence-Based Multimodal Dialogue Systems Capable of Emotion Recognition. _Multimodal Technologies and Interaction_ 9, 3 (2025), 28. doi:10.3390/mti9030028 

- [6] Erik Cambria, Xulang Zhang, Rui Mao, Melvin Chen, and Kenneth Kwok. 2024. SenticNet 8: Fusing emotion AI and commonsense AI for interpretable, trustworthy, and explainable affective computing. In _International Conference on HumanComputer Interaction_ . Springer, 197–216. 

- [7] Kathleen Koosje de Boer. 2024. _Towards Interpretable Multimodal Models for Emotion Recognition_ . Master’s thesis. Utrecht University. Master’s thesis, Department of Information and Computing Sciences. 

- [8] Samira Hazmoune and Fateh Bougamouza. 2024. Using transformers for multimodal emotion recognition: Taxonomies and state of the art review. _Engineering Applications of Artificial Intelligence_ 133 (2024), 108339. 

- [9] Brian Hosler, Davide Salvi, Anthony Murray, Fabio Antonacci, Paolo Bestagini, Stefano Tubaro, and Matthew C. Stamm. 2021. Do Deepfakes Feel Emotions? A Semantic Approach to Detecting Deepfakes via Emotional Inconsistencies. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops_ . 1013–1022. 

- [10] Fernando Jia, Yuteng Fu, Jade Zheng, and Florence Li. 2025. Embodied AI Agent for Co-creation Ecosystem: Elevating Human-AI Co-creation through Emotion Recognition and Dynamic Personality Adaptation. (2025). 

- [11] Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A Dobre, and Merouane Debbah. 2025. From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications. _arXiv preprint arXiv:2505.22311_ (2025). 

- [12] Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A Dobre, and Merouane Debbah. 2025. From large ai models to agentic ai: A tutorial on future intelligent communications. _arXiv preprint arXiv:2505.22311_ (2025). 

- [13] Jeevani Singireddy, Botlagunta Preethish Nandan, Phanish Lakkarasu, Venkata Narasareddy Annapareddy, and Jai Kiran Reddy Burugulla. 2025. Emotional Intelligence in Artificial Agents: Leveraging Deep Multimodal Big Data for Contextual Social Interaction and Adaptive Behavioral Modelling. _Metallurgical and Materials Engineering_ 31, 4 (2025), 599–615. 

- [14] Wei Zeng, Hengshu Zhu, Chuan Qin, Han Wu, Yihang Cheng, Sirui Zhang, Xiaowei Jin, Yinuo Shen, Zhenxing Wang, Feimin Zhong, et al. 2025. ApplicationDriven Value Alignment in Agentic AI Systems: Survey and Perspectives. _arXiv preprint arXiv:2506.09656_ (2025). 

- [15] Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Lutao Jiang, Haiwei Xue, Bin Ren, Danda Paudel, Nicu Sebe, Luc Van Gool, and Xuming Hu. 2025. Retrieval augmented generation and understanding in vision: A survey and new outlook. _arXiv preprint arXiv:2503.18016_ (2025). 

