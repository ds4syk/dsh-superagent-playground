
## 0 | 7769 cites | 2023 | preprint | seeds=ijepa
**DINOv2: Learning Robust Visual Features without Supervision**

Maxime Oquab, Timothée Darcet, Théo Moutakanni | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2304.07193 | https://openalex.org/W4366208220

Abstract: The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could greatly simplify the use of images in any system by producing all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning. This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. We revisit existing approaches and combine different techniques to scale our pretraining in terms of data and model size. Most of the technical contributions aim at accelerating and stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy et al., 2020) with 1B parameters and distill it into a series of smaller models that surpass the best available all-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the benchmarks at image and

## 1 | 2274 cites | 2023 | article | seeds=worldmodels
**Diffusion Models: A Comprehensive Survey of Methods and Applications**

Ling Qiang Yang, Zhilong Zhang, Yang Song | ACM Computing Surveys

DOI: https://doi.org/10.1145/3626235 | https://openalex.org/W4387195417

Abstract: Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design. In this survey, we provide an overview of the rapidly expanding body of work on diffusion models, categorizing the research into three key areas: efficient sampling, improved likelihood estimation, and handling data with special structures. We also discuss the potential for combining diffusion models with other generative models for enhanced results. We further review the wide-ranging applications of diffusion models in fields spanning from computer vision, natural language processing, temporal data modeling, to interdisciplinary applications in other scientific disciplines. This survey aims to provide a contextualized, in-depth look at the state of diffusion models, identifying the key areas of focus and pointing to potential areas for further exploration. Github: https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy

## 2 | 1016 cites | 2025 | preprint | seeds=dreamer,dreamerv3,worldmodels
**Cosmos World Foundation Model Platform for Physical AI**

NVIDIA, :, Niket Agarwal | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2501.03575 | https://openalex.org/W4406191642

Abstract: Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make Cosmos open-source and our models open-weight with permissive licenses available via https://github.com/nvidia-cosmos/cosmos-predict1.

## 3 | 1002 cites | 2024 | article | seeds=dreamer,dreamerv2,worldmodels
**A comprehensive survey on pretrained foundation models: a history from BERT to ChatGPT**

Ce Zhou, Qian Li, Chen Li | International Journal of Machine Learning and Cybernetics

DOI: https://doi.org/10.1007/s13042-024-02443-6 | https://openalex.org/W4404658388

Abstract: 

## 4 | 871 cites | 2023 | article | seeds=dreamer,rwm,planet
**Model-based Reinforcement Learning: A Survey**

Thomas M. Moerland, Joost Broekens, Aske Plaat | Foundations and Trends® in Machine Learning

DOI: https://doi.org/10.1561/2200000086 | https://openalex.org/W3038822267

Abstract: Sequential decision making, commonly formalized as Markov Decision Process (MDP) optimization, is an important challenge in artificial intelligence. Two key approaches to this problem are reinforcement learning (RL) and planning. This survey is an integration of both fields, better known as model-based reinforcement learning. Model-based RL has two main steps. First, we systematically cover approaches to dynamics model learning, including challenges like dealing with stochasticity, uncertainty, partial observability, and temporal abstraction. Second, we present a systematic categorization of planning-learning integration, including aspects like: where to start planning, what budgets to allocate to planning and real data collection, how to plan, and how to integrate planning in the learning and acting loop. After these two sections, we also discuss implicit model-based RL as an end-to-end alternative for model learning and planning, and we cover the potential benefits of model-based RL. Along the way, the survey also draws connections to several related RL fields, like hierarchical RL and transfer learning. Altogether, the survey presents a broad conceptual overview of the combinati

## 5 | 844 cites | 2024 | article | seeds=dreamer,dreamerv2,rwm
**End-to-End Autonomous Driving: Challenges and Frontiers**

Li Chen, Penghao Wu, Kashyap Chitta | IEEE Transactions on Pattern Analysis and Machine Intelligence

DOI: https://doi.org/10.1109/tpami.2024.3435937 | https://openalex.org/W4401109681

Abstract: The autonomous driving community has witnessed a rapid growth in approaches that embrace an end-to-end algorithm framework, utilizing raw sensor input to generate vehicle motion plans, instead of concentrating on individual tasks such as detection and motion prediction. End-to-end systems, in comparison to modular pipelines, benefit from joint feature optimization for perception and planning. This field has flourished due to the availability of large-scale datasets, closed-loop evaluation, and the increasing need for autonomous driving algorithms to perform effectively in challenging scenarios. In this survey, we provide a comprehensive analysis of more than 270 papers, covering the motivation, roadmap, methodology, challenges, and future trends in end-to-end autonomous driving. We delve into several critical challenges, including multi-modality, interpretability, causal confusion, robustness, and world models, amongst others. Additionally, we discuss current advancements in foundation models and visual pre-training, as well as how to incorporate these techniques within the end-to-end driving framework.

## 6 | 795 cites | 2023 | conference-paper | seeds=dreamer,dreamerv2,rwm,worldmodels
**Reasoning with Language Model is Planning with World Model**

Shibo Hao, Yi Gu, Haodi Ma | Conference on Empirical Methods in Natural Language Processing (EMNLP)

DOI: https://doi.org/10.18653/v1/2023.emnlp-main.507 | https://openalex.org/W4389520747

Abstract: Large language models (LLMs) have shown remarkable reasoning capabilities, particularly with chain-of-thought (CoT) prompting.However, LLMs sometimes still struggle with problems that are easy for humans, such as generating action plans to achieve given goals in an environment, or performing complex math or logical reasoning.The deficiency stems from the key fact that LLMs lack an internal world model to predict the world state (e.g., environment status, intermediate variable values) and simulate long-term outcomes of actions.This prevents LLMs from performing deliberate planning akin to human brains, which involves exploring alternative reasoning paths, anticipating future states and rewards, and iteratively refining existing reasoning steps.To overcome the limitations, we propose a new LLM reasoning framework, Reasoning via Planning (RAP).RAP repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm based on Monte Carlo Tree Search for strategic exploration in the vast reasoning space.During reasoning, the LLM (as agent) incrementally builds a reasoning tree under the guidance of the LLM (as world model) and rewards, and effi

## 7 | 762 cites | 2023 | article | seeds=dreamer,dreamerv3
**Reinforcement learning algorithms: A brief survey**

Ashish Kumar Shakya, G. N. Pillai, Sohom Chakrabarty | Expert Systems with Applications

DOI: https://doi.org/10.1016/j.eswa.2023.120495 | https://openalex.org/W4377695283

Abstract: 

## 8 | 672 cites | 2026 | preprint | seeds=dreamer,dreamerv2,dreamerv3,vjepa2
**World Action Models are Zero-shot Policies**

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2602.15922 | https://openalex.org/W7130552073

Abstract: State-of-the-art Vision-Language-Action (VLA) models excel at semantic generalization but struggle to generalize to unseen physical motions in novel environments. We introduce DreamZero, a World Action Model (WAM) built upon a pretrained video diffusion backbone. Unlike VLAs, WAMs learn physical dynamics by predicting future world states and actions, using video as a dense representation of how the world evolves. By jointly modeling video and action, DreamZero learns diverse skills effectively from heterogeneous robot data without relying on repetitive demonstrations. This results in over 2x improvement in generalization to new tasks and environments compared to state-of-the-art VLAs in real robot experiments. Crucially, through model and system optimizations, we enable a 14B autoregressive video diffusion model to perform real-time closed-loop control at 7Hz. Finally, we demonstrate two forms of cross-embodiment transfer: video-only demonstrations from other robots or humans yield a relative improvement of over 42% on unseen task performance with just 10-20 minutes of data. More surprisingly, DreamZero enables few-shot embodiment adaptation, transferring to a new embodiment with o

## 9 | 441 cites | 2024 | preprint | seeds=worldmodels
**The Platonic Representation Hypothesis**

Minyoung Huh, Brian Cheung, Wang, Tongzhou | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2405.07987 | https://openalex.org/W4396914181

Abstract: We argue that representations in AI models, particularly deep networks, are converging. First, we survey many examples of convergence in the literature: over time and across multiple domains, the ways by which different neural networks represent data are becoming more aligned. Next, we demonstrate convergence across data modalities: as vision models and language models get larger, they measure distance between datapoints in a more and more alike way. We hypothesize that this convergence is driving toward a shared statistical model of reality, akin to Plato's concept of an ideal reality. We term such a representation the platonic representation and discuss several possible selective pressures toward it. Finally, we discuss the implications of these trends, their limitations, and counterexamples to our analysis.

## 10 | 439 cites | 2025 | preprint | seeds=worldmodels
**WorldVLA: Towards Autoregressive Action World Model**

Jun Cen, Chaohui Yu, Yuan, Hangjie | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2506.21539 | https://openalex.org/W4415183980

Abstract: We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the cu

## 11 | 435 cites | 2023 | conference-paper | seeds=dreamerv3
**Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents**

Zihao Wang, Shaofei Cai, Guanzhou Chen | neural information processing systems

DOI: https://doi.org/10.52202/075280-1480 | https://openalex.org/W7133201869

Abstract: We investigate the challenge of task planning for multi-task embodied agents in open-world environments. Two main difficulties are identified: 1) executing plans in an open-world environment (e.g., Minecraft) necessitates accurate and multi-step reasoning due to the long-term nature of tasks, and 2) as vanilla planners do not consider how easy the current agent can achieve a given sub-task when ordering parallel sub-goals within a complicated plan, the resulting plan could be inefficient or even infeasible. To this end, we propose "$\underline{D}$escribe, $\underline{E}$xplain, $\underline{P}$lan and $\underline{S}$elect" ($\textbf{DEPS}$), an interactive planning approach based on Large Language Models (LLMs). DEPS facilitates better error correction on initial LLM-generated $\textit{plan}$ by integrating $\textit{description}$ of the plan execution process and providing self-$\textit{explanation}$ of feedback when encountering failures during the extended planning phases. Furthermore, it includes a goal $\textit{selector}$, which is a trainable module that ranks parallel candidate sub-goals based on the estimated steps of completion, consequently refining the initial plan. Our ex

## 12 | 415 cites | 2024 | conference-paper | seeds=ijepa
**Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**

Shengbang Tong, Zhuang Liu, Yuexiang Zhai | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings

DOI: https://doi.org/10.1109/cvpr52733.2024.00914 | https://openalex.org/W4402754023

Abstract: Is vision good enough for language? Recent advancements in multimodal models primarily stem from the powerful reasoning abilities of large language models (LLMs). However, the visual component typically depends only on the instance-level contrastive language-image pre-training (CLIP). Our research reveals that the visual capabilities in recent MultiModal LLMs (MLLMs) still exhibit systematic shortcomings. To understand the roots of these errors, we explore the gap between the visual embedding space of CLIP and vision-only self-supervised learning. We identify “CLIP-blind pairs”- images that CLIP perceives as similar despite their clear visual differences. With these pairs, we construct the Multimodal Visual Patterns (MMVP) benchmark. MMVP exposes areas where state-of-the-art systems, including GPT-4V, struggle with straightforward questions across nine basic visual patterns, often providing incorrect answers and hallucinated explanations. We further evaluate various CLIP-based vision-and-language models and found a notable correlation between visual patterns that challenge CLIP models and those problematic for multimodal LLMs. As an initial effort to address these issues, we propos

## 13 | 408 cites | 2026 | article | seeds=ijepa
**A Survey on Vision--Language--Action Models for Embodied AI**

Yueen Ma, Zixing Song, Yuzheng Zhuang | IEEE Transactions on Neural Networks and Learning Systems

DOI: https://doi.org/10.1109/tnnls.2025.3650584 | https://openalex.org/W4398796510

Abstract: Embodied AI is widely recognized as a cornerstone of artificial general intelligence (AGI) because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models (LLMs) and vision-language models (VLMs), a new category of multimodal models-referred to as vision-language-action (VLA) models-has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, si

## 14 | 407 cites | 2023 | preprint | seeds=dreamerv3
**A Survey of Reinforcement Learning from Human Feedback**

Timo Kaufmann, Paul Weng, Viktor Bengs | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2312.14925 | https://openalex.org/W4390215524

Abstract: Reinforcement learning from human feedback (RLHF) is a variant of reinforcement learning (RL) that learns from human feedback instead of relying on an engineered reward function. Building on prior work on the related setting of preference-based reinforcement learning (PbRL), it stands at the intersection of artificial intelligence and human-computer interaction. This positioning provides a promising approach to enhance the performance and adaptability of intelligent systems while also improving the alignment of their objectives with human values. The success in training large language models (LLMs) has impressively demonstrated this potential in recent years, where RLHF has played a decisive role in directing the model's capabilities towards human objectives. This article provides an overview of the fundamentals of RLHF, exploring how RL agents interact with human feedback. While recent focus has been on RLHF for LLMs, our survey covers the technique across multiple domains. We provide our most comprehensive coverage in control and robotics, where many fundamental techniques originate, alongside a dedicated LLM section. We examine the core principles that underpin RLHF, how algorit

## 15 | 406 cites | 2023 | preprint | seeds=dreamerv3
**Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models**

Andy Zhou, Yan, Kai, Michal Shlapentokh-Rothman | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2310.04406 | https://openalex.org/W4389422760

Abstract: While language models (LMs) have shown potential across a range of decision-making tasks, their reliance on simple acting processes limits their broad deployment as autonomous agents. In this paper, we introduce Language Agent Tree Search (LATS) -- the first general framework that synergizes the capabilities of LMs in reasoning, acting, and planning. By leveraging the in-context learning ability of LMs, we integrate Monte Carlo Tree Search into LATS to enable LMs as agents, along with LM-powered value functions and self-reflections for proficient exploration and enhanced decision-making. A key feature of our approach is the incorporation of an environment for external feedback, which offers a more deliberate and adaptive problem-solving mechanism that surpasses the constraints of existing techniques. Our experimental evaluation across diverse domains, including programming, interactive question-answering (QA), web navigation, and math, validates the effectiveness and generality of LATS in decision-making while maintaining competitive or improved reasoning performance. Notably, LATS achieves state-of-the-art pass@1 accuracy (92.7%) for programming on HumanEval with GPT-4 and demonst

## 16 | 379 cites | 2024 | preprint | seeds=dreamerv3,worldmodels
**GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation**

Chilam Cheang, Guangzeng Chen, Ya Jing | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2410.06158 | https://openalex.org/W4403344670

Abstract: We present GR-2, a state-of-the-art generalist robot agent for versatile and generalizable robot manipulation. GR-2 is first pre-trained on a vast number of Internet videos to capture the dynamics of the world. This large-scale pre-training, involving 38 million video clips and over 50 billion tokens, equips GR-2 with the ability to generalize across a wide range of robotic tasks and environments during subsequent policy learning. Following this, GR-2 is fine-tuned for both video generation and action prediction using robot trajectories. It exhibits impressive multi-task learning capabilities, achieving an average success rate of 97.7% across more than 100 tasks. Moreover, GR-2 demonstrates exceptional generalization to new, previously unseen scenarios, including novel backgrounds, environments, objects, and tasks. Notably, GR-2 scales effectively with model size, underscoring its potential for continued growth and application. Project page: \url{https://gr2-manipulation.github.io}.

## 17 | 376 cites | 2025 | preprint | seeds=vjepa
**UniVLA: Learning to Act Anywhere with Task-centric Latent Actions**

Qingwen Bu, Yanting Yang, Jisong Cai | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2505.06111 | https://openalex.org/W4417086207

Abstract: A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute an

## 18 | 375 cites | 2024 | conference-paper | seeds=dreamer,dreamerv2,dreamerv3
**DriveDreamer: Towards Real-World-Drive World Models for Autonomous Driving**

Xiaofeng Wang, Zheng Hong Zhu, Guan Huang | Lecture notes in computer science

DOI: https://doi.org/10.1007/978-3-031-73195-2_4 | https://openalex.org/W4404722496

Abstract: 

## 19 | 363 cites | 2023 | preprint | seeds=dreamerv2,dreamerv3
**Learning Interactive Real-World Simulators**

Yang, Sherry, Yilun Du, Kamyar Ghasemipour | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2310.06114 | https://openalex.org/W4387560742

Abstract: Generative models trained on internet data have revolutionized how text, image, and video content can be created. Perhaps the next milestone for generative models is to simulate realistic experience in response to actions taken by humans, robots, and other interactive agents. Applications of a real-world simulator range from controllable content creation in games and movies, to training embodied agents purely in simulation that can be directly deployed in the real world. We explore the possibility of learning a universal simulator (UniSim) of real-world interaction through generative modeling. We first make the important observation that natural datasets available for learning a real-world simulator are often rich along different dimensions (e.g., abundant objects in image data, densely sampled actions in robotics data, and diverse movements in navigation data). With careful orchestration of diverse datasets, each providing a different aspect of the overall experience, we can simulate the visual outcome of both high-level instructions such as "open the drawer" and low-level controls from otherwise static scenes and objects. We use the simulator to train both high-level vision-langu

## 20 | 359 cites | 2025 | article | seeds=ijepa,worldmodels
**Aligning Cyber Space With Physical World: A Comprehensive Survey on Embodied AI**

Yang Liu, Weixing Chen, Yongjie Bai | IEEE/ASME Transactions on Mechatronics

DOI: https://doi.org/10.1109/tmech.2025.3574943 | https://openalex.org/W4412722243

Abstract: Embodied artificial intelligence (Embodied AI) is crucial for achieving artificial general intelligence (AGI) and serves as a foundation for various applications (e.g., intelligent mechatronics systems, smart manufacturing) that bridge cyberspace and the physical world. Recently, the emergence of multimodal large models (MLMs) and World models (WMs) have attracted significant attention due to their remarkable perception, interaction, and reasoning capabilities, making them a promising architecture for embodied agents. In this survey, we give a comprehensive exploration of the latest advancements in Embodied AI. Our analysis firstly navigates through the forefront of representative works of embodied robots and simulators, to fully understand the research focuses and their limitations. Then, we analyze four main research targets: 1) embodied perception, 2) embodied interaction, 3) embodied agent, and 4) sim-to-real adaptation, covering state-of-the-art methods, essential paradigms, and comprehensive datasets. In addition, we explore the complexities of MLMs in virtual and real embodied agents, highlighting their significance in facilitating interactions in digital and physical enviro

## 21 | 353 cites | 2023 | preprint | seeds=dreamerv3
**Ghost in the Minecraft: Generally Capable Agents for Open-World Environments via Large Language Models with Text-based Knowledge and Memory**

Xizhou Zhu, Yuntao Chen, Hao Tian | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2305.17144 | https://openalex.org/W4378768661

Abstract: The captivating realm of Minecraft has attracted substantial research interest in recent years, serving as a rich platform for developing intelligent agents capable of functioning in open-world environments. However, the current research landscape predominantly focuses on specific objectives, such as the popular "ObtainDiamond" task, and has not yet shown effective generalization to a broader spectrum of tasks. Furthermore, the current leading success rate for the "ObtainDiamond" task stands at around 20%, highlighting the limitations of Reinforcement Learning (RL) based controllers used in existing methods. To tackle these challenges, we introduce Ghost in the Minecraft (GITM), a novel framework integrates Large Language Models (LLMs) with text-based knowledge and memory, aiming to create Generally Capable Agents (GCAs) in Minecraft. These agents, equipped with the logic and common sense capabilities of LLMs, can skillfully navigate complex, sparse-reward environments with text-based interactions. We develop a set of structured actions and leverage LLMs to generate action plans for the agents to execute. The resulting LLM-based agent markedly surpasses previous methods, achieving 

## 22 | 342 cites | 2026 | preprint | seeds=dreamer,dreamerv2,dreamerv3
**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2601.16163 | https://openalex.org/W7125496427

Abstract: Recent video generation models demonstrate remarkable ability to capture complex physical interactions and scene evolution over time. To leverage their spatiotemporal priors, robotics works have adapted video models for policy learning but introduce complexity by requiring multiple stages of post-training and new architectural components for action generation. In this work, we introduce Cosmos Policy, a simple approach for adapting a large pretrained video model (Cosmos-Predict2) into an effective robot policy through a single stage of post-training on the robot demonstration data collected on the target platform, with no architectural modifications. Cosmos Policy learns to directly generate robot actions encoded as latent frames within the video model's latent diffusion process, harnessing the model's pretrained priors and core learning algorithm to capture complex action distributions. Additionally, Cosmos Policy generates future state images and values (expected cumulative rewards), which are similarly encoded as latent frames, enabling test-time planning of action trajectories with higher likelihood of success. In our evaluations, Cosmos Policy achieves state-of-the-art perform

## 23 | 342 cites | 2026 | preprint | seeds=dreamerv2,dreamerv3,vjepa2,worldmodels
**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**

Lucas Maes, Quentin Le Lidec, Damien Scieur | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2603.19312 | https://openalex.org/W7140141371

Abstract: Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pre-trained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48x faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

## 24 | 330 cites | 2025 | article | seeds=worldmodels
**Causal Machine Learning: A Survey and Open Problems**

Jean Kaddour, Aengus Lynch, Qi Liu | Foundations and Trends® in Optimization

DOI: https://doi.org/10.1561/2400000052 | https://openalex.org/W4413642769

Abstract: Causal Machine Learning (CausalML) is an umbrella term for machine learning methods that formalize the data- generation process as a causal model. This perspective enables one to reason about the effects of changes to this process (interventions) and what would have happened in hindsight (counterfactuals). We categorize work in CausalML into five groups according to the problems they address: (1) causal supervised learning, (2) causal generative modeling, (3) causal explanations, (4) causal fairness, and (5) causal reinforcement learning. We systematically compare approaches in each category and point out open problems. Further, we review field-specific applications in computer vision, natural language processing, and graph representation learning. Finally, we provide an overview of causal benchmarks and a discussion of the state of this nascent field, including recommendations for future work.

## 25 | 327 cites | 2024 | preprint | seeds=ijepa
**An Introduction to Vision-Language Modeling**

Florian Bordes, Richard Yuanzhe Pang, Ajay, Anurag | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2405.17247 | https://openalex.org/W4399114997

Abstract: Following the recent popularity of Large Language Models (LLMs), several attempts have been made to extend them to the visual domain. From having a visual assistant that could guide us through unfamiliar environments to generative models that produce images using only a high-level text description, the vision-language model (VLM) applications will significantly impact our relationship with technology. However, there are many challenges that need to be addressed to improve the reliability of those models. While language is discrete, vision evolves in a much higher dimensional space in which concepts cannot always be easily discretized. To better understand the mechanics behind mapping vision to language, we present this introduction to VLMs which we hope will help anyone who would like to enter the field. First, we introduce what VLMs are, how they work, and how to train them. Then, we present and discuss approaches to evaluate VLMs. Although this work primarily focuses on mapping images to language, we also discuss extending VLMs to videos.

## 26 | 320 cites | 2024 | article | seeds=dreamer
**A Comprehensive Survey: Evaluating the Efficiency of Artificial Intelligence and Machine Learning Techniques on Cyber Security Solutions**

Merve Ozkan-Okay, Erdal Akin, Ömer Aslan | IEEE Access

DOI: https://doi.org/10.1109/access.2024.3355547 | https://openalex.org/W4390970201

Abstract: Given the continually rising frequency of cyberattacks, the adoption of artificial intelligence methods, particularly Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL), has become essential in the realm of cybersecurity. These techniques have proven to be effective in detecting and mitigating cyberattacks, which can cause significant harm to individuals, organizations, and even countries. Machine learning algorithms use statistical methods to identify patterns and anomalies in large datasets, enabling security analysts to detect previously unknown threats. Deep learning, a subfield of ML, has shown great potential in improving the accuracy and efficiency of cybersecurity systems, particularly in image and speech recognition. On the other hand, RL is again a subfield of machine learning that trains algorithms to learn through trial and error, making it particularly effective in dynamic environments. We also evaluated the usage of ChatGPT-like AI tools in cyber-related problem domains on both sides, positive and negative. This article provides an overview of how ML, DL, and RL are applied in cybersecurity, including their usage in malware detection, intrusion

## 27 | 303 cites | 2024 | article | seeds=dreamer,planet
**Industrial Internet of Things Intelligence Empowering Smart Manufacturing: A Literature Review**

Yujiao Hu, Qingmin Jia, Yuan Yao | IEEE Internet of Things Journal

DOI: https://doi.org/10.1109/jiot.2024.3367692 | https://openalex.org/W4391953359

Abstract: The fiercely competitive business environment and increasingly personalized customization needs are driving the digital transformation and upgrading of the manufacturing industry. IIoT intelligence, which can provide innovative and efficient solutions for various aspects of the manufacturing value chain, illuminates the path of transformation for the manufacturing industry. It’s time to provide a systematic vision of IIoT intelligence. However, existing surveys often focus on specific areas of IIoT intelligence, leading researchers and readers to have biases in their understanding of IIoT intelligence, that is, believing that research in one direction is the most important for the development of IIoT intelligence, while ignoring contributions from other directions. Therefore, this paper provides a comprehensive overview of IIoT intelligence. We first conduct an in-depth analysis of the inevitability of manufacturing transformation and study the successful experiences from the practices of Chinese enterprises. Then we give our definition of IIoT intelligence and demonstrate the value of IIoT intelligence for industries in fucntions, operations, deployments, and application. Afterwar

## 28 | 292 cites | 2023 | conference-paper | seeds=dreamerv3
**Harms from Increasingly Agentic Algorithmic Systems**

Alan Chan, Rebecca Salganik, Alva Markelius | ACM Conference on Fairness, Accountability, and Transparency (FAccT)

DOI: https://doi.org/10.1145/3593013.3594033 | https://openalex.org/W4376632242

Abstract: Research in Fairness, Accountability, Transparency, and Ethics (FATE)1 has established many sources and forms of algorithmic harm, in domains as diverse as health care, finance, policing, and recommendations. Much work remains to be done to mitigate the serious harms of these systems, particularly those disproportionately affecting marginalized communities. Despite these ongoing harms, new systems are being developed and deployed, typically without strong regulatory barriers, threatening the perpetuation of the same harms and the creation of novel ones. In response, the FATE community has emphasized the importance of anticipating harms, rather than just responding to them. Anticipation of harms is especially important given the rapid pace of developments in machine learning (ML). Our work focuses on the anticipation of harms from increasingly agentic systems. Rather than providing a definition of agency as a binary property, we identify 4 key characteristics which, particularly in combination, tend to increase the agency of a given algorithmic system: underspecification, directness of impact, goal-directedness, and long-term planning. We also discuss important harms which arise fro

## 29 | 274 cites | 2023 | preprint | seeds=dreamer,dreamerv2
**Foundation Models for Decision Making: Problems, Methods, and Opportunities**

Sherry X. Yang, Ofir Nachum, Yilun Du | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2303.04129 | https://openalex.org/W4323651349

Abstract: Foundation models pretrained on diverse data at scale have demonstrated extraordinary capabilities in a wide range of vision and language tasks. When such models are deployed in real world environments, they inevitably interface with other entities and agents. For example, language models are often used to interact with human beings through dialogue, and visual perception models are used to autonomously navigate neighborhood streets. In response to these developments, new paradigms are emerging for training foundation models to interact with other agents and perform long-term reasoning. These paradigms leverage the existence of ever-larger datasets curated for multimodal, multitask, and generalist interaction. Research at the intersection of foundation models and decision making holds tremendous promise for creating powerful new systems that can interact effectively across a diverse range of applications such as dialogue, autonomous driving, healthcare, education, and robotics. In this manuscript, we examine the scope of foundation models for decision making, and provide conceptual tools and technical background for understanding the problem space and exploring new research directi

## 30 | 268 cites | 2024 | preprint | seeds=dreamer,dreamerv2,dreamerv3,worldmodels
**DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning**

Gaoyue Zhou, Pan, Hengkai, Yann LeCun | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2411.04983 | https://openalex.org/W4404360744

Abstract: The ability to predict future outcomes given control actions is fundamental for physical reasoning. However, such predictive models, often called world models, remains challenging to learn and are typically developed for task-specific solutions with online policy learning. To unlock world models' true potential, we argue that they should 1) be trainable on offline, pre-collected trajectories, 2) support test-time behavior optimization, and 3) facilitate task-agnostic reasoning. To this end, we present DINO World Model (DINO-WM), a new method to model visual dynamics without reconstructing the visual world. DINO-WM leverages spatial patch features pre-trained with DINOv2, enabling it to learn from offline behavioral trajectories by predicting future patch features. This allows DINO-WM to achieve observational goals through action sequence optimization, facilitating task-agnostic planning by treating goal features as prediction targets. We demonstrate that DINO-WM achieves zero-shot behavioral solutions at test time on six environments without expert demonstrations, reward modeling, or pre-learned inverse models, outperforming prior state-of-the-art work across diverse task families 

## 31 | 268 cites | 2023 | review | seeds=ijepa
**Large AI Models in Health Informatics: Applications, Challenges, and the Future**

Jianing Qiu, Lin Li, Jiankai Sun | IEEE Journal of Biomedical and Health Informatics

DOI: https://doi.org/10.1109/jbhi.2023.3316750 | https://openalex.org/W4386973901

Abstract: Large AI models, or foundation models, are models recently emerging with massive scales both parameter-wise and data-wise, the magnitudes of which can reach beyond billions. Once pretrained, large AI models demonstrate impressive performance in various downstream tasks. A prime example is ChatGPT, whose capability has compelled people's imagination about the far-reaching influence that large AI models can have and their potential to transform different domains of our lives. In health informatics, the advent of large AI models has brought new paradigms for the design of methodologies. The scale of multi-modal data in the biomedical and health domain has been ever-expanding especially since the community embraced the era of deep learning, which provides the ground to develop, validate, and advance large AI models for breakthroughs in health-related areas. This article presents a comprehensive review of large AI models, from background to their applications. We identify seven key sectors in which large AI models are applicable and might have substantial influence, including: 1) bioinformatics; 2) medical diagnosis; 3) medical imaging; 4) medical informatics; 5) medical education; 6) p

## 32 | 265 cites | 2023 | preprint | seeds=dreamerv3
**TD-MPC2: Scalable, Robust World Models for Continuous Control**

Nicklas Hansen, Hao Su, Xiaolong Wang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2310.16828 | https://openalex.org/W4387964172

Abstract: TD-MPC is a model-based reinforcement learning (RL) algorithm that performs local trajectory optimization in the latent space of a learned implicit (decoder-free) world model. In this work, we present TD-MPC2: a series of improvements upon the TD-MPC algorithm. We demonstrate that TD-MPC2 improves significantly over baselines across 104 online RL tasks spanning 4 diverse task domains, achieving consistently strong results with a single set of hyperparameters. We further show that agent capabilities increase with model and data size, and successfully train a single 317M parameter agent to perform 80 tasks across multiple task domains, embodiments, and action spaces. We conclude with an account of lessons, opportunities, and risks associated with large TD-MPC2 agents. Explore videos, models, data, code, and more at https://tdmpc2.com

## 33 | 264 cites | 2024 | preprint | seeds=worldmodels
**World Model on Million-Length Video And Language With Blockwise RingAttention**

Hao Liu, Wilson Yan, Matei Zaharia | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2402.08268 | https://openalex.org/W4391833450

Abstract: Enabling long-context understanding remains a key challenge in scaling existing sequence models -- a crucial component in developing generally intelligent models that can process and operate over long temporal horizons that potentially consist of millions of tokens. In this paper, we aim to address these challenges by providing a comprehensive exploration of the full development process for producing 1M context language models and video-language models, setting new benchmarks in language retrieval and new capabilities in long video understanding. We detail our long context data curation process, progressive context extension from 4K to 1M tokens, and present an efficient open-source implementation for scalable training on long sequences. Additionally, we open-source a family of 7B parameter models capable of processing long text documents and videos exceeding 1M tokens.

## 34 | 263 cites | 2025 | preprint | seeds=vjepa2,worldmodels
**Advances and Challenges in Foundation Agents: From Brain-Inspired Intelligence to Evolutionary, Collaborative, and Safe Systems**

Bang Liu, Xinfeng Li, Jiayi Zhang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2504.01990 | https://openalex.org/W4409316940

Abstract: The advent of large language models (LLMs) has catalyzed a transformative shift in artificial intelligence, paving the way for advanced intelligent agents capable of sophisticated reasoning, robust perception, and versatile action across diverse domains. As these agents increasingly drive AI research and practical applications, their design, evaluation, and continuous improvement present intricate, multifaceted challenges. This book provides a comprehensive overview, framing intelligent agents within modular, brain-inspired architectures that integrate principles from cognitive science, neuroscience, and computational research. We structure our exploration into four interconnected parts. First, we systematically investigate the modular foundation of intelligent agents, systematically mapping their cognitive, perceptual, and operational modules onto analogous human brain functionalities and elucidating core components such as memory, world modeling, reward processing, goal, and emotion. Second, we discuss self-enhancement and adaptive evolution mechanisms, exploring how agents autonomously refine their capabilities, adapt to dynamic environments, and achieve continual learning throu

## 35 | 261 cites | 2023 | article | seeds=worldmodels
**Diffusion Probabilistic Modeling for Video Generation**

Ruihan Yang, Prakhar Kumar Srivastava, Stephan Mandt | Entropy

DOI: https://doi.org/10.3390/e25101469 | https://openalex.org/W4221146061

Abstract: Denoising diffusion probabilistic models are a promising new class of generative models that mark a milestone in high-quality image generation. This paper showcases their ability to sequentially generate video, surpassing prior methods in perceptual and probabilistic forecasting metrics. We propose an autoregressive, end-to-end optimized video diffusion model inspired by recent advances in neural video compression. The model successively generates future frames by correcting a deterministic next-frame prediction using a stochastic residual generated by an inverse diffusion process. We compare this approach against six baselines on four datasets involving natural and simulation-based videos. We find significant improvements in terms of perceptual quality and probabilistic frame forecasting ability for all datasets.

## 36 | 256 cites | 2023 | article | seeds=dreamer
**Exploration in Deep Reinforcement Learning: From Single-Agent to Multiagent Domain**

Jianye Hao, Tianpei Yang, Hongyao Tang | IEEE Transactions on Neural Networks and Learning Systems

DOI: https://doi.org/10.1109/tnnls.2023.3236361 | https://openalex.org/W4319165238

Abstract: Deep reinforcement learning (DRL) and deep multiagent reinforcement learning (MARL) have achieved significant success across a wide range of domains, including game artificial intelligence (AI), autonomous vehicles, and robotics. However, DRL and deep MARL agents are widely known to be sample inefficient that millions of interactions are usually needed even for relatively simple problem settings, thus preventing the wide application and deployment in real-industry scenarios. One bottleneck challenge behind is the well-known exploration problem, i.e., how efficiently exploring the environment and collecting informative experiences that could benefit policy learning toward the optimal ones. This problem becomes more challenging in complex environments with sparse rewards, noisy distractions, long horizons, and nonstationary co-learners. In this article, we conduct a comprehensive survey on existing exploration methods for both single-agent RL and multiagent RL. We start the survey by identifying several key challenges to efficient exploration. Then, we provide a systematic survey of existing approaches by classifying them into two major categories: uncertainty-oriented exploration an

## 37 | 256 cites | 2023 | review | seeds=planet
**An Information-Theoretic Perspective on Intrinsic Motivation in Reinforcement Learning: A Survey**

Arthur Aubret, Laëtitia Matignon, Salima Hassas | Entropy

DOI: https://doi.org/10.3390/e25020327 | https://openalex.org/W4319966548

Abstract: The reinforcement learning (RL) research area is very active, with an important number of new contributions, especially considering the emergent field of deep RL (DRL). However, a number of scientific and technical challenges still need to be resolved, among which we acknowledge the ability to abstract actions or the difficulty to explore the environment in sparse-reward settings which can be addressed by intrinsic motivation (IM). We propose to survey these research works through a new taxonomy based on information theory: we computationally revisit the notions of surprise, novelty, and skill-learning. This allows us to identify advantages and disadvantages of methods and exhibit current outlooks of research. Our analysis suggests that novelty and surprise can assist the building of a hierarchy of transferable skills which abstracts dynamics and makes the exploration process more robust.

## 38 | 255 cites | 2026 | preprint | seeds=vjepa2,worldmodels
**Cosmos 3: Omnimodal World Models for Physical AI**

NVIDIA, :, Aditi | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2606.02800 | https://openalex.org/W7163380451

Abstract: We introduce Cosmos 3, a family of omnimodal world models designed to jointly process and generate language, image, video, audio, and action sequences within a unified mixture-of-transformers architecture. By supporting highly flexible input-output configurations, Cosmos 3 seamlessly unifies critical modalities for Physical AI -- effectively subsuming vision-language models, video generators, world simulators, and world-action models into a single framework. Our evaluation demonstrates that Cosmos 3 establishes a new state-of-the-art across a diverse suite of understanding and generation tasks, demonstrating omnimodal world models as scalable, general-purpose backbones for embodied agents. Our post-trained Cosmos 3 models were ranked as the best open-source Text-to-Image and Image-to-Video models by Artificial Analysis, and the best policy model by RoboArena at the time the technical report was written. To accelerate open research and deployment in Physical AI, we make our code, model checkpoints, curated synthetic datasets, and evaluation benchmark available under the Linux Foundation's OpenMDW-1.1 License at https://github.com/nvidia/cosmos and https://huggingface.co/collections/

## 39 | 252 cites | 2024 | preprint | seeds=dreamer
**Diffusion Models Are Real-Time Game Engines**

Dani Valevski, Yaniv Leviathan, Moab Arar | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2408.14837 | https://openalex.org/W4402704721

Abstract: We present GameNGen, the first game engine powered entirely by a neural model that also enables real-time interaction with a complex environment over long trajectories at high quality. When trained on the classic game DOOM, GameNGen extracts gameplay and uses it to generate a playable environment that can interactively simulate new trajectories. GameNGen runs at 20 frames per second on a single TPU and remains stable over extended multi-minute play sessions. Next frame prediction achieves a PSNR of 29.4, comparable to lossy JPEG compression. Human raters are only slightly better than random chance at distinguishing short clips of the game from clips of the simulation, even after 5 minutes of auto-regressive generation. GameNGen is trained in two phases: (1) an RL-agent learns to play the game and the training sessions are recorded, and (2) a diffusion model is trained to produce the next frame, conditioned on the sequence of past frames and actions. Conditioning augmentations help ensure stable auto-regressive generation over long trajectories, and decoder fine-tuning improves the fidelity of visual details and text.

## 40 | 250 cites | 2025 | preprint | seeds=dreamer,vjepa2,worldmodels
**World Simulation with Video Foundation Models for Physical AI**

Nvidia Nvidia, :, Arslan Ali | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2511.00062 | https://openalex.org/W4415935378

Abstract: We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accel

## 41 | 248 cites | 2025 | article | seeds=dreamerv2,dreamerv3,worldmodels
**Agentic Large Language Models, a Survey**

Aske Plaat, Max J. van Duijn, Bas van Stein | Journal of Artificial Intelligence Research

DOI: https://doi.org/10.1613/jair.1.18675 | https://openalex.org/W4409318329

Abstract: Background: There is great interest in agentic LLMs, large language models that act as agents. Objectives: We review the growing body of work in this area and provide a research agenda. Methods: Agentic LLMs are LLMs that (1) reason, (2) act, and (3) interact. We organize the literature according to these three categories. Results: The research in the first category focuses on reasoning, reflection, and retrieval, aiming to improve decision making; the second category focuses on action models, robots, and tools, aiming for agents that act as useful assistants; the third category focuses on multi-agent systems, aiming for collaborative task solving and simulating interaction to study emergent social behavior. We find that works mutually benefit from results in other categories: retrieval enables tool use, reflection improves multi-agent collaboration, and reasoning benefits all categories. Conclusions: We discuss applications of agentic LLMs and provide an agenda for further research. Important applications are in medical diagnosis, logistics and financial market analysis. Meanwhile, self-reflective agents playing roles and interacting with one another augment the process of scienti

## 42 | 244 cites | 2024 | article | seeds=dreamer,dreamerv2,planet
**Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods**

Yuji Cao, Huan Qi Zhao, Yuheng Cheng | IEEE Transactions on Neural Networks and Learning Systems

DOI: https://doi.org/10.1109/tnnls.2024.3497992 | https://openalex.org/W4393763811

Abstract: With extensive pretrained knowledge and high-level general capabilities, large language models (LLMs) emerge as a promising avenue to augment reinforcement learning (RL) in aspects, such as multitask learning, sample efficiency, and high-level task planning. In this survey, we provide a comprehensive review of the existing literature in LLM-enhanced RL and summarize its characteristics compared with conventional RL methods, aiming to clarify the research scope and directions for future studies. Utilizing the classical agent-environment interaction paradigm, we propose a structured taxonomy to systematically categorize LLMs' functionalities in RL, including four roles: information processor, reward designer, decision-maker, and generator. For each role, we summarize the methodologies, analyze the specific RL challenges that are mitigated and provide insights into future directions. Finally, the comparative analysis of each role, potential applications, prospective opportunities, and challenges of the LLM-enhanced RL are discussed. By proposing this taxonomy, we aim to provide a framework for researchers to effectively leverage LLMs in the RL field, potentially accelerating RL applic

## 43 | 232 cites | 2025 | preprint | seeds=dreamerv3,vjepa2
**A Survey of Reinforcement Learning for Large Reasoning Models**

Kaiyan Zhang, Yuxin Zuo, He, Bingxiang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2509.08827 | https://openalex.org/W4417080162

Abstract: In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: https://github.com/Tsi

## 44 | 227 cites | 2024 | article | seeds=worldmodels
**Human-in-the-Loop Reinforcement Learning: A Survey and Position on Requirements, Challenges, and Opportunities**

Carl Orge Retzlaff, Srijita Das, Christabel Wayllace | Journal of Artificial Intelligence Research

DOI: https://doi.org/10.1613/jair.1.15348 | https://openalex.org/W4391399751

Abstract: Artificial intelligence (AI) and especially reinforcement learning (RL) have the potential to enable agents to learn and perform tasks autonomously with superhuman performance. However, we consider RL as fundamentally a Human-in-the-Loop (HITL) paradigm, even when an agent eventually performs its task autonomously. In cases where the reward function is challenging or impossible to define, HITL approaches are considered particularly advantageous. The application of Reinforcement Learning from Human Feedback (RLHF) in systems such as ChatGPT demonstrates the effectiveness of optimizing for user experience and integrating their feedback into the training loop. In HITL RL, human input is integrated during the agent’s learning process, allowing iterative updates and fine-tuning based on human feedback, thus enhancing the agent’s performance. Since the human is an essential part of this process, we argue that human-centric approaches are the key to successful RL, a fact that has not been adequately considered in the existing literature. This paper aims to inform readers about current explainability methods in HITL RL. It also shows how the application of explainable AI (xAI) and specific

## 45 | 225 cites | 2023 | preprint | seeds=dreamerv2
**Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation**

Wu, Hongtao, Ya Jing, Chilam Cheang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2312.13139 | https://openalex.org/W4390136276

Abstract: Generative pre-trained models have demonstrated remarkable effectiveness in language and vision domains by learning useful representations. In this paper, we extend the scope of this effectiveness by showing that visual robot manipulation can significantly benefit from large-scale video generative pre-training. We introduce GR-1, a straightforward GPT-style model designed for multi-task language-conditioned visual robot manipulation. GR-1 takes as inputs a language instruction, a sequence of observation images, and a sequence of robot states. It predicts robot actions as well as future images in an end-to-end manner. Thanks to a flexible design, GR-1 can be seamlessly finetuned on robot data after pre-trained on a large-scale video dataset. We perform extensive experiments on the challenging CALVIN benchmark and a real robot. On CALVIN benchmark, our method outperforms state-of-the-art baseline methods and improves the success rate from 88.9% to 94.9%. In the setting of zero-shot unseen scene generalization, GR-1 improves the success rate from 53.3% to 85.4%. In real robot experiments, GR-1 also outperforms baseline methods and shows strong potentials in generalization to unseen sc

## 46 | 216 cites | 2026 | preprint | seeds=dreamerv3,vjepa2,worldmodels
**Advancing Open-source World Models**

Robbyant Team, Zelin Gao, Qiuyu Wang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2601.20540 | https://openalex.org/W7126064400

Abstract: We present LingBot-World, an open-sourced world simulator stemming from video generation. Positioned as a top-tier world model, LingBot-World offers the following features. (1) It maintains high fidelity and robust dynamics in a broad spectrum of environments, including realism, scientific contexts, cartoon styles, and beyond. (2) It enables a minute-level horizon while preserving contextual consistency over time, which is also known as "long-term memory". (3) It supports real-time interactivity, achieving a latency of under 1 second when producing 16 frames per second. We provide public access to the code and model in an effort to narrow the divide between open-source and closed-source technologies. We believe our release will empower the community with practical applications across areas like content creation, gaming, and robot learning.

## 47 | 213 cites | 2025 | preprint | seeds=worldmodels
**A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges**

Zongxia Li, Wu, Xiyang, Hongyang Du | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2501.02189 | https://openalex.org/W4406133928

Abstract: Large vision-language models (VLMs) have evolved rapidly from contrastive image-text encoders and adapter-based assistants into natively multimodal foundation models that support long-context reasoning, agentic workflows, and, increasingly, unified perception-generation-action loops. Since 2025, the frontier has consolidated around a small number of large pretraining families, including GPT, Gemini, Claude, Grok, Qwen, Gemma, DeepSeek, Kimi, and MiniMax, while evaluation has shifted from short-form visual question answering toward spatial, temporal, embodied, and calibration-sensitive benchmarks. In this survey, we provide an updated overview of VLMs through 2026 with an emphasis on three developments. First, we trace the architectural evolution from contrastive or bridged two-tower models to LLM-backbone adapter models, native multimodal-input models, omni-modal unified input/output models, and emerging world-action models. Second, we summarize how benchmarks expanded from classical OCR, VQA, and chart understanding toward long-video reasoning, embodied evaluation, reward-model judging, and domain-specific decision making. Third, we review the growing role of post-training, includ

## 48 | 212 cites | 2024 | article | seeds=dreamer,dreamerv2,planet
**A survey on model-based reinforcement learning**

Fan-Ming Luo, Tian Xu, Hang Lai | Science China Information Sciences

DOI: https://doi.org/10.1007/s11432-022-3696-5 | https://openalex.org/W4391259941

Abstract: 

## 49 | 210 cites | 2023 | preprint | seeds=dreamer,dreamerv3
**Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models**

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2310.10639 | https://openalex.org/W4387725599

Abstract: If generalist robots are to operate in truly unstructured environments, they need to be able to recognize and reason about novel objects and scenarios. Such objects and scenarios might not be present in the robot's own training data. We propose SuSIE, a method that leverages an image-editing diffusion model to act as a high-level planner by proposing intermediate subgoals that a low-level controller can accomplish. Specifically, we finetune InstructPix2Pix on video data, consisting of both human videos and robot rollouts, such that it outputs hypothetical future "subgoal" observations given the robot's current observation and a language command. We also use the robot data to train a low-level goal-conditioned policy to act as the aforementioned low-level controller. We find that the high-level subgoal predictions can utilize Internet-scale pretraining and visual understanding to guide the low-level goal-conditioned policy, achieving significantly better generalization and precision than conventional language-conditioned policies. We achieve state-of-the-art results on the CALVIN benchmark, and also demonstrate robust generalization on real-world manipulation tasks, beating strong b

## 50 | 206 cites | 2024 | conference-paper | seeds=dreamerv3
**Driving Into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving**

Yuqi Wang, Jiawei He, Lue Fan | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings

DOI: https://doi.org/10.1109/cvpr52733.2024.01397 | https://openalex.org/W4402778069

Abstract: In autonomous driving, predicting future events in advance and evaluating the foreseeable risks empowers autonomous vehicles to better plan their actions, enhancing safety and efficiency on the road. To this end, we propose Drive-Wm, the first driving world model compatible with existing end-to-end planning models. Through a joint spatial-temporal modeling facilitated by view factorization, our model generates high-fidelity multiview videos in driving scenes. Building on its powerful generation ability, we showcase the potential of applying the world model for safe driving planning for the first time. Particularly, our Drive-Wm enables driving into multiple futures based on distinct driving maneuvers, and determines the optimal trajectory according to the image-based rewards. Evaluation on real-world driving datasets verifies that our method could generate high-quality, consistent, and controllable multiview videos, opening up possibilities for real-world simulations and safe planning.

## 51 | 195 cites | 2023 | conference-paper | seeds=dreamerv2
**Learning Universal Policies via Text-Guided Video Generation**

Yilun Du, Sherry Yang, Bo Dai | neural information processing systems

DOI: https://doi.org/10.52202/075280-0403 | https://openalex.org/W7133231380

Abstract: A goal of artificial intelligence is to construct an agent that can solve a wide variety of tasks. Recent progress in text-guided image synthesis has yielded models with an impressive ability to generate complex novel images, exhibiting combinatorial generalization across domains. Motivated by this success, we investigate whether such tools can be used to construct more general-purpose agents. Specifically, we cast the sequential decision making problem as a text-conditioned video generation problem, where, given a text-encoded specification of a desired goal, a planner synthesizes a set of future frames depicting its planned actions in the future, after which control actions are extracted from the generated video. By leveraging text as the underlying goal specification, we are able to naturally and combinatorially generalize to novel goals. The proposed policy-as-video formulation can further represent environments with different state and action spaces in a unified space of images, which, for example, enables learning and generalization across a variety of robot manipulation tasks. Finally, by leveraging pretrained language embeddings and widely available videos from the internet

## 52 | 190 cites | 2025 | review | seeds=ijepa
**Early warning of complex climate risk with integrated artificial intelligence**

Markus Reichstein, Vitus Benson, Jan Blunk | Nature Communications

DOI: https://doi.org/10.1038/s41467-025-57640-w | https://openalex.org/W4408503131

Abstract: Abstract As climate change accelerates, human societies face growing exposure to disasters and stress, highlighting the urgent need for effective early warning systems (EWS). These systems monitor, assess, and communicate risks to support resilience and sustainable development, but challenges remain in hazard forecasting, risk communication, and decision-making. This perspective explores the transformative potential of integrated Artificial Intelligence (AI) modeling. We highlight the role of AI in developing multi-hazard EWSs that integrate Meteorological and Geospatial foundation models (FMs) for impact prediction. A user-centric approach with intuitive interfaces and community feedback is emphasized to improve crisis management. To address climate risk complexity, we advocate for causal AI models to avoid spurious predictions and stress the need for responsible AI practices. We highlight the FATES (Fairness, Accountability, Transparency, Ethics, and Sustainability) principles as essential for equitable and trustworthy AI-based Early Warning Systems for all. We further advocate for decadal EWSs, leveraging climate ensembles and generative methods to enable long-term, spatially re

## 53 | 188 cites | 2026 | preprint | seeds=vjepa2
**StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing**

StarVLA Community | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2604.05014 | https://openalex.org/W7151757339

Abstract: Building generalist embodied agents requires integrating perception, language understanding, and action, which are core capabilities addressed by Vision-Language-Action (VLA) approaches based on multimodal foundation models, including recent advances in vision-language models and world models. Despite rapid progress, VLA methods remain fragmented across incompatible architectures, codebases, and evaluation protocols, hindering principled comparison and reproducibility. We present StarVLA, an open-source codebase for VLA research. StarVLA addresses these challenges in three aspects. First, it provides a modular backbone--action-head architecture that supports both VLM backbones (e.g., Qwen-VL) and world-model backbones (e.g., Cosmos) alongside representative action-decoding paradigms, all under a shared abstraction in which backbone and action head can each be swapped independently. Second, it provides reusable training strategies, including cross-embodiment learning and multimodal co-training, that apply consistently across supported paradigms. Third, it integrates major benchmarks, including LIBERO, SimplerEnv, RoboTwin~2.0, RoboCasa-GR1, and BEHAVIOR-1K, through a unified evaluat

## 54 | 187 cites | 2025 | preprint | seeds=dreamerv3
**ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving**

Li, Yongkang, Kaixin Xiong, X. Y. Guo | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2506.08052 | https://openalex.org/W4415337225

Abstract: Recent studies have explored leveraging the world knowledge and cognitive capabilities of Vision-Language Models (VLMs) to address the long-tail problem in end-to-end autonomous driving. However, existing methods typically formulate trajectory planning as a language modeling task, where physical actions are output in the language space, potentially leading to issues such as format-violating outputs, infeasible actions, and slow inference speeds. In this paper, we propose ReCogDrive, a novel Reinforced Cognitive framework for end-to-end autonomous Driving, unifying driving understanding and planning by integrating an autoregressive model with a diffusion planner. First, to instill human driving cognition into the VLM, we introduce a hierarchical data pipeline that mimics the sequential cognitive process of human drivers through three stages: generation, refinement, and quality control. Building on this cognitive foundation, we then address the language-action mismatch by injecting the VLM's learned driving priors into a diffusion planner to efficiently generate continuous and stable trajectories. Furthermore, to enhance driving safety and reduce collisions, we introduce a Diffusion 

## 55 | 186 cites | 2023 | conference-paper | seeds=dreamer
**Language-Driven Representation Learning for Robotics**

Siddharth Karamcheti, Suraj Nair, Annie Chen | Robotics: Science and Systems

DOI: https://doi.org/10.15607/rss.2023.xix.032 | https://openalex.org/W4385431288

Abstract: Recent work in visual representation learning for robotics demonstrates the viability of learning from large video datasets of humans performing everyday tasks. Leveraging methods such as masked autoencoding and contrastive learning, these representations exhibit strong transfer to policy learning for visuomotor control. But, robot learning encompasses a diverse set of problems beyond control including grasp affordance prediction, language-conditioned imitation learning, and intent scoring for human-robot collaboration, amongst others. First, we demonstrate that existing representations yield inconsistent results across these tasks: masked autoencoding approaches pick up on low-level spatial features at the cost of high-level semantics, while contrastive learning approaches capture the opposite. We then introduce Voltron, a framework for language-driven representation learning from videos and associated captions. Voltron trades off language-conditioned visual reconstruction to learn low-level visual patterns, and visually-grounded language generation to encode high-level semantics. We also construct an evaluation suite spanning five distinct robot learning problems -a unified platf

## 56 | 185 cites | 2025 | conference-paper | seeds=dreamer,dreamerv2,dreamerv3
**DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation**

Guosheng Zhao, Xiaofeng Wang, Zheng Zhu | Proceedings of the AAAI Conference on Artificial Intelligence

DOI: https://doi.org/10.1609/aaai.v39i10.33130 | https://openalex.org/W4409366191

Abstract: World models have demonstrated superiority in autonomous driving, particularly in the generation of multi-view driving videos. However, significant challenges still exist in generating customized driving videos. In this paper, we propose DriveDreamer-2, which incorporates a Large Language Model (LLM) to facilitate the creation of user-defined driving videos. Specifically, a trajectory generation function library is developed to produce trajectories that conform to user descriptions. Subsequently, an HDMap generator is designed to learn the mapping from trajectories to road structures. Ultimately, we propose the Unified Multi-View Model (UniMVM) to enhance temporal and spatial coherence in the generated multi-view driving videos. To the best of our knowledge, DriveDreamer-2 is the first world model to generate customized driving videos, and it can generate uncommon driving videos (e.g., vehicles abruptly cut in) in a user-friendly manner. Besides, experimental results demonstrate that the generated videos enhance the training of driving perception methods (e.g., 3D detection and tracking). Furthermore, video generation quality of DriveDreamer-2 surpasses other state-of-the-art metho

## 57 | 182 cites | 2024 | review | seeds=dreamer,dreamerv2
**Deep learning in computational mechanics: a review**

Leon Herrmann, Stefan Kollmannsberger | Computational Mechanics

DOI: https://doi.org/10.1007/s00466-023-02434-4 | https://openalex.org/W4390839314

Abstract: Abstract The rapid growth of deep learning research, including within the field of computational mechanics, has resulted in an extensive and diverse body of literature. To help researchers identify key concepts and promising methodologies within this field, we provide an overview of deep learning in deterministic computational mechanics. Five main categories are identified and explored: simulation substitution, simulation enhancement, discretizations as neural networks, generative approaches, and deep reinforcement learning. This review focuses on deep learning methods rather than applications for computational mechanics, thereby enabling researchers to explore this field more effectively. As such, the review is not necessarily aimed at researchers with extensive knowledge of deep learning—instead, the primary audience is researchers on the verge of entering this field or those attempting to gain an overview of deep learning in computational mechanics. The discussed concepts are, therefore, explained as simple as possible.

## 58 | 181 cites | 2024 | conference-paper | seeds=dreamer,dreamerv3
**Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability**

Shenyuan Gao, Jiazhi Yang, Li Chen | neural information processing systems

DOI: https://doi.org/10.52202/079017-2906 | https://openalex.org/W4415796242

Abstract: World models can foresee the outcomes of different actions, which is of paramount importance for autonomous driving. Nevertheless, existing driving world models still have limitations in generalization to unseen environments, prediction fidelity of critical details, and action controllability for flexible application. In this paper, we present Vista, a generalizable driving world model with high fidelity and versatile controllability. Based on a systematic diagnosis of existing methods, we introduce several key ingredients to address these limitations. To accurately predict real-world dynamics at high resolution, we propose two novel losses to promote the learning of moving instances and structural information. We also devise an effective latent replacement approach to inject historical frames as priors for coherent long-horizon rollouts. For action controllability, we incorporate a versatile set of controls from high-level intentions (command, goal point) to low-level maneuvers (trajectory, angle, and speed) through an efficient learning strategy. After large-scale training, the capabilities of Vista can seamlessly generalize to different scenarios. Extensive experiments on multip

## 59 | 181 cites | 2025 | article | seeds=ijepa
**Exploring scalable medical image encoders beyond text supervision**

Fernando Pérez‐García, Harshita Sharma, Sam Bond-Taylor | Nature Machine Intelligence

DOI: https://doi.org/10.1038/s42256-024-00965-w | https://openalex.org/W4406353025

Abstract: 

## 60 | 180 cites | 2025 | article | seeds=dreamer,dreamerv2,dreamerv3,vjepa,vjepa2,worldmodels
**Understanding World or Predicting Future? A Comprehensive Survey of World Models**

Jingtao Ding, Yunke Zhang, Yu Shang | ACM Computing Surveys

DOI: https://doi.org/10.1145/3746449 | https://openalex.org/W4411735734

Abstract: The concept of world models has garnered significant attention due to advancements in multimodal large language models such as GPT-4 and video generation models such as Sora, which are central to the pursuit of artificial general intelligence. This survey offers a comprehensive review of the literature on world models. Generally, world models are regarded as tools for either understanding the present state of the world or predicting its future dynamics. This review presents a systematic categorization of world models, emphasizing two primary functions: (1) constructing internal representations to understand the mechanisms of the world, and (2) predicting future states to simulate and guide decision-making. Initially, we examine the current progress in these two categories. We then explore the application of world models in key domains, including autonomous driving, robotics, and social simulacra, with a focus on how each domain utilizes these aspects. Finally, we outline key challenges and provide insights into potential future research directions. We summarize the representative articles along with their code repositories in https://github.com/tsinghua-fib-lab/World-Model .

## 61 | 179 cites | 2025 | review | seeds=rwm
**Generative artificial intelligence: a historical perspective**

Ran He, Jie Cao, Tieniu Tan | National Science Review

DOI: https://doi.org/10.1093/nsr/nwaf050 | https://openalex.org/W4407861126

Abstract: Generative artificial intelligence (GAI) has recently achieved significant success, enabling anyone to create texts, images, videos and even computer codes while providing insights that might not be possible with traditional tools. To stimulate future research, this work provides a brief summary of the ongoing and historical developments in GAI over the past 70 years. The achievements are grouped into four categories: (i) rule-based generative systems that follow specialized rules and instructions, (ii) model-based generative algorithms that produce new content based on statistical or graphical models, (iii) deep generative methodologies that utilize deep neural networks to learn how to generate new content from data and (iv) foundation models that are trained on extensive datasets and capable of performing a variety of generative tasks. This paper also reviews successful generative applications and identifies open challenges posed by remaining issues. In addition, this paper describes potential research directions aimed at better utilizing, understanding and harnessing GAI technologies.

## 62 | 177 cites | 2026 | preprint | seeds=vjepa2
**DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos**

Shenyuan Gao, William Liang, Kaiyuan Zheng | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2602.06949 | https://openalex.org/W7128357184

Abstract: Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.81 FPS and further improves context consistency. Our work enables several important applications based on generative world models, including l

## 63 | 177 cites | 2024 | preprint | seeds=worldmodels
**Tree Search for Language Model Agents**

Jing Yu Koh, Stephen McAleer, Daniel Fried | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2407.01476 | https://openalex.org/W4400337884

Abstract: Autonomous agents powered by language models (LMs) have demonstrated promise in their ability to perform decision-making tasks such as web automation. However, a key limitation remains: LMs, primarily optimized for natural language understanding and generation, struggle with multi-step reasoning, planning, and using environmental feedback when attempting to solve realistic computer tasks. Towards addressing this, we propose an inference-time search algorithm for LM agents to explicitly perform exploration and multi-step planning in interactive web environments. Our approach is a form of best-first tree search that operates within the actual environment space, and is complementary with most existing state-of-the-art agents. It is the first tree search algorithm for LM agents that shows effectiveness on realistic web tasks. On the challenging VisualWebArena benchmark, applying our search algorithm on top of a GPT-4o agent yields a 39.7% relative increase in success rate compared to the same baseline without search, setting a state-of-the-art success rate of 26.4%. On WebArena, search also yields a 28.0% relative improvement over a baseline agent, setting a competitive success rate of

## 64 | 174 cites | 2024 | preprint | seeds=dreamer,dreamerv2,dreamerv3
**How Far is Video Generation from World Model: A Physical Law Perspective**

Kang, Bingyi, Yue, Yang, Rui Lu | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2411.02385 | https://openalex.org/W4404355547

Abstract: OpenAI's Sora highlights the potential of video generation for developing world models that adhere to fundamental physical laws. However, the ability of video generation models to discover such laws purely from visual data without human priors can be questioned. A world model learning the true law should give predictions robust to nuances and correctly extrapolate on unseen scenarios. In this work, we evaluate across three key scenarios: in-distribution, out-of-distribution, and combinatorial generalization. We developed a 2D simulation testbed for object movement and collisions to generate videos deterministically governed by one or more classical mechanics laws. This provides an unlimited supply of data for large-scale experimentation and enables quantitative evaluation of whether the generated videos adhere to physical laws. We trained diffusion-based video generation models to predict object movements based on initial frames. Our scaling experiments show perfect generalization within the distribution, measurable scaling behavior for combinatorial generalization, but failure in out-of-distribution scenarios. Further experiments reveal two key insights about the generalization me

## 65 | 171 cites | 2023 | article | seeds=worldmodels
**Recent advances in applying deep reinforcement learning for flow control: Perspectives and future directions**

Colin Vignon, Jean Rabault, Ricardo Vinuesa | Physics of Fluids

DOI: https://doi.org/10.1063/5.0143913 | https://openalex.org/W4321764793

Abstract: Deep reinforcement learning (DRL) has been applied to a variety of problems during the past decade and has provided effective control strategies in high-dimensional and non-linear situations that are challenging to traditional methods. Flourishing applications now spread out into the field of fluid dynamics and specifically active flow control (AFC). In the community of AFC, the encouraging results obtained in two-dimensional and chaotic conditions have raised the interest to study increasingly complex flows. In this review, we first provide a general overview of the reinforcement-learning and DRL frameworks, as well as their recent advances. We then focus on the application of DRL to AFC, highlighting the current limitations of the DRL algorithms in this field, and suggesting some of the potential upcoming milestones to reach, as well as open questions that are likely to attract the attention of the fluid mechanics community.

## 66 | 165 cites | 2025 | preprint | seeds=dreamer,dreamerv2,worldmodels,planet
**Training Agents Inside of Scalable World Models**

Danijar Hafner, Wilson Yan, Timothy Lillicrap | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2509.24527 | https://openalex.org/W4415336982

Abstract: World models learn general knowledge from videos and simulate experience for training behaviors in imagination, offering a path towards intelligent agents. However, previous world models have been unable to accurately predict object interactions in complex environments. We introduce Dreamer 4, a scalable agent that learns to solve control tasks by reinforcement learning inside of a fast and accurate world model. In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin. The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture. Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos. We propose the challenge of obtaining diamonds in Minecraft from only offline data, aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow. This task requires choosing sequences of over 20,000 mouse and keyboard action

## 67 | 165 cites | 2024 | review | seeds=rwm,planet
**Self-Supervised Multimodal Learning: A Survey**

Yongshuo Zong, Oisin Mac Aodha, Timothy M. Hospedales | IEEE Transactions on Pattern Analysis and Machine Intelligence

DOI: https://doi.org/10.1109/tpami.2024.3429301 | https://openalex.org/W4401387482

Abstract: Multimodal learning, which aims to understand and analyze information from multiple modalities, has achieved substantial progress in the supervised regime in recent years. However, the heavy dependence on data paired with expensive human annotations impedes scaling up models. Meanwhile, given the availability of large-scale unannotated data in the wild, self-supervised learning has become an attractive strategy to alleviate the annotation bottleneck. Building on these two directions, self-supervised multimodal learning (SSML) provides ways to learn from raw multimodal data. In this survey, we provide a comprehensive review of the state-of-the-art in SSML, in which we elucidate three major challenges intrinsic to self-supervised learning with multimodal data: 1) learning representations from multimodal data without labels, 2) fusion of different modalities, and 3) learning with unaligned data. We then detail existing solutions to these challenges. Specifically, we consider 1) objectives for learning from multimodal unlabeled data via self-supervision, 2) model architectures from the perspective of different multimodal fusion strategies, and 3) pair-free learning strategies for coars

## 68 | 162 cites | 2025 | conference-paper | seeds=worldmodels
**Navigation World Models**

Amir Bar, Gaoyue Zhou, Danny Tran | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings

DOI: https://doi.org/10.1109/cvpr52734.2025.01472 | https://openalex.org/W4413147084

Abstract: Navigation is a fundamental skill of agents with visual-motor capabilities. We introduce a Navigation World Model (NWM), a controllable video generation model that predicts future visual observations based on past observations and navigation actions. To capture complex environment dynamics, NWM employs a Conditional Diffusion Transformer (CDiT), trained on a diverse collection of egocentric videos of both human and robotic agents, and scaled up to 1 billion parameters. In familiar environments, NWM can plan navigation trajectories by simulating them and evaluating whether they achieve the desired goal. Unlike supervised navigation policies with fixed behavior, NWM can dynamically incorporate constraints during planning. Experiments demonstrate its effectiveness in planning trajectories from scratch or by ranking trajectories sampled from an external policy. Furthermore, NWM leverages its learned visual priors to imagine trajectories in unfamiliar environments from a single input image, making it a flexible and powerful tool for next-generation navigation systems1.

## 69 | 153 cites | 2026 | conference-paper | seeds=ijepa,vjepa,vjepa2,worldmodels
**V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**

Lorenzo Mur-Labadia, Matthew J. Muckley, Amir Bar | Lecture notes in computer science

DOI: https://doi.org/10.1007/978-3-032-37369-4_36 | https://openalex.org/W7137837393

Abstract: 

## 70 | 152 cites | 2026 | conference-paper | seeds=vjepa2
**mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs**

Jonas Pai, Liam Achenbach, Oliver Sanchez | Robotics: Science and Systems

DOI: https://doi.org/10.15607/rss.2026.xxii.077 | https://openalex.org/W7116084174

Abstract: Prevailing Vision-Language-Action Models (VLAs) for robotic manipulation are built upon vision-language backbones pretrained on large-scale, but disconnected static web data. As a result, despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories. This reliance creates an unsustainable data burden, necessitating continuous, large-scale expert data collection to compensate for the lack of innate physical understanding. We contend that while vision-language pretraining effectively captures semantic priors, it remains blind to physical causality. A more effective paradigm leverages video to jointly capture semantics and visual dynamics during pretraining, thereby isolating the remaining task of low-level control. To this end, we introduce mimic-video, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations. The decoder serves as an Inverse Dynamics Model (IDM), generating low-level robot actions from the latent representation of video-space action plans. Our extensive evaluatio

## 71 | 152 cites | 2024 | conference-paper | seeds=worldmodels
**OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving**

Wenzhao Zheng, Weiliang Chen, Yuanhui Huang | Lecture notes in computer science

DOI: https://doi.org/10.1007/978-3-031-72624-8_4 | https://openalex.org/W4403771534

Abstract: 

## 72 | 151 cites | 2025 | preprint | seeds=dreamer,dreamerv2
**Ctrl-World: A Controllable Generative World Model for Robot Manipulation**

Yanjiang Guo, Lucy Xiaoyang Shi, Chen, Jianyu | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2510.10125 | https://openalex.org/W4415273528

Abstract: Generalist robot policies can now perform a wide range of manipulation skills, but evaluating and improving their ability with unfamiliar objects and instructions remains a significant challenge. Rigorous evaluation requires a large number of real-world rollouts, while systematic improvement demands additional corrective data with expert labels. Both of these processes are slow, costly, and difficult to scale. World models offer a promising, scalable alternative by enabling policies to rollout within imagination space. However, a key challenge is building a controllable world model that can handle multi-step interactions with generalist robot policies. This requires a world model compatible with modern generalist policies by supporting multi-view prediction, fine-grained action control, and consistent long-horizon interactions, which is not achieved by previous works. In this paper, we make a step forward by introducing a controllable multi-view world model that can be used to evaluate and improve the instruction-following ability of generalist robot policies. Our model maintains long-horizon consistency with a pose-conditioned memory retrieval mechanism and achieves precise action

## 73 | 151 cites | 2024 | conference-paper | seeds=dreamerv3
**HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation**

Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin | Robotics: Science and Systems

DOI: https://doi.org/10.15607/rss.2024.xx.061 | https://openalex.org/W4402354138

Abstract: Fig. 1: Humanoid robots equipped with dexterous hands hold immense promise for integration into real-world human environments.Nonetheless, harnessing the full potential of humanoid robots presents numerous challenges, such as the intricate control of robots with complex dynamics, sophisticated coordination among various body parts, and addressing long-horizon complex tasks envisioned for these robots.We present HumanoidBench, a simulated humanoid robot benchmark consisting of 15 whole-body manipulation and 12 locomotion tasks, such as shelf rearrangement, package unloading, and maze navigation.Abstract-Humanoid robots hold great promise in assisting humans in diverse environments and tasks, due to their flexibility and adaptability leveraging human-like morphology.However, research in humanoid robots is often bottlenecked by the costly and fragile hardware setups.To accelerate algorithmic research in humanoid robots, we present a high-dimensional, simulated robot learning benchmark, HumanoidBench, featuring a humanoid robot equipped with dexterous hands and a variety of challenging whole-body manipulation and locomotion tasks.Our findings reveal that state-of-the-art reinforcement 

## 74 | 150 cites | 2024 | article | seeds=worldmodels
**A Review on Recent Trends of Bioinspired Soft Robotics: Actuators, Control Methods, Materials Selection, Sensors, Challenges, and Future Prospects**

Abhirup Sarker, Md. Tamzid Islam, Md. Robiul Islam | Advanced Intelligent Systems

DOI: https://doi.org/10.1002/aisy.202400414 | https://openalex.org/W4402998604

Abstract: Bioinspired soft robotics is an emerging field that aims to develop flexible and adaptive robots inspired by the movement and capabilities of biological organisms. This review article examines recent advances in materials, actuation mechanisms, sensors, and control strategies and discusses the challenges and future prospects of bioinspired soft robotics. Key innovations highlighted include pneumatic, elastomer actuators, variable‐length shape memory alloy tendons, closed‐loop control with soft sensors, and the incorporation of soft materials including shape memory polymers and conductive composites. Challenges in soft robotics such as achieving complex motion control, incorporating feedback systems, modeling soft material dynamics, and replicating biological muscle efficiency with artificial muscles are also discussed. Promising future directions are explored including the integration of biodegradable materials, machine learning‐based control algorithms, and leveraging data‐driven techniques for modeling and control. Building on progress in multi‐functional materials, manufacturing techniques, and bioinspired design principles, soft robots hold considerable promise for expanding ro

## 75 | 149 cites | 2025 | article | seeds=ijepa
**SARATR-X: Toward Building a Foundation Model for SAR Target Recognition**

Weijie Li, Wei Yang, Yuenan Hou | IEEE Transactions on Image Processing

DOI: https://doi.org/10.1109/tip.2025.3531988 | https://openalex.org/W4406921704

Abstract: Despite the remarkable progress in synthetic aperture radar automatic target recognition (SAR ATR), recent efforts have concentrated on detecting and classifying a specific category, e.g., vehicles, ships, airplanes, or buildings. One of the fundamental limitations of the top-performing SAR ATR methods is that the learning paradigm is supervised, task-specific, limited-category, closed-world learning, which depends on massive amounts of accurately annotated samples that are expensively labeled by expert SAR analysts and have limited generalization capability and scalability. In this work, we make the first attempt towards building a foundation model for SAR ATR, termed SARATR-X. SARATR-X learns generalizable representations via self-supervised learning (SSL) and provides a cornerstone for label-efficient model adaptation to generic SAR target detection and classification tasks. Specifically, SARATR-X is trained on 0.18 M unlabelled SAR target samples, which are curated by combining contemporary benchmarks and constitute the largest publicly available dataset till now. Considering the characteristics of SAR images, a backbone tailored for SAR ATR is carefully designed, and a two-ste

## 76 | 148 cites | 2023 | preprint | seeds=dreamerv3
**Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis**

Hubert Siuzdak | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2306.00814 | https://openalex.org/W4379259581

Abstract: Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourier-based time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complex-valued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared to prevailing time-domain neural vocoding approaches. The source code and model weights have been open-sourced at https://github.com/gemelo-ai/vocos.

## 77 | 148 cites | 2023 | article | seeds=rwm
**Privacy-Preserving and Communication-Efficient Energy Prediction Scheme Based on Federated Learning for Smart Grids**

Mahmoud M. Badr, Mohamed M. E. A. Mahmoud, Yuguang Michael Fang | IEEE Internet of Things Journal

DOI: https://doi.org/10.1109/jiot.2022.3230586 | https://openalex.org/W4313591693

Abstract: Energy forecasting is important because it enables infrastructure planning and power dispatching while reducing power outages and equipment failures. It is well-known that federated learning (FL) can be used to build a global energy predictor for smart grids without revealing the customers’ raw data to preserve privacy. However, it still reveals local models’ parameters during the training process, which may still leak customers’ data privacy. In addition, for the global model to converge, it requires multiple training rounds, which must be done in a communication-efficient way. Moreover, most existing works only focus on load forecasting while neglecting energy forecasting in net-metering systems. To address these limitations, in this article, we propose a privacy-preserving and communication-efficient FL-based energy predictor for net-metering systems. Based on a data set for real power consumption/generation readings, we first propose a multidata-source hybrid deep learning (DL)-based predictor to accurately predict future readings. Then, we repurpose an efficient inner-product functional encryption (IPFE) scheme for implementing secure data aggregation to preserve the customers

## 78 | 142 cites | 2025 | conference-paper | seeds=ijepa
**REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers**

Xingjian Leng, Jaskirat Singh, Yunzhong Hou | IEEE/CVF International Conference on Computer Vision (ICCV)

DOI: https://doi.org/10.1109/iccv51701.2025.01697 | https://openalex.org/W4415161210

Abstract: In this paper we tackle a fundamental question: "Can we train latent diffusion models together with the variational auto-encoder (VAE) tokenizer in an end-to-end manner?" Traditional deep-learning wisdom dictates that end-to-end training is often preferable when possible. However, for latent diffusion transformers, it is observed that end-to-end training both VAE and diffusion-model using standard diffusion-loss is ineffective, even causing a degradation in final performance. We show that while diffusion loss is ineffective, end-to-end training can be unlocked through the representation-alignment (REPA) loss -- allowing both VAE and diffusion model to be jointly tuned during the training process. Despite its simplicity, the proposed training recipe (REPA-E) shows remarkable performance; speeding up diffusion model training by over 17x and 45x over REPA and vanilla training recipes, respectively. Interestingly, we observe that end-to-end tuning with REPA-E also improves the VAE itself; leading to improved latent space structure and downstream generation performance. In terms of final performance, our approach sets a new state-of-the-art; achieving FID of 1.12 and 1.69 with and witho

## 79 | 138 cites | 2025 | conference-paper | seeds=dreamer,dreamerv2,worldmodels
**WorldMem: Long-term Consistent World Simulation with Memory**

Zeqi Xiao, Yushi Lan, Yifan Zhou | neural information processing systems

DOI: https://doi.org/10.52202/085713-1659 | https://openalex.org/W4417089341

Abstract: World simulation has gained increasing popularity due to its ability to model virtual environments and predict the consequences of actions. However, the limited temporal context window often leads to failures in maintaining long-term consistency, particularly in preserving 3D spatial consistency. In this work, we present WorldMem, a framework that enhances scene generation with a memory bank consisting of memory units that store memory frames and states (e.g., poses and timestamps). By employing a memory attention mechanism that effectively extracts relevant information from these memory frames based on their states, our method is capable of accurately reconstructing previously observed scenes, even under significant viewpoint or temporal gaps. Furthermore, by incorporating timestamps into the states, our framework not only models a static world but also captures its dynamic evolution over time, enabling both perception and interaction within the simulated world. Extensive experiments in both virtual and real scenarios validate the effectiveness of our approach.

## 80 | 135 cites | 2025 | conference-paper | seeds=dreamerv3
**Demonstrating MuJoCo Playground**

Kevin Zakka, Baruch Tabanpour, Qiayuan Liao | Robotics: Science and Systems

DOI: https://doi.org/10.15607/rss.2025.xxi.020 | https://openalex.org/W4414079055

Abstract: We introduce MuJoCo Playground, a fully open- source framework for robot learning built with MJX, with the express goal of streamlining simulation, training, and sim- to-real transfer onto robots. With a simple pip install playground, researchers can train policies in minutes on a single GPU. Playground supports diverse robotic platforms, including quadrupeds, humanoids, dexterous hands, and robotic arms, enabling zero-shot sim-to-real transfer from both state and pixel inputs. This is achieved through an integrated stack comprising a physics engine, batch renderer, and training en- vironments. Along with video results, the entire framework is freely available at playground.mujoco.org.

## 81 | 135 cites | 2025 | article | seeds=dreamerv3
**Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications**

Kento Kawaharazuka, Jihoon Oh, Jun Yamada | IEEE Access

DOI: https://doi.org/10.1109/access.2025.3609980 | https://openalex.org/W4414198610

Abstract: Amid growing efforts to leverage advances in large language models (LLMs) and visionlanguage models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data c

## 82 | 133 cites | 2025 | preprint | seeds=dreamer,worldmodels
**Unified Vision-Language-Action Model**

Yuqi Wang, Xinghang Li, Wenxuan Wang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2506.19850 | https://openalex.org/W4414685219

Abstract: Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA 

## 83 | 130 cites | 2026 | conference-paper | seeds=ijepa,vjepa2
**VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model**

Jingwen Sun, Wenyao Zhang, Zekun Qi | Lecture notes in computer science

DOI: https://doi.org/10.1007/978-3-032-37196-6_26 | https://openalex.org/W7128621118

Abstract: 

## 84 | 129 cites | 2023 | preprint | seeds=dreamerv3
**Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks**

Haoqi Yuan, Chi Zhang, Hongcheng Wang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2303.16563 | https://openalex.org/W4361806741

Abstract: We study building multi-task agents in open-world environments. Without human demonstrations, learning to accomplish long-horizon tasks in a large open-world environment with reinforcement learning (RL) is extremely inefficient. To tackle this challenge, we convert the multi-task learning problem into learning basic skills and planning over the skills. Using the popular open-world game Minecraft as the testbed, we propose three types of fine-grained basic skills, and use RL with intrinsic rewards to acquire skills. A novel Finding-skill that performs exploration to find diverse items provides better initialization for other skills, improving the sample efficiency for skill learning. In skill planning, we leverage the prior knowledge in Large Language Models to find the relationships between skills and build a skill graph. When the agent is solving a task, our skill search algorithm walks on the skill graph and generates the proper skill plans for the agent. In experiments, our method accomplishes 40 diverse Minecraft tasks, where many tasks require sequentially executing for more than 10 skills. Our method outperforms baselines by a large margin and is the most sample-efficient dem

## 85 | 128 cites | 2024 | preprint | seeds=ijepa,vjepa
**Large Concept Models: Language Modeling in a Sentence Representation Space**

LCM Team, Loïc Barrault, Paul-Ambroise Duquenne | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2412.08821 | https://openalex.org/W4405354515

Abstract: LLMs have revolutionized the field of artificial intelligence and have emerged as the de-facto tool for many tasks. The current established technology of LLMs is to process input and generate output at the token level. This is in sharp contrast to humans who operate at multiple levels of abstraction, well beyond single words, to analyze information and to generate creative content. In this paper, we present an attempt at an architecture which operates on an explicit higher-level semantic representation, which we name a concept. Concepts are language- and modality-agnostic and represent a higher level idea or action in a flow. Hence, we build a "Large Concept Model". In this study, as proof of feasibility, we assume that a concept corresponds to a sentence, and use an existing sentence embedding space, SONAR, which supports up to 200 languages in both text and speech modalities. The Large Concept Model is trained to perform autoregressive sentence prediction in an embedding space. We explore multiple approaches, namely MSE regression, variants of diffusion-based generation, and models operating in a quantized SONAR space. These explorations are performed using 1.6B parameter models 

## 86 | 126 cites | 2026 | preprint | seeds=vjepa2
**Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory**

Zile Wang, Zexiang Liu, Jiaxing Li | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2604.08995 | https://openalex.org/W7154071213

Abstract: With the advancement of interactive video generation, diffusion models have increasingly demonstrated their potential as world models. However, existing approaches still struggle to simultaneously achieve memory-enabled long-term temporal consistency and high-resolution real-time generation, limiting their applicability in real-world scenarios. To address this, we present Matrix-Game 3.0, a memory-augmented interactive world model designed for 720p real-time longform video generation. Building upon Matrix-Game 2.0, we introduce systematic improvements across data, model, and inference. First, we develop an upgraded industrial-scale infinite data engine that integrates Unreal Engine-based synthetic data, large-scale automated collection from AAA games, and real-world video augmentation to produce high-quality Video-Pose-Action-Prompt quadruplet data at scale. Second, we propose a training framework for long-horizon consistency: by modeling prediction residuals and re-injecting imperfect generated frames during training, the base model learns self-correction; meanwhile, camera-aware memory retrieval and injection enable the base model to achieve long horizon spatiotemporal consistenc

## 87 | 126 cites | 2025 | preprint | seeds=vjepa2
**A Survey on Vision-Language-Action Models: An Action Tokenization Perspective**

Zhong, Yifan, Bai, Fengshuo, Shaofei Cai | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2507.01925 | https://openalex.org/W4417042701

Abstract: The remarkable advancements of vision and language foundation models in multimodal understanding, reasoning, and generation has sparked growing efforts to extend such intelligence to the physical world, fueling the flourishing of vision-language-action (VLA) models. Despite seemingly diverse approaches, we observe that current VLA models can be unified under a single framework: vision and language inputs are processed by a series of VLA modules, producing a chain of \textit{action tokens} that progressively encode more grounded and actionable information, ultimately generating executable actions. We further determine that the primary design choice distinguishing VLA models lies in how action tokens are formulated, which can be categorized into language description, code, affordance, trajectory, goal state, latent representation, raw action, and reasoning. However, there remains a lack of comprehensive understanding regarding action tokens, significantly impeding effective VLA development and obscuring future directions. Therefore, this survey aims to categorize and interpret existing VLA research through the lens of action tokenization, distill the strengths and limitations of each

## 88 | 118 cites | 2025 | preprint | seeds=dreamer,worldmodels
**MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft**

Guo, Junliang, Ye Yang, He, Tianyu | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2504.08388 | https://openalex.org/W4414828502

Abstract: World modeling is a crucial task for enabling intelligent agents to effectively interact with humans and operate in dynamic environments. In this work, we propose MineWorld, a real-time interactive world model on Minecraft, an open-ended sandbox game which has been utilized as a common testbed for world modeling. MineWorld is driven by a visual-action autoregressive Transformer, which takes paired game scenes and corresponding actions as input, and generates consequent new scenes following the actions. Specifically, by transforming visual game scenes and actions into discrete token ids with an image tokenizer and an action tokenizer correspondingly, we consist the model input with the concatenation of the two kinds of ids interleaved. The model is then trained with next token prediction to learn rich representations of game states as well as the conditions between states and actions simultaneously. In inference, we develop a novel parallel decoding algorithm that predicts the spatial redundant tokens in each frame at the same time, letting models in different scales generate $4$ to $7$ frames per second and enabling real-time interactions with game players. In evaluation, we propos

## 89 | 118 cites | 2025 | conference-paper | seeds=dreamerv3,worldmodels
**IRASim: A Fine-Grained World Model for Robot Manipulation**

Fangqi Zhu, Hongtao Wu, Song Guo | IEEE/CVF International Conference on Computer Vision (ICCV)

DOI: https://doi.org/10.1109/iccv51701.2025.00917 | https://openalex.org/W7160076956

Abstract: World models allow autonomous agents to plan and explore by predicting the visual outcomes of different actions. However, for robot manipulation, it is challenging to accurately model the fine-grained robot-object interaction within the visual space using existing methods which overlooks precise alignment between each action and the corresponding frame. In this paper, we present IRASim, a novel world model capable of generating videos with fine-grained robot-object interaction details, conditioned on historical observations and robot action trajectories. We train a diffusion transformer and introduce a novel frame-level action-conditioning module within each transformer block to explicitly model and strengthen the action-frame alignment. Extensive experiments show that: (1) the quality of the videos generated by our method surpasses all the baseline methods and scales effectively with increased model size and computation; (2) policy evaluations using IRASim exhibit a strong correlation with those using the ground-truth simulator, highlighting its potential to accelerate real-world policy evaluation; (3) testing-time scaling through model-based planning with IRASim significantly enh

## 90 | 117 cites | 2026 | article | seeds=dreamerv3
**State space model for new generation network alternative to transformers: a survey**

Xiao Wang, Shiao Wang, Yuhe Ding | Visual Intelligence

DOI: https://doi.org/10.1007/s44267-026-00133-1 | https://openalex.org/W4394866661

Abstract: Abstract In the post-deep learning era, the Transformer architecture has demonstrated its powerful performance in pre-trained large models and various downstream tasks. However, the enormous computational demands of Transformers have deterred many researchers. To reduce the complexity of attention models, numerous efforts have been made to design more efficient methods. Among them, the state space model (SSM), as a possible replacement for the self-attention based Transformer model, has drawn more and more attention in recent years. This paper provides the first comprehensive review of SSM works and offers experimental comparisons and analyses to better demonstrate SSM’s features and advantages. First, we provide a detailed description of the principles to help the readers quickly grasp the key ideas of the SSM. Then, we will review existing SSMs and their various applications, including those in the fields of natural language processing, computer vision, graphs, multi-modal and multi-media, point cloud and event stream, and other domains. Next, we provide statistical comparisons and analyses of these models to help readers understand how different structures affect performance on 

## 91 | 117 cites | 2024 | article | seeds=dreamerv3
**Artificial intelligence research: A review on dominant themes, methods, frameworks and future research directions**

Kingsley Ofosu‐Ampong | Telematics and Informatics Reports

DOI: https://doi.org/10.1016/j.teler.2024.100127 | https://openalex.org/W4392560024

Abstract: This article presents an analysis of artificial intelligence (AI) in information systems and innovation-related journals to determine the current issues and stock of knowledge in AI literature, research methodology, frameworks, level of analysis and conceptual approaches to identify research gaps for future investigations. A total of 85 peer-reviewed articles from 2020 to 2023 were used in the analysis. The findings show that extant literature is skewed towards the prevalence of technological issues and highlights the relatively lower focus on other themes, such as contextual knowledge co-creation issues, conceptualisation, and application domains. While there have been increasing technological issues with artificial intelligence, the three identified areas of security concern are data security, model security and network security. Furthermore, the review found that contemporary AI, which continually drives the boundaries of computational capabilities to tackle increasingly intricate decision-making challenges, distinguishes itself from earlier iterations in two primary aspects that significantly affect organisational learning in dealing with AI's potential: autonomy and learnabili

## 92 | 117 cites | 2023 | preprint | seeds=dreamerv3
**Vision-Language Models as Success Detectors**

Yuqing Du, Ksenia Konyushkova, Misha Denil | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2303.07280 | https://openalex.org/W4324299512

Abstract: Detecting successful behaviour is crucial for training intelligent agents. As such, generalisable reward models are a prerequisite for agents that can learn to generalise their behaviour. In this work we focus on developing robust success detectors that leverage large, pretrained vision-language models (Flamingo, Alayrac et al. (2022)) and human reward annotations. Concretely, we treat success detection as a visual question answering (VQA) problem, denoted SuccessVQA. We study success detection across three vastly different domains: (i) interactive language-conditioned agents in a simulated household, (ii) real world robotic manipulation, and (iii) "in-the-wild" human egocentric videos. We investigate the generalisation properties of a Flamingo-based success detection model across unseen language and visual changes in the first two domains, and find that the proposed method is able to outperform bespoke reward models in out-of-distribution test scenarios with either variation. In the last domain of "in-the-wild" human videos, we show that success detection on unseen real videos presents an even more challenging generalisation task warranting future work. We hope our initial results

## 93 | 116 cites | 2024 | article | seeds=worldmodels
**A survey on reinforcement learning in aviation applications**

Pouria Razzaghi, Amin Tabrizian, Wei Guo | Engineering Applications of Artificial Intelligence

DOI: https://doi.org/10.1016/j.engappai.2024.108911 | https://openalex.org/W4400650558

Abstract: 

## 94 | 114 cites | 2023 | preprint | seeds=ijepa
**Open-Sourcing Highly Capable Foundation Models: An evaluation of risks, benefits, and alternative methods for pursuing open-source objectives**

Elizabeth Seger, Noemi Dreksler, Richard Moulange | SSRN Electronic Journal

DOI: https://doi.org/10.2139/ssrn.4596436 | https://openalex.org/W4387508266

Abstract: 

## 95 | 114 cites | 2025 | article | seeds=rwm
**Igniting Language Intelligence: The Hitchhiker’s Guide from Chain-of-Thought Reasoning to Language Agents**

Zhuosheng Zhang, Yao Yao, Aston Zhang | ACM Computing Surveys

DOI: https://doi.org/10.1145/3719341 | https://openalex.org/W4407922851

Abstract: Large language models (LLMs) have dramatically enhanced the field of language intelligence, as demonstrably evidenced by their formidable empirical performance across a spectrum of complex reasoning tasks. Additionally, theoretical proofs have illuminated their emergent reasoning capabilities, providing a compelling showcase of their advanced cognitive abilities in linguistic contexts. Critical to their remarkable efficacy in handling complex reasoning tasks, LLMs leverage the intriguing chain-of-thought (CoT) reasoning techniques, obliging them to formulate intermediate steps en route to deriving an answer. The CoT reasoning approach has not only exhibited proficiency in amplifying reasoning performance but also in enhancing interpretability, controllability, and flexibility. In light of these merits, recent research endeavors have extended CoT reasoning methodologies to nurture the development of autonomous language agents, which adeptly adhere to language instructions and execute actions within varied environments. This survey article orchestrates a thorough discourse, penetrating vital research dimensions, encompassing (i) the foundational mechanics of CoT techniques, with a fo

## 96 | 109 cites | 2023 | conference-paper | seeds=dreamer,dreamerv2,worldmodels,planet
**Structured World Models from Human Videos**

Russell Mendonca, Shikhar Bahl, Deepak Pathak | Robotics: Science and Systems

DOI: https://doi.org/10.15607/rss.2023.xix.012 | https://openalex.org/W4385430451

Abstract: In this paper, we tackle the problem of learning complex, general behaviors directly in the real world.We propose an approach for robots to efficiently learn manipulation skills using only a handful of real-world interaction trajectories from many different settings.Inspired by the success of learning from large-scale datasets in the fields of computer vision and natural language, our belief is that in order to efficiently learn, a robot must be able to leverage internet-scale, human video data.Humans interact with the world in many interesting ways, which can allow a robot to not only build an understanding of useful actions and affordances but also how these actions affect the world for manipulation.Our approach builds a structured, human-centric action space grounded in visual affordances learned from human videos.Further, we train a world model on human videos and finetune on a small amount of robot interaction data without any task supervision.We show that this approach of affordance-space world models enables different robots to learn various manipulation skills in complex settings, in under 30 minutes of interaction.Videos can be found at https://human-world-model.github.io

## 97 | 109 cites | 2023 | article | seeds=dreamer
**Tutorial on Amortized Optimization**

Brandon Amos | Foundations and Trends® in Machine Learning

DOI: https://doi.org/10.1561/9781638282099 | https://openalex.org/W4382243918

Abstract: Optimization is a ubiquitous modeling tool and is often deployed in settings which repeatedly solve similar instances of the same problem. Amortized optimization methods use learning to predict the solutions to problems in these settings, exploiting the shared structure between similar problem instances. These methods have been crucial in variational inference and reinforcement learning and are capable of solving optimization problems many orders of magnitudes times faster than traditional optimization methods that do not use amortization. This tutorial presents an introduction to the amortized optimization foundations behind these advancements and overviews their applications in variational inference, sparse coding, gradient-based meta-learning, control, reinforcement learning, convex optimization, optimal transport, and deep equilibrium networks. The source code for this tutorial is available at https://github.com/facebookresearch/amortized-optimization-tutorial.

## 98 | 107 cites | 2025 | preprint | seeds=worldmodels
**RELIC: Interactive Video World Model with Long-Horizon Memory**

Yicong Hong, Yiqun Mei, Chongjian Ge | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2512.04040 | https://openalex.org/W4417036201

Abstract: A truly interactive world model requires three key ingredients: real-time long-horizon streaming, consistent spatial memory, and precise user control. However, most existing approaches address only one of these aspects in isolation, as achieving all three simultaneously is highly challenging-for example, long-term memory mechanisms often degrade real-time performance. In this work, we present RELIC, a unified framework that tackles these three challenges altogether. Given a single image and a text description, RELIC enables memory-aware, long-duration exploration of arbitrary scenes in real time. Built upon recent autoregressive video-diffusion distillation techniques, our model represents long-horizon memory using highly compressed historical latent tokens encoded with both relative actions and absolute camera poses within the KV cache. This compact, camera-aware memory structure supports implicit 3D-consistent content retrieval and enforces long-term coherence with minimal computational overhead. In parallel, we fine-tune a bidirectional teacher video model to generate sequences beyond its original 5-second training horizon, and transform it into a causal student generator using 

## 99 | 106 cites | 2025 | conference-paper | seeds=dreamer
**Video World Models with Long-term Spatial Memory**

Tong Wu, Shuai Yang, Ryan Po | neural information processing systems

DOI: https://doi.org/10.52202/085713-1651 | https://openalex.org/W4416138746

Abstract: Emerging world models autoregressively generate video frames in response to actions, such as camera movements and text prompts, among other control signals. Due to limited temporal context window sizes, these models often struggle to maintain scene consistency during revisits, leading to severe forgetting of previously generated environments. Inspired by the mechanisms of human memory, we introduce a novel framework to enhancing long-term consistency of video world models through a geometry-grounded long-term spatial memory. Our framework includes mechanisms to store and retrieve information from the long-term spatial memory and we curate custom datasets to train and evaluate world models with explicitly stored 3D memory mechanisms. Our evaluations show improved quality, consistency, and context length compared to relevant baselines, paving the way towards long-term consistent world generation.

## 100 | 105 cites | 2025 | article | seeds=dreamer
**A survey on physics informed reinforcement learning: Review and open problems**

Chayan Banerjee, Kien Nguyen, Clinton Fookes | Expert Systems with Applications

DOI: https://doi.org/10.1016/j.eswa.2025.128166 | https://openalex.org/W4410361159

Abstract: The fusion of physical information in machine learning frameworks has revolutionized many application areas. This involves enhancing the learning process by incorporating physical constraints and adhering to physical laws. This work explores their utility for reinforcement learning applications. A thorough review of the literature on the fusion of physics information or physics priors in reinforcement learning approaches, commonly referred to as physics-informed reinforcement learning (PIRL), is presented. A novel taxonomy is introduced with the reinforcement learning pipeline as the backbone to classify existing works, compare and contrast them, and derive crucial insights. Existing works are analyzed with regard to the representation/form of the governing physics modeled for integration, their specific contribution to the typical reinforcement learning architecture, and their connection to the underlying reinforcement learning pipeline stages. Core learning architectures and physics incorporation biases (i.e., observational, inductive, and learning) of existing PIRL approaches are identified and used to further categorize the works for better understanding and adaptation. By prov

## 101 | 105 cites | 2024 | preprint | seeds=dreamer,worldmodels
**Pandora: Towards General World Model with Natural Language Actions and Video States**

Xiang, Jiannan, Guangyi Liu, Gu, Yi | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2406.09455 | https://openalex.org/W4399758271

Abstract: World models simulate future states of the world in response to different actions. They facilitate interactive content creation and provides a foundation for grounded, long-horizon reasoning. Current foundation models do not fully meet the capabilities of general world models: large language models (LLMs) are constrained by their reliance on language modality and their limited understanding of the physical world, while video models lack interactive action control over the world simulations. This paper makes a step towards building a general world model by introducing Pandora, a hybrid autoregressive-diffusion model that simulates world states by generating videos and allows real-time control with free-text actions. Pandora achieves domain generality, video consistency, and controllability through large-scale pretraining and instruction tuning. Crucially, Pandora bypasses the cost of training-from-scratch by integrating a pretrained LLM (7B) and a pretrained video model, requiring only additional lightweight finetuning. We illustrate extensive outputs by Pandora across diverse domains (indoor/outdoor, natural/urban, human/robot, 2D/3D, etc.). The results indicate great potential of 

## 102 | 103 cites | 2024 | article | seeds=dreamer,dreamerv2,dreamerv3,worldmodels
**World Models for Autonomous Driving: An Initial Survey**

Yanchen Guan, Haicheng Liao, Zhenning Li | IEEE Transactions on Intelligent Vehicles

DOI: https://doi.org/10.1109/tiv.2024.3398357 | https://openalex.org/W4396754486

Abstract: In the rapidly evolving landscape of autonomous driving, the capability to accurately predict future events and assess their implications is paramount for both safety and efficiency, critically aiding the decision-making process. World models have emerged as a transformative approach, enabling autonomous driving systems to synthesize and interpret vast amounts of sensor data, thereby predicting potential future scenarios and compensating for information gaps. This paper provides an initial review of the current state and prospective advancements of world models in autonomous driving, spanning their theoretical underpinnings, practical applications, and ongoing research efforts aimed at overcoming existing limitations. Highlighting the significant role of world models in advancing autonomous driving technologies, this survey aspires to serve as a foundational reference for the research community, facilitating swift access to and comprehension of this burgeoning field, and inspiring continued innovation and exploration.

## 103 | 102 cites | 2023 | conference-paper | seeds=dreamerv3,worldmodels
**A Dynamic Multi-Scale Voxel Flow Network for Video Prediction**

Xiaotao Hu, Zhewei Huang, Ailin Huang | IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings - IEEE Computer Society Conference on Computer Vision and Pattern Recognition/Proceedings

DOI: https://doi.org/10.1109/cvpr52729.2023.00593 | https://openalex.org/W4386083005

Abstract: The performance of video prediction has been greatly boosted by advanced deep neural networks. However, most of the current methods suffer from large model sizes and require extra inputs, e.g., semantic/depth maps, for promising performance. For efficiency consideration, in this paper, we propose a Dynamic Multi-scale Voxel Flow Network (DMVFN) to achieve better video prediction performance at lower computational costs with only RGB images, than previous methods. The core of our DMVFN is a differentiable routing module that can effectively perceive the motion scales of video frames. Once trained, our DMVFN selects adaptive sub-networks for different inputs at the inference stage. Experiments on several benchmarks demonstrate that our DMVFN is an order of magnitude faster than Deep Voxel Flow [35] and surpasses the state-of-the-art iterative-based OPT [63] on generated image quality.

## 104 | 101 cites | 2025 | preprint | seeds=dreamerv2
**FLARE: Robot Learning with Implicit World Modeling**

Ruijie Zheng, Jing Wang, Reed, Scott | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2505.15659 | https://openalex.org/W4415329187

Abstract: We introduce $\textbf{F}$uture $\textbf{LA}$tent $\textbf{RE}$presentation Alignment ($\textbf{FLARE}$), a novel framework that integrates predictive latent world modeling into robot policy learning. By aligning features from a diffusion transformer with latent embeddings of future observations, $\textbf{FLARE}$ enables a diffusion transformer policy to anticipate latent representations of future observations, allowing it to reason about long-term consequences while generating actions. Remarkably lightweight, $\textbf{FLARE}$ requires only minimal architectural modifications -- adding a few tokens to standard vision-language-action (VLA) models -- yet delivers substantial performance gains. Across two challenging multitask simulation imitation learning benchmarks spanning single-arm and humanoid tabletop manipulation, $\textbf{FLARE}$ achieves state-of-the-art performance, outperforming prior policy learning baselines by up to 26%. Moreover, $\textbf{FLARE}$ unlocks the ability to co-train with human egocentric video demonstrations without action labels, significantly boosting policy generalization to a novel object with unseen geometry with as few as a single robot demonstration. 

## 105 | 101 cites | 2024 | article | seeds=ijepa
**A Review of Deep Learning-Based Methods for Road Extraction from High-Resolution Remote Sensing Images**

Ruyi Liu, Junhong Wu, Wenyi Lu | Remote Sensing

DOI: https://doi.org/10.3390/rs16122056 | https://openalex.org/W4399432762

Abstract: Road extraction from high-resolution remote sensing images has long been a focal and challenging research topic in the field of computer vision. Accurate extraction of road networks holds extensive practical value in various fields, such as urban planning, traffic monitoring, disaster response and environmental monitoring. With rapid development in the field of computational intelligence, particularly breakthroughs in deep learning technology, road extraction technology has made significant progress and innovation. This paper provides a systematic review of deep learning-based methods for road extraction from remote sensing images, focusing on analyzing the application of computational intelligence technologies in improving the precision and efficiency of road extraction. According to the type of annotated data, deep learning-based methods are categorized into fully supervised learning, semi-supervised learning, and unsupervised learning approaches, each further divided into more specific subcategories. They are comparatively analyzed based on their principles, advantages, and limitations. Additionally, this review summarizes the metrics used to evaluate the performance of road ext

## 106 | 99 cites | 2024 | preprint | seeds=dreamer,dreamerv3
**Learning and Leveraging World Models in Visual Representation Learning**

Quentin Garrido, Mahmoud Assran, Nicolas Ballas | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2403.00504 | https://openalex.org/W4392427358

Abstract: Joint-Embedding Predictive Architecture (JEPA) has emerged as a promising self-supervised approach that learns by leveraging a world model. While previously limited to predicting missing parts of an input, we explore how to generalize the JEPA prediction task to a broader set of corruptions. We introduce Image World Models, an approach that goes beyond masked image modeling and learns to predict the effect of global photometric transformations in latent space. We study the recipe of learning performant IWMs and show that it relies on three key aspects: conditioning, prediction difficulty, and capacity. Additionally, we show that the predictive world model learned by IWM can be adapted through finetuning to solve diverse tasks; a fine-tuned IWM world model matches or surpasses the performance of previous self-supervised methods. Finally, we show that learning with an IWM allows one to control the abstraction level of the learned representations, learning invariant representations such as contrastive methods, or equivariant representations such as masked image modelling.

## 107 | 99 cites | 2024 | preprint | seeds=dreamer,dreamerv2,dreamerv3
**WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens**

Xiaofeng Wang, Zheng Dong Zhu, Guan Huang | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2401.09985 | https://openalex.org/W4391047144

Abstract: World models play a crucial role in understanding and predicting the dynamics of the world, which is essential for video generation. However, existing world models are confined to specific scenarios such as gaming or driving, limiting their ability to capture the complexity of general world dynamic environments. Therefore, we introduce WorldDreamer, a pioneering world model to foster a comprehensive comprehension of general world physics and motions, which significantly enhances the capabilities of video generation. Drawing inspiration from the success of large language models, WorldDreamer frames world modeling as an unsupervised visual sequence modeling challenge. This is achieved by mapping visual inputs to discrete tokens and predicting the masked ones. During this process, we incorporate multi-modal prompts to facilitate interaction within the world model. Our experiments show that WorldDreamer excels in generating videos across different scenarios, including natural scenes and driving environments. WorldDreamer showcases versatility in executing tasks such as text-to-video conversion, image-tovideo synthesis, and video editing. These results underscore WorldDreamer's effectiv

## 108 | 99 cites | 2024 | article | seeds=ijepa
**Predicting gradient is better: Exploring self-supervised learning for SAR ATR with a joint-embedding predictive architecture**

Weijie Li, Wei Yang, Tianpeng Liu | ISPRS Journal of Photogrammetry and Remote Sensing

DOI: https://doi.org/10.1016/j.isprsjprs.2024.09.013 | https://openalex.org/W4402734376

Abstract: The growing Synthetic Aperture Radar (SAR) data can build a foundation model using self-supervised learning (SSL) methods, which can achieve various SAR automatic target recognition (ATR) tasks with pretraining in large-scale unlabeled data and fine-tuning in small-labeled samples. SSL aims to construct supervision signals directly from the data, minimizing the need for expensive expert annotation and maximizing the use of the expanding data pool for a foundational model. This study investigates an effective SSL method for SAR ATR, which can pave the way for a foundation model in SAR ATR. The primary obstacles faced in SSL for SAR ATR are small targets in remote sensing and speckle noise in SAR images, corresponding to the SSL approach and signals. To overcome these challenges, we present a novel joint-embedding predictive architecture for SAR ATR (SAR-JEPA) thatleverages local masked patches to predict the multi-scale SAR gradient representations of an unseen context. The key aspect of SAR-JEPA is integrating SAR domain features to ensure high-quality self-supervised signals as target features. In addition, we employ local masks and multi-scale features to accommodate various smal

## 109 | 99 cites | 2025 | article | seeds=rwm,planet
**Deep Reinforcement Learning: A Chronological Overview and Methods**

Juan R. Terven | AI

DOI: https://doi.org/10.3390/ai6030046 | https://openalex.org/W4407903031

Abstract: Introduction: Deep reinforcement learning (deep RL) integrates the principles of reinforcement learning with deep neural networks, enabling agents to excel in diverse tasks ranging from playing board games such as Go and Chess to controlling robotic systems and autonomous vehicles. By leveraging foundational concepts of value functions, policy optimization, and temporal difference methods, deep RL has rapidly evolved and found applications in areas such as gaming, robotics, finance, and healthcare. Objective: This paper seeks to provide a comprehensive yet accessible overview of the evolution of deep RL and its leading algorithms. It aims to serve both as an introduction for newcomers to the field and as a practical guide for those seeking to select the most appropriate methods for specific problem domains. Methods: We begin by outlining fundamental reinforcement learning principles, followed by an exploration of early tabular Q-learning methods. We then trace the historical development of deep RL, highlighting key milestones such as the advent of deep Q-networks (DQN). The survey extends to policy gradient methods, actor–critic architectures, and state-of-the-art algorithms such a

## 110 | 97 cites | 2024 | article | seeds=dreamer,dreamerv2,worldmodels
**On Transforming Reinforcement Learning With Transformers: The Development Trajectory**

Shengchao Hu, Li Shen, Ya Zhang | IEEE Transactions on Pattern Analysis and Machine Intelligence

DOI: https://doi.org/10.1109/tpami.2024.3408271 | https://openalex.org/W4399310716

Abstract: Transformers, originally devised for natural language processing (NLP), have also produced significant successes in computer vision (CV). Due to their strong expression power, researchers are investigating ways to deploy transformers for reinforcement learning (RL), and transformer-based models have manifested their potential in representative RL benchmarks. In this paper, we collect and dissect recent advances concerning the transformation of RL with transformers (transformer-based RL (TRL)) to explore the development trajectory and future trends of this field. We group the existing developments into two categories: architecture enhancements and trajectory optimizations, and examine the main applications of TRL in robotic manipulation, text-based games (TBGs), navigation, and autonomous driving. Architecture enhancement methods consider how to apply the powerful transformer structure to RL problems under the traditional RL framework, facilitating more precise modeling of agents and environments compared to traditional deep RL techniques. However, these methods are still limited by the inherent defects of traditional RL algorithms, such as bootstrapping and the "deadly triad". Traj

## 111 | 97 cites | 2023 | article | seeds=dreamer,dreamerv2,worldmodels,planet
**Combining physics and deep learning to learn continuous-time dynamics models**

Michael Lutter, Jan Peters | The International Journal of Robotics Research

DOI: https://doi.org/10.1177/02783649231169492 | https://openalex.org/W3204169784

Abstract: Deep learning has been widely used within learning algorithms for robotics. One disadvantage of deep networks is that these networks are black-box representations. Therefore, the learned approximations ignore the existing knowledge of physics or robotics. Especially for learning dynamics models, these black-box models are not desirable as the underlying principles are well understood and the standard deep networks can learn dynamics that violate these principles. To learn dynamics models with deep networks that guarantee physically plausible dynamics, we introduce physics-inspired deep networks that combine first principles from physics with deep learning. We incorporate Lagrangian mechanics within the model learning such that all approximated models adhere to the laws of physics and conserve energy. Deep Lagrangian Networks (DeLaN) parametrize the system energy using two networks. The parameters are obtained by minimizing the squared residual of the Euler–Lagrange differential equation. Therefore, the resulting model does not require specific knowledge of the individual system, is interpretable, and can be used as a forward, inverse, and energy model. Previously these properties w

## 112 | 97 cites | 2026 | preprint | seeds=dreamerv3
**A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents**

Tianxin Wei, Ting-Wei Li, Zhining Liu | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2601.12538 | https://openalex.org/W7125263063

Abstract: Reasoning is a fundamental cognitive process underlying inference, problem-solving, and decision-making. While large language models (LLMs) demonstrate strong reasoning capabilities in closed-world settings, they struggle in open-ended and dynamic environments. Agentic reasoning marks a paradigm shift by reframing LLMs as autonomous agents that plan, act, and learn through continual interaction. In this survey, we organize agentic reasoning along three complementary dimensions. First, we characterize environmental dynamics through three layers: foundational agentic reasoning, which establishes core single-agent capabilities including planning, tool use, and search in stable environments; self-evolving agentic reasoning, which studies how agents refine these capabilities through feedback, memory, and adaptation; and collective multi-agent reasoning, which extends intelligence to collaborative settings involving coordination, knowledge sharing, and shared goals. Across these layers, we distinguish in-context reasoning, which scales test-time interaction through structured orchestration, from post-training reasoning, which optimizes behaviors via reinforcement learning and supervised 

## 113 | 97 cites | 2024 | preprint | seeds=worldmodels
**WorldSimBench: Towards Video Generation Models as World Simulators**

Yiran Qin, Zhelun Shi, Jiwen Yu | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2410.18072 | https://openalex.org/W4404308559

Abstract: Recent advancements in predictive models have demonstrated exceptional capabilities in predicting the future state of objects and scenes. However, the lack of categorization based on inherent characteristics continues to hinder the progress of predictive model development. Additionally, existing benchmarks are unable to effectively evaluate higher-capability, highly embodied predictive models from an embodied perspective. In this work, we classify the functionalities of predictive models into a hierarchy and take the first step in evaluating World Simulators by proposing a dual evaluation framework called WorldSimBench. WorldSimBench includes Explicit Perceptual Evaluation and Implicit Manipulative Evaluation, encompassing human preference assessments from the visual perspective and action-level evaluations in embodied tasks, covering three representative embodied scenarios: Open-Ended Embodied Environment, Autonomous, Driving, and Robot Manipulation. In the Explicit Perceptual Evaluation, we introduce the HF-Embodied Dataset, a video assessment dataset based on fine-grained human feedback, which we use to train a Human Preference Evaluator that aligns with human perception and exp

## 114 | 95 cites | 2025 | preprint | seeds=dreamerv3
**Yume: An Interactive World Generation Model**

Xuehui Mao, Lin, Shaoheng, Zhen Li | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2507.17744 | https://openalex.org/W4414886176

Abstract: Yume aims to use images, text, or videos to create an interactive, realistic, and dynamic world, which allows exploration and control using peripheral devices or neural signals. In this report, we present a preview version of \method, which creates a dynamic world from an input image and allows exploration of the world using keyboard actions. To achieve this high-fidelity and interactive video world generation, we introduce a well-designed framework, which consists of four main components, including camera motion quantization, video generation architecture, advanced sampler, and model acceleration. First, we quantize camera motions for stable training and user-friendly interaction using keyboard inputs. Then, we introduce the Masked Video Diffusion Transformer~(MVDT) with a memory module for infinite video generation in an autoregressive manner. After that, training-free Anti-Artifact Mechanism (AAM) and Time Travel Sampling based on Stochastic Differential Equations (TTS-SDE) are introduced to the sampler for better visual quality and more precise control. Moreover, we investigate model acceleration by synergistic optimization of adversarial distillation and caching mechanisms. We

## 115 | 95 cites | 2024 | preprint | seeds=worldmodels
**Sora as a World Model? A Complete Survey on Text-to-Video Generation**

Fachrina Dewi Puspitasari, Chaoning Zhang, Cho, Joseph | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2403.05131 | https://openalex.org/W4392678331

Abstract: The evolution of video generation from text, from animating MNIST to simulating the world with Sora, has progressed at a breakneck speed. Here, we systematically discuss how far text-to-video generation technology supports essential requirements in world modeling. We curate 250+ studies on text-based video synthesis and world modeling. We then observe that recent models increasingly support spatial, action, and strategic intelligences in world modeling through adherence to completeness, consistency, invention, as well as human interaction and control. We conclude that text-to-video generation is adept at world modeling, although homework in several aspects, such as the diversity-consistency trade-offs, remains to be addressed.

## 116 | 92 cites | 2025 | article | seeds=ijepa
**Artificial General Intelligence (AGI)-Native Wireless Systems: A Journey Beyond 6G**

Walid Saad, Omar Hashash, Christo Kurisummoottil Thomas | Proceedings of the IEEE

DOI: https://doi.org/10.1109/jproc.2025.3526887 | https://openalex.org/W4408520711

Abstract: Building the next-generation wireless systems that could support services such as the metaverse, digital twins (DTs), and holographic teleportation is challenging to achieve exclusively through incremental advances to conventional wireless technologies like metasurfaces or holographic antennas. While the 6G concept of artificial intelligence (AI)-native networks promises to overcome some of the limitations of existing wireless technologies, current developments of AI-native wireless systems rely mostly on conventional AI tools such as auto-encoders and off-the-shelf artificial neural networks. However, those tools struggle to manage and cope with the complex, nontrivial scenarios faced in real-world wireless environments and the growing quality-of-experience (QoE) requirements of the aforementioned, emerging wireless use cases. In contrast, in this article, we propose to fundamentally revisit the concept of AI-native wireless systems, equipping them with the common sense necessary to transform them into artificial general intelligence (AGI)-native systems. Our envisioned AGI-native wireless systems acquire common sense by exploiting different cognitive abilities such as reasoning a

## 117 | 92 cites | 2024 | article | seeds=worldmodels
**Agent design pattern catalogue: A collection of architectural patterns for foundation model based agents**

Yue Liu, Sin Kit Lo, Qinghua Lu | Journal of Systems and Software

DOI: https://doi.org/10.1016/j.jss.2024.112278 | https://openalex.org/W4404355908

Abstract: Foundation model-enabled generative artificial intelligence facilitates the development and implementation of agents, which can leverage distinguished reasoning and language processing capabilities to takes a proactive, autonomous role to pursue users’ goals. Nevertheless, there is a lack of systematic knowledge to guide practitioners in designing the agents considering challenges of goal-seeking (including generating instrumental goals and plans), such as hallucinations inherent in foundation models, explainability of reasoning process, complex accountability, etc. To address this issue, we have performed a systematic literature review to understand the state-of-the-art foundation model-based agents and the broader ecosystem. In this paper, we present a pattern catalogue consisting of 18 architectural patterns with analyses of the context, forces, and trade-offs as the outcomes from the previous literature review. We propose a decision model for selecting the patterns. The proposed catalogue can provide holistic guidance for the effective use of patterns, and support the architecture design of foundation model-based agents by facilitating goal-seeking and plan generation. • A coll

## 118 | 91 cites | 2023 | article | seeds=rwm
**Graph neural networks-based scheduler for production planning problems using reinforcement learning**

Mohammed Sharafath Abdul Hameed, Andreas Schwung | Journal of Manufacturing Systems

DOI: https://doi.org/10.1016/j.jmsy.2023.06.005 | https://openalex.org/W4380768317

Abstract: 

## 119 | 91 cites | 2024 | preprint | seeds=worldmodels
**GameGen-X: Interactive Open-world Game Video Generation**

Haoxuan Che, Xuanhua He, Liu, Quande | arXiv (Cornell University)

DOI: https://doi.org/10.48550/arxiv.2411.00769 | https://openalex.org/W4404345918

Abstract: We introduce GameGen-X, the first diffusion transformer model specifically designed for both generating and interactively controlling open-world game videos. This model facilitates high-quality, open-domain generation by simulating an extensive array of game engine features, such as innovative characters, dynamic environments, complex actions, and diverse events. Additionally, it provides interactive controllability, predicting and altering future content based on the current clip, thus allowing for gameplay simulation. To realize this vision, we first collected and built an Open-World Video Game Dataset from scratch. It is the first and largest dataset for open-world game video generation and control, which comprises over a million diverse gameplay video clips sampling from over 150 games with informative captions from GPT-4o. GameGen-X undergoes a two-stage training process, consisting of foundation model pre-training and instruction tuning. Firstly, the model was pre-trained via text-to-video generation and video continuation, endowing it with the capability for long-sequence, high-quality open-domain game video generation. Further, to achieve interactive controllability, we des
