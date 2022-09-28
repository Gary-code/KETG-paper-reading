# Knowledge-enriched Text Generation paper reading

😎 Awesome list of papers about knowledge-enhanced Question generation with notes.

:white_check_mark: : **already reading carefully**

:fire:: **high citation in recent years**

:hammer_and_wrench:: **available code**

> Content

[TOC]



---

## :grey_question: Question Generation

---

### :mountain_snow: **Textual Question Generating Crosstalk**

**一、利用答案和语言特征**

1. **两篇Ground Breaking Work**

:white_check_mark: :fire: **Neural question generation from text: A preliminary study**, in EMNLP 2017. [[pdf](https://arxiv.org/abs/1704.01792)] 

* 在编码时额外考虑了答案位置与语法信息，取得了更好的性能。(现在来看非常**basic**重要的信息！)
  * word case 做训练时候的teacher forcing
  * answer position feature
  * lexical features
    * **POS**
    * **NER**

```mermaid
graph LR
en((encoder)) --bi-GRU--> fe((feature-Rich)) --> word-vecotr
fe --> lexcial-feature-embedding-vectors --> POS+NER
fe --> answer-position-embedding --> BIO-tagging

word-vecotr --> 双向的隐藏层
POS+NER --> 双向的隐藏层
BIO-tagging --> 双向的隐藏层

de((decoder)) --带注意力机制,使用加性注意力--> maxout-hidden+具体需要看reference论文
de --> GRU

de --> Copy-Mechanism,一样使用加性注意力 --> 计算出概率从source句子中直接copy单词
```



:white_check_mark: :fire: :hammer_and_wrench: **Learning to Ask: Neural Question Generation for Reading Comprehension**, in ACL 2017. [[pdf]](https://arxiv.org/abs/1705.00106) [[official code (torch)](https://github.com/xinyadu/nqg)]
* 将端到端训练的神经网络应用于问题生成
* 采用seq2seq+attention模型架构
* 摆脱了转换规则与模版的局限，取得了相比于传统方法更好的性能
* 加入了paragraph-level


```mermaid
graph LR
任务难点 --更加接近于人类--> 同义词替换+知识引入 --> 相关工作 --> 过去:rule-based 
相关工作 --> 其他数据映射自然语言

Seq2Seq --> en((encoder)) --bidirectional--> soft计算注意力分数 --> lstm((LSTM))  --> only-sentence
lstm --> sentence+paragraph --> truncate截断,当然更好的方法是切片

Seq2Seq --> de((decoder)) --word-level-prediction--> LSTM((LSTM)) --> 隐藏层初始化 --basic-model --> 句子encoder的最后隐藏层
LSTM --oours--> 句子+段落的encoder输出


```

2. **答案编码**

:white_check_mark: :fire: **Improving Neural Question Generation using Answer Separation**, in AAAI 2019.  [[pdf](https://arxiv.org/abs/1809.02393)] 
* 很多基础操作
* 在答案上做了简单高效的预处理
  * Mask 原文中的答案
  * 对答案中的关键信息做抽取，计算attention

3. **语言特征强化**

> 传统的有**POS**（词性标注）和**NER**（命名实体识别）。后续还有一些更加细微的处理

:fire: **Learning to Generate Questions by Learning What not to Generate**, in WWW 2019.  [[pdf](https://arxiv.org/pdf/1902.10418.pdf)] 

* clue 和 copy的机制
* ![image-20220703112845472](https://s2.loli.net/2022/07/03/hoa1kTVzIpXGeQK.png)
* ![image-20220703113749291](https://s2.loli.net/2022/07/03/bdToKkIADvemV6B.png)
* 文章贡献
  * 帮助模型决策什么时候生成，什么时候copy
  * 生成多个问题

4. 疑问词类型（question type）

**Question-type Driven Question Generation**, in EMNLP 2019.  [[pdf](https://arxiv.org/pdf/1909.00140.pdf)]

* 引入对疑问词的预测模块，并且加入对应的损失函数
* ![image-20220703121312077](https://s2.loli.net/2022/07/03/uhcs1erWpQ9UFKL.png)
* ![image-20220703121503229](https://s2.loli.net/2022/07/03/GTO7j5c1AkXNnqY.png)
* ![image-20220703121520488](https://s2.loli.net/2022/07/03/seIfghRSbvpO68a.png)
* [损失函数引文](https://aclanthology.org/P17-1099.pdf)： <img src="https://s2.loli.net/2022/07/03/UZv17XkMyLRPOgd.png" alt="image-20220703122045296" style="zoom: 50%;" />

**二、段落级别特征**

1. **强化段落级别文本的特征**

:fire: :hammer_and_wrench: **Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks**, in EMNLP 2018. [[pdf](https://github.com/seanie12/neural-question-generation)] [[torch](https://github.com/seanie12/neural-question-generation)]

* 主要贡献都在模型上面，基于seq2seq设计：![image-20220704150006013](https://s2.loli.net/2022/07/04/trhe5uTRkUqyfsx.png)

  * gate self-attention: 个人觉得是一套很常用的框架，可以学习一下，也非常简单

  * Maxout ==**Pointer**== & Decoding **全新的处理 copy 机制**  (有空可以自行去看看代码！)

    * 之前copy得分：$\operatorname{sc}^{\text {copy }}\left(y_{t}\right)=\left\{\begin{array}{l}\sum_{k, \text { where } x_{k}=y_{t}} r_{t, k}, \quad y_{t} \in \chi \\ -i n f, \text{otherwise}\end{array} \quad\right.$ , 问题在于若文章中某个单词重复出现多次，则对该单词copy也会多，影响语句通顺。

    * 为此改进为Maxout Pointer：
      $$
      \operatorname{sc}^{\text {copy }}\left(y_{t}\right)= \begin{cases}\max _{k, \text { where } x_{k}=y_{t}} r_{t, k}, & y_{t} \in \chi \\ -i n f, & \text { otherwise }\end{cases}
      $$

:fire: **Natural Question Generation with Reinforcement Learning Based Graph-to-Sequence Model**,in ICLR 2020. [[pdf](https://arxiv.org/abs/1908.04942)] [[torch](https://github.com/hugochan/RL-based-Graph2Seq-for-NQG.)]

* 将passage和answer的表示（包含bert向量，glove向量，词汇特征等）进行多次反复的交互进行编码（**非常细节**的deep alignment network）
* 利用GNN来编码（使用了两种方式）：
  * 对sentence做**dependency parsing**，然后相邻的句子链接得到passage的图
  * 通过self attention的方式得到passage 的图（权值矩阵）

:fire: **Improving Question Generation With to the Point Context**, in EMNLP 2019.  [[pdf](https://aclanthology.org/D19-1317.pdf)]

* 联合建模非结构化句子（原文）和结构化答案相关关系（ answer-relevant relation 预先从句子中提取）以生成问题(**抓取重点上下文**)
* 作者发现上下文中，距离ansewr比较远的词并不一定不重要，相对的跟answer紧贴的词也有很多无关的，为了捕捉这种关系，作者使用**OpenIE**这个工具抽取上下文中存在的**关系三元组**。

**三、多任务训练**

**Multi-Task Learning with Language Modeling for Question Generation**, in EMNLP 2019. [[pdf](http://aclanthology.lst.uni-saarland.de/D19-1337.pdf)]

* 把语言模型（预测前后词）和QG作为multi-task一起进行训练。
* 两个任务是层级的关系，先进行语言模型的预测，然后将语言模型的hidden作为特征提供给后面seq2seq

:fire: **Improving Question Generation with Sentence-level Semantic Matching and Answer Position Inferring**, in AAAI 2019.  [[pdf](https://arxiv.org/abs/1912.00879)]

* 出发点是是**解决生成错误的疑问词**和**copy原文中无关词**的问题

* 作者认为生成错误词的原因是没有正确的利用**answer position**信息，copy无关词的原因是缺乏**局部语义信息**。

* 为了分别缓解这两个问题，作者也是设计了两个辅助任务：

  * 语义匹配分类：这个任务的设计出发点也是SQuAD的数据特点，对于一个passage存在多个answer-question训练数据，模型对这样的数据容易产生一些宽泛不具体的问题。所以作者把passage-question作为正样本，passage-random selected question， random selected passage-question作为负样本进行分类任务。

  ![img](https://pic3.zhimg.com/80/v2-6e137d01fd833693fcc6cf526a39fb8e_720w.jpg)

  * answer-postion位置预测：了让模型更好的利用answer信息，设计了一个预测answer在上下文中start和end位置的模型（pointer network），其中基础的编码部分采用BiDAF的方式。
  * 然后**QG和这两个辅助任务一起训练**，效果可。


****

### :sunrise: Visual QG

:hammer_and_wrench: **[No Visual] Entity Guided Question Generation with Contextual Structure and Sequence Information Capturing**, in AAAI 2021. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17544)] [[torch](https://github.com/VISLANG-Lab/EGSS)]

* Multi-feature Encoder: 使用了POS（词性标注）+ NER（关系抽取）

:hammer_and_wrench: **Multiple Objects-Aware Visual Question Generation**, in ACM MM 2021. [[pdf](https://dl.acm.org/doi/abs/10.1145/3474085.3476969)]
* **写作上写得很实在，很容易懂**，有很多承上启下的句子。
* 首次将**对象**融入到问题生成任务当中

![image-20220629230122528](https://s2.loli.net/2022/06/29/6Mn4HPG9ZiCjOqv.png)

:hammer_and_wrench:  **Difficulty-Controllable Visual Question Generation**, in APWeb-WAIM 2021. [[pdf](https://link.springer.com/content/pdf/10.1007/978-3-030-85896-4_26.pdf)]

* **难度可控**的问题生成：采用了教育学领域收集好的问题难度标签(DIF), 详见[链接](https://www.apims.net/index.php/apims/article/view/9)
* 在VQA2.0数据集的基础上构建了一个包含区分为容易和难的问题数据集
  * 引入两个VQA的模型来进行回答，都回答对的为容易，都回答错误就是难的

* ![image-20220629231350190](https://s2.loli.net/2022/06/29/POftb69si7hnINX.png)
  * 其中Difficulty Variable就是$\{0, 1\}$



:hammer_and_wrench: **Learning to Caption Images Through a Lifetime by Asking Questions**, in ICCV 2019.  [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9009050)] [[torch](https://github.com/fidler-lab/Caption-Lifetime-by-Asking-Questions)]

* 将Caption 和 VQG 一起来做，提升生成的性能


### :video_camera: Video QG

**Video Question Generation via Semantic Rich Cross-Modal Self-Attention Networks Learning**, in ICASSP 2020. [[pdf](https://ieeexplore.ieee.org/document/9053476)]

* 使用了**[TVQA](https://paperswithcode.com/dataset/tvqa)**数据集，is based on 6 popular TV shows and consists of **152,545 QA pairs** from **21,793 clips**.
* 总体没什么创新的

**Multi-Turn Video Question Generation via Reinforced Multi-Choice Attention Network**, in T-CSVT 2021.[[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9161024)]

* Multi-Turn（M-VQG）：结合多轮对话+视频信息
* 优点： 利用动态场景信息，问题可回答性，对话记录信息抽取
* 方法：baseline方法，强化学习（看不懂）

**End-to-End Video Question-Answer Generation with Generator-Pretester Network**, in T-CSVT 2021. [[pdf](https://arxiv.org/pdf/2101.01447.pdf)]

* 引入一问一答的形式，生成问题和答案，然后测试答案是否正确
* 硬件平台：NVIDIA DGX-1（8 * V100）

### :sun_with_face: QG examples

:white_check_mark:  :hammer_and_wrench:  **Mixture Content Selection for Diverse Sequence Generation**, in EMNLP 2019.[[pdf](https://arxiv.org/abs/1909.01953)] [[torch](https://github.com/clovaai/FocusSeq2Seq)]

:hammer_and_wrench: **Radial Graph Convolutional Network for Visual Question Generation**, in IEEE Transactions on Neural Networks and Learning Systems 2020. [[pdf](https://ieeexplore.ieee.org/document/9079208)] [[torch](https://github.com/Wangt-CN/VQG-GCN)]

## :bookmark_tabs: Question Answering

---

### :sunflower: Visual

> 在2022年的今天，VQA任务不太可能从刷分的角度来入手了 [[Blog链接](https://www.zhihu.com/question/419828408/answer/1595386400)]
>
> - VQA任务是什么
>
> - 介绍之前的模型和方法
>
> - 欢迎来到Transformer的时代
>
> - - 2019：尝试多模态表征
>   - 2020：拥抱多模态表征
>   - 2021：统一构架的探索

machine reading comprehension (**MRC**)和question answering (QA)的关系其实是相对独立的。Pure VQA任务一般是没有引入额外的**文本内容**，只是单纯的有$\{图， 问句， 回答\}$。而Multimodal MRC任务，实际上就只是引入了**额外的context**作为VQA任务的知识，并且更加注重于自然语言的理解。MRC的主要**任务类型**一共有四种，分别为:

* 完形填空（Cloze Style）
* 多项选择（Multiple Choice）
* 片段抽取（Span Prediction）
* 自由作答（Free-form Answer）

**[非深度学习方法] Answer-Type Prediction for Visual Question Answering**，in CVPR 2016. [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780907)]

* 预测问题类别（人为标定）的概率再回答问题
* 利用**贝叶斯算法**对目标的空间关系进行建模，计算出每个答案的概率
* 其有效性不如简单的基线模型；部分原因在于其**依赖语义分割的结果**



**Differential Attention for Visual Question Answering**, in CVPR 2018. [[pdf](https://arxiv.org/pdf/1804.00298.pdf)]

* 解决为了让模型更加关注到**人类所关注**的区域

![image-20220910151132747](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220910151132747.png)



:fire: :hammer_and_wrench: **[因果关系] Visual Commonsense R-CNN**, in CVPR 2020. [[pdf](https://arxiv.org/abs/2002.12204)] [[torch](https://github.com/Wangt-CN/VC-R-CNN)] [[blog](https://zhuanlan.zhihu.com/p/111306353)]

> 出自[MReal](https://mreallab.github.io/)， 张含望老师团队的工作，非常Solid的一篇工作
>
> * 目标是训练基于`Faster-RCNN`训练一个更强的`feature extractor`可以捕获视觉上的常识信息。
> * 这篇论文实在**太多细节和推理**了，建议看我自己的**GoodNote上的笔记**！

* 动机

  * 现在的模型无法学习到视觉常识（**Commonsense**）：人和椅子 -> 人可以坐在椅子上。但在NLP中，常识的信息已经放在特征里面了

    ![image-20220913155431514](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220913155431514.png)

  * 数据集的偏差会导致无法捕捉到常识信息
    * 真正的**视觉关系**无法描述（左图）
    * 给出的**解释**不够正确（右图）

  **因果理论就是用来发现==现象背后的不变规律==的，是一种鲁棒的预测。这与常识本身不就很相似吗，我们人类也是从生活中不断总结积累这些不变的、鲁棒的经验或者因果规律，并把他们叫做常识。** 比如，看见凳子知道可以坐，看见pizza知道可以吃。

**Association 和 Intervention（分层）的计算**
$$
\begin{gathered}
P(Y \mid X)=\sum_z P(Y \mid X, z) P(z \mid X)=\frac{P(Y, X)}{P(X)} \\
P(Y \mid d o(X))=\sum_z P(Y \mid X, z) P(z)=\sum_z \frac{P(Y, X, z) P(z)}{P(X, z)}
\end{gathered}
$$
其中 $X, Y, z$分别代表了图片中的object label，同时这里我们用物体出现的频率来代替概率，比如 $P(Sink|Hair drier)$就是用“含有$Sink$和$Hair drier$两者的图片数”比上“只含有Hair drier的图片数”计算得到的。画出两者计算结果差异的对比图（只标明了20类）：

![img](https://pic2.zhimg.com/80/v2-ddffe6ddaaf70839faa1d62a9ef25291_720w.jpg)

* 两个Case的分析
  * $Sink 和 drier$，想要探寻在**已知吹风机**的情况下，去预测水池的可能性大小 $P(Sink|drier)$
    * **场景因素**考虑在内，对不同的场景进行分层（因为场景就是由object组成的），得到实际的因果效应，比单纯Association算的数值要低
  * 人和马桶，探寻“马桶”和“人”之间可能存在的因果效应
    * 数据集中人和马桶一起出现的样本其实不多（也不会有很多人在马桶旁边拍照）
    * 如果想要做出更robust的预测，我们就需要考虑混杂因子**confounder**， 比如瓶子、水池、杯子等等。按照confounder 行**分层计算**，最后再加权求和。

* 方法（因果干预**Intervention**）

  * 代理任务（无监督学习）：**给定RoI X的feature去预测RoI Y的类别**

  * 包括很多潜在的**混杂因子**，如果直接预测周围物体Y就不可避免的会被上文提到的混杂因子**confounder**所影响。根据我们刚刚介绍的**“do算子”**的理论，解决的办法也不难，只要能找到confounder然后对他们使用**backdoor理论**进行控制即可。

  * 混杂因子是什么？ 我们直接把整个数据集上的**object RoI特征（Faster RCNN中来）在每个类别上取平均**，当作这个类别的表示，进而构建出一个 **类别数x1024** 的confounder字典作为$Z$（比如MSCOCO有80类，就是 80x1024），它包含着所有可能的混杂因子。

  * 后门调整

    ![img](https://raw.githubusercontent.com/Gary-code/pic/main/img/v2-514061ff24e803c016324ead8bcf84b1_720w.jpg)

    * 我们把confounder dictionary $Z$中的物体$z_i$“borrow”到当前图片中，注意这里的物体$z_i$不需要是当前图片中存在的，所以是一种global层面的定义。
    * 然后把借来的$z_i$“put”到$X, Y$周围和$X, Y$对比，例如上图中的把 sink、handbag、chair等等移到 toilet 和 person 周围进行backdoor的计算。

  * 模型

    * 整个intervention整合成一路context predictor。
    * 同时为了不让网络忘掉识别RoI本身类别的能力，context predictor的基础上又保留了原先的自身类别预测**self predictor**。

  ![img](https://pic2.zhimg.com/80/v2-d3a05b26274f54a8bd785209f1b6a4c1_720w.jpg)

  注意：VC R-CNN的实现和原先的Faster R-CNN相比，**去除了RPN网络**（Region Proposal Network），不再训练网络propose边界框，而是直接将数据集**ground-truth的bounding box坐标输入到其中**，直接提取region的特征。而在训练完成后的feature提取阶段，相对应的，只要给定图片和bounding box坐标，都可以获得对应的VC特征。就这样，我们利用bottomup特征已有的边界框坐标提取VC特征后，将其并在先前的bottomup特征上作为新的特征。我们在传统的 Vision&Language 三大任务上挑选了经典model和SOTA model进行了测试，发现在各个任务上都取得了明显的提升，尤其是在image captioning上的提升尤其大。同时为了验证性能的提升不是由于参数增多带来的，我们还在原有特征上并上了ablative的特征（单独object特征，用correlation计算的特征），具体可以参考论文的实验部分。

:hammer_and_wrench: **MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering**, in CVPR 2022.  [[pdf](https://arxiv.org/pdf/2203.09138.pdf)] [[torch](https://github.com/AndersonStra/MuKEA.)]

* 动机

  * 过去基于知识的，都只是考虑了文本上的知识，缺乏对多模态知识的理解

  ![image-20220901165323764](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901165323764.png)

* 主要贡献

  * 端到端的多模态知识表示 $(Entity, relation, answer)$
  * **pre-training and fine-tuning** strategy to accumulate both **out-domain and in-domain** knowledge

![image-20220901165520112](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901165520112.png)

* 细节

  * 三个**损失函数**的设计

    * `Triplet TransE Loss`: 保持embedding的结构（通过对比学习）

    $$
    \mathcal{L}_{\text {TransE }}=\sum_{t^{+} \in \mathcal{A}^{+}} \sum_{t^{-} \in \mathcal{A}^{-}}\left[\gamma+\mathrm{d}\left(h+\boldsymbol{h}, \boldsymbol{t}^{+}\right)-\mathrm{d}\left(\boldsymbol{h}+\boldsymbol{r}, \boldsymbol{t}^{-}\right)\right]_{+}
    $$

    * `Triplet Consistency Loss`： 保证严格的**拓扑关系**

    $$
    \mathcal{L}_{\mathrm{Tri}}=\operatorname{MSE}\left(h+r, t^{+}\right)
    $$

    * `Semantic Consistency Loss`: 保持在语义空间中的表达一致性

    $$
    {P\left(t^{+}\right)=\operatorname{softmax}\left((T)^{T}(h+r)\right)} \\{\mathcal{L}_{\mathrm{Sem}}=-\log \left(P\left(t^{+}\right)\right)}
    $$

    

  * 预训练和微调策略

    * 先在`VQA 2.0`数据集上进行预训练来收集视觉主导的知识
    * 在`KB-VQA`数据集上进行微调

  * 关于尾部`Entity`

    * 训练的时候直接做`teacher-forcing`
    * 推理的时候计算$\mathbf{h}_{inf}+\mathbf{r}_{inf}$ 与 `look up` table $\mathbf{T}$的最小距离

    $$
    \boldsymbol{t}_{\inf f}=\underset{\boldsymbol{t}_i \in T}{\arg \min } \mathrm{d}\left(\boldsymbol{h}_{\text {inf } f}+\boldsymbol{r}_{\text {inf } f}, \boldsymbol{t}_{\mathrm{i}}\right)
    $$

    

![image-20220901170039718](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901170039718.png)

**[VCR] Explicit Cross-Modal Representation Learning for Visual Commonsense Reasoning**, in TMM 2022. [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9465732)

* 动机：为了加强`VCR`任务的**reasoning**过程，不再那么隐式
* 方法

![image-20220923223408003](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220923223408003.png)

* 例子

![image-20220923223502035](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220923223502035.png)



### :sunny: Textual

:fire:  :hammer_and_wrench: **[Question Answering] Commonsense for Generative Multi-Hop Question Answering Tasks**, in EMNLP 2018. [[pdf]](https://arxiv.org/abs/1809.06309) [[tensorflow]](https://github.com/yicheng-w/CommonSenseMultiHopQA)

:hammer_and_wrench: **[Dialogue System] Improving Knowledge-aware Dialogue Generation via Knowledge Base Question Answering**, in AAAI 2020. [[pdf]](https://arxiv.org/abs/1912.07491) [[torch]](https://github.com/siat-nlp/TransDG)

**[Question Answering] Using Local Knowledge Graph Construction to Scale Seq2Seq Models to Multi-Document Inputs**, in EMNLP 2019. [[pdf\]](https://arxiv.org/abs/1910.08435)

:fire: :hammer_and_wrench: **[Question Answering] ** **Improving Multi-hop Question Answering over Knowledge Graphs usingKnowledge Base Embeddings**, in ACL 2020. [[pdf](https://aclanthology.org/2020.acl-main.412/)] [[torch](https://github.com/malllabiisc/EmbedKGQA)]

:hammer_and_wrench: **Found a Reason for me? Weakly-supervised Grounded Visual Question Answering using Capsules**, in CVPR 2021.  [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Urooj_Found_a_Reason_for_me_Weakly-supervised_Grounded_Visual_Question_Answering_CVPR_2021_paper.pdf)] [[torch](https://github.com/aurooj/ WeakGroundedVQA_Capsules.git)]

* 不用faster-rcnn
* 训练输入是问题和答案，输出是预测答案对应的**grouding area**。

**KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base**, in ACL 2022. [[pdf](https://aclanthology.org/2022.acl-long.422.pdf)] [[project](https://github.com/shijx12/ KQAPro_Baselines)]

* 更加复杂的数据量更大的引入知识的数据集
  * 并且给出了两种reasoning的过程
  * 可以做QA和**语义解析**服务
  * 利用更加复杂的模版和知识生成问题

![image-20220912160811327](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220912160811327.png)

* [详细介绍的blog](https://blog.csdn.net/weixin_47903246/article/details/124649493)



:hammer_and_wrench: **[VCR] Heterogeneous Graph Learning for Visual Commonsense Reasoning**, in NIPS 2019. [[pdf](https://arxiv.org/abs/1910.11475)] [[torch](https://github.com/yuweijiang/HGL-pytorch)]

* 与传统的`VQA`不太一样，R: 解释（Reason）
  * 三个子任务分别是: $Q \rightarrow A$, $QA \rightarrow R$, $Q \rightarrow AR$
* 方法：构建异构图

![image-20220921151513456](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220921151513456.png)

![image-20220921151544188](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220921151544188.png)



:hammer_and_wrench: **[VCR] Connective Cognition Network for Directional Visual Commonsense Reasoning**，in NIPS 2019.  [[pdf](https://proceedings.neurips.cc/paper/2019/file/8a56257ea05c74018291954fc56fc448-Paper.pdf)] [[torch](https://github.com/AmingWu/CCN)]

* 与上一篇论文思想比较类似，参考神经科学当中将神经元整合起来的思想

* 做法

![image-20220921171105320](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220921171105320.png)

* 模型相关细节

![image-20220921171210400](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220921171210400.png)

* 第一part中**连接**的构建

![image-20220921171342266](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220921171342266.png)

## :book: Paraphrase

---



:hammer_and_wrench: **[Sentence Discrimination] Learning Semantic Sentence Embeddings using Sequential Pair-wise Discriminator**,in COLING 2018. [[pdf](https://aclanthology.org/C18-1230/)] [[torch](https://github.com/badripatro/PQG)]

:hammer_and_wrench: **[Hierarchical Sketch&Paraphrase Generation] Hierarchical Sketch Induction for Paraphrase Generation**, in ACL 2022.[[pdf](https://aclanthology.org/2022.acl-long.178.pdf)] [[torch](https://github.com/tomhosking/hrq-vae)]

### :whale2: Related Big Model

:fire: :hammer_and_wrench: **[Cross-Modal&Contrastive Learning] UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning**, in ACL(long paper) 2021. [[pdf](https://aclanthology.org/2021.acl-long.202/)] [[project from Baidu](https://unimo-ptm.github.io/)]

:hammer_and_wrench: **[MultiModal] UniT: Multimodal Multitask Learning with a Unified Transformer**, ICCV 2021. [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_UniT_Multimodal_Multitask_Learning_With_a_Unified_Transformer_ICCV_2021_paper.pdf)] [[project from Fair](https://mmf.sh/)]

## :framed_picture: Image Caption

---



:white_check_mark: :hammer_and_wrench: **[Image Caption] Generating Diverse and Descriptive Image Captions Using Visual Paraphrases**, in ICCV 2019. [[pdf](https://ieeexplore.ieee.org/document/9010984)] [[torch](https://github.com/pkuliu/visual-paraphrases-captioning)]

* 该论文研究了目前图像的文本描述的**多样性**和**具体性**缺乏的问题，提出了一种基于视觉复述的两阶段解码的模型。
  * 给定图像输入，该模型首先生成初步的句子，再将其改写为内容更加多样和丰富的描述。在MS COCO图像描述数据集上的实验显示，方法可以显著提升文本描述的**多样性**和**具体性**。

  * 重点探索**visual paraphrases** 角色 + **scoring function**
  
    * ```mermaid
      graph LR
      与人类相比 --文章中有example--> 缺少多样性和具体性 --> 两阶段视觉复述方法 --> MSCOCO数据集
      ```
  
  
  * 故事展开:
  
    * ```mermaid
      graph LR
      标准 -->流畅+相关+多样+具体 --多样性--> 形容词
      流畅+相关+多样+具体 --多样性--> 细节,with
      形容词 --> Pa((Paraphrase))
      细节,with --> Pa
      Pa --> visual-paraphrase
      visual-paraphrase --> sentence_pairs --> 两阶段编码
      ```
  
    * ```mermaid
      graph LR
      相关工作 --caption--> 多caption.vs.单caption --paraphrases--> 未处理特征和视觉信息 --两阶段编码--> 中间seq.vs.2captions 
      ```
  
  
  * 模型方法：
  
    * ```mermaid
      graph LR
      选择视觉复述caption对 --> 评分函数 --> 设计三个Attention操作,学习到多模态知识 --> 最后softmax输出
      ```

* 更多细节可见我个人的[slide](https://kdocs.cn/l/conDzdschwAn)

:white_check_mark: ::fire: :hammer_and_wrench: **[Text Generation & Image Caption] Show, Control and Tell: A Framework for Generating Controllable and Grounded Captions**, in CVPR 2019. [[pdf](https://openaccess.thecvf.com/content_CVPR_2019/html/Cornia_Show_Control_and_Tell_A_Framework_for_Generating_Controllable_and_CVPR_2019_paper.html)] [[torch](https://github.com/aimagelab/show-control-and-tell)]

![](https://s2.loli.net/2022/04/09/COnvomETrl6GRf2.png)

![](https://s2.loli.net/2022/04/09/7ASmXcCazOh9GsU.png)

```mermaid
  graph LR
  外部信号控制 --> 图像中的一组区域块 --> core((核心))
  core --> 改变chunk的顺序
  core --> 改变图像的区域
  model((模型)) --基于区域的特征与状态--> LSTM((LanguageModel,两层LSTM)) --第一层--> 计算attention --注意--> 所有区域的特征向量进行mean-pooling作为图像的总体特征I
  LSTM --第二层--> 预测下一个单词
  model --何时切换到下一个图像区域--> 块转移门 --计算gt--> 基于第一层LSTM的状态设立一个chunk-sentinel --> 类似计算ht对sc_t-rt的attention
  model --视觉词or文本次--> AdaptiveAttention --> 设置一个visual-sentinel --> 类似计算ht对sv_t-rt的attention --> attention的结果,可以计算出当前时刻模型正在关注的上下文特征ct
  model --无序集合排序--> 排序网络 --> R中包含N个区域集 --全连接层--> 每个区域集的特征映射为N维向量,然后拼接在一起 --Sinkhorn算子--> 软置换矩阵 
  每个区域集的特征映射为N维向量,然后拼接在一起 --> 最小化软置换与真实结果之间的均方误差
  每个区域集的特征映射为N维向量,然后拼接在一起 --测试匈牙利算法进行匹配--> 软置换矩阵转化为最终的置换,以此来对R进行排序
```

 [详细讲解](https://zhuanlan.zhihu.com/p/150667499)

:hammer_and_wrench: **Length-Controllable Image Captioning**, in ECCV 2020 by [Qi Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Q) and  Mingkui Tan.  [[pdf](https://arxiv.org/abs/2007.09580)] [[torch](https://github.com/bearcatt/LaBERT)]

* 动机
  * 为了让句子更加粗略或者细节，提出**长度可控**的caption生成
  * 过去由于方法是自回归的，所以计算复杂度会随着句子长度上升而上升。（模型上的创新）

![image-20220831210727673](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831210727673.png)

之前的SOTA方法可能会遗漏一些关键的信息，如果我想要更加细节点的描述，他们无法生成。

* 方法

  > 过去由于方法是自回归的，所以计算复杂度会随着句子长度上升而上升。在这里提出了 non-autoregressive的方法。

  * 获取句子长度信息（level -> $[L_{low}, L_{high}]$）做embedding

  * 提出Decode 阶段 (non-autoregressive) **LaBERT**

    * 使用位置信息来预测mask

    * 使用长度信息来预测unmask

    * 推理的时候鼓励生成**更长的句子**

      * exponentially decay: $p_i\left(s_i=[\mathrm{EOS}]\right) \leftarrow \gamma^{L_{\text {high }}-i} p_i\left(s_i=[\mathrm{EOS}]\right), \forall i \in\left[L_{\text {low }}, L_{\text {high }}\right]$

        ![image-20220831212334860](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831212334860.png)

      * 每一步都会对最低置信度的单词进行mask

:hammer_and_wrench: **Human-like Controllable Image Captioning with Verb-specific Semantic Roles**, in CVPR 2021.  [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Human-Like_Controllable_Image_Captioning_With_Verb-Specific_Semantic_Roles_CVPR_2021_paper.pdf)] [[torch](https://github.com/mad-red/VSR-guided-CIC)]

* 与上面两篇工作可控性的对比

![image-20220831213129731](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831213129731.png)

* 动机

  * 事件兼容性，两个不兼容的事件不应该合在一起

  * 采样的兼容性，不合理的采样不应该出现在句子当中

  * 对于上面的case： 

    ```python
    verb=sit, Arg1="thing sitting", Arg2="sitting position" 
    verb=read, Arg0="reader", Arg1="thing read"
    ```

* 方法上是先抽取出来约束的标签，再decoder

![image-20220831213747447](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831213747447.png)

:star: **MAGIC: Multimodal relAtional Graph adversarIal inferenCe for Diverse and Unpaired Text-Based Image Captioning**, in AAAI 2022.  [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/20243)]

* 动机
  * 为了caption的生成更加丰富多样，并且**无需过多的标注数据**！
  * caption直接做到**场景文本**级别
* 方法
  * Unpired Captioning的方法（其实就是`GAN`的思想）
  * 学到了模态内部，跨模态之间的关系
  * Unpaired 学习的范式，无需过多监督信号

![image-20220904163643769](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220904163643769.png)





:hammer_and_wrench: :fire: **Show, Edit and Tell: A Framework for Editing Image Captions**, in CVPR 2020.  [[pdf](https://arxiv.org/abs/2003.03107)] [[torch](https://github.com/fawazsammani/show-edit-tell)]

* 直接对生成的caption进行编辑修改

:hammer_and_wrench: **Towards Accurate Text-based Image Captioning with Content Diversity Exploration**, in CVPR 2021. [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Towards_Accurate_Text-Based_Image_Captioning_With_Content_Diversity_Exploration_CVPR_2021_paper.pdf)]  [[torch](https://github.com/guanghuixu/AnchorCaptioner)]

* 动机
  * Caption生成的多样性
  * 挑战
    * 不知道应该如何选择文本信息
    * 文本和图片之间的关系
    * 多样性caption的生成

![image-20220831221056898](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831221056898.png)



* 模型方法

![image-20220831221253743](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831221253743.png)



**Improving OCR-based Image Captioning by Incorporating Geometrical Relationship**, in CVPR 2021.  [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Improving_OCR-Based_Image_Captioning_by_Incorporating_Geometrical_Relationship_CVPR_2021_paper.pdf)]

* 动机
  * 无法建立OCR抽出来东西之间的关系
* 方法
  * 通过高度，宽度、距离、IoU和方向构建相应的OCR

![image-20220831223252058](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831223252058.png)



 :hammer_and_wrench: **Towards Unique and Informative Captioning of Images**, in ECCV 2020.  [[pdf](https://link.springer.com/content/pdf/10.1007/978-3-030-58571-6_37.pdf)] [[torch](https://github.com/princetonvisualai/SPICE-U)]

* 目前问题：

![image-20220901103845451](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901103845451.png)

* 关键贡献，做了一个**新的评价指标**

:hammer_and_wrench: **Comprehensive Image Captioning via Scene Graph Decomposition**, in ECCV 2020.  [[pdf](https://link.springer.com/content/pdf/10.1007/978-3-030-58568-6_13.pdf)] [[torch](https://pages.cs.wisc.edu/~yiwuzhong/Sub-GC.html)]

* 场景图分解来实现多样性

![image-20220901104357419](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901104357419.png)

![image-20220901104406842](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901104406842.png)

:hammer_and_wrench: **In Defense of Scene Graphs for Image Captioning**, in ICCV 2021.  [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9710596)]  [[torch](https://github.com/ Kien085/SG2Caps)]

* 动机
  * 弥补文本场景图还有视觉场景图直接的Gap
  * 以往的工作在训练captioner时，往往用**TSG作为输入**，测试时再换成VSG
  * VG数据集上学得的场景图中relationship多是has, on这类**无意义的关系**
  * VSG与TSG并不兼容  （两个场景图之间）

![image-20220831224945278](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831224945278.png)

* 基本思想
  *  close the **semantic gap** between the two scene graphs
  * 使用**HOI信息增强VSG**，并引入object location信息提升VSG的表达能力

![image-20220831224458911](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831224458911.png)

* 具体方法

  * VSG构建
    * VG数据集训练一个VSG generator，同以往工作一样对MSCOCO中的图片生成VSG。与此同时，作者又在MSCOCO上训练了一个object detector，对图片检测出一系列的物体。
  * VSG编码
    * 随后使用HOI inference对与人相关的物体进行关系及属性的预测。最后取原始VSG与HOI (检测到的物体) graph的并集作为最终VSG。
    * 使用多个GCN对其进行编码，不同类型的节点使用不同的GCN参数。
  * decode阶段 (Up-down)
    * 仅仅使用scene graph，不使用任何视觉特征，SG2Caps模型便可以取得有竞争力的描述生成结果。
  * case展示

  ![image-20220831230157943](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220831230157943.png)



:hammer_and_wrench: **Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning**, in CVPR 2022. [[pdf]()] [[torch](https://github.com/GT-RIPL/Xmodal-Ctx)]

* 关注到更多级别的信息

![image-20220901105828069](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901105828069.png)

* 方法上主要加入了Crop

![image-20220901105902348](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901105902348.png)

:hammer_and_wrench: **Comprehending and Ordering Semantics for Image Captioning**, in CVPR. [[pdf](https://arxiv.org/pdf/2206.06930.pdf)] [[torch](https://github.com/YehLi/xmodaler/tree/master/configs/image_caption/cosnet)]

* **语义的语言排序**（不单单是对象）同样很重要

![image-20220901111757395](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901111757395.png)

* 方法

![image-20220901111817536](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901111817536.png)

:hammer_and_wrench: **DIFNet: Boosting Visual Information Flow for Image Captioning**, in CVPR 2022.  [[pdf](DIFNet: Boosting Visual Information Flow for Image Captioning)] [[torch](https://github.com/mrwu-mac/DIFNet)]

* 考虑了信息流的信息

![image-20220901111002940](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901111002940.png)

![image-20220901111014355](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901111014355.png)

:hammer_and_wrench: **Injecting Semantic Concepts into End-to-End Image Captioning**, in CVPR 2022.  [[pdf](https://arxiv.org/abs/2112.05230)]  [[torch](https://github.com/jacobswan1/ViTCAP)]

* 端到端的训练，detector-free 和加入语义concept
* 过去的工作

![image-20220901104826507](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901104826507.png)

* 加入Concept
  * 通过抽取caption中的动名词或者通过知识蒸馏得到一些concept作为**伪标签**做分类

![image-20220901104847149](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901104847149.png)



:hammer_and_wrench: **Show, Deconfound and Tell: Image Captioning with Causal Inference**, in CVPR 2022.  [[pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Show_Deconfound_and_Tell_Image_Captioning_With_Causal_Inference_CVPR_2022_paper.pdf)] [[torch](https: //github.com/CUMTGG/CIIC)]

* 解决数据集中大量出现了，模型**short-cut path** 的问题

![image-20220901110327977](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901110327977.png)

* 主要为了解决两个Caption存在的问题
  * 识别**对象错误**（长头发的男人识别成了女人）
  * 描述得**不够关键和详细**

* Encoder阶段（解决分类准确性的问题）
  * 基于Faster-RCNN得到无偏的物体分类

![image-20220901110456322](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220901110456322.png)

![image-20220915113730411](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220915113730411.png)

![image-20220915113654808](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220915113654808.png)

* decoder阶段考虑生成单词的bias

![image-20220915114452498](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220915114452498.png)



![image-20220915114520789](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220915114520789.png)



**[因果关系 + 强化学习] Dependent Multi-Task Learning with Causal Intervention for Image Captioning**, in IJCAI 2021.  [[pdf](https://www.ijcai.org/proceedings/2021/0312.pdf)] 

> 说实话这篇论文写作**有点太复杂了，很难看懂**

* 解决caption生成**反事实**与**不够详细**的问题

![image-20220914112625963](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220914112625963.png)

* **因果干预分析过程比较复杂，详见论文**

![image-20220914112637246](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220914112637246.png)



## :sunglasses: Video Understanding

### :video_camera: Features Fusion

:white_check_mark: :fire: :hammer_and_wrench: **[TSN] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**, in ECCV 2016.  [[pdf](https://arxiv.org/abs/1608.00859)]  [[torch](https://github.com/yjxiong/temporal-segment-networks)]

* 抽取所有帧是不现实的，TSN将其**等间隔分**为$K$个片段（i.e., $K=16$）,在每个片段中谁寄抽取一帧作为输入

* 提供了非常常用的数据争强方式和一些训练时候的trick（主要包括location jittering, horizontal flipping, corner cropping, and scale jittering）

* 仍然利用双流的思路，让每个片段信息最后通过一个共识网络再Fusion

  ![](https://pic4.zhimg.com/80/v2-67b66b3618606af8b81d1f77b1f92a3b_1440w.jpg)

:white_check_mark: :fire: :hammer_and_wrench: **[TRN] Temporal Relation Reasoning in Videos**, in ECCV 2018.  [[pdf]()] [[torch](https://github.com/zhoubolei/TRN-pytorch)]

![img](https://pic4.zhimg.com/80/v2-86fa6c271c9d2dfad07d4603ed457a83_1440w.jpg)

* 融合尺度确定 (需要多少个视频帧来融合)【如图所示】有2，3，4这三种尺度
* 每个尺度下需要多少组视频帧
* 在应用多尺度TRN的时候，一般会额外增加一个全帧的尺度，即12帧特征全部concat到一起，以充分利用有效信息。
* **平衡**效果和计算速度，**简单好用**

:white_check_mark: :fire: :hammer_and_wrench: **[TSM] TSM: Temporal Shift Module for Efficient Video Understanding**, in ICCV 2019.  [[pdf](https://arxiv.org/abs/1811.08383)] [[torch](https://github.com/mit-han-lab/temporal-shift-module)]

* 对某些通道shift，得到前一帧或者后一帧的特征

  ![image-20220709174802714](https://s2.loli.net/2022/07/09/cb1f8LWpV9JotO3.png)

* 由于shift是有损失的，为此设计残差来进行弥补（原来的与残差的对比）

![image-20220709174842935](https://s2.loli.net/2022/07/09/2dTjLBJwQ5FUber.png)

:white_check_mark: :fire: :hammer_and_wrench: **[LRCN] Long-term Recurrent Convolutional Networks for Visual Recognition and Description**, in CVPR 2015.  [[pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)] [[torch](https://github.com/garythung/torch-lrcn)]

* CNN抽出来的帧特征再放进去`LSTM`得到每帧的时序特征

> 关于视频特征抽取，下面讲一下`netvlad`系列的结构,NextVlad就是专门针对视频帧融合来做的优化。
>
> [相关博客链接](https://zhuanlan.zhihu.com/p/385512915)

:fire: :hammer_and_wrench: **[NetVLAD] NetVLAD: CNN architecture for weakly supervised place recognition**, in CVPR 2016.  [[pdf](https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf)] [[torch (simple)](https://github.com/lyakaap/NetVLAD-pytorch)]

* VLAD算法（实际上就是Kmeans）：
  $$
  V(j, k)=\sum_{i=1}^{N} a_{k}\left(x_{i}\right)\left(x_{i}(j)-c_{k}(j)\right), \quad k \in K, j \in D
  $$

* 本文使用`CNN`模拟该VLAD算法的过程

  * 平滑化$\alpha$ 使其变成一个0-1分布的权重参数，使用$1 \times 1$卷积+softmax 进行该过程，平滑推导公式$\bar{a}_{k}\left(\mathbf{x}_{i}\right)=\frac{e^{-\alpha\left\|\mathbf{x}_{i}-\mathbf{c}_{k}\right\|^{2}}}{\sum_{k^{\prime}} e^{-\alpha\left\|\mathbf{x}_{i}-\mathbf{c}_{k^{\prime}}\right\|^{2}}}$

  

  ![image-20220712104829439](https://s2.loli.net/2022/07/12/aFlJSCw8dvBnzEH.png)

   

:fire: :hammer_and_wrench: **[NextVLAD] NeXtVLAD: An Efficient Neural Network to Aggregate Frame-level Features for Large-scale Video Classification**, in ECCV workshop 2018.  [[pdf](https://arxiv.org/pdf/1811.05014.pdf)] [[tensorflow](https://github.com/linrongc/youtube-8m)]

> 同时，还有[关于多模态（视频-文本）Transformer模型的博客链接](https://zhuanlan.zhihu.com/p/388361095)





### :timer_clock: Temporal Grounding

> 我们使用一个十分经典的任务（Temporal Grounding）来看看视频的特征是如何利用的

:fire::hammer_and_wrench: **[Video-NLP] Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language**, in AAAI 2020.  [[pdf](https://arxiv.org/pdf/1912.03590.pdf)] [[torch](https://github.com/microsoft/VideoX)]

> extend version: **MS-2D-TAN**, in TPAMI 2021.  [[pdf](https://arxiv.org/pdf/2012.02646.pdf)]  [[torch](https://github.com/microsoft/VideoX)]

* 2D: start time & end time 构成的邻接矩阵

![image-20220723153302775](https://s2.loli.net/2022/07/23/lUBL6uPHRFCzvxt.png)

* **核心思想**：

  * 使用max pooling （本文使用） 或者 stack convolution的获取moment feature，如上图`2D Temporal Feature Map Extraction`所示
  * 由于这样子的计算开销太大了，使用**特定的采样方式**进行调整，详见论文！（距离近的采样多一点，远的采样少一点）
  * 多模态融合（`Hadamard product`）

  $$
  \mathbf{F}=\left\|\left(\mathbf{w}^{S} \cdot \mathbf{f}^{S} \cdot \mathbb{1}^{T}\right) \odot\left(\mathbf{W}^{M} \cdot \mathbf{F}^{M}\right)\right\|_{F}
  $$

  * 损失计算时候，对`IoU`进行一个scale变成监督信号

  $$
  y_{i}= \begin{cases}0 & o_{i} \leq t_{\min } \\ \frac{o_{i}-t_{\min }}{t_{\max }-t_{\min }} & t_{\min }<o_{i}<t_{\max } \\ 1 & o_{i} \geq t_{\max }\end{cases}
  $$

  * BCE loss:

  $$
  L o s s=\frac{1}{C} \sum_{i=1}^{C} y_{i} \log p_{i}+\left(1-y_{i}\right) \log \left(1-p_{i}\right)
  $$

  

:fire: :hammer_and_wrench: **Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding**, in AAAI 2022. [[pdf](https://arxiv.org/abs/2109.04872)] [[torch](https://github.com/MCG-NJU/MMN)] [[blog](https://zhuanlan.zhihu.com/p/446203594)]

* 主干网络是沿用[TDN](https://arxiv.org/abs/2012.10071)

* 使用了**metric learning**的方法并且引入**负样本**来做Temporal Grounding的任务

  * 视频间的负样本（`IoU`来标定监督信号`yi`，与`2D-TAN`一样处理得来的，记得`scale`一下）
  * 文本中的负样本，从其他视频的文本语句当中选取出来

* 贡献

  * 构造了新的监督信号：视频间的正负样本(`IoU`来采样)， 句子和视频对应的正负样本（负样本句子从别的视频抽取过来）
  * 一个视频只需要建模一次，大大节省训练时间，以往的fusion方法都是要文本-视频帧对来建模

* Trick

  * 为了编码公平，使用预训练好的`DistilBERT`来进行编码句子

* 损失函数计算

  * 和`2D-TDN`一样的`BCE_loss`
  * 类似于`InfoNCE loss`的设计对比损失

  $$
  \begin{aligned}
  &p\left(i_{s} \mid v\right)=\frac{\exp \left(\left(\mathbf{f}_{i}^{S T} \mathbf{f}^{V}-m\right) / \tau_{v}\right)}{\exp \left(\left(\mathbf{f}_{i}^{S T} \mathbf{f}^{V}-m\right) / \tau_{v}\right)+\sum_{j \neq i}^{N_{s}} \exp \left(\mathbf{f}_{j}^{S T} \mathbf{f}^{V} / \tau_{v}\right)} \\
  &p\left(i_{v} \mid s\right)=\frac{\exp \left(\left(\mathbf{f}_{i}^{V T} \mathbf{f}^{S}-m\right) / \tau_{s}\right)}{\exp \left(\left(\mathbf{f}_{i}^{V T} \mathbf{f}^{S}-m\right) / \tau_{s}\right)+\sum_{j \neq i}^{N_{v}} \exp \left(\mathbf{f}_{j}^{V T} \mathbf{f}^{S} / \tau_{s}\right)} \\
  &L_{m m}=-\left(\sum_{i=1}^{N} \log p\left(i_{v} \mid s_{i}\right)+\sum_{i=1}^{N} \log p\left(i_{s} \mid v_{i}\right)\right)
  \end{aligned}
  $$

  

### :man_student: Video Question Answer

**Invariant Grounding for Video Question Answering**, in CVPR 2022 oral.  [[pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)] [[torch](https://github.com/yl3800/IGV)]

> 这篇文章感觉是一篇很标准的`CVPR`的中规中矩文章，写作用词上非常出色的

* 先做了Grounding的检测，检测出问题相关帧（有因果关系`Casual`）还有无关帧（补偿帧`Complement`）
* 构建负样本到无关帧当中，使用`memory bank`来存储所有样本 (因此要注意存储的特征维度不能太大)

**Video as Conditional Graph Hierarchy for Multi-Granular Question Answering**，in AAAI 2022. [[pdf](https://arxiv.org/abs/2112.06197)] [[torch](https://arxiv.org/abs/2112.06197)]

* 现有的方法对视频问题的回答缺乏**可解释性**
* 构建了两种视角来看问题
  * bottom-up， 不同的视频特征决定了不同的属性level（实体，原子，动作，事件）
  * up-bottom，问题中的不同单词，关联了不同的level
* 构建**图神经网络**来模拟这些level思考的过程

![image-20220910231620242](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220910231620242.png)

:fire: :hammer_and_wrench: **[交通事故QA数据集] SUTD-TraffificQA: A Question Answering Benchmark and an Effificient Network for Video Reasoning over Traffific Events**, in CVPR 2021. [[pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_SUTD-TrafficQA_A_Question_Answering_Benchmark_and_an_Efficient_Network_for_CVPR_2021_paper.pdf)] [[project](https://github.com/SUTDCV/SUTD-TrafficQA[)]

* 对比起其他QA数据集
  * 需要模型有因果推理和认知发展（cognitive development）
* 处理方法
  * 对粗细两种粒度进行识别和计算（不同的CNN网络），大大加快了模型的运算时间

![image-20220912152326067](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220912152326067.png)

**Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering**, in TPAMI 2022. [[pdf](https://arxiv.org/pdf/2207.12647.pdf)]

> 这篇论文模型较为复杂，所以这里只讲诉其核心思想

* 动机

  * 现有方法只关注了很简单的事件，比如说看电影，无法关注真正事件级的因果关系

    ![image-20220912224156198](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220912224156198.png)

  * 语言和图像当中的干扰因素（Confounder）

    * 过于关注一些显式的东西，忽略了一些很重要的东西

  ![image-20220912224356724](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220912224356724.png)

* 方法

![image-20220912224635966](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220912224635966.png)

* 详细细节见论文！



### :writing_hand: Video Caption

**[Video Caption] VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2101.12059)]

:hammer_and_wrench: :fire: **[Video Caption] Robust Change Captioning**, in ICCV 2019. [[pdf](https://arxiv.org/pdf/1901.02527.pdf)] [[torch](https://github.com/Seth-Park/RobustChangeCaptioning)]

* 输入为前后图像对，五种变化类型（color/material change,adding/dropping/moving an object）
* 提出一个有视点变化的数据集[CLEVR-Change](https://cs.stanford.edu/people/jcjohns/clevr/)（80K图片对），并在无视点变化的数据集[Spot-the-Diff](https://github.com/harsh19/spot-the-diff)取得SOTA效果。
* 模型：Dual 注意力， 分辨**视点变化**， 其实是通过输入两张差不多的图片，提前标定好数据集获得的，有点被坑的意思![image-20220522213419579](https://s2.loli.net/2022/05/22/fiUArgZIjlzw4p1.png)

:hammer_and_wrench: :fire: **[Video Caption] Semantic Grouping Network for Video Captioning**, in AAAI 2021. [[pdf](https://arxiv.org/pdf/2102.00831.pdf)] [[torch](https://github.com/hobincar/SGN)]

* ![image-20220621204108736](https://s2.loli.net/2022/06/21/DMmzxs7dKwyU6BE.png)

:hammer_and_wrench: :fire: **Hierarchical Context-aware Network for Dense Video Event Captioning**, in ACL 2021. [[pdf](https://aclanthology.org/2021.acl-long.156/)] [[torch](https://github.com/KirkGuo/HCN)]

* **局部信息**+**全局信息**结合生成dense caption （输入包括**video** 和 **transcript**）
* 为此设计了两套`Attention`机制
  * falt attention + cross attention

![image-20220820000708776](https://s2.loli.net/2022/08/20/wdO5eWZcxgqsItT.png)

* 设计了门机制来decode（之前的文本信息与未来的文本信息）



```mermaid
  graph LR
  SG(Semantic-Grouping) --去掉冗余phrase--> 相似度计算
  SG --attention机制 --> 对其phrase和frame --> 加入对比损失,计算没有包含negative的概率
```

**对比损失**$\mathcal{L}_{c a}=\sum_{(V, Y) \in \mathcal{D}} \sum_{t} \sum_{i}^{M_{t}}\left(-\log p_{c a}\left(s_{i, t}\right)\right)$, $p_{c a}\left(s_{i, t}\right)=\sum_{j=1}^{N} \alpha_{i, j, t}^{p o s}$    ($\alpha^{pos}$ 为正样本时候对齐注意力的权重) 



##  :apple: Causality Learning

:fire: :hammer_and_wrench: **Distilling Causal Effect of Data in Class-Incremental Learning**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2103.01737)] [[torch](https://github.com/JoyHuYY1412/DDE_CIL)]

* [[模型公式解释](https://zhuanlan.zhihu.com/p/358340627)]  [[论文介绍](https://www.163.com/dy/article/G4OHT10U0511DPVD.html)]

* 动机

  * 对撞节点的存在使得模型对新数据会产生灾难性遗忘

  * 过去的方法当中

    ![image-20220918101920897](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220918101920897.png)

    ![img](https://pic1.zhimg.com/80/v2-0a7452df24b00f8e7d186f0dcf0c0680_720w.jpg)

    

    		* **data reply**: 需要较大的**存储**空间；**distillation**: 不是**端到端**的表示学习。因此作者考虑吧是否存在一种端到端影响的蒸馏方法。

* 思路

  1. 构建**因果角度下的类别增量学习**过程
  2. 分析**灾难性遗忘发生的原因**（**causal effect** lost）
  3. 分析现有工作如何实现有效的抗遗忘。 在这些基础上，我们发现**控制对撞节点**是一种尚未利用、但非常有效的抗遗忘方法，在各种类别增量学习的设定上取得了稳定的提升。
  4. 同时解决了数据（新旧类别）采样分布**不均匀**导致的**bias**问题。

* **文章细节详见开头的博客链接**

  

**Learning Causal Effects on Hypergraphs**, Best Paper of KDD 2022. [[pdf](https://arxiv.org/pdf/2207.04049.pdf)]

* **超图**上进行因果影响分析，对比传统的以**两两节点**为图的更加有意义。
* [[博客链接](https://zhuanlan.zhihu.com/p/564481108)] [[方法细节](https://zhuanlan.zhihu.com/p/567996036)]



## :abc: Scene Text Recognization

:hammer_and_wrench: **From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network**, in ICCV 2021. [[pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2108.09661)] [[torch](https://link.zhihu.com/?target=https%3A//github.com/wangyuxin87/VisionLAN)]

* 过去的场景文本识别需要：视觉特征抽取器 + 语言模型

* 本文直接在视觉空间进行语言建模（类似人类，语言信息是可以学习的）
  * 对字符级别的Mask操作![image-20220701212346925](https://s2.loli.net/2022/07/01/ZLFUIkb41S782GD.png)
    * 训练过程，采用弱监督互补学习![image-20220701212430601](https://s2.loli.net/2022/07/01/kc3K7XxAN6SfRut.png)

:hammer_and_wrench: **Visual Semantics Allow for Textual Reasoning Better in Scene Text Recognition**, in AAAI 2022.  [[pdf]([https://arxiv.org/abs/2112.12916](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2112.12916))] [[torch]([https://github.com/adeline-cs/GTR](https://link.zhihu.com/?target=https%3A//github.com/adeline-cs/GTR))]
* 加入一个GCN强化了视觉学习的过程，并且做了一个fusion



## :label: NER

>  Named Entity Recognition

**当前竞赛NER任务的baseline：**

- BERT + BILSTM + CRF
- [博客连接 NER铁打的baseline](https://zhuanlan.zhihu.com/p/166496466)

:fire: **Bidirectional LSTM-CRF Models for Sequence Tagging**, in 2015. [[pdf](https://arxiv.org/pdf/1508.01991v1.pdf)] [[code](https://paperswithcode.com/paper/bidirectional-lstm-crf-models-for-sequence)

* 使用BiLSTM+CRF做NER的开山之作
* [相关博客连接](https://zhuanlan.zhihu.com/p/166496466)

:hammer_and_wrench: :fire: **Fast and Accurate Entity Recognition with Iterated Dilated Convolutions**, in EMNLP 2017.  [[pdf](https://arxiv.org/pdf/1702.02098.pdf)] [[tensorflow](https://github.com/iesl/dilated-cnn-ner)]

* Iterated Dilated Convolutions 空洞卷积 


![image-20220825002628063](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220825002628063.png)

* 核心思想

  * 传统卷积的问题：pooling层会**损失信息**，降低精度。那么不加pooling层会使**感受野变小**，学不到全局的特征。如果单纯的去掉pooling层、扩大卷积核的话，这样纯粹的扩大卷积核势必导致**计算量的增大**。
  * CNN也可以解决长距离依赖问题

![image-20220825004251541](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220825004251541.png)

* 速度比以前的`Bi-LSTM-CRF`快了非常多，而且精度没有下滑

:hammer_and_wrench: :fire: **BOND: BERT-Assisted Open-Domain Named Entity Recognition with Distant Supervision** , in KDD 2020.  [[pdf](https://arxiv.org/pdf/2006.15509.pdf)] [[torch](https://github.com/cliang1453/BOND)]

* 远距离监督的问题

  * 噪声，负样本不好生成
  * 完整性不足
  * `trade-off` 在标注准确度和覆盖范围之间

* 想法

  * 第一阶段使用`RoBERTa`微调，适应`NER`任务

    * 使用`Early stopping` 方法防止数据过拟合还有对**未知数据增强泛化能力**
    * 首先通过`POS`识别潜在实体，然后通过语料库计算最小损失确定实体

    ![img](https://pic2.zhimg.com/80/v2-0951f0afefa53b2efc269f92136a3ae9_720w.jpg)

  * 第二阶段自我学习框架 （teacher-student模型2）

    * 应对**嘈杂和不完整标注**的挑战
    * teacher生成伪标签交给student去预测
    * **重新加权的高置信度软标签**
    * ==第二阶段的核心就是增强数据的置信程度！==

![img](https://pic4.zhimg.com/80/v2-8788aa61ea19b041e763b597d844ebcb_720w.jpg)

两阶段的BOND框架

* 在阶段I中，经过预训练的BERT用于早停的远距离NER任务
* 在阶段II中，首先从阶段I中学习的模型初始化student模型和teacher模型。然后使用teacher模型生成的伪标签对student模型进行训练。同时，由早停的student迭代更新teacher模型。



:hammer_and_wrench: :fire: **[A Boundary-aware Neural Model for Nested Named Entity Recognition](https://aclanthology.org/D19-1034.pdf)** , in EMNLP 2019.  [[pdf](https://aclanthology.org/D19-1034.pdf)] [[torch](https://github.com/thecharm/boundary-aware-nested-ner)]

* 解决`Nested NER`的问题
* 使用多任务学习方法
  * 预测边界
  * 预测实体分类

:hammer_and_wrench: :fire: **Cross-Domain NER using Cross-Domain Language** , in ACL 2020.  [[pdf](https://aclanthology.org/P19-1236/)] [[torch](https://github.com/jiachenwestlake/Cross-Domain_NER)]

* 解决**跨领域无监督**的标注问题
* 核心思想

![img](https://pic4.zhimg.com/80/v2-52cb3242e0a708e48b29ab2f8d81e027_720w.jpg)

最底下的一层是数据层，标准情况一下总共有四份语料，分别对应**两个domain下的两个task（NER和语言建模）**。其中Source Domain（即保证有标记数据用于NER的domain）对应之前提到的News Domain，因为论文中Source Domain使用的是新闻数据。另外如果是无监督抽取Target Domain数据则只有三份语料。

由底向上第二层是Word Embedding层，论文中的Word Embedding结合了词级别和字符级别的向量表示。即把词向量和一个词的字符序列形成的矩阵经过CNN处理后的**向量concatenate起来**。

第三层是双向LSTM，用于序列处理第二层的数据，生成前后向hidden state。

第四层LM和CRF，即task model层。我们可以看到第四层有三个模块，两个是用于NER的CRF模型，分别对应Source Domain和Target Domain。另一个是基于第三层BiLSTM的语言模型，NSSoftmax是指这个语言模型利用**Negative Sampling Softmax的方式进行训练**。



* 最关键的地方就是`Bi-LSTM`的参数是生成的，不同domain的不同task需要的LSTM的参数和$I$有关

$$
\begin{equation}
\theta_{\mathrm{LSTM}}^{d, t}=\mathbf{W} \otimes \mathbf{I}_{d}^{D} \otimes \mathbf{I}_{t}^{T},
\end{equation}
$$





**NER的未来**

既然模型打不动了，然后我找了找 ACL2020做NER的论文，看看现在的NER还在做哪些事情，主要分几个方面

1. **多特征**：实体识别不是一个特别复杂的任务，不需要太深入的模型，那么就是加特征，特征越多效果越好，所以字特征、词特征、词性特征、句法特征、KG表征等等的就一个个加吧，甚至有些中文 NER 任务里还加入了拼音特征、笔画特征...... 心有多大，特征就有多多
2. **多任务**：很多时候做 NER 的目的并不仅是为了 NER，而是服务于一个更大的目标或系统，比如信息抽取、问答系统等等。如果把整个大任务做一个端到端的模型，就需要做成一个多任务模型，把 NER 作为其中一个子任务；另外，单纯的 NER 也可以做成多任务，比如实体类型过多时，仅用一个序列标注任务来同时抽取实体与判断实体类型，会有些力不从心，就可以拆成两个子任务来做
3. **时令大杂烩**：把当下比较流行的深度学习话题或方法跟 NER 结合一下，比如结合强化学习的 NER、结合 **few-shot learning** 的 NER、结合多模态信息的 NER、结合跨语种学习的 NER 等等的，具体就不提了 (Few-shot + Cross-domain是个不错的选项！)

作者：王岳王院长
链接：https://zhuanlan.zhihu.com/p/166496466
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
