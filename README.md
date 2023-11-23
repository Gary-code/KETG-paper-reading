

# Knowledge-enriched Text Generation paper reading

😎 Awesome list of papers about knowledge-enhanced Question generation with notes.

:white_check_mark: : **already reading carefully**

:fire:: **high citation in recent years**

:hammer_and_wrench:: **available code**

> Content

[TOC]





## :bookmark_tabs: Question Generation & Answering & Reasoning

---

### :mountain_snow: **Textual QG**

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

:hammer_and_wrench: **Varifocal Question Generation for Fact-checking**, in EMNLP 2022. [[pdf](https://arxiv.org/abs/2210.12400)] [[torch]()]

> 使用QG辅助做Fact-checking任务（很有用的一个应用场景），也有对应的数据集

![image-20221113161641588](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221113161641588.png)





:fire: :hammer_and_wrench: **Generative Language Models for Paragraph-Level Question Generation**, in EMNLP 2022 oral. [[pdf](https://arxiv.org/abs/2210.03992)] [[demo](https://autoqg.net/#/)]

> 大模型做QG的Benchmark
>
> * 非常solid的一项工作

* 动机
  * 之前都是基于评价指标来进行计算的，如`BLEU`等
  * 存在很多方面的约束，比如可回答性，输入不一样等
* 贡献
  * 统一到一个benchmark当中
    * 特定领域的dataset
    * 多种语言（8种）
  * 在LM中进行微调，在上一点提到的数据中进行验证
  * 多方面验证
    * 自动评价指标
    * 人工评价指标



:hammer_and_wrench: **Educational Question Generation of Children Storybooks via Question Type Distribution Learning and Event-Centric Summarization**, in ACL 2022. [[pdf](https://arxiv.org/abs/2203.14187)] [[torch](https://github.com/zhaozj89/Educational-Question-Generation)]

* 动机

  * 从童话故事当中生成`hugh-cognitive-demand`的问题很有意义
  * 过去都是`low-dognitive-demand (LCD)` 问题描述，比如说 他是谁？
  * 故事的事件之间是有联系的，需要问出action，causal relationship的问题

* 方法

  ![image-20221114170953173](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221114170953173.png)



:hammer_and_wrench: **CQG: A Simple and Effective Controlled Generation Framework for Multi-hop Question Generation**, in ACL 2022.  [[pdf](https://aclanthology.org/2022.acl-long.475.pdf)] [[torch](https://github.com/sion-zcfei/CQG)]

* 动机

  * 过去的多挑QG方法无法保证问题的**复杂程度**（complexity）
  * 两大挑战
    * 建立不同文档信息的联系
    * complex chains of entities

  ![image-20221115112431373](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221115112431373.png)

* 方法

  * 先试用GAT抽取多文档之间联系，`Standford corenlp toolkit`来建立实体图

    * 重点在于找到关键的实体，和gt进行loss的计算

  * 设计flag tag来约束chain of entities，保证问题生成的复杂程度（注意：并不是`teacher forcig`）
    $$
    \operatorname{flag}_i^t= \begin{cases}0 & x_i \text { is not a constrain } \\ 1 & x_i \text { does not appear in } y_{1: t} \\ 2 & x_i \text { appear in } y_{1: t}\end{cases}
    $$

![image-20221115112909494](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221115112909494.png)

  * 模型图

  ![image-20221115155002446](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221115155002446.png)




:hammer_and_wrench: **[Meta Learning] DSM: Question Generation over Knowledge Base via Modeling Diverse Subgraphs with Meta-learner**, in EMNLP 2022. [[pdf](https://xiaojingzi.github.io/publications/EMNLP22-Guo-et-al-DSM.pdf)] [[torch](https://github.com/RUCKBReasoning/DSM)]

> COLING 2022 也有一篇[CQG](https://aclanthology.org/2022.acl-long.475/)使用**Meta Learning**做QG的，和这篇很像

* 动机

  * 不同的问题实际上有**不同的知识语义结构**
  * 过去的模型往往都是使用**一个方法来拟合这些不同语义的问题**，但是这样子会导致模型训练困难（实际上还可以解决数据imbalance的问题，COLING2022 CQG中的动机就是这个）

  ![image-20230218215343124](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230218215343124.png)

* 方法模型，类似**MAML的元学习器**， [MAML简单解析](https://blog.csdn.net/weixin_42392454/article/details/109891791)

![image-20230218215656301](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230218215656301.png)



:hammer_and_wrench: **Modeling What-to-ask and How-to-ask for Answer-unaware Conversational Question Generation**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.03088)] [[torch](https://github.com/dxlong2000/SG-CQG)]

> 有点教育学 + AI的意思，但实际上还是比较简单的做法

* 动机：What-to-ask and How-to-ask （**教育学**）
* 方法（RoBERTa$_{large}$ 作为**QT classifier**， 预训练好的T5$_{base}$作为主干网络参数）

![image-20230510163004578](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230510163004578.png)



:hammer_and_wrench: **Synthetic Question Value Estimation for Domain Adaptation of Question Answering**, in ACL 2023. [[pdf](https://aclanthology.org/2022.acl-long.95.pdf)] [[torch](https://github. com/xiangyue9607/QVE)]

* 动机
  * 目标领域的QA，需要生成目标领域的QG（设置有点复杂）

![image-20230718215810631](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230718215810631.png)

* 模型方法

![image-20230718215838581](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230718215838581.png)


****

### :sunrise: Visual QG

:fire: **Generating Natural Questions About an Image**, in ACL 2022.

> 第一篇VQG的工作

* 动机

  * 与caption相比，VQG更希望稳一些超出语言表达本身，具有逻辑推理，**常识知识**的一些问题

  ![image-20221111221831899](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221111221831899.png)

  ![image-20221111221858832](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221111221858832.png)

* 方法

  * 很简单的方法，就是一些简单的baseline尝试

:hammer_and_wrench: **[No Visual] Entity Guided Question Generation with Contextual Structure and Sequence Information Capturing**, in AAAI 2021. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/17544)] [[torch](https://github.com/VISLANG-Lab/EGSS)]

* Multi-feature Encoder: 使用了POS（词性标注）+ NER（关系抽取）



:white_check_mark:  :hammer_and_wrench:  **Mixture Content Selection for Diverse Sequence Generation**, in EMNLP 2019.[[pdf](https://arxiv.org/abs/1909.01953)] [[torch](https://github.com/clovaai/FocusSeq2Seq)]

:hammer_and_wrench: **Radial Graph Convolutional Network for Visual Question Generation**, in IEEE Transactions on Neural Networks and Learning Systems 2020. [[pdf](https://ieeexplore.ieee.org/document/9079208)] [[torch](https://github.com/Wangt-CN/VQG-GCN)]

### :sunflower: VQG

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

:hammer_and_wrench: **Inferential Visual Question Generation**, in MM 2022.  [[pdf](https://dl.acm.org/doi/10.1145/3503161.3548055)] [[dataset & code](https://github.com/bcxbg/InVQG)]

> 无需模型训练的，就是人为**规定函数模版**规则的方法生成的。

* 动机
  * 过去VQG的方法生成问题不够challenge 或者依赖于人工标记
* 过去的VQG
  * 只是简单的反转一下VQA
  * 数据集当中有很多long tail和语义联系太弱了
  * 生成问题太过普通，而且缺少关键信息
  * 与Caption不同的是，VQG需要生成具有挑战性的问题
* 方法

![image-20221015215029327](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221015215029327.png)

* 例子
  * 增加可交互性

![image-20221015215356890](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221015215356890.png)

![截屏2022-10-15 21.54.35](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-15%2021.54.35.png)

:hammer_and_wrench: **Multi-VQG: Generating Engaging Questions for Multiple Images**, in EMNLP 2022. [[pdf](https://arxiv.org/abs/2211.07441)]] [[dataset](https://github.com/AcademiaSinicaNLPLab/MVQG-Dataset-of-Generating-Engaging-Questions-for-Multiple-Images)]

> Baseline就是`VL-T5`，详见论文

* 动机
  * 过去的VQG主要是针对一张图片进行提问的，往往回答的都是**很简单的事实**。而人类通常在构造问题之前会**构造一个故事**，类似于发推特
  * 过去的VQG减少了了参与者的**回答意愿**
  * **单个图片**缺乏**事件信息**，缺乏对事件信息**时序**的理解

<img src="https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230127235522552.png" alt="image-20230127235522552" />

* dataset的特点
  * 故事背后包含**故事信息**
  * 五张图片 + 一个故事描述
  * 数据集标注方法
    * 写出top5对象和事件
    * 给出一个描述
    * 想想如果想将**这五张图片发上去推特**会问什么
* 更多Cases

![image-20230128205258419](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230128205258419.png)



:hammer_and_wrench: **Why Did the Chicken Cross the Road? Rephrasing and Analyzing Ambiguous Questions in VQA**, in ACL 2023. [[pdf](https://arxiv.org/abs/2211.07516)] [[dataset&torch](https://github.com/esteng/ambiguous_vqa)]

* 动机
  * 标注一个新的VQA数据集（只有验证和测试时候的），因为这个工作是想用来评估当下VQG模型的性能
  * 缓解一下VQG任务当中一些很ambiguity的问题
    * 视觉信息没出现
    * 问法不明确

| ![image-20230621151706413](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230621151706413.png) | ![image-20230621151725513](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230621151725513.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



* 方法模型
  * Baseline验证一下而已
  * 未来可以用来做**VQG改写**的任务



:fire: :hammer_and_wrench: **Location-Aware Visual Question Generation with Lightweight Models**, in EMNLP 2023. [[pdf](https://arxiv.org/pdf/2310.15129v1.pdf)] [[torch](https://github.com/academiasinicanlplab/locavqg)]

> 很搞的一个动机

* 动机
  * 避免司机的疲劳驾驶，让移动设备根据当前驾驶环境给司机提问

![image-20231105103551827](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231105103551827.png)

* 数据集构建过程（利用GPT-4）

![image-20231105103601176](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231105103601176.png)

---



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

:hammer_and_wrench: **[2023.3未开源] Learning to Retrieve Videos by Asking Questions**, in MM 2022. [[pdf](https://arxiv.org/abs/2205.05739)] [[torch](https://github.com/avinashsai/ViRED)]

* 动机

  * 缩小检索的范围，更加interactive

  ![image-20230317100606225](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230317100606225.png)



* 和过去方法的不同

![image-20230317100636723](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230317100636723.png)

* 模型方法

![image-20230317100707179](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230317100707179.png)





### :city_sunset: VQA

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



:hammer_and_wrench:  **Knowledge-Grounded Self-Rationalization via Extractive and Natural Language Explanations**, in ICML 2022. [[pdf](https://arxiv.org/abs/2106.13876)] [[code (not released in 2022/11/20)](https://github.com/majumderb/rexc)]

> 注意这篇论文早于👇下面的两篇论文，所以结果上和下面两篇论文有明显的差距

* 动机

  * 借助**关键信息抽取**还有**解释生成**的方法来预测答案会更好

* 方法（这篇论文方法上文章没有做很详细的，描述，具体代码也没有公开，所以无法很详细的解析）

  ![image-20221120100005850](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221120100005850.png)

  * 抽取关键元素
    * 使用`HardKuma`吩咐的方法来学习latent selectors 选择合适的元素
  * 知识抽取
  * 知识选择（方法和第一点一致）
  * 生成解释+预测答案（**先**生成解释**后**预测答案）

  ![image-20221120100330138](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221120100330138.png)

  



:hammer_and_wrench: **NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks**, in CVPR 2022. [[pdf](https://arxiv.org/abs/2203.05081)] [[pdf](https://github.com/fawazsammani/nlxgpt)]

* 动机
  * 之前方法分为先生成`A`再生成`E`两步走, 缺乏对reasoning过程的考虑
  * 缺少一些对E和A相关性还有对bias程度估计的评价指标
  * 过去和文章方法上的对比

![image-20221001182330655](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221001182330655.png)

* 方法（及其简单）
  * GPT-2模型是在大规模的image-caption数据集中蒸馏出来的
  * Vision Encoder 使用了`CLIP`

![image-20221001182426390](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221001182426390.png)

* 新的自动化评价指标

  * 评估`E`和`A`的相关性

  ![image-20221001182614962](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221001182614962.png)

  * 评估模型Bias程度

  ![image-20221001182654034](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221001182654034.png)

  红色为小于0的值，都置为0，然后计算distance的平均值，**值越小，模型的Bias越小**

:hammer_and_wrench: **[Viisual Explanation]  Chunk-aware Alignment and Lexical Constraint for VisualEntailment with Natural Language Explanations**, in MM 2022. [[pdf](https://arxiv.org/abs/2207.11401)] [[Talk](https://www.youtube.com/watch?v=nAHIZOQSiXg)] [[torch](https://github.com/HITsz-TMG/ExplainableVisualEntailment)]

* 是上面一篇工作的进一步研究
* 任务介绍
  * 给定一个图片文本对 ----> 关系（**entailment或者答案**）+ 解释

![image-20220930221347803](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220930221347803.png)

* 动机

  * 缺少对Chunk级别的高层语义信息的理解
  * 没有决策的过程，只是简单的融合特征

* 方法（注意：使用了**Oscar**这个预训练模型）

  ![image-20220930221438196](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20220930221438196.png)

:hammer_and_wrench: **SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering**, in CVPR 2022. [[pdf](https://arxiv.org/abs/2204.02285)] [[torch](https://github.com/vipulgupta1011/swapmix/)]

* 动机

  * 作者从一个新的角度来研究 VQA 模型的鲁棒性：visual context。并表示，这些模型过度依赖visual context，即图像中不相关的物体，来进行预测。

  ![img](https://pic2.zhimg.com/80/v2-184656fad6e2bfbecb20726736d1283d_720w.webp)

* 发现

  * 虽然之前的工作从**语言上下文**的角度研究了VQA鲁棒性，但在这项工作中，我们从另一个角度研究了VQA模型的鲁棒性：**视觉上下文**。
  * 通过**扰乱不相关的上下文**，超过45%的预测的正确答案都会被改变。表明VQA模型高度依赖图像中的上下文，因此容易受到上下文干扰的影响。
  * 过度依赖上下文在**很大程度上取决于视觉表征**的质量
    * 个完美的视觉模型对上下文的依赖要小得多。我们通过使用**groundtruth对象和属性编码**替换视觉表征来实现这一点

* 贡献

  * 是第一个从视觉上下文的角度研究VQA稳健性的人，通过简单的**上下文扰动策略SwapMix**，对两个有代表性的VQA模型的健壮性进行了基准测试，并发现它们过度依赖视觉上下文。
  * 发现一个完美的视觉模型对视觉上下文的依赖要小得多，为模型提供**完美的视觉编码**，并观察模型稳健性的改善。
  * 定义了**两个指标**，**上下文依赖性和有效准确性**，并通过使用SwapMix作为数据增强技术显示了改进。

* 方法

  ![image-20221103155743460](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221103155743460.png)

  

**Weakly Supervised Relative Spatial Reasoning for Visual Question Answering**, in ICCV 2021. [[pdf](https://arxiv.org/abs/2109.01934)]

> [博客链接](https://www.cnblogs.com/lhiker/articles/15630482.html)

* 动机

  * 视觉推理的一个关键方面是**空间理解**，它涉及到理解对象的相对位置，即隐式地学习场景的几何形状。
  * 过去的预训练大模型在推理时候，都无法理解2D图像当中的**空间信息**。
  * 为此设计了两个目标作为空间推理（SR）的代理
    * 对象质心估计
    * 相对位置估计

  ![image-20221106145708208](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221106145708208.png)

* 贡献

  * 提出两个子任务，理解2D图像当中的几何信息
  * 展现了强大的`zero-shot`能力，只需要10%的训练数据进行训练
  * OOD（Out of Distribution）的泛化能力也很强

* 方法

  * 预处理工作

    * 抽取图像的深度（预训练好的`AdaBins`模型）

      * 质心表示$(x_c, y_c, z_c),$ $z_c代表深度$
      * 相对位置表示: 质心之间向量的减法，同时$dist(A，B)=−dist(B，A)$

    * 在上述两个任务中，预测都是**实值向量**。评估了这些任务的两个变体：

      * 一个**回归任务**，其中模型预测$\mathbb{R}^{3}_{[-1, 1]}$中的实值向量
      * **bin分类**，为此，我们将所有三个维度的实值范围划分为C个log-scale bins。第c个bin的bin宽由下式（使用超参数$\lambda=1.5$)给出：

      $$
      b_c=\frac{1}{\lambda^{C-\left|c-\frac{C}{2}\right|+1}}-\frac{1}{\lambda^{C-\left|c-\frac{C}{2}\right|+2}} \forall c \in\{0 . . C-1\}
      $$

      * 对数尺度的bins对更近的距离有更高的分辨率（更多的bins），对更远的距离有更低的分辨率（更少的bins）
      * 最简单的bin分类形式是一个具有bin间隔的三类分类任务$[−1,0)、[0]、(0,1]$

  * 弱监督代理任务 $SR$

    * 质心预测: $\mathcal{L}_{S R-r e g}=\mathcal{L}_{M S E}\left(f_{r e g}(v), y_{r e g}\right)$
      * 将视觉信息压缩为$36 \times 3$来和`gt`(预处理得到的) 进行计算
    * 相对位置评估: $\mathcal{L}_{S R \text {-bin }}=\mathcal{L}_{C E}\left(f_{\text {bin }}(V), y_{b i n}\right)$
      * 训练一个两层前馈网络$f_bin$来预测每个维度上每个对象的$36\times C \times D$个$bin$类，其中$C$是类的数量，$D$为3

  * Patches视觉信息

    * 能更好利用空间信息(这里说的是平面空间信息，有位置编码)

    ![image-20221106152759382](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221106152759382.png)

:fire: :hammer_and_wrench: **An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA**, in AAAI 2022. [[pdf](https://arxiv.org/abs/2109.05014)] [[torch](https://github.com/microsoft/PICa)] [[博客链接](https://zhuanlan.zhihu.com/p/433110834)]

![image-20221108163040036](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221108163040036.png)

* 探索`PLM`对知识的理解能力做VQA，不需要根据知识库进行检索
* `few-shot`的形式



:fire: **CLIP Models are Few-shot Learners: Empirical Studies on VQA and Visual Entailment**, in ACL 2022. [[pdf](https://arxiv.org/abs/2203.07190)]

> 实验性的文章

* 动机

  * 为了验证CLIP做VQA任务还有Visual Entailment任务的zero-shot性能
  * 提出VQA中few-shot下的微调策略

* 方法

  * Zero-shot VQA

  ![image-20221204155946020](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221204155946020.png)

  * Zero-shot Visual Entailment

  ![image-20221204160041611](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221204160041611.png)

  * **Few-shot VQA**
    * 分成`question type` $\times$ `answer type` 路（如果按照词汇表来分就太多了）, $65 \times 3 = 195$ ways
    * 微调细节详见论文（微调`CLIP`模型的一部分参数）

:fire: :hammer_and_wrench: **A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge**, in ECCV 2022. [[home page](https://allenai.org/project/a-okvqa/home)]

> OK-VQA的升级版本

* 动机

  * 过往的数据集顶多就是检索数据库里面的知识，没有做到对图片常识的推理（更加丰富的知识）
  * 我们实际上需要，识别图片，理解问题，找到知识，**推理**出对应的答案
  * 过去的数据集
    * FVQA：缺乏推理的过程，而且和图片不怎么相关
    * KVQA：通常是实体的知识，在维基百科上面的知识，而且主要是问任务方面的，没有常识的知识
    * OK- VQA：有bias，知识太过简单（e.g., What is the capital of this country?）, 而且**缺乏推理**
    * VCR：重点是电影场景中人的意图

* 数据集知识类型

  ![image-20221110165948489](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221110165948489.png)

  * **Commonsense** - learn from social behavior, 有点**推理**的意思 (e.g., many donuts being made in a cart implies they are for sale rather than for personal consumption).
  * **Visual** - Knowledge of concepts represented visually， 和**图片相关** (e.g., muted color pallets are associated with the 1950s).
  * **Knowledge bases** - **检索知识库的知识**, Knowledge obtained from textbooks, Wikipedia and other textual sources (e.g., hot dogs were invented in Austria).
  * **Physical** - **日常生活的知识，Fact**, Knowledge about the physics of the world (e.g., shaded areas have a lower temperature than other areas)

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



:fire: **[Knowledge-Based] KRISP: Integrating Implicit and Symbolic Knowledge for Open-Domain Knowledge-Based VQA**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2012.11014)] [[博客链接](https://zhuanlan.zhihu.com/p/392431083)]

* 动机

  * 验证利用**外部知识+隐式知识**结合的做`QA`的能力

  * **隐式知识**可以从基于大规模语料预训练的模型有效地学习。

  * 而**显示的知识**可以从知识库中的明确的、符号化的知识中学习。

  * 将两种模型进行集成，即可同时结合隐式知识与显式知识进行推理。

    ![image-20221112091608964](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221112091608964.png)

* 方法

  * 构建外部知识库，由于原来的知识太多了，根据规则对知识进行一定的筛选

    * DBPedia, ConceptNet , VisualGenome and hasPart KB ，进行节点过滤，只保留包含图像目标检测label的节点，最后的知识图包含8000节点和36000条边，采用RGCN作为卷积模型

      ![img](https://pic4.zhimg.com/80/v2-4299f57ff66a828d0cf5e5eb869e5f17_1440w.webp)

  * 模型设计（很简单）

  ![截屏2022-11-12 09.18.44](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-11-12%2009.18.44.png)



:hammer_and_wrench: :fire: **[多模态版本COT]** **Learn to Explain: Multimodal Reasoning via *Thought Chains for Science Question Answering**, in NIPS 2022. [[pdf](https://arxiv.org/abs/2209.09513)] [[博客](https://blog.csdn.net/EasyAIForum/article/details/127627611)] [[data&code](https://scienceqa.github.io.)]

> 手速实在太快了

* 动机

  * AI做问题回答的时候应该具有类似人的思考方式，当下的大语言模型更多是黑盒
    * 链式思维方式
    * few-shot的迁移能力

* 贡献

  * 提出一个多模态的包含多步解释的数据集（科学问题，21,208个例子）
  * COT的方式生成解释
  * 结合语言模型探索GPT+COT的上限

  ![image-20221122112213974](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112213974.png)

* 数据集特点

![image-20221122112301059](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112301059.png)

![image-20221122112322813](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112322813.png)

* 方法

  * 图像处理就简单的转成Caption，作为visual context

  * 使用COT改进UnifiedQA (微调)

    * 原来：->A
    * 改进后: -> ALE (Answer, Lecture, Explanation) 

  * 使用GPT-3+COT

    * 设计Prompt（few-shot的方式）

    ![image-20221122112544202](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112544202.png)

    

* 实验结果（具体结果可以看论文或者博客）

  * 解释生成的效果（是不是写错了？？？）

  ![image-20221122112645359](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112645359.png)

  * 模型上限探索+不同的ALE位置探索

  ![image-20221122112715645](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221122112715645.png)



:hammer_and_wrench: **A Unified End-to-End Retriever-Reader Framework for Knowledge-based VQA**, in MM 2022. [[pdf]()] [[torch](https://github.com/guoyang9/UnifER)]

> 这篇论文实验味道比较重，挺**严谨**的一个工作

* 动机
  * 过去的VQA模型主要是集中在利用**显示知识**还有**隐式知识**上面
    * 过去的显示知识往往存在noisy，因为没有对应的标签来约束
    * 隐式知识又大部分是利用了预训练的语言模型，而没有人探究**多模态预训练模型**是否也有隐式知识
  * 模型上，传统的都是检索知识，再做predict answer。是两个单独的模块
    * 本文将两个模块联系起来，用一个统一的模块解决

* 方法

![image-20230128223057226](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230128223057226.png)

![image-20230128223221039](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230128223221039.png)

* **推理**时候
  * 直接**检索出来最好的那个知识**，然后去回答即可

* 结论
  * 预训练模型当中的**隐式知识更加重要**
  * 具体见**论文的实验** 

:hammer_and_wrench: **A Multi-Modal Context Reasoning Approach for Conditional Inference on Joint Textual and Visual Clues**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.04530)] [[torch](https://github.com/YunxinLi/Multimodal-Context-Reasoning)]

* 动机

  * 过去这个多模态推理的任务都是用VLM的方式，很少考虑到multimodal context推理的能力（**缺少这种pre-context的机制来激活模型**）
  * PMR数据集中的例子

  ![image-20230510161851173](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230510161851173.png)

* 贡献
  * 多模态的in-context能力
  * 激发P LM做多模态推理的能力
* 方法

![image-20230510162012719](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230510162012719.png)

:fire: :hammer_and_wrench: **Combo of Thinking and Observing for Outside-Knowledge VQA**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.06407)] [[torch](https://github.com/PhoebusSi/Thinking-while-Observing)]

* 动机
  * 过去的知识性VQA要么忽视了textual knowledge in natural-language space
  * 要么完全放弃了视觉信息
  * 这里希望两者结合起来，约束多模态空间into语言空间当中

![image-20230516111041254](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516111041254.png)

* 过去工作的汇总

![image-20230516111123423](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516111123423.png)

* 方法：超级无敌大杂烩

![image-20230516111152375](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516111152375.png)



:hammer_and_wrench: **[I-T Retrieval] CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2303.13440)] [[torch](https://aneeshan95.github.io/Sketch_LVM/)]

* 动机
  * 第一次ZS的 Sketch-based 的图片检索
  * 方法上设计了一些独特的loss（很低资源的显卡都可以跑起来）

![image-20230724101958886](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230724101958886.png)

* 方法
  * 整体框架

![image-20230724102027823](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230724102027823.png)

* 领域迁移的loss

![image-20230724102052645](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230724102052645.png)



:hammer_and_wrench: **[T-I Retrieval] Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2302.03084)] [[torch](https://github.com/google-research/composed_image_retrieval)]

* 动机
  * Zero-shot的开放世界图文检索

![image-20230724103100073](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230724103100073.png)

* 方法

![image-20230724103119337](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230724103119337.png)



:hammer_and_wrench: **Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task**, in AAAI 2023. [[pdf](https://arxiv.org/abs/2208.12037)] [[Dataset&torch](https://github.com/showlab/CLVQA)]

* 动机
  * 和CVPR 2023的那一篇一样，都是VQA连续学习的
  * 做法上为了保证数据隐私，memory要用生成的内容
  * 提出benchmark主要分为场景图（6类场景） + 功能（6类功能）

![image-20230725103249913](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725103249913.png)

![image-20230725103403996](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725103403996.png)

* 方法（个人还是觉得CVPR 2023的那个做法好一些）

![image-20230725103420284](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725103420284.png)

![image-20230725103514264](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725103514264.png)



* 实验（不同的Task Order）

![image-20230725103637538](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725103637538.png)

:hammer_and_wrench: :fire: **PROMPTCAP: Prompt-Guided Task-Aware Image Captioning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2211.09699)] [[torch](https://yushi-hu.github.io/promptcap_demo/)]

> 这篇论文我关注很久了，估计是投CVPR没中，中了ICCV

* 动机
  * 如何用ChatGPT辅助知识性视觉问答

![image-20230829205901778](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829205901778.png)

* 训练方法

![image-20230829210040620](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829210040620.png)

* Prompt的设计

![image-20230829210305089](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829210305089.png)

* 模型推理过程

![image-20230829210335689](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829210335689.png)

* 实验结果

![image-20230829210409210](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829210409210.png)



:hammer_and_wrench: **SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images**, in AAAI 2023. [[pdf](https://arxiv.org/abs/2301.04883)] [[torch](https://github. com/nttmdlab-nlp/SlideVQA)]

* 动机
  * PPT VQA数据集：很好的解决了为什要进行多图像的QA或者QG的动机
  * 过去的文档VQA往往也只是关注一张图片
  * 目前模型的结果和人类的结果相比相差还非常大
* 数据集样例
  * 单跳，多跳，数据推到的问题

![image-20230912151704417](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230912151704417.png)

* 模型方法

![image-20230912151818638](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230912151818638.png)



:hammer_and_wrench: :fire: **Encyclopedic VQA: Visual questions about detailed properties of fine-grained categories**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2306.09224)] [[dataset](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa)]

* 动机：提出一个开放世界超级难的VQA数据集

![image-20231018210801188](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018210801188.png)

:hammer_and_wrench: :fire: **Improving Zero-shot Visual Question Answering via Large Language Models with Reasoning Question Prompts**, in MM 2023. [[pdf](https://dl.acm.org/doi/10.1145/3581783.3612389)] [[torch](https://github.com/ECNU-DASE-NLP/RQP)]

* 动机
  * LLM做VQA，提出应该把问题分开来

![image-20231114211254215](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114211254215.png)

* 方法

![image-20231114211314688](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114211314688.png)



### :sunny: Textual-QA

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

**[自然语言推理] Generated Knowledge Prompting for Commonsense Reasoning**, in ACL 2022. [[pdf](https://aclanthology.org/2022.acl-long.225.pdf)] [[torch](https://github.com/liujch1998/GKP)

> 这篇论文的思想和`few-shot` COT很像 

* 动机

  * 探索LM对知识的理解能力来回答逻辑推理的问题
  * **不再需要**外部知识库进行检索，或者根据特定任务的监督信号进行知识的融合

* 方法

  * 让LM在few-shot情况下来生成解释的`prompt`

    * 首先根据任务规定一下prompt模版，每个任务五个模板（few-shot）

    ![image-20221106225100256](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221106225100256.png)

    * 继续输入问题到那个占位符当中，LM自动生成知识，扔回去作为样本集合$K_q=\{k_1, k2, ..., k_M\}$

  * 知识融合

    * 问题和每个样本集合中的知识$k$进行融合, `concat`操作

    $$
    q_0=q, q_1=\left[k_1 \| q\right], \ldots, q_M=\left[k_M \| q\right]
    $$

  * 最后选择最合适的对，扔进去下一个LM进行推理，最后生成答案




:hammer_and_wrench: **GeoMLAMA: Geo-Diverse Commonsense Probing on Multilingual Pre-Trained Language Models**, in EMNLP Oral. [[pdf](https://arxiv.org/abs/2205.12247)] [[benchmark](https://github.com/WadeYin9712/GeoMLAMA)]

> [Probe介绍博客](https://zhuanlan.zhihu.com/p/362385123)
>
> * Probe探究了神经网络的**内部机制**如何对auxiliary linguistic tasks (or probe tasks, or ancillary tasks)进行**分类**
>
> 具体地，以BERT举例，对于一个在训练在主任务上的大型神经网络，Probe是一个插入在其中间层的浅层神经网络，通常是一个分类器层。Probe有助于探查不同层捕获的信息。**使用辅助任务对Probe进行训练和验证**，**以发现是否捕获了此类辅助信息**。原文作者给了一个图片示例：
>
> ![img](https://pic2.zhimg.com/80/v2-726af6b149c175f13a931cc48cdbeb75_720w.webp)

* 动机：**常识知识在不同的地理位置**中，会有所不一样

![image-20230203111838400](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230203111838400.png)

* 数据集标注方法

![image-20230203111924641](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230203111924641.png)





:hammer_and_wrench: :fire: **Automatic Chain of Thought Prompting in Large Language Models**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2210.03493)] [[torch](https://github.com/amazon-science/auto-cot)] [[rebuttal](https://openreview.net/forum?id=5NTt8GFjUHkr)]

* 动机
  * 过去`COT`主要是两种
    * 完全`Zero-shot COT`，就简单说一句“Let’s think step by step” 就让GPT-3直接生成
    * `Manual COT` 人为手动标定一些Q-C的`demonstration` 做 in-context learning
  * `Zero-shot COT` 性能不够好，`Manual COT` 需要手工标注，并且对demonstration的领域很敏感（以往都是根据特定领域进行标注）

![image-20230210172030892](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230210172030892.png)

* 作者希望这些demonstration直接让`Zero-shot COT`生成，但是实验过程中发现，demonstration对**多样性的要求很高**，所以考虑使用**问题聚类**的方式来先分类
  * **多样性高的demonstration**才会效果好，详见论文。做了详细的实验和说明
  * 因为如果多样性不高，demonstration生成的chain的答案就是有错的，**如果都是同一类的话，基本上就是demonstration答案全是错的**，更别说后面生成目标问题的答案了

![image-20230210172719091](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230210172719091.png)

* 方法模型
  * ⚠️注意人工筛选是指用一些规则进行筛选，详见论文

![image-20230210172754187](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230210172754187.png)





**Iteratively Prompt Pre-trained Language Models for Chain of Thought**, in EMNLP 2022. [[pdf](https://arxiv.org/abs/2203.08383)] [[torch](https://github.com/sunlab-osu/IterPrompt)]

* 动机

  * `CoT`的其中一种实现方式

  ![image-20230323172041557](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230323172041557.png)

* 方法

![image-20230323172136781](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230323172136781.png)



**Inductive Relation Prediction with Logical Reasoning Using Contrastive Representations**, in EMNLP 2022. [[pdf](https://aclanthology.org/2022.emnlp-main.286/)

* 动机

![image-20230327110331795](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230327110331795.png)



* 方法

![image-20230327110354474](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230327110354474.png)



:hammer_and_wrench: **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.04091)] [[code](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting[)]

* 过去的`COT`存在的问题
  * calculation errors
  * missing-step error
  * semantic misunderstanding errors

![image-20230510163540908](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230510163540908.png)

* 我们的方法和过去对比

![image-20230510163605207](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230510163605207.png)



:fire: **Think Twice: Measuring the Effificiency of Eliminating Prediction Shortcuts of Question Answering Models**, in ACL 2023. [[pdf](https://arxiv.org/pdf/2305.06841.pdf)]

* QA数据集中，LLM **bias和debias方法的研究**
* debias的一些方法
  * Resampling
  * **LearnedMixin**
    *  uses a biased model as a complement of the trained debiased model in a weighted composition.
  * **Confifidence Regularization**
    * aims to reduce the model’s confifidence, i.e. **the predicted score over samples marked as biased.**
    * 本文实验中：use BERT-BASE for both the teacher and bias model.



## :icecream: Vision-Language Text Generation

### :book: Paraphrase

:hammer_and_wrench: **[Sentence Discrimination] Learning Semantic Sentence Embeddings using Sequential Pair-wise Discriminator**,in COLING 2018. [[pdf](https://aclanthology.org/C18-1230/)] [[torch](https://github.com/badripatro/PQG)]

:hammer_and_wrench: **[Hierarchical Sketch&Paraphrase Generation] Hierarchical Sketch Induction for Paraphrase Generation**, in ACL 2022.[[pdf](https://aclanthology.org/2022.acl-long.178.pdf)] [[torch](https://github.com/tomhosking/hrq-vae)]

---



### :framed_picture: Image Caption

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

**NOC-REK: Novel Object Captioning with Retrieved Vocabulary from External Knowledge**, in CVPR 2022. [[pdf](https://arxiv.org/pdf/2203.14499.pdf)]

* 任务描述

  * 新物体描述(novel object captioning)，即让模型描述出训练**描述语料中没有出现过的物体**。
  * 如图，传统的caption训练如左下框所示，训练语料里没有ramp这个词，测试时自然也不会生成含有**ramp**的句子。

  ![img](https://raw.githubusercontent.com/Gary-code/pic/main/img/v2-f66020b9a4dad5f7a4ef1108c57468f5_720w.webp)

  * 在实际场景中，模型往往会见到在训练数据中没见过的新物体，此时传统的方法不能做到对新物体生成描述，不符合我们对模型应用的期待。

* 当前的方法

  * **采集更多的数据**，让训练语料包含尽可能多的物体并重新训练。然而数据的采集和标注是麻烦且昂贵的，并且每次加入新物体，模型需都需要**重新训练**，很不优雅。
  * 如先对图片做物体检测，获得ROI和object tags，再结合其训练caption模型。（当前**主流方法**）
  * 然而，所用物体检测模型的能力也是有限的，见到的**新物体极有可能超出检测模型的能力范围**。

* 动机

  * 人类认知物体有两种方式，一是靠**外观的匹配**，见过即认识；二是靠定义，哪怕没见过一个物体，**凭借物体的定义**，我们往往也能理解识别。
  * 本文设计了一个词语检索模块，将**==新物体定义作为外部知识==**引入caption模型，与caption模块一起端到端地训练。

* 方法

  ![image-20221005102539630](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221005102539630.png)

  * 匹配`loss`的计算

    * 为了鼓励模型引入新的类，我们将padding（和Region feature数量一致）的空类的15%随机替换成从外部知识库中随机挑选的词作为GT

    * 计算$Hungarian$ loss (匈牙利损失)
      $$
      \mathcal{L}_{\mathrm{H}}(\mathcal{Y}, \mathcal{V})=\sum_{i=1}^K-\log \operatorname{sim}\left(\mathbf{y}_i, \hat{\mathbf{v}}_{\hat{\sigma}(i)}\right)
      $$
      

**Image Captioning with Novel Topics Guidance and Retrieval-based Topics Re-weighting**, in TMM 2022. [[pdf](https://ieeexplore.ieee.org/document/9869686)]

* 利用主题(**Topic**)模型来做image caption

  * 探索图片对象和Topic之间的关系

* 什么是主题 [链接](https://zhuanlan.zhihu.com/p/41683009)

  * 三个list：【鸡胸肉，蛋白粉，饮食控制】、【跑步机，椭圆机，龙门架】、【减肥，体重下降，精神状态】。那么，我们也能大概知道每个词背后的主题是什么了。
  * 我们可以将Topic Model 看待为一个**Cluster问题**。而我们要做的就是将一些特征明显的词抓取出来。

* 方法

  ![截屏2022-10-08 19.14.43](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-08%2019.14.43.png)

  * Topic是通过`NMF`(NLTK中有)来从caption中提前获取的，数量为200。
  * RTR是推理时候才使用的，为了让**主题和图像信息更加相关**
  * ETP学习对象和主题之间的相似度
  * STP模块嵌入在每个LSTMCell上面，为了在对应的时间步选择合适的主题进行caption



**[主题 + Image Caption] Show, Rethink, And Tell: Image Caption Generation With Hierarchical Topic Cues**, in ICME 2021. [[pdf](https://ieeexplore.ieee.org/document/9428353)]

* 动机（caption本身就具备多个或者一个主题的特征）【目前觉得topic的作用就是生成更加多语义信息+生成更加精确】

  * 之前方法没有建立不同类型图片特征的联系，caption不够准确
  * decoder时候注意力机制对于每个单词只会操作一次

* 方法

  * 根据Faster-RCNN特征构建出主题->关键词级别的特征（每个检测区域都会检索出对应的主题）
  * decoder时候加入对主题的attention操作（可以生成一些不常用的描述，如little boy中的little）

  ![截屏2022-10-17 16.08.26](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-17%2016.08.26.png)

:hammer_and_wrench: **Explicit Image Caption Editing**, in ECCV 2022. [[pdf](https://arxiv.org/pdf/2207.09625.pdf)] [[torch](https://github.com/baaaad/ECE)]

* 任务：Image Caption编辑，修改得更加合理

  * 过去的方法都是隐式的修改，缺少可解释性，更像是re-writing的任务，句子的结构也容易被破坏

  ![image-20221012105504301](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221012105504301.png)

* 数据集构建

  * Ref-Cap和GT-Cap都需要是人工标记的（以前的方法Ref-Cap是模型生成的）
  * COCO数据集
    * 每张图的5个cap都作为GT，然后根据通过图像和cap计算和其他（不是这张图）caption相似度计算，选择最相关的
  * e-SNLI-VE（Flicker30K）数据集
    * contradiction和entailment分别作为Ref和GT即可

* 方法 (模型是基于**BERT**)

  * 引入编辑算子`<ADD>, <DELETE>, <KEEP>`

    ![image-20221012110107268](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221012110107268.png)

    ![](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221012110107268.png)

* Case Study

![image-20221012110327880](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221012110327880.png)

:hammer_and_wrench: **Object-Centric Unsupervised Image Captioning**, in ECCV 2022.  [[pdf](https://arxiv.org/abs/2112.00969)] [[torch](https://github.com/zihangm/obj-centric-unsup-caption)]

* 主要动机
  * 人工标注的image caption annotation太浪费时间了

* 贡献
  * **无监督**的Image Caption + **多语言**

* 如何无监督

  * 使用网络爬下来的语料库，还有数据集中固有的图像信息来做
  * 但`BLEU4`很低，只有6左右

  ![image-20221105204258603](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221105204258603.png)

:hammer_and_wrench: **Human-Centric Image Captioning**, in PR 2022. [[pdf](https://pdf.sciencedirectassets.com/272206/1-s2.0-S0031320322X00034/1-s2.0-S0031320322000267/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGoaCXVzLWVhc3QtMSJIMEYCIQDGI9Pe2Vg29Uf127g8bgoLH2in5NivB%2FYNbpJFB7t6pwIhAIJI6BJ3FO8IVZf7anhzx1JlrQnoA%2BvNPnTnFXynNyuLKswECEMQBRoMMDU5MDAzNTQ2ODY1IgyuR0sHCTJ7ZSggWKQqqQRnRT7Gv%2F7BVa9ChvOv%2Bv5L7LGF8GiKDTAVH%2FlK624uvf7X%2B8Rb0sn9tn7aGPgMwDLI8wG3Xu0uZKtLTtWkTE8zzNxzxDCTkNqLc7ePo3gd5kKdrjOCgXRyPF3K%2FKu%2FHiy5b%2FkKhvaXoSIv49WpU4Mw3jduHux0yg74CmV3I087VDVR6y0Z%2Fn1MGartEpeGdbwBoy%2F0Z59JHQF7rjquT4cG5isYwRK0nnr%2B79L%2BtWkL6uIaFyYk%2Bi%2B9GtknfOyGpuOQ4aF%2FVoFmdAzCsqwJAGjqS30nAaK7SiiiFtIoYpRPT5woODRU8SJ68UI4nHt0Es%2FEQJWTF59qx%2Fpl8SlN2sI%2FDIfN4aBDgwtF7ZPC3vgpwmcxBg67LjXS5Lwsha0Ui1PlDfo4EyrzzBHnE2wimBvn%2BU5L4xokqI246LjOALNUEUH8fPRm3ZnN9b2IWKqDFkJWwQ76T7FN2hnGPu%2F3JYf%2FX6mOJXaSVsumevPlqTn4oE1gQHZ%2FmP8uNc2feI36D4y17pFycY3HBZp5COJ3XlrSmGIBTFQFmXwt7JxQ14peTqI1GFR%2FTNZqTP9Uv6UkfQCdhYkK7dkemhKJpKtALjJz%2F0J1RIwaetYxcSMo%2F4kQDowiRJpQY7Y6OsRqgz7aqxCxPp9PNVIR64T9p%2Bj4VJdzsLs7B6%2BDWHoq0Sz99UQhVWupnVDXjH%2F8HLIaWQq5l6soAeVRmG8Z8t28pXM1pzgaoPzY%2Fdvl36r0MNqIv5oGOqgB8Mx5pBXQYDZWHBBnVJkdllGOlXSy8W7DgEygt929CiDsF%2F%2Fz4aCPqCmaN8WERin7vVZgIAwGD8dyHFgfCKJ0wY8%2BxnqhtHVeRq4FHknBpQJlBc50wlk9Jd4H4SIl%2FqASs5J7ssdt7UqhLhim%2FEosHKxTPOhVlHkd7HlgjOYebUvlAnxsOuqyy9LoMtdnpRwnCaCzqdOahqTFZ0bY1KoGgjLvfX306cS0&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221019T105552Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYV2BRT6MM%2F20221019%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=cdb598601cea787f991b427a00715b9977c06eb5ca33f744d087db73f1e3f9d3&hash=4a2e7e1603daf30647f492f3181b0a0e2260dc043d979436c01ef5d645e16465&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0031320322000267&tid=spdf-176b5542-edc3-49a9-b2fd-3d5bf73304c5&sid=19abb8f59650524d941be66636a7de94bc37gxrqb&type=client&ua=51570d5e09030508575a&rr=75c8fbdf599b8b41)] [[code & dataset](https://github.com/JohnDreamer/HCCM/)]

* 动机【动机不是很强，说得不够有道理】

  * 当前方法缺乏对人类行为动作的描述，只是简单的探索前景物体
  * 无视了这个关键物体的一些细节还有和其他物体之间的关系

* 方法

  * 标注**数据集**，已公开

    * 人工标注了人的身体部位的boxes

      ![截屏2022-10-20 16.08.38](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-20%2016.08.38.png)

    * 背景物体和活动物体的特征区分是通过算和人之间的similarity来得到的

:hammer_and_wrench: **Matching Visual Features to Hierarchical Semantic Topics for Image Paragraph Captioning**, in IJCV 2022. [[pdf](https://arxiv.org/abs/2105.04143)] [[torch](https://github.com/dandanguo1993/vtcm-based-image-paragraph-caption)]

> 引入Topic来做paragraph Caption的生成

* 动机

  * 降低生成Caption的难度
  * 以往的Topic模型做Caption存在冗余信息
  * 只有单层Topic不够丰富
  * Caption的多样性，多层Topic的Keyword是用户想要强调的东西

* 方法

  * 使用**`end-to-end`的方法**，即训练主题模型，又生成Caption，和以往的Two-stage不一样。
    * 以往都是单层的Topic模型
    * 两阶段方法，先用对象在预训练好的Topic模型里面检索出来Topic，缺乏对图像信息的考虑来丢弃Topic中的冗余信息。
  * 学习到了**层次化**的Topic表示
  * 可插拔的

  ![image-20221023213041474](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221023213041474.png)

* Case Study

![image-20221023213130865](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221023213130865.png)



:hammer_and_wrench: :fire: **ClipCap: CLIP Prefix for Image Captioning**, on arXiv (2021). [[pdf](https://arxiv.org/abs/2111.09734)] [[torch](https://github.com/rmokady/CLIP_prefix_caption)]

> 模型非常简单，跑起来也很简单。一台1080 Ti 就可以训练了！

* 动机

  * 过去decode句子的时候都是以前面生成的单词作为condition，同样的，我们可以使用图片信息来作为condition
  * 对于images & captions $\{x^i, c^i\}^N_{i=1}$

  $$
  \max _\theta \sum_{i=1}^N \sum_{j=1}^{\ell} \log p_\theta\left(c_j^i \mid x^i, c_1^i, \ldots, c_{j-1}^i\right)
  $$

* 模型架构

![image-20221225095940586](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221225095940586.png)



**Controllable Image Captioning via Prompting**, in AAAI 2023. [[pdf](https://arxiv.org/abs/2212.01803)]

* 动机

  * 通过学习**不同的动态prompt**，生成不同风格的图像字幕

  ![image-20230305214259376](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230305214259376.png)

* 方法极其简单

  * 预训练+微调结构
  * **预训练都是BLIP的那一套**

![image-20230305214515377](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230305214515377.png)



:hammer_and_wrench: **Learning Distinct and Representative Modes for Image Captioning**, in NIPS 2022. [[pdf](https://arxiv.org/abs/2209.08231)] [[torch](https://github.com/bladewaltz1/ModeCap)]

* 动机
  * Caption的可控性 + 多样性

![image-20230319202536999](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230319202536999.png)

* 方法

![image-20230319202651561](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230319202651561.png)



**Incorporating Unlikely Negative Cues for Distinctive Image Captioning**, in IJCAI 2023. [[pdf](https://www.ijcai.org/proceedings/2023/0083.pdf)]

* 动机
  * 不要生成那么generic（通用）的Caption
  * 使用unlikely Training的方法（就是避免让模型学习到负样本的行为）

![image-20230829203433135](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829203433135.png)

* 模型方法

![image-20230829203508036](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829203508036.png)



:hammer_and_wrench: **Transferable Decoding with Visual Entities for Zero-Shot Image Captioning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.16525)]] [[torch](https://github.com/FeiElysia/ViECap)]

* 动机
  * 减少没见过物体的**对象幻觉**现象和bias问题
    * OOD问题
  * 要让模型在in-domain和cross-domain都表现良好的性能
  * **只用纯文本来进行训练**

![image-20230829204925597](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829204925597.png)



* 模型方法（纯文本进行训练 + zero-shot inference）
  * 加一个CLIP entity clsssifier来识别实体

![image-20230829205102812](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829205102812.png)

* 主要对比的是之前两篇同样是纯文本训练的Caption论文（EMNLP Findings 2022 和 ICLR 2022）

![image-20230829205237122](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829205237122.png)



:hammer_and_wrench: **With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2308.12383)] [[torch](https://github.com/aimagelab/PMA-Net)]

* 动机
  * 过去的传统attention不会学习到别的样本当中的语义信息
  * 提出原型网络来做Image Caption

![image-20230829215053304](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829215053304.png)

* 模型方法

![image-20230829215139441](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230829215139441.png)



:hammer_and_wrench: **[多模态机器翻译] CLIPTrans: Transferring Visual Knowledge with Pre-trained Models for Multimodal Machine Translation**, in ICCV 2023. [[pdf](https://vcg.seas.harvard.edu/publications/cliptrans-transferring-visual-knowledge-with-pre-trained-models-for-multimodal-machine-translation/paper)] [[torch](https://github.com/devaansh100/CLIPTrans)]

* 动机
  * 多模态机器翻译推理的时候不需要使用到图片
  * 过去的方法在低资源的语言上表现不好（本身标注就少）
  * 究其原因可能是文本和图像的embedding没有很充分的对齐

![image-20230905111542477](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230905111542477.png)

* 模型方法
  * 两阶段训练迁移学习（**caption -> 翻译**）

![image-20230905111625639](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230905111625639.png)

* 修改了Decoder

![image-20230905111848743](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230905111848743.png)

:hammer_and_wrench: **MultiCapCLIP: Auto-Encoding Prompts for Zero-Shot Multilingual Visual Captioning**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.664/)] [[torch](https://github.com/yangbang18/MultiCapCLIP)]

* 动机
  * 多语言的Image Caption
  * 多语言的Caption标注困难
* 方法
  * 在训练的时候只是使用文本数据，推理才使用图像推理

![image-20230907213754699](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230907213754699.png)

* 例子展示

![image-20230907215730610](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230907215730610.png)



:hammer_and_wrench: **[视觉信息增强LM文本生成能力] Learning to Imagine: Visually-Augmented Natural Language Generation**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.526/)] [[torch](https://github.com/RUCAIBox/LIVE)]

> ⚠️**事先声明：这篇论文实际上是做纯文本的，输入的是文本，输出的也是文本，只不过是想联想生成一些图片来增强模型文本生成的能力**

* 动机
  * 人类在写作的时候会联想一些画面来辅助写作，比如说滑雪，就会想到一些对应的场景
  * 利用图片增强LM文本生成的能力
* 模型（**Stable Diffusion来生成图片**）

![image-20230918095120803](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918095120803.png)



:hammer_and_wrench: **[多模态摘要生成] CFSum: A Coarse-to-Fine Contribution Network for Multimodal Summarization**, in ACL 2023. [[pdf]()] [[torch](https://github.com/xiaomin418/CFSum)]

* 动机
  * 过去的模型没有很好或者有选择性的利用图像信息
  * 无法关注到文本中的实体和图像之间的对应关系

![image-20230918113933715](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918113933715.png)

* 模型方法

![image-20230918114129452](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918114129452.png)



:fire: :hammer_and_wrench: **OxfordTVG-HIC: Can Machine Make Humorous Captions from Images?**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.11636)] [[project](https://torrvision.com/tvghic/)]

* 动机
  * 创建了一个幽默的Caption数据集
  * 发现是符合幽默的良性冲突理论的 [[知乎](https://zhuanlan.zhihu.com/p/400646267)]
  * 而且本文不是用传统的BLEU等指标进行测评的！（具体见论文）

![image-20231013113213641](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013113213641.png)

![image-20231013113317492](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013113317492.png)

* 方法：增加选择的多样性，因为本来数据就是很多样的 

![image-20231013113230477](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013113230477.png)

:hammer_and_wrench: **ALIP: Adaptive Language-Image Pre-training with Synthetic Caption**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2308.08428)]] [[torch](https://github.com/deepglint/ALIP)]

* 动机
  * 过去的Caption在海量数据生成时候，会存在I-T noise的情况

![image-20231021160149360](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231021160149360.png)

* 方法：用OFA生成合理的Caption，控制weight设计loss来训练（具体见论文）

![image-20231021160251481](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231021160251481.png)



:hammer_and_wrench: **Noise-aware Learning from Web-crawled Image-Text Data for Image Captioning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2212.13563)] [[torch](https://github.com/kakaobrain/noc)]

* 动机
  * 过去网上的数据存在noise的问题，图文不匹配
  * 过去的方法需要filter再训练（类似BLIP），但是会让模型学不到丰富的知识

![image-20231021160448062](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231021160448062.png)

* 方法

![image-20231021160523997](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231021160523997.png)



:fire: :hammer_and_wrench: **Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation**, in NeurIPS 2023. [[pdf]](https://arxiv.org/abs/2306.13460) [[torch](https://github.com/yuezih/SMILE)]

* 动机
  * 图像描述任务（为给定图像生成自然语言描述）长期以来受到「输出过于平凡」的问题的困扰，即模型倾向于为不同图像生成缺乏细节的相似描述。
  * 传统的最大似然估计（MLE）所提供的严格监督并不完全适合图像描述模型的优化。
    * **直观地将其优化过程解耦为使模型生成更丰富的“丰富性优化”和使模型生成更简洁的“简洁性优化”，并假设这两种优化的相互抗衡最终使得模型生成平凡描述**
  * 提出了一个新的学习目标——**半渗透最大似然估计（SMILE）**，它允许“丰富性优化”而阻止“简洁性优化”，从而鼓励模型生成更丰富、包含更多细节的描述。

![img](https://pic4.zhimg.com/80/v2-d5760f29fa07ee9ffd9b76d04ab4d2a7_1440w.webp)

![img](https://pic1.zhimg.com/80/v2-715054ccd2eeb037b04f4937ea818b6c_1440w.webp)

* 方法（在BLIP上加伤SMILE在Caption数据集上面做微调）

![image-20231109162455341](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231109162455341.png)

:hammer_and_wrench: **Exploring Diverse In-Context Configurations for Image Captioning**, in NeurIPS 2023. [[pdf](https://arxiv.org/abs/2305.14800)] [[torch](https://github.com/yongliang-wu/ExploreCfg)]

> 探索In-context Learning如何做Image Caption，有点奇怪的论文🤔

* 动机
  * 发现few-shot的样本，往往和caption的质量相关

![image-20231112202436751](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231112202436751.png)

* 探索了四种图像选择的策略

![image-20231112202526320](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231112202526320.png)

:hammer_and_wrench: **Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models**, in NeurIPS 2023.  [[pdf](https://arxiv.org/abs/2305.19595)]

> 看起来像是做Caption过滤的

* 动机
  * 过去VLM生成Caption会有一些bias的问题
  * 主要归结于：数据集中Caption的质量 + Caption的密度（是否描述完整）。互联网数据的锅

![image-20231114114703984](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114114703984.png)

* 方法

![image-20231114114725730](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114114725730.png)

:fire: :hammer_and_wrench: **Caption Anything: Interactive Image Description with Diverse Multimodal Controls**, 2023-05. [[pdf](https://arxiv.org/abs/2305.02677)] [[torch](https://github.com/ttengwang/Caption-Anything)]

* 动机（分割+可控信号的caption）

![image-20231116101406270](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116101406270.png)

* 方法（Training Free）

![image-20231116101429416](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116101429416.png)

![image-20231116101440390](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116101440390.png)

:fire: :hammer_and_wrench: **Attractive Storyteller: Stylized Visual Storytelling with Unpaired Text**, in ACL 2023. [[pdf]](https://aclanthology.org/2023.acl-long.619/)] [[torch](https://github.com/DingyiYang/StyleVSG)]

* 动机：生成有style的故事（很少有标注）

![image-20231120213011448](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231120213011448.png)

* 方法

![image-20231120213035079](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231120213035079.png)

## :sunglasses: Video Understanding

### :video_camera: Features Learning

:white_check_mark: :fire: :hammer_and_wrench: **[TSN] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition**, in ECCV 2016.  [[pdf](https://arxiv.org/abs/1608.00859)]  [[torch](https://github.com/yjxiong/temporal-segment-networks)]

* 抽取所有帧是不现实的，TSN将其**等间隔分**为$K$个片段（i.e., $K=16$),在每个片段中谁寄抽取一帧作为输入

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



:hammer_and_wrench: **[PEFT&动作识别] AIM: Adapting Image Models for Efficient Video Action Recognition**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2302.03024)] [[torch](https://adapt-image-models.github.io/)]

> **大模型时代**下很好的尝试`Parameter Effectient Fine-tuning`

* 动机
  * 到底可不可以直接**冻住一个很好的Image Model**，只是**加一些很少量的可学习的参数做视频理解**的任务呢？
  
  ![image-20230401174941604](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230401174941604.png)
  
* 方法（很巧妙很简单）

![image-20230401175010208](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230401175010208.png)



:hammer_and_wrench: **Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground**, in ICCV 2023 Oral. [[pdf](https://arxiv.org/abs/2211.12883)] [[torch](https://github.com/lihaoxin05/StillMix)]

* 动机 & 贡献
  *  前景还有后景都有可能造成动作识别的bias
    * 前景：衣服颜色
    * 后景：背景，排球场
  * 提出benchmark来评估bias

![image-20230902181442456](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902181442456.png)

* debias的方法（做数据增强）- 从bank中采样一帧来插值
  * bank里面都是要求某个action的概率要高，但是本身不包含动作信息的图像

![image-20230902181709607](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902181709607.png)



---



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

* 



:fire: **Event-Guided Procedure Planning from Instructional Videos with Text Supervision**, in ICCV 2023. [[pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Event-Guided_Procedure_Planning_from_Instructional_Videos_with_Text_Supervision_ICCV_2023_paper.pdf)]

* 动机
  * 过去的方法没有很好的考虑到事件信息
  * 实际上visual states和action text label之间是有gap的
    * 在某些label中出现的物品，在一开始和结束时候的state是没有出现的

![image-20231120205912793](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231120205912793.png)

![image-20231120205928481](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231120205928481.png)

* 方法（增加多一个event-aware的loss）

![image-20231120205959318](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231120205959318.png)

---



### :man_student: Video Question Answer

:fire: :hammer_and_wrench: **Invariant Grounding for Video Question Answering**, in CVPR 2022 Best Paper Finalist.  [[pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Invariant_Grounding_for_Video_Question_Answering_CVPR_2022_paper.pdf)] [[torch](https://github.com/yl3800/IGV)]

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



:hammer_and_wrench: :fire: **Discovering Spatio-Temporal Rationales for Video Question Answering**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.12058)] [[torch](https://github.com/yl3800/TranSTR)]

* 动机
  * 解决长视频和复杂视频的QA
  * 问题还有视频帧存在冗余信息

![image-20230902182028333](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902182028333.png)

* 模型方法（**自适应采样帧**）
  * 核心就是找到关键信息，不要冗余的信息

![image-20230902182128865](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902182128865.png)

:fire: **Redundancy-aware Transformer for Video Question Answering**, in MM 2023. [[pdf](https://arxiv.org/abs/2308.03267)]

> 和上一篇论文动机很类似的，都是同一个作者，CVPR 22最佳论文候选那个作者

* 模型方法

![image-20230902182900805](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902182900805.png)

:fire: :hammer_and_wrench: **Tem-adapter: Adapting Image-Text Pretraining for Video Question Answer**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2308.08414)] [[torch](https://github.com/XLiu443/Tem-adapter)]

* 动机
  * 讲CLIP的图文预训练方法，适配到VideoQA领域
  * 直接适配存在两个天然的gap

![image-20230902182553786](/Users/gary/Library/Application Support/typora-user-images/image-20230902182553786.png)

* 过去的方法

![image-20230902182618433](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902182618433.png)

* 我们的方法

![image-20230902182647169](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902182647169.png)



:fire: :hammer_and_wrench: **A-CQUIRED: A Dataset for Answering Counterfactual Questions In Real-Life Videos**, in EMNLP 2023 [[pdf](https://arxiv.org/abs/2311.01620)] [[dataset&code](https: //github.com/PlusLabNLP/acquired)]

* 动机
  * 提出一个更加多样化的VideoQA数据集，更多反事实问题

![image-20231114205907629](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114205907629.png)

* 数据集介绍

![image-20231114205924567](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114205924567.png)

---



### :writing_hand: Video Caption

**[Video Caption] VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2101.12059)]

:hammer_and_wrench: :fire: **[Video Caption] Robust Change Captioning**, in ICCV 2019. [[pdf](https://arxiv.org/pdf/1901.02527.pdf)] [[torch](https://github.com/Seth-Park/RobustChangeCaptioning)]

* 输入为前后图像对，五种变化类型（color/material change,adding/dropping/moving an object）
* 提出一个有视点变化的数据集[CLEVR-Change](https://cs.stanford.edu/people/jcjohns/clevr/)（80K图片对），并在无视点变化的数据集[Spot-the-Diff](https://github.com/harsh19/spot-the-diff)取得SOTA效果。
* 模型：Dual 注意力， 分辨**视点变化**， 其实是通过输入两张差不多的图片，提前标定好数据集获得的，有点被坑的意思![image-20220522213419579](https://s2.loli.net/2022/05/22/fiUArgZIjlzw4p1.png)

:hammer_and_wrench: :fire: **[Video Caption] Semantic Grouping Network for Video Captioning**, in AAAI 2021. [[pdf](https://arxiv.org/pdf/2102.00831.pdf)] [[torch](https://github.com/hobincar/SGN)]

![image-20220621204108736](https://s2.loli.net/2022/06/21/DMmzxs7dKwyU6BE.png)

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



## :eye: Grounding

:fire: **Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model**, in CVPR 2022. [[pdf](https://arxiv.org/abs/2203.14940)] [[torch](https://github.com/dyabel/detpro)]

> 将CoOP（图像分类）论文思想放入OD当中

* 动机
  * 人工设计Prompt较为麻烦
  * 前景和背景分离在OD中很重要
  * 前景的上下文分级也很重要（通过IOU来判定）
* 模型方法（具体见论文）
  * IOU来分理处前景背景
  * 损失函数设计
    * **背景的损失和哪个class都不相似**
  * 通过IOU来对前景分级处理

* 连续型Prompt学习：

![image-20230307224847531](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230307224847531.png)

* 嵌入ViLD框架当中

![image-20230307224949152](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230307224949152.png)



:fire: :hammer_and_wrench: **Multi-Modal Classifiers for Open-Vocabulary Object Detection**, in ICML 2023. [[pdf](https://arxiv.org/abs/2306.05493)] [[code](https://www.robots.ox.ac.uk/~vgg/research/mm-ovod/)]

* 动机
  * 视觉和文本段相互辅助**开放词表的OD**
  * 类似一个Empirical Study

![image-20230616164215541](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230616164215541.png)

* **视觉和文本**端的处理

![image-20230616164353443](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230616164353443.png)

##  :apple: Causality Learning

:fire: :star: **Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond**, in TACL 2022. [[pdf](https://arxiv.org/abs/2109.00725)] [[blog](https://github.com/badbadcode/Causality-NLP-Reading-List/blob/master/notes/Feder%20et%20al_2021_Causal%20Inference%20in%20Natural%20Language%20Processing%20-%20Estimation%2C%20Prediction%2C%20Beyond.md)]

* **关于NLP + Causality 的综述，写得非常详细非常好！！！**

:hammer_and_wrench: **CauAIN: Causal Aware Interaction Network for Emotion Recognition in Conversations**, in IJCAI 2022. [[pdf](https://www.ijcai.org/proceedings/2022/0628.pdf)] [[torch](https://github.com/circle-hit/CauAIN)]

> 在对话当中找**情感相关的因果线索**

* 动机：

  * 现有情感分析研究中，对情感的识别往往仅依据当前句子，而忽略了**对话历史中存在的能够帮助识别当前情感的深层线索**
    * 对话某一方自身的话语中，存在可退里情感的因果联系 **(Intra-cause)**
    * 对话双方的话语中，存在可以帮助推理对方情感的交互因果联系 **(Inter-cause)**
  * 目前还没有相关的标注有情感线索的数据集，因此论文提出了一种利用常识知识自主**寻找线索**的方法来确定线索所在句子

  ![img](https://img-blog.csdnimg.cn/ad9e9bbada84442795c3953952768e74.png)

* 方法

  * 利用 ATOMIC 常识知识语料库，获得对话历史每句的 6 种因果线索

    * 3 种来自自身因果线索 (Intra-cause)，xEffect，xReact，xWant
    * 3 种来自交互因果线索 (Inter-cause)，oEffect，oReact，oWant

    ![image-20230104121326042](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230104121326042.png)

  * 利用因果线索，建模对话历史

    * RoBERTa + GRU 建模对话历史文本
    * 利用因果线索，获取对话历史每句与当前句情感的相关程度分数
    * 利用相关程度分数加权后的向量进行分类获取情感

    ![image-20230104121406446](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230104121406446.png)

:hammer_and_wrench: **Knowledge-Bridged Causal Interaction Network for Causal Emotion Entailment**, in AAAI 2023. [[pdf](https://arxiv.org/abs/2212.02995)] [[torch](https://github.com/circle-hit/KBCIN)]

> 这份工作和上面👆的[IJCAI 2022:CauAIN: Causal Aware Interaction Network for Emotion Recognition in Conversations](https://www.ijcai.org/proceedings/2022/0628.pdf)那个工作**idea是一样**的，只不过做了两个不同的任务而已

* 主要动机都是找寻对话当中的**因果线索**

  * 引入**常识知识**来弥补gap

  ![image-20230130121710045](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230130121710045.png)

* 方法模型

  * 数据来源 （AOTMIC-2020）

  ![image-20230130121639558](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230130121639558.png)

  * 模型

  ![image-20230130121841363](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230130121841363.png)

:fire: :hammer_and_wrench: **Everything Has a Cause: Leveraging Causal Inference in Legal Text Analysis**, in NAACL 2021. [[pdf](https://aclanthology.org/2021.naacl-main.155/)]  [[torch](https://github.com/xxxiaol/GCI/)]

* 动机
  * 将因果推理应用到非结构化的文本数据当中，帮助法律从业者决策
* 方法
  * 概览：从事实描述文本中自动构建因果图，用因果推理来辅助法律决策的制定，本文中similar charge disambiguation任务上测试了该框架的效果。
    1. 用关键词抽取（用YAKE+IDF计算单词对罪名的重要性），来识别出事实描述中的key factors。
    2. 将相似的key factors聚类到组中，每个组视为一个独立节点。（图的节点的每个组和罪名）
    3. 用对未识别变量鲁棒（无监督抽取可能导致关键词不完全，因果发现时有未识别confounder）的causal discovery algorithm（Greedy Fast Causal Inference (GFCI)）来构造因果图。（输出是Partial Ancestral Graph (PAG)）（在附录中可以看到，这种算法能够识别出隐factor）（限制：1. 禁止罪名节点出边。2. 以案例（事件描述文本）的时间顺序来限制因果关系）（抽样因果图）
    4. 估算每条边的causal strength来减少不可靠边的影响。（保持Confounder不变）（方法：Average Treatment Effect (ATE)）（估算ATE的方法：Propensity Score Matching (PSM) 在treated/untreated group之间构建相似样本对）
    5. 将**因果知识结合到NN**中：①在NN attention weights上加入causal strength限制（加损失函数）。②在因果图上抽取出的因果链上使用RNN。

![image-20230308160457133](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230308160457133.png)

* 两种方式将因果推理和神经网络相结合

![image-20230308160530921](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230308160530921.png)



:hammer_and_wrench: :fire: **Deconfounded Video Moment Retrieval with Causal Intervention**, in SIGIR 2021. [[pdf](https://arxiv.org/abs/2106.01534)] [[torch](https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval)]

* 动机

  * 目前视频定位当中存在一些Bias
    * 数据集**长尾分布**
      * 某些Query动作出现频率太高了
    * Query和locations之间存在**太强的依赖**
      * open往往都是视频开头，close一般往往都是对应视频结束

* 方法

  ![image-20230110165317100](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110165317100.png)

  * 模型图

  ![image-20230110165441609](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110165441609.png)

:fire: :hammer_and_wrench: **Interventional Video Grounding with Dual Contrastive Learning**, in CVPR 2021. [[pdf](https://arxiv.org/pdf/2106.11013.pdf)] [[torch](https://github.com/nanguoshun/IVG)] （2023年1月仍为开源）

* 动机

  * 和上面SIGIR那一篇论文有点类似，但是通过loss来进行因果推断
  * 过去模型往往只理解实体，而没有关注到一些动作

  ![image-20230110165932709](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110165932709.png)

* 方法

  * 通过对比学习来进行更好交互

  ![image-20230110170055802](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110170055802.png)

  * 因果推理模块（通过loss来调整）

    * 因果图

    ![image-20230110165959351](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110165959351.png)

    * 通过抽取三元组近似$z$，具体见论文
    * loss计算

    ![image-20230110170418014](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230110170418014.png)





:fire: :hammer_and_wrench: **Two Causal Principles for Improving Visual Dialog**, in CVPR 2020. [[pdf](https://arxiv.org/abs/1911.10496)] [[torch](https://github.com/simpleshinobu/visdial-principles)] [[zhihu](https://zhuanlan.zhihu.com/p/363411361)]

> 含老师团队**第一篇因果推理**方向的文章
>
> **一句话总结**：
>
> * 从因果图角度审视视觉对话任务，切断对话历史与答案的直接因果效应，添加混杂因子[用户偏好]构建描述真实世界的因果图。

* 动机

  * **VisDial本质上并非带有History的VQA**。作者以因果推断的角度入手，引入两条因果原则，将所有的VisDial基线模型提升到了SOTA。

* 两个原则

  * $P_1$: `H`对话历史记录，不应该直接影响`A`

    ![image-20221124113615064](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124113615064.png)

  * $P_2$: **用户偏好**会导致回答的不一样

    ![image-20221124113806608](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124113806608.png)

    * 由于用户偏好不可观测，因此作者通过三种方式来**近似采样**。

      ![image-20221124114039418](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124114039418.png)

  * 构建两个原则的**因果图**：

    ![image-20221124113458538](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124113458538.png)

* 具体后门调整及其概率运算公式，**详见论文及其附录**

![image-20221124114137317](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124114137317.png)

![image-20221124114218219](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221124114218219.png)

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



:hammer_and_wrench: **[指代表达] Deconfounded Visual Grounding**, in AAAI 2022. [[pdf](https://arxiv.org/abs/2112.15324)] [[torch](https://github.com/JianqiangH/Deconfounded_VG)] (2023.1.17未开源)

* 动机

  * visual grounding任务过去存在bias，比如说，羊往往都是出现在图片中间

  ![](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230117235029986.png)

  * 方法

    * 构建因果图

      ![image-20230117235128346](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230117235128346.png)

      * 通过后门路径可以观察，主要bias来源与`R`语言信息，因为其实视觉上的bias其他一些方向论文都有所解决了

    * 由于$G$不可观测，因此需要**近似采样**

      * 通过**生成式模型采样**，这里有很丰富的理论依据具体见论文。$\hat{G}=Enc(R), R'=Dec(\hat{G})$
      * 通过因果干预得到unbias的$R'$即可
      * 模型图

      ![image-20230117235726811](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230117235726811.png)

:fire: :hammer_and_wrench:  **Unbiased Scene Graph Generation from Biased Training**, in CVPR 2020. [[pdf](https://arxiv.org/abs/2002.11949)] [[torch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)] [[zhihu](https://zhuanlan.zhihu.com/p/109657521)]

* 动机

  * 期望从**有偏见的训练中（Biased Training）利用无偏预测获得无偏见的场景图（Unbiased Scene Graph）**
  * 现有方法问题
    * human **walk on/ sit on/ lay on** beach等包含丰富信息的谓语简单“概括”为human **on** beach或将**behind/ in front of**“概括”为**near**
    * 无法很好应用到下游任务上面

* 贡献

  * 设计了一个无偏预测的推理算法**Causal TDE Inference**（非训练方法，模型“不可见”，适用于任何SGG模型）
  * 设计了一个新的通用SGG框架**[Scene-Graph-Benchmark.pytorch](https://link.zhihu.com/?target=https%3A//github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)**，其使用[maskrcnn-benchmark](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/maskrcnn-benchmark)进行底层目标检测，集成了目前最全的SG metrics（包括Recall、Mean Recall、No Graph Constraint Recall、Zero Shot Recall等）。该框架提供重写的各种SGG baseline模型（如MOTIFS、VTranE、VCTree），有着当之无愧的State-of-The-Art SGCls和SGGen结果。

  ![img](https://pic4.zhimg.com/80/v2-f0657ac8afb0b8fbcbe0fcf65e4944f3_720w.webp)

* 问题定义（**有偏见的数据标注的原因**）

  * **the long-tail theory：person** carry bag确实比**dog** carry bag的数量多
  * **bounded rationality（有限理性）：**在人类标注关系时，更倾向于标注简单的关系，即标注person **beside** table而不是person **eating on** table（**关系不够准确层面**）
  * **language or reporting bias：**我们更喜欢说person **on** bike，而不是person **ride on** bike（**只是简单描述了真实事件的一部分**）

* 无偏的思想（**content：内因，context：外因**）

  * **人类**在有偏见的大自然中生长，在**拥抱好的context的同时，避免不好的context**，并与content一起做出无偏见的决定。

  * 其潜在的机制是**基于因果关系的（causality-based）：**决策是通过追求由**content引起的主要因果效应**，而不是追求由**context引起的副作用**来做出的。然而，**机器**是基于可能性的（likelihood-based），会产生有偏结果。

  * 故论文认为，无偏预测的关键是教会机器如何区分主要作用（main effect）和副作用

    * **content**：object和subject的**visual features**
    * **context**：object-subject union regions的**visual features**以及object、subject的**类别标签**

  * 为了在无偏预测中追求主要作用，论文提出赋予机器**反事实思维（counterfactual thinking）:**
    ***If i had not seen** the content, would I still make the same prediction?*

  * **反事实思维**：**事实与反事实之间的比较，将会自然地消除context偏差的影响，因为context是两者之间唯一不变的东西**。

  * 如图，左侧图片是所谓的**事实场景**，也可以说是**原始场景**；右侧图片是**反事实场景**，就是**将原始场景中content（狗和冲浪板的视觉特征）去除，其他部分（如scene和object classes）保持不变，就像object的视觉特征从未出现过。**通过这两者的比较，我们可以专注于关系的主要视觉影响，同时也不丢失context。

     ![img](https://pic2.zhimg.com/80/v2-d811612b212103ed4852016136302139_720w.webp)

* 方法

  * **有偏&无偏**训练架构

  ![image-20221009153731849](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221009153731849.png)

  * 无偏见+反事实的因果图构建

  ![image-20221009153828669](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221009153828669.png)

TDE（Total Direct Effect）方法**没有引入任何额外的参数**，也可以说没有针对模型的有偏训练进行任何改动，其使用原始SGG模型进行了两次预测，将两次预测的结果进行**差值运算**，最终得到无偏见的预测。所以**TDE方法是模型“不可见”的，广泛适用于各种SGG模型**。



:fire: :hammer_and_wrench: **Long-Tailed Classification by Keeping the Good and Removing the Bad Momentum Causal Effect**, in NIPS 2020. [[pdf](https://arxiv.org/abs/2009.12991)] [[torch](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch)] [[zhihu](https://zhuanlan.zhihu.com/p/259569655)]

* 动机

  * 提出了一种崭新的**长尾问题**的通用解决思路。而且**实现非常简单**，能够广泛适用于各种**不同类型的任务**
  * 过去解决长尾分布的方法**存在的问题**
    * 【**未卜先知**】虽然利用数据集分布的**re-sampling**和**re-weighting**训练方法可以一定程度上缓解长尾分布的问题。然而这种利用其实是违背现实学习场景的，他们都需要在训练/学习之前，了解“未来”将要看到的数据分布，这显然不符合人类的学习模式，也因此无法适用于各种动态的数据流。
    * 【**两阶段**，不是end-to-end】目前长尾分类最优的[Decoupling](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.09217)算法依赖于**2-stage**的分步训练，这显然不符合深度学习end-to-end的训练传统，而论文本身也没有提出让人信服的理由解释**为什么特征提取backbone需要在长尾分布下学**，而偏偏classifier又需要re-balancing的学。
    * 【**迁移能力**不足】长尾分布下简单的图片分类问题和其他复杂问题（诸如物体检测和实例分割）研究的割裂，目前长尾分布下图片分类问题的算法日趋复杂，**导致很难运用于本来框架就很繁琐的检测分割等任务**。而我觉得长尾问题的本质都是相似的，真正的解决方案一定是简洁的，可以通用的。
  * **基于上面这些问题，也就最终诞生了我们的这篇工作。我们提出的[De-confound-TDE](https://link.zhihu.com/?target=https%3A//kaihuatang.github.io/Files/long-tail.pdf)的优势如下：**
    1. 我们的训练过程**完全不依赖于提前获取的数据分布**，只需要在传统训练框架的基础上统计一个特征的移动平均向量，并且这个平均特征在训练中并不会参与梯度计算（只在测试时使用）。这也就解决了传统长尾分类方法依赖“提前获取未来数据分布”的问题。
    2. 尽管我们的测试过程和训练过程有所不同，但我们的**模型是一次训练到位的**，并不需要依赖繁琐的多步训练，这大大简化了拓展至其他任务时的修改成本。
    3. 并且，我们成功的将这个方法运用于图片分类（ImageNet-LT，Long-tailed CIFAR-10/-100）和物体检测/实例分割（LVIS dataset）等多个任务，均取得了最优的结果（截止至我们投稿也就是2020年5月）。**这证明了我们的方法可以作为继re-balancing之后又一个在长尾数据下通用的Strong Single-Stage Baseline**。

* 核心思想

  * 需要利用原始的长尾分布来学习特征提取的原因在于，大量的尾部类别其实不足以提供足够的样本来学习鲁棒的特征表达。人类描述罕见的物体时，往往是通过和已知常见类的比较，**比如会说狮鹫是有着狮子的身体，鹰的翅膀和头的生物，而不必要单独拿一堆狮鹫的图片出来，让你死记硬背住狮鹫的长相**。

    ![img](https://pic2.zhimg.com/80/v2-d453e3a2168bc0085970561adb1f2b91_720w.webp)

  * **发现优化器的动量项时，这货不就是在训练数据时引入数据分布，从而产生shortcut的元凶么**。

  * 方法，详细的推到有点复杂，相近论文或者知乎

    ![img](https://pic1.zhimg.com/80/v2-aeb0b5c18e021b302263ffd7e49587c4_720w.webp)

  * **方法速成步骤**
    * **基本只要改classifier**，不需要任何额外的训练步骤或复杂的采样算法

  ![image-20230211224744104](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230211224744104.png)

  

:fire: :hammer_and_wrench: **Interventional Few-Shot Learning**, in NIPS 2020. [[pdf](https://arxiv.org/pdf/2009.13000v2.pdf)] [[torch](https://github.com/yue-zhongqi/ifsl)] [[blog](https://zhuanlan.zhihu.com/p/584951599)]

> 博客讲得很全面，具体可以见博客对本文的讲解
>
> * 方法**挺会玩**的！

* 动机

  * few-shot learning过去往往收到**预训练知识**confounder的影响

  ![image-20230206215108229](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230206215108229.png)

  * 过去few-shot learning 方法

  ![image-20230206215028756](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230206215028756.png)

  

* 问题建模（探索Many-shot Learning 和 Few-shot在因果关系上的不一致地方）
  * 有点难解释具体见论文

![image-20230206215234701](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230206215234701.png)

![image-20230206215302623](/Users/gary/Library/Application Support/typora-user-images/image-20230206215302623.png)

* 问题核心

$$
P(Y \mid d o(X=\boldsymbol{x}))=\sum_d P(Y \mid X=\boldsymbol{x}, D=d, C=g(\mathbf{x}, d)) P(D=d)
$$

* **模型方法**（真会玩！）

![img](https://pic1.zhimg.com/80/v2-7f68252d630d8d70b2bfd9562ea560d4_720w.webp)

:fire: :hammer_and_wrench: **Introspective Distillation for Robust Question Answering**, in NIPS 2021. [[pdf](https://arxiv.org/abs/2111.01026)] [[torch](https://github.com/yuleiniu/introd)] [[zhihu](https://zhuanlan.zhihu.com/p/445125531)]

* 动机
  * 希望模型可以同时在**ID（in-domain）和OOD场景下获得良好的表现**
  * 通过对专门捕捉ID和OOD偏置的两位老师模型**进行知识融合**(内省)并蒸馏出一个学生模型来完成这种平衡
* 方法

![image-20230211230411971](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230211230411971.png)



:fire: :hammer_and_wrench: **Distilling Causal Effect of Data in Class-Incremental Learning**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2103.01737)] [[torch](https://github.com/JoyHuYY1412/DDE_CIL)]

* [[模型公式解释](https://zhuanlan.zhihu.com/p/358340627)]  [[论文介绍](https://www.163.com/dy/article/G4OHT10U0511DPVD.html)]

* 动机

  * 对撞节点的存在使得模型对新数据会产生灾难性遗忘

    * 当在一个数据集当中训练好的模型扔到一个新的数据当中微调时候会发生遗忘

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

:fire: :hammer_and_wrench: **Causal Attention for Vision-Language Tasks**, in CVPR 2021. [[pdf](https://arxiv.org/pdf/2103.03493.pdf)] [[torch](https://github.com/yangxuntu/catt)] [[Blog](https://www.cnblogs.com/gongqk/p/14772297.html)]

> 我的想法和博客最后说的类似，似乎本文**推理推得不是特别彻底**，但是仍然是一篇优秀的论文

* 动机：解决当前`VL`模型当中训练数据集当中bias

  ![截屏2022-10-11 11.09.42](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-11%2011.09.42.png)

* 思想

  * 构建因果图

  ![image-20221011111108239](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221011111108239.png)

  * 方法（**前门调整机制**做**Causal Attention**，此处不做展开讨论，详见论文）

    ![截屏2022-10-11 11.12.27](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-11%2011.12.27.png)

    ![截屏2022-10-11 11.12.56](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-11%2011.12.56.png)

* 结果case

  ![截屏2022-10-11 11.13.42](https://raw.githubusercontent.com/Gary-code/pic/main/img/%E6%88%AA%E5%B1%8F2022-10-11%2011.13.42.png)

:fire: :hammer_and_wrench: **Counterfactual Zero-Shot and Open-Set Visual Recognition**, in CVPR 2021. [[pdf](https://arxiv.org/abs/2103.00887)] [[torch](https://github.com/yue-zhongqi/gcm-cf)] [[zhihu](https://zhuanlan.zhihu.com/p/365089242)]

* **生成式的因果模型，博客很详细，详见知乎！**

* 方法

  * **样本特征和类别特征之间解耦**

    ![img](https://pic2.zhimg.com/80/v2-d3135e2e7959fcbd10218d71db857e61_720w.webp)

  ![img](https://pic4.zhimg.com/80/v2-2319bf19f8737e74ad7b7504903d78df_720w.webp)

:fire: :hammer_and_wrench: **Counterfactual VQA: A Cause-Effect Look at Language Bias**, in CVPR 2021. [[pdf](https://arxiv.org/pdf/2006.04315.pdf)] [[torch](https://github.com/yuleiniu/cfvqa)]

* 动机

  * 过去方法主要是消除VQA中语言先验信息的bias
    * 数据增广
    * 生成视觉或者文本的解释
    * 反事实样本生成
    * 集成方法
  * 语言当中的Bias有好有坏，我们做的应该**分辨出好的bias和坏的bias**
    * 坏的bias：香蕉🍌有可能是绿色的，但回答往往是黄色
    * 好的bias：问 What Color的时候，往往模型都会想到视觉词汇，缩减了答案的空间，避免生成很离谱的答案

* 方法（基于ensemble的`VQA`模型做的一个**推理框架**，相当简单）

  ![image-20230205205153879](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230205205153879.png)

  ![image-20230205205846733](/Users/gary/Library/Application Support/typora-user-images/image-20230205205846733.png)

* 因果推理角度解释（具体见论文）

![image-20230205205700958](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230205205700958.png)



**Learning Causal Effects on Hypergraphs**, Best Paper of KDD 2022. [[pdf](https://arxiv.org/pdf/2207.04049.pdf)]

* **超图**上进行因果影响分析，对比传统的以**两两节点**为图的更加有意义。
* [[博客链接](https://zhuanlan.zhihu.com/p/564481108)] [[方法细节](https://zhuanlan.zhihu.com/p/567996036)]

**Entropic Causal Inference: Graph Identifiability**, in ICML 2022. [[pdf](https://proceedings.mlr.press/v162/compton22a.html)]

> 新的**因果推断架构**，可以**不借助intervention**
>
> 理论性很强的一篇文章

* 熵因果推断：

  * 通过寻找数据的信息-理论上最简单的结构解释，即**最小熵模型**，从观测数据中学习两个变量之间的因果图。

  * **在这个工作中，首先推广了松弛假设下的因果图可辨识性结果**。

  * 然后，我们展示了第一个可识别的结果，使用熵的方法学习**超过两个节点的因果图**。

    * 该方法利用了一个属性，即一个源节点和它的后代节点之间的祖先关系可以用二元熵测试来确定。
    * 提供了一种基于此特性的普通图的**有序剥离算法**。我们还提出了一个**启发式算法**，小图显示了较强的经验性能。
    * 最后，在真实的数据集上测试的算法。

    ![img](https://pic4.zhimg.com/80/v2-05352bd30cf8d5f8373c2ae67c3203c3_720w.webp)

:hammer_and_wrench: **[2023.01未开源] Disentangle and Remerge: Interventional Knowledge Distillation for Few-Shot Object Detection from A Conditional Causal Perspective**, in AAAI 2023. [[pdf](https://arxiv.org/abs/2208.12681)] [[torch](https://github.com/ZYN-1101/DandR)]

> 本文**方法上和实验上**还有很多亮点，这里主要就说**因果推理**

* 动机
  * 知识蒸馏做Few-shot任务时候，往往大模型存在的知识**很难迁移**到下游任务当中
* 因果图

![image-20230129234912585](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230129234912585.png)

* 模型

![image-20230129235019260](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230129235019260.png)



**Learning to Imagine: Integrating Counterfactual Thinking in Neural Discrete Reasoning**, in ACL 2022. [[pdf](https://aclanthology.org/2022.acl-long.5/)]

* 动机

  * 当前神经离散推理在反事实推理上效果不佳
  * 借用**因果推断**的理论可以做反事实推理

  ![image-20230130170116064](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230130170116064.png)

* 方法

  * 两步走
    * **识别事实**
    * 通过离散的操作进行**干预**
  * 模型

  ![image-20230130170205387](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230130170205387.png)







:hammer_and_wrench: :star2: **Causality Inspired Representation Learning for Domain Generalization**, in CVPR 2022 **Oral**. [[pdf](https://arxiv.org/abs/2203.14237)] [[torch](https://github.com/BIT-DA/CIRL)]

> **傅立叶变换结合因果推理**，超级solid的工作

* 动机

  * 过去领域泛化的方法都是**基于统计依赖**，没有学习到真正的**因果关系**
  * 任务的数据表征往往可以分成**causal factors $S$ and uncausal factors $U$**
  * 因果推断应用进来的三个原则
    1. $S$ 和 $U$ 相互分离
    2. $s_1, s_2,...,s_n$之间相互独立
    3. $X \rightarrow Y$之间因果充分，因果的表征是充分的

* 方法

  * 前提：傅立叶变换当中

    * 相位：高层的语义信息
    * 振幅：低层的统计信息

    ![image-20230222154015844](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230222154015844.png)

  * 核心思想与流程概览，**具体见论文**

    * 对振幅进行干预，使得干预前后的表征尽可能的相似
    * 表征当中每个维度的表征$s_i$尽可能不相近
    * 使用分类器和掩码器之间的对抗检测因果信息较少的维度，分别**最大化较好维度的分类器和最小化较好维度的分类器**

    ![image-20230222154338573](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230222154338573.png)

    * **总优化目标**：

    $$
    \min _{\hat{g}, \hat{h}_{1}, \hat{h}_{2}} \mathcal{L}_{c l s}^{s u p}+\mathcal{L}_{c l s}^{i n f}+\tau \mathcal{L}_{F a c}, \quad \min _{\hat{w}} \mathcal{L}_{c l s}^{s u p}-\mathcal{L}_{c l s}^{i n f},
    $$



:fire: **[CLIP] EI-CLIP: Entity-aware Interventional Contrastive Learning for E-commerce Cross-modal Retrieval**, in CVPR 2022. [[pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_EI-CLIP_Entity-Aware_Interventional_Contrastive_Learning_for_E-Commerce_Cross-Modal_Retrieval_CVPR_2022_paper.pdf)]

* 动机

  * 在电商产品当中，language的semantic和general domain当中的不太一样
  * 比如：Golden Goose在电商产品中可能是一个品牌

  ![image-20230306100430190](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230306100430190.png)

  * **产品中不同的metadata对最后检索的结果贡献不一样**

  ![image-20230306100443859](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230306100443859.png)

* 方法
  * CLIP直接检索无法搞定那些电商领域的语义信息，所以do一下文本的embedding
  * 过去的confounder设计都是直接考虑了全部文本数据，这里使用metadata进行考虑，同时加权选出**重要的confounder**

![image-20230306102010632](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230306102010632.png)

**[领域泛化] GCISG: Guided Causal Invariant Learning for Improved Syn-to-real Generalization**, in ECCV 2022. [[pdf](https://arxiv.org/abs/2208.10024)]

> 个人认为这篇论文写得有点夸张了，实际上就那么回事。

* 动机

  * 和上面论文一样，需要解耦出**任务无关的style特征**和**任务相关的style特征**

* 因果发现

  * CNN网络对**texture（纹理，如style）特征有偏见**， [参考文献](https://arxiv.org/abs/1811.12231)
  * 因果图

  ![image-20230227101118298](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230227101118298.png)

* 方法（很简单）

![image-20230227101000348](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230227101000348.png)





:hammer_and_wrench: **Causality-aware Concept Extraction based on Knowledge-guided Prompting**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.01876)] [[torch](https://github.com/siyuyuan/KPCE)]

> 方法极其简单

* 动机

  * PLM在概念抽取当中往往会存在Bias（概念抽取就是把某个span抽取出来）

  ![image-20230505221541856](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230505221541856.png)

  ![image-20230505221654869](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230505221654869.png)

* 因果分析

  * 由于语言模型里面的先验知识没办法观测，所以采用前门调整的机制（加多一个中介）

  ![image-20230505221755518](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230505221755518.png)

* 方法（极其简单，加多一个分类出来的Topic）

  ![image-20230505221901782](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230505221901782.png)



:hammer_and_wrench: **COLA: Contextualized Commonsense Causal Reasoning from the Causal Inference Perspective**, in ACL 2023.  [[pdf](https://arxiv.org/abs/2305.05191)] [[torch](https://github. com/HKUST- KnowComp/COLA)]

> 和2022年那篇ICML论文ROCK做的任务是一样的，就是纯文本当中的**因果事件检测**

* 动机

  * 有上下文（Context）的因果发现，为此构建了一个数据集

  ![image-20230513220020021](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230513220020021.png)

* 方法（理论依据详见论文）

![image-20230513220210226](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230513220210226.png)



:fire: **CFL: Causally Fair Language Models Through Token-level Attribute Controlled Generation**, in ACL Finding 2023. [[pdf](https://arxiv.org/abs/2306.00374)]

> 有点晦涩写的，不是很看得懂

* 动机
  * 使用因果图来解决LM有毒性的问题
  * 过去detoxification的方法会过分探索毒性与受保护的虚假关系
  * 过去方法总结：

![image-20230607103152478](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230607103152478.png)

* 思路

![image-20230607103224807](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230607103224807.png)

* 模型方法

  1. 三个toxicity attributes的检测：offense，abuse，hate
  2. 反事实生成 + 计算ATE（平均化当前词的TE）

  ![image-20230607103351552](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230607103351552.png)

  3. 构建SCM + 求 Loss

![image-20230607103515590](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230607103515590.png)

* 实验：微调GPT-2还有OPT



:fire: :hammer_and_wrench: **Causal-Debias: Unifying Debiasing in Pretrained Language Models and Fine-tuning via Causal Invariant Learning**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.232/)] [[torch](https://github.com/myZeratul/Causal-Debias)]

> 通过学习因果不变性，来缓解PLM在fine-tune时候的bias问题

* 动机
  * 过去的方法虽然可以在Pre-trained就可以debias，但是bias的问题（刻板的联想，性别，racial等）会在fine-tune时候重现

![image-20230727090904224](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230727090904224.png)

* 方法

![image-20230727091050764](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230727091050764.png)

$do(N=n)$求invariant loss
$$
\min \mathcal{L}_{\text {invariant }}=\mathbb{E}_n(\mathcal{R})+\operatorname{Var}_n(\mathcal{R})
$$
![image-20230727091347770](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230727091347770.png)



:fire: :hammer_and_wrench: **Preserving Commonsense Knowledge from Pre-trained Language Models via Causal Inference**, in ACL 2023 Oral. [[pdf](https://arxiv.org/abs/2306.10790)] [[torch](https://github.com/zzz47zzz/CET)]

* 动机
  * 预训练模型Fine-tune的时候容易存在灾难性遗忘
    * **文章中提到Fine-tune与灾难性遗忘在一起的时候，主要是为了解决target task**
    * 而CL + 灾难性遗忘在一起的时候会考虑所有task
* 因果分析

![image-20230729104513604](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230729104513604.png)

* 模型方法
  * 有点类似CL中Replay的方法，用KNN找最相邻的样本（但这个样本哪里来呢，感觉并不是预训练里面来的？）

![image-20230729105649434](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230729105649434.png)

* Fine-tune的目标函数

![image-20230729105714303](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230729105714303.png)



**[VQA] Reducing Vision-Answer Biases for Multiple-Choice VQA**, in TIP 2023. [[pdf](https://ieeexplore.ieee.org/abstract/document/10214252/)]

* 动机
  * 过去的VQA模型容易存在Bias，不看问题就来选
  * 原因是标注的长尾分布
  * 如何影响模型：通过多模态融合特征

![image-20230824202610504](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230824202610504.png)

* 方法（后门调整 + 多模态特征的反事实loss）

![image-20230824202719842](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230824202719842.png)

**Causal Intervention and Counterfactual Reasoning for Multi-modal Fake News Detection**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.37/)]

* 动机
  * 过去的方法从训练集中学习到的label-specific的特征，导致了bias
  * 因果推理 + 反事实 引入到多模态fake new检测中

![image-20230902195944598](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902195944598.png)

* 因果图分析

![image-20230902200058716](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902200058716.png)

* 模型方法（公式比较多，可以看一下原文）

![image-20230902200152045](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902200152045.png)

:hammer_and_wrench: :fire: **A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.32/)] [[code](https://github.com/alestolfo/causal-math)]

> 非常Perfect的关于因果推理 + LLM + 数学推理的Empirical Study

* 动机
  * 研究每个LLM在不同的数学问题输入干预下的敏感度和鲁棒性

![image-20230905105852822](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230905105852822.png)

* 因果推理框架，研究LLM

![image-20230905105935990](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230905105935990.png)

* 具体实验结果，可以看论文

:hammer_and_wrench: **Random Boxes Are Open-world Object Detectors**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2307.08249)] [[torch](https://github.com/scuwyh2000/RandBox)]

* 动机
  * 过去的Open-world的OD方法太关注known的对象，而不关注unknown的对象
  * 提出random的box，有两个优点
    * 防止了训练被confounded（randomization和known的object分布相互独立）
    * 无偏的训练鼓励更多的proposal探索

![image-20230920105054836](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920105054836.png)

![image-20230920105249978](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920105249978.png)

* 过去OWOD方法

![image-20230920110822856](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920110822856.png)

* 因果分析（切断后门路径$R \leftarrow D \rightarrow Y$）

![image-20230920105358323](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920105358323.png)



* 模型方法（具体见论文）

![image-20230920110744901](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920110744901.png)



:fire: :hammer_and_wrench: **Mitigating Adversarial Vulnerability through Causal Parameter Estimation by Adversarial Double Machine Learning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.07250)] [[torch](https://github.com/ByungKwanLee/Double-Debiased-Adversary)]

* 动机
  * 过去的模型仍然会受到不可消除的对抗漏洞
  * 使用Double Machine Learning的方法来评估因果参数

![image-20230930112534237](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930112534237.png)

* 因果建模（关键是那个$\theta$）

![image-20230930121446835](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930121446835.png)

* 方法（具体见论文，数学推导有点复杂）

![image-20230930121524762](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930121524762.png)



:fire: **Variational Causal Inference Network for Explanatory Visual Question Answering**, in ICCV 2023. [[pdf](https://openaccess.thecvf.com/content/ICCV2023/html/Xue_Variational_Causal_Inference_Network_for_Explanatory_Visual_Question_Answering_ICCV_2023_paper.html)]

> 本文公式推导较为复杂，具体可以见论文

* 动机
  * 保证生成的解释和Answer之间的一支新，并且可以动态捕获visual token和question token生成解释

![image-20231003110257827](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231003110257827.png)

* 因果分析

 ![image-20231003110348348](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231003110348348.png)

* 方法（具体见论文）
  * 核心：让测试和训练时候的分布不要偏移太厉害，用一个KL散度规范

![image-20231003110505687](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231003110505687.png)

* 损失函数（具体见论文）

![image-20231003110557738](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231003110557738.png)



:fire: **A Multi-modal Debiasing Model with Dynamical Constraint for Robust Visual Question Answering**, in ACL Findings 2023. [[pdf](https://aclanthology.org/2023.findings-acl.311/)]

* 动机：过去直接相减的debias方法，损害了ID的性能表现

![image-20231015110456869](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231015110456869.png)

* 方法（具体见论文，有点道理，又不是太合理）（个人觉得和Backdoor Debiasing的论文思路比较类似）

![image-20231015110559283](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231015110559283.png)

![image-20231015110629597](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231015110629597.png)



:fire: **Neuro-Symbolic Procedural Planning with Commonsense Prompting**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2206.02928)]

* Prompt + Causal的
* 方法有点奇怪，我还没看懂



**A Causal Inference Look at Unsupervised Video Anomaly Detection**, in AAAI 2022. [[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/20053)]

* 动机
  * 无监督的视频AD有bias问题（伪标签带来的）
* 分析

![image-20231114205543073](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114205543073.png)

## :happy: Emotion

**Understanding Chat Messages for Sticker Recommendation in Messaging Apps**， in AAAI 2020. [[pdf](https://arxiv.org/abs/1902.02704)]

* 贡献
  * 开发了一个表情包推荐的APP（根据对话来推荐）
  * 数据集和代码都没有开源
* 方法（两阶段）

![image-20230529102818500](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529102818500.png)



:hammer_and_wrench: :fire: **Learning to Respond with Stickers: A Framework of Unifying Multi-Modality in Multi-Turn Dialog**, in WWW 2020. [[pdf](https://arxiv.org/abs/2003.04679)] [[torch](https://github.com/gsh199449/stickerchat)]

* 贡献
  * 提出**340K的多轮对话和表情包对数据集**

![image-20230529104712626](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529104712626.png)

* 方法（挺简单）

![image-20230529104740082](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529104740082.png)



**SER30K: A Large-Scale Dataset for Sticker Emotion Recognition**, in MM 2022 Oral. [[pdf](https://dl.acm.org/doi/abs/10.1145/3503161.3548407)] [[torch](https://github.com/nku-shengzheliu/SER30K)]

* 贡献
  * **表情包情感分类数据集**

![image-20230529105113780](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529105113780.png)

* 表情包的特性和一些相关的应用

![image-20230529105330713](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529105330713.png)



* 数据集特点（sticker emotion recognition）

  * 包括情感和对话的文本标注（但是这个文本**其实只有一句话**）
  * 数据集统计

  ![image-20230529105634054](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529105634054.png)

* 方法（Baseline）

![image-20230529105703308](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529105703308.png)

* Case Study

![image-20230529105733843](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529105733843.png)



:hammer_and_wrench: :fire: **Selecting Stickers in Open-Domain Dialogue through Multitask Learning**, in ACL Finding 2022. [[pdf](https://aclanthology.org/2022.findings-acl.241/)] [[torch](https://github.com/nonstopfor/Sticker-Selection)]

* 表情包推荐例子

![image-20230529154209146](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529154209146.png)



* 方法（多任务学习的方法）

![image-20230529154301746](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230529154301746.png)



## :old_key: Traditional NLP Task

### :label: NER

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

---



### :older_man: Text Classification

:fire: :hammer_and_wrench: **Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classifification**, in ACL 2022. [[pdf](https://arxiv.org/abs/2108.02035)] [[torch]](https://github.com/thunlp/KnowledgeablePromptTuning)] [[知乎博客](https://zhuanlan.zhihu.com/p/398009000)]

> 将外部知识融入当prompt-tuning当中做TC任务

* 什么是 Prompt-tuning
  * 之前的Prompt-tuning方法可被用于文本分类任务，具体方式是通过构建标签词表，将分类问题转化为一个预测和标签相关词的问题。因此，这样的问题可以构建一个含有[MASK]的模板，然后让MLM（掩码语言模型）去预测[MASK]位置的单词。至此，分类任务被转化为了一个掩码语言建模问题。
  * 下面给出一个例子，当我们要对一个句子进行分类时，可以尝试构建下面的模板：**A [MASK] question: x**
  * 比如MLM预测出在[MASK]位置概率最高的词是science，那该句可以被分类为SCIENCE类别。
* 动机
  * 过去prompy-tuning方法，MLM在[MASK]位置可以预测出的单词是很多的，然而类别数只有特定数量的，因此该问题很重要的一个部分是如何==**构建一个单词表到类别标签的映射**==。这个映射能让MLM在预测到类别标签的相关词时，就可以被分到指定类别去。
  * 这样的一个映射，过去通常是由人来手工编辑或使用梯度下降搜索。**但显然这样会带来覆盖范围不全导致的高偏差和高方差**。**知识库的组织结构，天然的带有范围关系**，在知识库的图结构中，相关联的实体会有边相连，不相关的实体可能要经过很多跳才能找到关联，或无关联。因此如果能将外部的知识库信息融入，构建一个映射器（本文称语言表达器），就可以一定程度上避免手工构造和梯度下降带来的高偏差和高方差问题。
* 方法

![image-20221202123732640](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221202123732640.png)

---



### :framed_picture: Topic 

:fire: **[层次化主题] Deep Latent Dirichlet Allocation with Topic-Layer-Adaptive Stochastic Gradient Riemannian MCMC**, in ICML 2017. [[pdf](https://proceedings.mlr.press/v70/cong17a.html)]

> 理论性非常强的论文
>
> * 详细的理论推导请见原文。

* case展示

![image-20221031220028391](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221031220028391.png)



## :mailbox: Knowledge&LLM&VLM

### :rocket: LLM

:fire: :hammer_and_wrench: **Quark: Controllable Text Generation with Reinforced [Un]learning**, in NIPS 2022. [[pdf](https://arxiv.org/abs/2205.13636)] [[torch](https://github.com/GXimingLu/Quark)]

* 动机
  * 整体思路和`InstructGPT`很类似，但是不再训练奖励模型，而是使用prompt来取代（评判生成的是否是好句子）
* 与Instruct GPT 不同的地方
  1. **抛弃训练打分模型（reword model）这个步骤**，觉得训练一个打分模型过于繁琐，增加训练成本
  2. 然而也要保证人工监督的信息（human feedback）传入生成模型中，论文提出使用一个==**打分标记**==（reward token）嵌入prompt中来控制生成，如此可以更为简洁地实现human feedback+reinforcement learning的训练技术策略。
* 模型方法
  1. **Exploration，**同样基于一个大规模预训练模型（GPT）产生一批prompt生成结果，接着对其评价打分，并存储一个数据池中（data pool）
  2. **Quantization，**利用打分的结果，给数据池中中的数据进行排序，然后进行层级量化分类（如分好，中、差类似）
  3. **Learning，**将已经分层的样本，给一个打分标记（图中[R3]）与prompt拼接在一起作为输入，对原始的预训练模型进行微调训练学习。

![image-20230412090327306](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230412090327306.png)

:fire: :hammer_and_wrench: **Large Language Models Are Human-Level Prompt Engineers**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2211.01910)] [[torch](https://sites.google.com/view/automatic-prompt-engineer)] [[zhihu](https://zhuanlan.zhihu.com/p/619112790)]

* 动机
  * 一句话来概括本文的思路: 利用训练数据构造demonstration，要求LLM在指定描述的情境下，产生对应的prompt集合，然后评估选择出最佳的instruction。
* 具体思路
  1. 使用LLM作为推理模型，基于**输入和输出的描述集合**，产生**指令候选集**。
  2. 对于候选集中的每个指令，计算输入和输出LLM下的得分。
  3. 使用迭代蒙特卡洛的搜索方式，通过提出语义相似的指令变体来改进最佳的指令候选，增加指令的多样性。

![image-20230508105808096](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230508105808096.png)

:fire: **[2023.5.6未开源] SCOTT: Self-Consistent Chain-of-Thought Distillation**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.01879)]

* 动机

  * 只有比较大的LLM才有COT的能力，而且大的LLM也会产生很多幻觉信息，导致蒸馏出来的小模型在COT能力上会很差

  ![image-20230506092740246](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230506092740246.png)

* 方法【teacher：GPT-neox20B student：T5 3B】

  * 大模型中采用Contrastive Decoding编码的策略（意味着你必须获得词概率的分布，模型可以frozen掉）
  * 小模型当中利用大模型生成的数据进行训练（同时让teacher生成一些反事实的也一起丢进去fine-tuning）

   ![image-20230506093023819](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230506093023819.png)

  * 细节

    * teacher的contrastive decoing的策略

    ![image-20230506093124595](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230506093124595.png)

    * student的反事实推理能力（都是由teacher产生的，具体来说就是把teacher中输入的答案换成错误的答案或者空的字符就可以了）

    ![image-20230506093251574](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230506093251574.png)

:fire: **Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes**

> 方法也太简单了，其实就是当前LLM的范式，用大语言模型生成的东西来蒸馏学习小模型

* 方法

![image-20230506095246261](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230506095246261.png)



:fire: **Poisoning Language Models During Instruction Tuning**, in ICML 2023. [[pdf](https://arxiv.org/abs/2305.00944)] [[torch](https://github.com/AlexWan0/Poisoning-Instruction-Tuned-Models[)]

* 动机

  * 在训练时候加入**脏数据攻击语言模型**

  ![image-20230508103121615](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230508103121615.png)



:fire: :hammer_and_wrench: **Outline, Then Details: Syntactically Guided Coarse-To-Fine Code Generation**, in ICML 2023. [[pdf](https://arxiv.org/abs/2305.00909)] [[torch](https://github.com/VITA-Group/ChainCoder)]

* 动机

  * 过去的代码生成都是一次性的，效果并不好，应该COT那样子生成
  * 忽略了句法结构作为先验

* 最大贡献

  * 提出基于句法树的Tokenizer（最重要的贡献）

* 方法

  * Tokenizer

  ![image-20230508194557470](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230508194557470.png)

  $S3$和$S4$是重点，tokenizer的encode伪代码如下：

  ![image-20230508194659740](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230508194659740.png)

* 模型（预训练+微调）

![image-20230508194740606](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230508194740606.png)



:hammer_and_wrench: **Distilling Script Knowledge from Large Language Models for Constrained Language Planning**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.05252)] [[torch](https://github.com/siyuyuan/coscript)]

* 动机：有约束的plan生成，比如说给一个糖尿病人做蛋糕应该不要加糖

![image-20230513212959042](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230513212959042.png)

* 有约束数据集生成方式与我们的方法（先过度生成再过滤）

![image-20230513213047184](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230513213047184.png)



:fire: :hammer_and_wrench: **BadPrompt: Backdoor Attacks on Continuous Prompts**, in NIPS 2022.  [[pdf](https://arxiv.org/abs/2211.14719)] [[torch](https://github.com/papersPapers/BadPrompt)]

* 动机
  * 没有人研究过prompt-based model 的后门攻击
  * few-shot的场景往往缓解了攻击，会导致攻击不够有效。所以需要更高级的攻击策略
  * 目标希望ASR和CA都比较高

![image-20230516212217595](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516212217595.png)

* 方法（具体见论文）

![image-20230516212243228](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516212243228.png)



:fire: :hammer_and_wrench: **RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.08844)] [[torch](https://github.com/feyzaakyurek/rl4f[)]

> 太棒的工作了！

* 动机
  * 过去反馈都是人来生成的，能不能用一个模型来生成，然后修改那种黑盒LLM

![image-20230516215115928](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516215115928.png)

* 方法（太棒的方法了）【批评的模型就是一个`T5-large` (0.77M)】

![image-20230516215150724](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230516215150724.png)



:fire: :hammer_and_wrench: **Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark**, in ACL 2023. [[pdf]](https://arxiv.org/abs/2301.10226)] [[torch](https: //github.com/yjw1029/EmbMarker)]

* 动机
  * 利用后门来保护语言模型的版权

![image-20230518162732603](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230518162732603.png)

* 方法

![image-20230518162756186](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230518162756186.png)



:hammer_and_wrench: :fire: **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**, in 2023 05. [[pdf](https://arxiv.org/abs/2305.10601)] [[torch](https://github.com/ysymyth/tree-of-thought-llm)]

* [知乎链接](https://zhuanlan.zhihu.com/p/631940032)
* 模型方法 TOT

![image-20230531094920757](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230531094920757.png)

* TOT解决问题例子

![image-20230531094947180](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230531094947180.png)

![image-20230531095005174](/Users/gary/Library/Application Support/typora-user-images/image-20230531095005174.png)

![image-20230531095334358](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230531095334358.png)



:hammer_and_wrench: **Plug-and-Play Knowledge Injection for Pre-trained Language Models**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.17691)] [[torch](https://github.com/ THUNLP/Knowledge-Plugin)]

* 动机
  * 过去在LM中加入外部知识可以提高模型在下游任务的性能
  * 但是往往需要重新训练或者微调LM，显得很麻烦
  * 本文提出只训练一个Mapping的网络，将外部知识投射到下游任务的模型当中（本文探索了很多种LM的模型）
* 模型方法

![image-20230602105324893](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230602105324893.png)



:hammer_and_wrench: :fire: **Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models’ Memories**, in ACL 2023. [[pdf](https://arxiv.org/abs/2306.05406)] [[torch](https://github.com/ Amano-Aki/Mixture-of-Domain-Adapters)]

* 动机
  * 希望用Adapter的方式将PLM在某些特定领域中小参数形式微调
* 方法（两阶段）

![image-20230611215226591](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230611215226591.png)

* 实验
  * Baseline选取就是其他的微调方法，比如LoRa等方式（实验结果详见论文）

:fire: :hammer_and_wrench: **Grounding Language Models to Images for Multimodal Inputs and Outputs**, in ICML 2023. [[pdf]](https://arxiv.org/abs/2301.13823) [[torch](https://jykoh.com/fromage)]

> 单卡A6000进行训练，很友好

* 动机
  * 同时处理文本和图片的输入和输出（注意，这里的图片是检索出来的）

![image-20230613094222645](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230613094222645.png)

* 方法（两种很简单的与训练方法）
  * CLIP和LLM都是冻结参数的

![image-20230613094249427](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230613094249427.png)

* 例子展示
  * 推理时候如何进行的，具体要见论文代码

![image-20230613094451233](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230613094451233.png)



:hammer_and_wrench: **[COT] Element-aware Summarization with Large Language Models: Expert-aligned Evaluation and Chain-of-Thought Method**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.482/)] [[code](https://github.com/Alsace08/SumCoT)]

> 相当于重构了现在的Summarization数据集

* 动机
  * 过去的benchmark的参考summarization往往结果是嘈杂的，主要是在事实幻觉和信息冗余方面
  * 我们希望就主要关🧘元素来生成摘要

![image-20230717204206612](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230717204206612.png)

* 方法（给定几个参考问题进行提问）

![image-20230717204227177](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230717204227177.png)



:fire: :hammer_and_wrench: **Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.74/)]  [[torch](https://github.com/microsoft/dp-transformers)]

* 动机
  * 文本生成的隐私性，不要让语言模型泄漏一些隐私信息
  * 引入差分隐私的方法来Fine-tune GPT-2
* 方法（具体我也不是很看得懂，可能要看代码才知道具体操作）

![image-20230725170115857](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725170115857.png)

![image-20230725170734255](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230725170734255.png)



:fire: :hammer_and_wrench: **Editing Large Language Models: Problems, Methods, and Opportunities**, in 2023.05. [[pdf](https://arxiv.org/abs/2305.13172)] [[torch](https://github.com/zjunlp/EasyEdit)]

* 动机
  * LLM知识定向编辑的empirical study

![image-20230902184416729](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902184416729.png)



:fire: :hammer_and_wrench: **LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2212.04088)] [[torch](https://dki-lab.github.io/LLM-Planner/)]

* 动机：LLM解决导航问题

![image-20231013234300711](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013234300711.png)

* 模型方法

![image-20231013234456103](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013234456103.png)



:fire: :hammer_and_wrench: **CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2309.01093)] [[torch](https://toneyaya.github.io/cotdet)]

* 动机
  * 解决Task-driven的OD任务，需要视觉知识作为桥梁，LLM单独很那做到

![image-20231116103832300](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116103832300.png)

* 方法（挺好的方法）

![image-20231116104002333](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116104002333.png)

### :hammer: Code

:hammer_and_wrench: **AST-Trans: Code Summarization with Efficient Tree-Structured Attention**, in ICSE 2022. [[pdf](https://dl.acm.org/doi/pdf/10.1145/3510003.3510224)] [[torch](https://github.com/zetang94/ICSE2022_AST_Trans)]

> 代码摘要的生成

* 动机
  * 过去代码转成AST的方式，Tree太长了很难让模型学到节点之间的关系
  * 这里我们只关注关键的两种节点：**祖先-后代节点**和**兄弟节点**

![image-20230518112438244](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230518112438244.png)

* 方法

  * 转换成矩阵有效的encoding

  ![image-20230518112515126](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230518112515126.png)

  * 后面还有一系列很细节的操作（不太清楚有没有道理）

* 实验
  * 复杂度分析
  * 很细节的消融
  * 这会议居然写10页的论文，真多



:fire: :hammer_and_wrench: **Multi-target Backdoor Attacks for Code Pre-trained Models**, in ACL 2023. [[pdf](https://arxiv.org/abs/2306.08350)] [[torch](https://github.com/Lyz1213/Backdoored_PPLM)]

> 代码仓库很完善，但这篇论文写作一般般吧

* 动机
  * 在预训练期间对代码模型注入后门
  * 针对两种任务：分类任务 + 生成任务
* 方法

![image-20230620211344890](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230620211344890.png)

* 攻击流程与思路
  1. 设置Trigger Set
     1. 自然语言的Trigger
     2. 代码Trigger
        * 必须保留原始意思
        * 永真的assert语句或者永假的if语句
        * 保证不会被编译器忽略（如，if(sin(0.7)<1)）
     3. Trigger Output (Target Label)
        1. 分类任务：特定标签就好
        2. 生成任务：增删改
  2. 预训练任务 (注入后门)
     1. Seq2Seq Learning
        1. 三分类（增删改）
        2. NL <---> Code
     2. Token表示学习，EOS做文章
  3. 模型部署
* 模型预训练与数据集
  * CodeT5和PLBART
  * CodeSearchNet的数据集来攻击
* 攻击下游任务（实验）
  * 代码understanding
  * 代码生成



### :rainbow: VLM

> 多模态预训练

:fire: :hammer_and_wrench: **[Cross-Modal&Contrastive Learning] UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning**, in ACL(long paper) 2021. [[pdf](https://aclanthology.org/2021.acl-long.202/)] [[project from Baidu](https://unimo-ptm.github.io/)]

:hammer_and_wrench: **[MultiModal] UniT: Multimodal Multitask Learning with a Unified Transformer**, ICCV 2021. [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_UniT_Multimodal_Multitask_Learning_With_a_Unified_Transformer_ICCV_2021_paper.pdf)] [[project from Fair](https://mmf.sh/)]



:fire: **:hammer_and_wrench:** **Leveraging Visual Knowledge in Language Tasks: An Empirical Study on Intermediate Pre-training for Cross-modal Knowledge Transfer**, in ACL 2022. [[pdf](https://aclanthology.org/2022.acl-long.196/)] [[torch (2022.12.01未开源)](https://github.com/INK-USC/CMKT)]

> 本文是一篇**实验性的文章**，实验的方法写得不错！
>
> 什么叫做：**intermediate pre-training**?
>
> * 在预训练好的模型上，补充一些数据集或者语料库继续预训练

* 动机

  * 目前的大规模语言模型在理解**日常物品属性**方面的能力还很缺乏，因为它们的预训练语料中**很少有这样的事实性知识**，即所谓的reporting bias，下图即是一个例子：

  ![image-20221201194524047](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221201194524047.png)

* 方法（以下两种方法来弥补PLM在这方面的缺陷）

  * text knowledge transfer: 即使用image caption进行知识迁移，设计了如下两个训练目标：

    * MLM：在image caption上进行MLM，相当于进行了domain adaptive的预训练，和Bert一样的设置
    * Text Contrastive Learning (TCL): 采用和SimCSE一样的方式，进行对比学习，batch中的其他样本都是负样本

  * cross-modal knowledge transfer: 即使用图片和文本以及`V&L`的训练来进行知识迁移，设计了如下几个训练目标：

    * Voken Classification: voken采用token层面的text2image检索来迁移视觉知识，它假设每一个token都有一个视觉域的voken与其对应，训练目标就是在所有预先设定好的voken中将正确的voken检索出来

    * Masked Language Modeling with Visual Clues: 给定图片作为线索，预测mask掉的token，比MLM多了图片作为输入，目标函数是一样的

    * Cross-Modal Contrastive Learning (CMCL): 和CLIP一样（不同在于这里把**视觉网络给冻住**），是跨模态的对比学习

      * 负样本构建方法（之一）（对抗性负样本）
        * 同时要计算生成负样本和原来句子的语义相似度，**过滤掉一些生成的假负样本**

      ![image-20221201195315054](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221201195315054.png)

      * 这些过滤的假负样本，也可以作为正样本使用，作为正样本！

    * Cross-Modal Knowledge Distillation (CMKD): 将在MSCOCO数据集上进行对比学习的多模态模型作为teacher model，将一个语言模型作为student，在纯文本语料Wiki103上进行知识蒸馏

  * 模型示意图

  ![image-20221201195203819](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20221201195203819.png)

* **结论**

  * **简单在captions数据集**上继续预训练可以取得知识迁移的效果
  * 跨模态的知识迁移在**很小的训练样本**情况下可以极大提高**下游任务**的性能
  * **对比学习的方法**对视觉知识（对象属性等，如第一张图说的）的学习是最好的

:hammer_and_wrench: :fire:**[Relation CLIP] RelCLIP: Adapting Language-Image Pretraining for Visual Relationship Detection via Relational Contrastive Learning**, in EMNLP 2022. [[pdf](https://aclanthology.org/2022.emnlp-main.317/)] [[torch]()]

* 动机
  * 让`CLIP`可以识别对象之间的relation

* 方法（很简单）

Commonsense Knowledge 就是**Conceptual Caption数据中来**

![image-20230321093338484](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230321093338484.png)

负样本收集：

![image-20230321094303920](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230321094303920.png)

:star: **[MM-KG + CLIP] Contrastive Language-Image Pre-Training with Knowledge Graphs**, in NIPS 2022. [[pdf](https://arxiv.org/abs/2210.08901)] [[rebuttal](https://openreview.net/forum?id=4T3kbrzfeR)]

* 动机

  * 以前的CLIP模型对自然语言当中复杂的语义特征，比如**颜色和位置**等
  * 将知识加入到**CLIP的预训练**过程当中，以**输入的形式**加进去的

  ![image-20230213181225549](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230213181225549.png)

  

  * **预训练数据集来源具体见论文**
  * 方法

  ![image-20230213181333946](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230213181333946.png)




:hammer_and_wrench: **PuMer: Pruning and Merging Tokens for Efficient Vision Language Models**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.17530)] [[torch](https://github.com/ csarron/PuMer)]

* 动机
  * Image或者Text的Token可能有的是多余的，会带来额外的计算代价
  * 应该把Token剪枝或者合并起来，高效对其多模态信息
* 方法（训练和推理框架一样）

![image-20230602112025512](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230602112025512.png)

:fire: **Learning to Estimate Shapley Values with Vision Transformers**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2206.05282)] [[torch](https://github.com/suinleelab/vit-shapley)]

> 很棒的一篇Shapley Values 结合 ViT的论文

* 动机
  * 验证random masking + 没有ground truth的loss也可以很好的训练出Explainer评估ViT中的Shapley Value
* 方法

![image-20230607002458191](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230607002458191.png)

:fire: **Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models**, in ICCV 2023. [[pdf](https://arxiv.org/pdf/2308.11186)]

* 动机

  * 传统的CLIP对于一些unseen class 泛化能力不是很好

  * CLIP + 外部知识 （个人觉得 CVPR的那几篇更好）

![image-20230902183343344](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902183343344.png)

* 模型方法 （连续 + 离散的知识prompt）

![image-20230902183402422](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230902183402422.png)



:hammer_and_wrench: **[多语言CLIP] mCLIP: Multilingual CLIP via Cross-lingual Transfer**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.728/)] [[torch](https://github.com/ huawei-noah/noah-research/NLP/mclip)]

* 动机：多语言的CLIP模型
* 方法
  1. 先训练好模型多语言的文本编码器
  2. 再将原来的CLIP进行三角蒸馏

![image-20230907223536040](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230907223536040.png)

![image-20230907223622103](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230907223622103.png)



:hammer_and_wrench: **Rethinking Multimodal Entity and Relation Extraction from a Translation Point of View**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.376/)] [[torch](https://github.com/thecharm/TMR)]

* 动机
  * 多模态信息对齐问题
  * 多模态信息对齐和跨语言散度很类似

![image-20230910094115148](/Users/gary/Library/Application Support/typora-user-images/image-20230910094115148.png)

* 模型方法（使用了外部的LAION-400这种数据来预训练模型）
  * Back-Translation：Diffusion模型生成图片
  * 低资源学习 + High-Resource Divergence Estimation

![image-20230910095421482](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910095421482.png)

![image-20230910095446366](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910095446366.png)

* Case Study

![image-20230910095543108](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910095543108.png)



:fire: :hammer_and_wrench: **Preserving Modality Structure Improves Multi-Modal Learning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2308.13077)] [[torch](https://github. com/Swetha5/Multi_Sinkhorn_Knopp)]

* 动机
  * 过去的模型往往忽略了模态特定的表征
* 方法
  * Multi-Assignment Sinkhorn-Knopp算法分配学习Anchor

![image-20230910104252077](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910104252077.png)



 :hammer_and_wrench: **[复杂的图像检索任务] A Neural Divide-and-Conquer Reasoning Framework for Image Retrieval from Linguistically Complex Text**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.02265)] [[torch](https://github.com/YunxinLi/NDCR)]

* 动机
  * 做复杂的这种检索要类似于人脑分支-组合的思想

![image-20230918113349556](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918113349556.png)

* 模型方法（感觉那个**神经符号推理**很值得参考）

![image-20230918113445960](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918113445960.png)

* **神经符号推理模块**

![image-20230918113517987](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918113517987.png)

* 实验结果（实际上所有模型目前结果都不太好）【**IMAGECODE数据集（里面好像也有视频的）**】

![image-20230918113620317](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918113620317.png)



:hammer_and_wrench: :fire: **Bayesian Prompt Learning for Image-Language Model Generalization**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2210.02390)] [[torch](https://github.com/saic-fi/Bayesian- Prompt-Learning)]

* 动机
  * 过去的prompt tuning方法对于unseen的domain表现不行，泛化能力不够。（suffer from distribution shift）
  * Bayesian PT的方法可以prevents learn- ing spurious features, and exploits transferable invariant features
* 方法

![image-20231007094433389](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231007094433389.png)

* 核心公式（需要先看**变分推理**网课）

![image-20231013111140013](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231013111140013.png)



:fire: :hammer_and_wrench: **A Unified Objective for Novel Class Discovery**, in ICCV 2021 Oral. [[pdf](https://arxiv.org/abs/2108.08536)] [[torch](https://ncd-uno.github.io/)] [[blog](https://mp.weixin.qq.com/s/sEhXntgjonNJuxMWKQ__mA)]

* 动机
  * 过去的方法往往都是现在labeled数据集上面预训练好模型，再设计clustering function在unlabeled上做作
  * 本文设计unified的Objective（很巧妙，也很牛逼）
* 方法

![image-20231114201847341](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231114201847341.png)

:fire: :hammer_and_wrench: **E2VPT: An Effective and Efficient Approach for Visual Prompt Tuning**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.13770)] [[torch](https://github.com/ChengHan111/E2VPT)]

* 动机
  * VPT改进版本：Prompt Tuning + Pruning的方法
* 模型（具体见论文）

![image-20231116102805839](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231116102805839.png)

:fire: **SINC: Self-Supervised In-Context Learning for Vision-Language Tasks**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2307.07742)]

* 动机
  * 探索VLM直接在预训练时候的in-context-learning能力
  * 过去ICL都是依赖于语言空间，会出现模版敏感 + 幻觉的问题
  * 而且LLM的训练代价很大

![image-20231122153701843](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231122153701843.png)

* 模型方法

![image-20231122153917503](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231122153917503.png)

* Prompt的设计

![image-20231122154115854](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231122154115854.png)

:fire: :hammer_and_wrench: **Make the U in UDA Matter: Invariant Consistency Learning for Unsupervised Domain Adaptation**, in NeurIPS 2023. [[pdf](https://arxiv.org/abs/2309.12742)] [[torch](https://github.com/yue-zhongqi/ICON)]

* 动机
  * UDA往往会有domain-specific和domain-invariant之间的spurious correlation的问题
  * 过去方法将target domain看作是一个辅助的数据，没有很好的利用起来
* 方法（和他们CVPR那篇视频异常检测的很像）

![image-20231123152426035](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231123152426035.png)

* 训练目标

![image-20231123152448484](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231123152448484.png)



### :running: Continual Learning

:hamburger: :hammer_and_wrench: :fire: **Learning to Prompt for Continual Learning**, in CVPR 2022. [[pdf](https://arxiv.org/abs/2112.08654)] [[code](https://github.com/google-research/l2p)]

> 官方代码是`tensorflow`的，github上有`pytorch`的实现

* 动机
  * 过去数据重放的方法有数据隐私还有大buffer的问题
  * 希望不需要那么大的buffer来存过去的数据，考虑使用Prompt Tuning的方法来做

![image-20230910091659822](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910091659822.png)

* 方法
  * 就是用预抽取的表征选择对应的`Prompt`来Tuning

![image-20230910091831179](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910091831179.png)

:hammer_and_wrench: **S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning**, in NeurIPS 2022. [[pdf](https://arxiv.org/abs/2207.12819)] [[torch](https://github.com/iamwangyabin/S-Prompts)]

* 动机（`S-Prompt`和`DualPrompt`是同时期的东东，他们两个并没有直接标价哦，S-Prompt主要比较的是L2P）
  * 目标是做DIL，让每个domain的prompt都存在一个子空间中，彼此不互相影响

![image-20230924102201828](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230924102201828.png)

* 方法

![image-20230924102352469](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230924102352469.png)



:fire: :hammer_and_wrench: **DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning**, in ECCV 2022. [[pdf](https://arxiv.org/abs/2204.04799)] [[code](https://github.com/google-research/l2p)]

> `L2P`的进阶工作，同组的工作

* 动机
  * L2P只考了一种Prompt，没有考虑任务特定的和任务间相似的特征
  * 人类在连续学习时候往往会考虑这两方面的能力
    * Complementary Learning Systems (CLS) [[引文1](https://pubmed.ncbi.nlm.nih.gov/7624455/),[引文2](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(16)30043-2)] (CLS) suggests that humans learn continually via the synergy between two learning systems: the hippocampus focuses on learning pattern-separated representation on specific experiences, and the neocortex focuses on learning more general and transferable representation from past experience sequences.
* 模型方法
  * 学习两种Prompt

![image-20230910092436783](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910092436783.png)



:fire: :hammer_and_wrench: **CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2211.13218)]] [[torch](https://github.com/GT-RIPL/CODA-Prompt)]

* 动机
  * 过去的PT-CL方法牺牲了新的任务性能，而且，其实并不是所有的任务都需要同样重要的prompt，应该prompt之前先做一下attention（本文中attention也是可学习的）

![image-20231018105108528](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018105108528.png)

* 方法

![image-20231018105148711](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018105148711.png)

![image-20231018105215146](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018105215146.png)



:hammer_and_wrench: :fire: **[2023-10 未开源] ConStruct-VL: Data-Free Continual Structured VL Concepts Learning**,in CVPR 2023. [[pdf](https://arxiv.org/abs/2211.09790)] [[torch](https://github.com/jamessealesmith/ConStruct-VL)]

* 动机
  * 过去的CL方法学概念还不是很行（Concept），所以本文提出Concept Learning的CL Benchmark

![image-20231018105622946](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018105622946.png)

* 模型方法 （LoRA高效微调 + 生成数据重放伪标签）

![image-20231018105644301](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018105644301.png)



:fire: **Introducing Language Guidance in Prompt-based Continual Learning**, in ICCV 2023. [[pdf](https://arxiv.org/pdf/2308.15827.pdf)]

> 可以和`L2P`以及`DualPrompt`无缝衔接起来

* 动机
  * 使用自然语言来引导Prompt的选择
  * 所有任务的样本表征都可以映射到相同的语义空间当中
* 方法

![image-20230910093733775](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230910093733775.png)

:fire: :hammer_and_wrench: **When Prompt-based Incremental Learning Does Not Meet Strong Pretraining**, in ICCV 2023. [[pdf](https://arxiv.org/pdf/2308.10445)] [[torch](https://github.com/TOM-tym/APG)]

* 动机
  * 过去的Prompt-base（L2P，DualPrompt）方法无法很好弥补预训练和下游任务之间的gap

![image-20230919164942525](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230919164942525.png)

* 模型方法（具体见论文）

![image-20230919165013021](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230919165013021.png)



:fire: :hammer_and_wrench: **Continual Pre-training of Language Models**, in ICLR 2023. [[pdf](https://pretrainedarxiv.org/abs/2302.03241)] [[torch](https://github.com/UIC-Liu-Lab/ContinualLM)]

* 动机
  * PLM到新的domain之后，要保证知识迁移 + 无灾难性遗忘
  * 过去的方法在PLM domain迁移时候不是很适用
* 方法（计算unit的importance）

![image-20230930092000716](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930092000716.png)



:fire: :hammer_and_wrench: :hushed: **SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2303.05118)] [[torch](https://github.com/GengDavid/SLCA)]

* 动机
  * CL的一大核心是`lr`

![image-20230930095356560](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930095356560.png)

* 方法
  * lr小一点
  * 每个classier都进行alignment处理（具体见论文）

![image-20230930095448740](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230930095448740.png)

:fire: :hammer_and_wrench: **Preventing Zero-Shot Transfer Degradation in Continual Learning of Vision-Language Models**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2303.06628)] [[torch](https://github.com/Thunderbeee/ZSCL)] [[知乎](https://zhuanlan.zhihu.com/p/643766605)]

* 动机
  * 过去CL方法不单单会忘记上一个任务，还会忘掉预训练的东西，导致很差的zero-shot迁移能力

![image-20231018104708625](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018104708625.png)

* 方法

![image-20231018104827078](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018104827078.png)

![](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231018104827078.png)

:fire: :hammer_and_wrench: **CTP: Towards Vision-Language Continual Pretraining via Compatible Momentum Contrast and Topology Preservation**, in ICCV 2023. [[pdf](https://arxiv.org/abs/2308.07146)] [[torch](https://github.com/KevinLight831/CTP)]

* 动机
  * 多模态预训练很少被探索
  * 和传统的CIL不太一样
    * VLP可以利用嘈杂的网络数据，并且不需要手动标注固定的类别
    * 新的挑战：新旧知识如何保存，更新多模态的encoder
  * 提出新的多模态VLCP的Benchmark
* 方法

![image-20231110200626396](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231110200626396.png)

:hammer_and_wrench: :fire: **Generating Instance-level Prompts for Rehearsal-free Continual Learning**, in ICCV 2023. [[pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Jung_Generating_Instance-level_Prompts_for_Rehearsal-free_Continual_Learning_ICCV_2023_paper.pdf)]] [[torch](https://github.com/naver-ai/dap-cl)]

* 动机
  * prompt不再是从pool里面去选择，而是直接去生成
  * 这样在一些不依赖ViT预训练参数的任务上表现更好，一些比较独特的domain上的表现

* 方法

![image-20231113212002719](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231113212002719.png)



:fire: :hammer_and_wrench: **Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality**, in NeurIPS 2023 Spotlight. [[pdf](https://arxiv.org/pdf/2310.07234.pdf)] [[torch](https://github.com/thu-ml/HiDe-Prompt)]

* 动机
  * 作者发现在自（无）监督的网络当中，MoCo等，传统的prompt的方法做CL效果不好
  * 提出全新的Unsupervised Learning + CL的范式
* 方法（within-task prediction, task-identity inference, and task-adaptive prediction）

![image-20231113212255755](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231113212255755.png)

:hammer_and_wrench: **Towards Continual Adaptation in Industrial Anomaly Detection**, in MM 2022. [[pdf](https://dl.acm.org/doi/abs/10.1145/3503161.3548232)] [[torch](https://github.com/vijaylee/Continual_Anomaly_Detection)]

* 动机
  * 第一个多工业异常检测（**二分类**） + CL的
  * 生成bridge了Unsupervised Learning + CL
    * 实际上是生成了负样本，没有什么特别的
* 方法（存储每个domain的分布信息）

![image-20231113212623593](/Users/gary/Library/Application Support/typora-user-images/image-20231113212623593.png)



:fire: :hammer_and_wrench: **Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection**, in ICCV 2023.  [[pdf]](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Augmented_Box_Replay_Overcoming_Foreground_Shift_for_Incremental_Object_Detection_ICCV_2023_paper.pdf)] [[torch](https://github.com/YuyangSunshine/ABR IOD.git)]

* 动机
  * Replay的方法在ODCL中不见得有效
  * 原因是Foreground-shift

![image-20231115205416533](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231115205416533.png)

* 方法：Replay+增广

![image-20231115205616322](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231115205616322.png)

---



### :world_map: Knowledge

:fire: **[多模态知识图谱综述] Multi-Modal Knowledge Graph Construction and Application: A Survey**, in 2022. [[pdf](https://arxiv.org/pdf/2202.05786.pdf)] [[zhihu](https://zhuanlan.zhihu.com/p/484096631)]



:hammer_and_wrench: **End-to-end Knowledge Retrieval with Multi-modal Queries**, in ACL 2023. [[pdf](https://arxiv.org/pdf/2306.00424.pdf)] [[dataset&torch](https://github.com/luomancs/ReMuQ)]

* 动机
  * 端到端的基于多模态的知识检索过去的工作探索不足
  * 过去往往就是先转成Caption再检索文本端知识，或者直接用图片检索知识

![image-20230604092623360](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230604092623360.png)

* 数据集
  * 基于WebQA提出新的知识检索数据集

![image-20230604092703138](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230604092703138.png)

* 方法（对比学习Loss）

![image-20230604092722382](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230604092722382.png)



:hammer_and_wrench: :fire: **Improving Empathetic Dialogue Generation by Dynamically Infusing Commonsense Knowledge**, in ACL Findings 2023. [[pdf](https://arxiv.org/abs/2306.04657)] [[torch](https://github.com/Hanscal/DCKS)]

* 动机
  * 过去也将常识知识引入进同理心对话里面，但是知识可能会冗余，需要去选择知识。

![image-20230612101031434](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230612101031434.png)

* 模型方法

![image-20230612101049910](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230612101049910.png)

* 知识选择策略（迭代来每次排除掉最不相关的知识）

![image-20230612101155806](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230612101155806.png)



:hammer_and_wrench: **HAHE: Hierarchical Attention for Hyper-Relational Knowledge Graphs in Global and Local Level**, in ACL 2023. [[pdf](https://arxiv.org/abs/2305.06588)] [[torch](https://github.com/LHRLAB/HAHE)]

> **超关系知识图谱**的预训练表示学习

* 什么是超关系知识图谱

![image-20230918095649749](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918095649749.png)

* 表示学习（一样的，还是两个常规任务）
  * 实体抽取
  * 关系预测
* 超图表示

![image-20230918095825890](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918095825890.png)

* 模型方法

![image-20230918095847947](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918095847947.png)

![image-20230918095920522](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230918095920522.png)



:hammer_and_wrench: **KILM: Knowledge Injection into Encoder-Decoder Language Models**, in ACL 2023. [[pdf](https://aclanthology.org/2023.acl-long.275.pdf)] [[torch](https://github.com/alexa/kilm)]

* 动机
  * 知识注入PLM当中（这个idea很一般吧）
* 模型方法

![image-20230920094808476](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20230920094808476.png)



### :ice_cream: Hallucination

:fire: :hammer_and_wrench: **Evaluating Object Hallucination in Large Vision-Language Models**, in EMNLP 2023. [[pdf](https://arxiv.org/abs/2305.10355)]] [[torch](https://github.com/RUCAIBox/POPE)] [[blog](https://mp.weixin.qq.com/s/bcIBN6raLcilgRcj-3ltTg)].

> 对象幻觉的评估测评指标

* 动机
  * 过去用Chair来评测，现在提出了一种基于**轮询的物体探测评测**方法 (Polling-based Object Probing Evaluation, POPE)
  * 实验结果表明 POPE 具有更好的稳定性，并且能够扩展到未标注数据集上
* 方法

![image-20231029180106082](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231029180106082.png)

* 结果
  * InstructBLIP的效果比较好

![image-20231029180139292](/Users/gary/Library/Application Support/typora-user-images/image-20231029180139292.png)

* 幻觉与VQA性能的关系

为了进一步研究幻觉问题和 LVLMs 性能之间的关系，我们在 A-OKVQA 和 GQA 上测试了部分 LVLMs 的性能。由于 LVLMs 的回答大都属于较长的开放式回答，我们难以继续使用传统的 VQA 评测方法。为此我们借助了 ChatGPT 来辅助测评，我们提供给 ChatGPT 的 prompt 如下:

`You are an examiner who can judge whether a student’s answer matches the correct answers. Next, I will provide you with the correct answer and a student’s answer. Please judge whether the student’s answer matches the correct answers.`

测评结果如下表所示。InstructBLIP 在幻觉和 VQA 任务中都取得了最好的表现，而 MiniGPT-4 和 LLaVA 在二者上的结果趋势并不一致。

认为这些差异和 LVLMs 的使用的指令形式有关，例如 MiniGPT-4 的指令是**较简单的图像描述任务**，而 LLaVA 的指令包含**更多较复杂的对话或推理任务**，使其更擅长处理 VQA 任务。总之，上述结果说明在评估现有 LVLMs 的性能时，幻觉和 VQA 性能都需要被考虑。

![图片](https://mmbiz.qpic.cn/mmbiz_png/G7ia3FZ0o0Oq5zb0h73dZrPafghJu43fL9C0LL6e6ZGJR53O0Y9ycJbI5xXoErMACWGOBOeDCdDgzZxdicyLibQNA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

:hammer_and_wrench: :fire: **Can We Edit Multimodal Large Language Models?**, in EMNLP 2023.  [[pdf](https://arxiv.org/abs/2310.08475)] [[torch](https://github.com/zjunlp/EasyEdit)]

* 动机
  * 提出多模态知识编辑的Benchmark（具体见论文）
  * 发现模型编辑的方法还可以，但还有很大进步空间
* 方法（模型编辑）

![image-20231113213749107](/Users/gary/Library/Application Support/typora-user-images/image-20231113213749107.png)



## :man_scientist: AI4Science

### :factory: Industrial

:fire: :hammer_and_wrench: **SimpleNet: A Simple Network for Image Anomaly Detection and Localization**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2303.15140)] [[torch](https://github.com/DonaldRR/SimpleNet)]

* 动机
  * 噪音在特征上面来加，生成负样本，而不是在图像上面加
* 方法（很Simple）

![image-20231113213001981](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231113213001981.png)

:fire: **Pushing the Limits of Fewshot Anomaly Detection in Industry Vision: Graphcore**, in ICLR 2023. [[pdf](https://arxiv.org/abs/2301.12082)]

* 动机
  * 开发了一种新的FSAD方法，称为GraphCore，该方法使用少量正常样本来实现新产品的快速训练和有竞争力的AD精度性能。一方面，通过利用少量数据，我们可以快速训练并加快异常推理的速度。另一方面，因为我们直接训练新产品样本，所以不会发生从旧产品到新产品的异常适应和迁移。
    - 提出了一种用于 FSAD 的特征增强方法，以研究 CNN 生成的视觉特征的特性。
    - 提出了一种新颖的**异常检测模型** **GraphCore**，将新的 VIIF 添加到基于内存库的 AD 范例中，这可以大大减少冗余视觉特征的数量。
    - 实验结果表明，所提出的 VIIF 是有效的，可以显着提高 FSAD 在 MVTec AD 和 MPDD 数据集上的性能。
* 方法（详见论文）

![图片](https://mmbiz.qpic.cn/mmbiz_png/ibPsADKUH1kuEfwNwXic6fTPEmu0gz7UrZZHoIIweowKxTiat19rE0wibcLJMdicxbmLKB55v1g6FBOwxLBElBAZGdw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

:fire: :hammer_and_wrench: **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation**, in CVPR 2023. [[pdf](https://arxiv.org/abs/2303.14814)] [[torch](https://github.com/caoyunkang/WinClip)]

* 动机
  * 直接用CLIP模型Zero-shot来做异常检测二分类问题
* 方法（Prompt的设计才是最重要的）
  * 作者设计了一个Prompt的模版（详见论文附录部分）

![image-20231113213509342](/Users/gary/Library/Application Support/typora-user-images/image-20231113213509342.png)
