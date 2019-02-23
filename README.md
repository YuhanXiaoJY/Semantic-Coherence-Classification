# Semantic Coherence Classification 

**Xiao Yuhan** 

------

[TOC]

  ### 任务描述  

给定一个段落，判断该段落是否连贯（语义、语法两方面）



------

### 一、文件说明  

我尝试了两类方法，一种是不使用学习、基于规则的判断法，另一种则是使用了机器学习。  

第一类方法做得很简单，效果不是很好。之所以用这个非机器学习法来做，是因为我觉得基于规则的判断很重要，很想将规则提取出来然后用SVM进行学习。可惜由于时间紧迫，并未能把特征搞定，SVM也就没有与规则相关的合适输入，没能将规则特征与机器学习结合起来。    

第二类方法里面，我主要尝试了SVM，CNN和LSTM。都基于word2vector来做。目前准确率最高的是CNN，正确率为65.7%左右  　　 

result.txt为测试集的label结果，是以CNN模型输出的  

 

#### 1、代码文件  

##### 1.1 机器学习代码文件：  

- dataProcessing.py：数据预处理  

- Word2Vec.py：根据预处理好的数据生成词向量，构建出词向量矩阵，然后处理出SVM所需要的段落向量，以及CNN和LSTM输入所需要的词索引（单词在词向量矩阵中的索引）  

- SVM_wv.py：使用SVM，来对输入的段落向量进行训练  

- CNN.py：使用CNN进行深度学习  

- Bi_LSTM.py: 使用双向LSTM进行深度学习  

##### 1.2 基于规则的代码文件：
- relevData.py：数据预处理  
- relevHandler.py：根据预处理好的数据进行连贯性判断   



#### 2、数据文件（代码运行所需要的以及生成的文件）  

##### 2.1 预处理文件  

- ```python
  dataProcessing.py
  需要：
  Data/rawData/train_data	&&	Data/rawData/valid_data	&&	Data/stopwordsList/stopwords.txt
  输出：
  Data/processedData/train_processed.txt	&&	Data/processedData/train_label.txt
  Data/processedData/valid_processed.txt	&&	Data/processedData/valid_label.txt
  ```

- ```
  Word2Vec.py
  需要：
  Data/word2vector/raw_word2vec.model（该部分代码自己生成的model文件）
  && Data/processedData/train_processed.txt	&&	Data/processedData/valid_processed.txt
  输出：
  Data/word2vector/raw_word2vec.model	&&	Data/word2vector/wv_for_weight_matrix.npy
  Data/word2vector/train_sentc_word2vec.npy	
  Data/word2vector/valid_sentc_word2vec.npy
  Data/word2vector/train_index.npy
  Data/word2vector/valid_index.npy
  ```

- ```
  relevData.py
  需要：
  Data/rawData/valid_data	&&	Data/stopwordsList/stopwords.txt
  输出：
  Data/processedData/valid_relevance.txt	&&	Data/processedData/valid_label.txt
  ```



##### 2.2 机器学习文件  

- ```
  SVM_wv.py
  需要：
  Data/word2vector/train_sentc_word2vec.npy	    Data/word2vector/valid_sentc_word2vec.npy
  ```

- ```
  CNN.py
  需要：
  Data/word2vector/wv_for_weight_matrix.npy
  Data/word2vector/train_index.npy	&&	Data/word2vector/valid_index.npy
  Data/processedData/train_label.txt	&&	Data/processedData/valid_label.txt
  ```

- ```
  Bi_LSTM.py
  需要：
  Data/word2vector/wv_for_weight_matrix.npy
  Data/word2vector/train_index.npy	&&	Data/word2vector/valid_index.npy
  Data/processedData/train_label.txt	&&	Data/processedData/valid_label.txt
  ```



##### 2.3  基于规则的代码文件  

- ```
  relevHandler.py
  需要：
  Data/processedData/valid_relevance.txt	&&	Data/processedData/valid_label.txt
  ```



------

###  二、代码说明  

####  1、预处理  

##### 1.1 dataProcessing.py（针对机器学习部分）  

- 函数```preprocess(dataType)```  

  - 完成预处理工作  
  - 读入train_data（valid_data）文件，将单词按空格分割，去除停用词，去除单纯的数字，抽取词干，将单词格式形式化，并化为小写字母  
  - 将处理好的语篇以及label写入train_processed.txt以及train_label.中txt（对valid_data也是如此操作）  

- 函数```read_stopwords(filePath)```  

  - 从Data/stopwordsList/stopwords.txt中读取停用词（自己设定的一些停用词），存到一个列表中  


##### 1.2 Word2Vec.py（针对机器学习部分）  

- 函数```SGM()```  

  - 调用gensim的word2vec库根据Data/processedData/train_processed.txt生成词向量，存到Data/word2vector/raw_word2vec.model中  
  -   使用的是Text8Corpus，用skim-gram model来生成词向量  

- 函数```fcount(line)```  

  - 计算一个语篇中每个单词的频率，存到一个字典中  

- 函数```wordVecHandler(dataType)```  

  - 以函数SGM ()生成的raw_word2vec.model为输入，将其所有的词向量取出来放到wv_for_weight_matrix.npy中，作为CNN和LSTM的词向量矩阵  
  - 计算SVM所需要的语篇段向量。实现方法比较直观，把每个单词的词向量乘以其在fcount()中算出的词频，然后累加起来作为语篇段向量。存到/word2vector/train_sentc_word2vec.npy和Data/word2vector/train_sentc_word2vec.npy中  
  - 最后调用函数CL_wvHandler  

- 函数CL_wvHandler(dataType)  

  - 为每个语篇形成一个索引列表，每个单词按顺序对应一个索引，索引到词向量矩阵中该单词的词向量
  - 将所有语篇的索引列表输出到Data/word2vector/train_index.npy和Data/word2vector/valid_index.npy中  

- 函数testwv(word)  

  - 用于检验word vector的质量，与其他部分没有太大关系  


##### 1.3 relevData.py（针对基于规则的部分）  

- 与dataProcessing.py大致相同，区别在于处理好的数据中仍包含```!  ?  .```等句尾标识符，这样方便relevHandler.py进行句子的划分  


#### 2、基于规则的部分  

- relevHandler.py  

  - **思想：**

    这部分只是简单的尝试。在这里主要是考虑句子之间相关度对连贯性的影响。一般情况下，很多连贯的句子都是相关的。因此，在一个语篇中，每两个句子之间给出一个相关度的score，放到该语篇对应的list中去。假设一个语篇有n个句子，那么它对应的list就有n-1个score。在此基础之上，还需要给出对连贯的定义。我对它的定义是：只要存在不连贯的句子，那么这个语篇就是不连贯的（类似于木桶原理）。因此，提取出每个list中的最小score，这个最小score就决定了这个语篇是否连贯。给定一个阈值，当score小于该阈值时，判定它为不连贯，否则连贯。

    关于每两个句子之间相关度的score，这里的处理比较简单，分为四类进行处理。一类是两个句子之间存在```also  about what which when```这类极弱相关词的重合，那么每存在这样一个重合，score+0.2。第二类是存在代词，即```he she it we our him his```这种类型的词，定义为弱相关，每存在这样一个重合，score+0.4。第三类是强相关词，如果有重合的词，且不属于上面的类型，那么score+2。最后是不相关词，用于平滑处理，防止两个句子一个重合的单词都没有（这是很可能出现的，预处理中将停用词都删了），score+0.1。

  - **实现：**  

    - 函数```getData```
      - 读入预处理数据，分句  
    - 函数```scorePara```  
      - 首先给出一个语篇中句子之间相关度的score，然后提取最小的score中，放到一个list中
    - 函数```test```  
      - 对语篇连贯性进行判断，并与验证集的label进行比较，得出正确率  
    - 函数```saveScore```  
      - 将每个语篇的score list保存  

  - **结果：**  

    准确度为53.66%，由于提取的规则太过简单，效果不好   

  - **未来得及实现的部分（想法）**  

    - 我之所以想尝试这样一种非深度学习法，是因为如果直接调用深度学习常用的库的话，我们做的其实就是输入，修改一些参数，然后等待输出。目前广泛使用的很多model具有普遍的适用性，但也因此对特定问题的处理能力弱化了。我个人是希望能将规则和机器学习结合起来，做一种启发式机器学习（其实就是提取好特征，然后用SVM）。目前是提取了特征了，不过看起来远远不够，一个文本的连贯性显然不只受句子之间的相关度影响，一方面是形式上，句子的结构是否正常（在这里连词是个很强的判定），另一方面是语义上，很有可能两个句子形式不一样且甚至没有重合的单词，但它在语义上就是通顺的。提取特征需要把这些都考虑到，然后形成一个综合的score。限于时间（期中季），我只实现了相关度的一部分，参数也不够好，因此结果不是很理想，也没有好意思扔给SVM去学。但我觉得这方面的提升应该很大，特征OK的话效果应该是超过深度学习的大部分模型的，后续有时间会尝试改善一下。  
    - 在这里我是根据类似于木桶原理那种想法判断语篇是否连贯。但还有一种方法也比较合理：可以建立一个点图，一个语篇就是一个图，一个句子对应一个点。如果两个句子之间是连贯的，那就把这两个点连起来。最后看这个图是否是连通图，如果连通则说明连贯，否则不连贯。该部分有想法，但未具体实现。  



#### 3、机器学习部分  

##### 3.1 基于语篇段向量的SVM    

- 选择SVM的原因：
  - 数据是带标签的，也即咱们这个作业是监督式学习，对机器学习来说比较友好  
  - 这是个二分类任务，给出连贯还是不连贯就好（二分类其实也可以用回归分析，根据一个阈值来进行二分就好了）
  - 做word2vec部分时得到了词向量，用简单方法的话可以很方便地做出语篇段向量。抱着试一试的态度把语篇段向量喂给SVM  

- 实现：

  基于sklearn库的SVM包  

  - 输入：每个语篇的段向量，以及对应的label
  - 输出：准确率  

- 结果：

  准确度最高的一次为57.19%，但最终版本基于的词向量和57.19%的那次并不相同。基于最终版本词向量的结果：

  参数nu=0.5时为56.23%， 参数nu=0.4时为55.75%    


##### 3.2 CNN模型  

- 基于keras提供的CNN相关函数实现，使用adam算法，batch大小为64，epoch设为6，输入维数54841，输出维数100。隐藏层的激活函数使用relu，最后输出时sigmoid一下  

- 结果  ：

  | epoch | 训练集的accuracy（训完最后一个batch对应的准确度） | 验证集的accuracy |
  | ----- | ------------------------------------------------- | ---------------- |
  | 1     | 57.47                                             | 63.56            |
  | 2     | 64.61                                             | 64.09            |
  | 3     | 66.87                                             | 64.30            |
  | 4     | 68.69                                             | 64.97            |
  | 5     | 70.94                                             | 65.55            |
  | 6     | 72.95                                             | 65.69            |

  当epoch比较大时，后面过拟合的现象比较严重。由于之前没有做过机器学习，搞L2正则化的时候出了各种一些奇怪的bug……在ddl的逼迫下，最后只能忍痛放弃  


##### 3.3 双向LSTM  

- 选择LSTM的原因  

  - 语篇连贯性存在继承关系、传递关系，用LSTM比较适合处理这种问题　　
  - 大名鼎鼎，想尝试做一下　　

- 实现上也是基于keras提供的库函数解决的，双向LSTM，使用adam算法，epoch为5，batch size为64

  输入维数54841，输出维数100，dropout为0.3，recurrent_dropout为0.3    

- 结果：  

  | epoch | 训练集的accuracy（训完最后一个batch对应的准确度） | 验证集的accuracy |
  | ----- | ------------------------------------------------- | ---------------- |
  | 1     | 60.30                                             | 63.31            |
  | 2     | 63.61                                             | 64.38            |
  | 3     | 65.61                                             | 63.96            |
  | 4     | 66.08                                             | 64.96            |
  | 5     | 67.47                                             | 63.19            |

  训练极慢！几乎花了CNN10倍的时间。  

  和CNN有同样的问题，过拟合现象比较严重。

  改进的话可以add一些LSTM，然后对dropout进行调参。  



------

### 三、可以进行的改进以及思考  

- 关于词向量的部分  
  - 可以看到，我喂给SVM的向量和CNN、LSTM的并不一样。给SVM的是一个语篇段向量，给CNN、LSTM的就是一串词向量。这两种方式其实都比较粗暴，对于这种连贯性的分析，直观上应当是对句子的考量，而非单词或者宽泛的文本。我在学习相关的知识时，看到一篇处理sentence embedding的论文，对sentence的实现既不复杂又优美：```	A Simple but Tough-to-Beat Baseline for Sentence Embeddings```。我也找到了它的代码实现：https://github.com/PrincetonML/SIF。本来想使用它的sentence embedding并做一些调整，最后因为时间问题不了了之。个人感觉如果能把这个sentence embedding做好，甚至不需要对规则方面的关联，就能达到不错的效果  
- 关于机器学习部分  
  - 过拟合的问题比较严重，可以考虑用L2正则化  
  - 参数方面如果能有更多时间调一调，效果应该更好  
  - 看了助教给的论文后想尝试用attention来做，不过没能有时间
- 关于规则部分  
  - 这部分的思考在上面已经提到了，不再赘述  



------

### 参考文献  

【1】A Simple but Tough-to-Beat Baseline for Sentence Embeddings

【2】Hierarchical Attention Network for Document Classification  

【3】Convolutional Neural Networks for Sentence Classification

【4】https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py  

【5】https://github.com/PrincetonML/SIF

【6】Dan Jurafsky的Speech and Language Processing： chapter7  

