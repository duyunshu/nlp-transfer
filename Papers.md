### A list of paper related to NLP transfer

- **How Transferable are Neural Networks in NLP Applications?**
   <https://arxiv.org/pdf/1603.06111.pdf >
   Authors look into the transferability of 6 datasets for 2 NLP tasks. Factors examed: 
     semantically similar data set vs. not similar dataset;
     layer-by-layer transfer;
     learning rate;
     when to transfer;
     combine initialization + multi-task;
     _Q: the way the authors defined "semantically" similar is not as appropriate --- they are just similar "task", not similar "sementics"?_
     
     
- **Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms**
   <https://arxiv.org/pdf/1805.09843.pdf >
   Authors look into 17 datasets for 4 NLP tasks, claim that Simple Word Embedding Models (SWEM) 
   sometimes outperform RNN?CNN based models. Various analysing in the importance of each component
   of the models. e.g., how important is word sequence, how important is semantic meaning, etc. 
   (related: **Simple Baseline for Visual Question Answering**
   <https://arxiv.org/pdf/1512.02167.pdf>
    for visual question answering problem, bag-of=words outperform CNN/RNN based model)
       
- **Strong Baselines for Neural Semi-supervised Learning under Domain Shift**
   <https://arxiv.org/abs/1804.09530>   
   Another paper stated that classic, baseline models (with some addition) out-perform state-of-the-art under sentiment analysis
   
- **Learning to select data for transfer learning with Bayesian Optimization**
   <http://aclweb.org/anthology/D17-1038>
   Authors propose using Bayesian optimizer to perform Source task selection given a target task 
   Another way to look at it: tread as active learning, ask target task to pick a suitable source 
   [related presentation](https://www.slideshare.net/SebastianRuder/transfer-learning-for-natural-language-processing)
   
   
- **Learning what to share between loosely related tasks**
   <https://arxiv.org/pdf/1705.08142.pdf>
   Authors proposed a "SLUICE NETWORK" that can learn what layer/parameters to share between main task and auxiliary task during the course of training.
   This is the MTL setting under transfer.
   However, I feel the amoung of performance improvement is very little compare to the increase of learning efforts. 
   
- **A Sensitivity Analysis of (and Practitioners� Guide to) Convolutional Neural Networks for Sentence Classification**
   <https://arxiv.org/pdf/1510.03820.pdf>
   Authors looked at how sensitive each component is for CNN text classification, could be useful when considering transfer. 
   
- **NLP's ImageNet moment has arrived**
   <https://thegradient.pub/nlp-imagenet/>
   A blog post summarize resent NLP and word embedding techqniues, proposed what an "ImageNet for NLP" might look like

   

   
   
### random thoughts 
PBT for source selection for a given target [Population Based Training (PBT)](https://arxiv.org/pdf/1711.09846.pdf)

Look at softmax layer for similarity, map back to sentence (emb or semantic) similarity

How much improvement is a "real" improvement? Seems a lot of paper just report <1% performance increase for their method and claim that the method "significantly outperformed" previous ones. 
While none of which provided a statisticlly significience test. 
   


   