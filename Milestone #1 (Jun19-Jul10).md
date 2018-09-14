### Milestone #1 (Exploratory): transfer in text classification task for StackExchange (SE) data 

Start date: Jun 19, 2018

End date: Jul 10, 2018

Status: Completed

Lead: Yunshu Du

Advisor: Nidhi Hegde

### Goal:
Build a good baseline for (short) text classification for StackExchange (SE) data

Understand how different model components affect performance

### Outcomes:
Built three baseline model for text classification (single-label): CNN, RNN, Fasttext, FT)

Emperically showed that using word embedding pretrain with fineture is the best way to train classification 

### Previous:
N/A

### Next:
Milestone #2

### Steps:
**Build baseline models**:

  - Embedding type: w2v, GloVe, (optional: BagofWords, skip-gram)
  
    - random initialization, pre-train in SE from scratch, use existing pretrained vectors  
      
  - Model type: 
        
    - CNN, RNN, FastText(FT, avg-pooling)
  
  - Training methods: 
      
    - Finetune, Freeze 

  - Add precise/recall as performance measurements in addition to accuracy 
    
  - Test on benchmark datasets, some examples here: <https://machinelearningmastery.com/datasets-natural-language-processing/>
  
    
| Model | Emb_layer |  pretrain_method  | Finetune |
|-------|:---------:|:-----------------:|----------|
| FT   |    Rand   |        N/A        | ~~Yes~~      |
|       |           |                   | No       |
|       |    W2V    |   Self-pretrain   | ~~Yes~~      |
|       |           |                   | No       |
|       |           |  Google-pretrain  | ~~Yes~~      |
|       |           |                   | No       |
|       |   GloVe   |   Self-pretrain   | ~~Yes~~      |
|       |           |                   | No       |
|       |           | Stanford-pretrain | ~~Yes~~      |
|       |           |                   | No       |
|-------|:---------:|:-----------------:|----------|
| ~~CNN~~   |    ~~Rand~~   |        ~~N/A~~        | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |    ~~W2V~~    |   ~~Self-pretrain~~   | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |           |  ~~Google-pretrain~~  | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |   ~~GloVe~~   |   ~~Self-pretrain~~   | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |           | ~~Stanford-pretrain~~ | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|-------|:---------:|:-----------------:|----------|
| ~~RNN - Vanilla~~  |    ~~Rand~~   |        ~~N/A~~        | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |    ~~W2V~~    |   ~~Self-pretrain~~   | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |           |  ~~Google-pretrain~~  | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |   ~~GloVe~~   |   ~~Self-pretrain~~   | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|       |           | ~~Stanford-pretrain~~ | ~~Yes~~      |
|       |           |                   | ~~No~~       |
|-------|:---------:|:-----------------:|----------|
| RNN- LSTM  |    Rand   |        N/A        | Yes      |
|       |           |                   | No       |
|       |    W2V    |   Self-pretrain   | Yes      |
|       |           |                   | No       |
|       |           |  Google-pretrain  | Yes      |
|       |           |                   | No       |
|       |   GloVe   |   Self-pretrain   | ~~Yes~~      |
|       |           |                   | No       |
|       |           | Stanford-pretrain | Yes      |
|       |           |                   | No       |

**Test transferrabilidy in SE data**:
      
  - Transfer across categories:
  
    - Layer-by-layer transfer
    
      - problem: there is no "layer" in the CNN --- the architecture only has one layer with multiple filfers, each layer use the same emb as input
      
    - identify transferrable nodes by looking inside the model (loss? weight distribution?)
    
      - Google visualize CNN features <https://distill.pub/2017/feature-visualization/>
      
    - are more "similar" categories more transferrable 

  - Transfer within one category:
  
    - possible problems: not enough training samples

### Success Criteria:
Build a baseline model compareable to <https://arxiv.org/pdf/1805.09843.pdf>

Identify some patterns for transferrability in SE data 


### Updates:
**Summary on CNN & RNN_vanilla embedding_type tests: w2v and GloVe**

![Summary_CNN_RNN_vanilla](images/2018-06-27_CNNRNN_emb_batchsize128_evalevery10.png) 

**Summary on FastText, compared with CNN/RNN rand-finetune**

![Summary_FT](images/2018-06-29_FT_emb_batchsize128_evalevery10.png)

_Observation (add plot on loss here): FT does not overfit like CNN/RNN (which needs early stop)_

1.Overall, Finetune is better than Freeze


  - CNN:

  ![train-w2v-finetune-freeze](images/train-w2v-finetune-freeze.png =160x120)
  ![google-w2v-finetune-freeze](images/google-w2v-finetune-freeze.png =160x120)
  ![train-glove-finetune-freeze](images/train-glove-finetune-freeze.png =160x120)
  ![stanford-glove-finetune-freeze](images/stanford-glove-finetune-freeze.png =160x120)

  - RNN-Vanilla:

  ![vanilla-train-w2v-finetune-freeze](images/vanilla-train-w2v-finetune-freeze.png =160x120)
  ![vanilla-google-w2v-finetune-freeze](images/vanilla-google-w2v-finetune-freeze.png =160x120)
  ![vanilla-train-glove-finetune-freeze](images/vanilla-train-glove-finetune-freeze.png =160x120)
  ![vanilla-stanford-glove-finetune-freeze](images/vanilla-stanford-glove-finetune-freeze.png =160x120)
  
  - CNN vs. RNN-Vanilla
  
    - Concern: Initial Accuracy for using pre-trained emb (Google, Stanford) should be about the same under each setting --- unless the rest of the model is initiallized differently
      (the percentage might be just "by chance"?) --- multiple runs and average. 
      For self-pretrained emb, could have variance as those training methods are not static.
    - RNN has higher "initial accuracy" for random initial compare to CNN (because of the model structure?).
    - Asymptotic improve rates are small for finetune methods in both CNN&RNN --- for a simple task here, random initial is just fine. But do provide some jumpstart. 
    - W2V seems to work better in CNN (highest accuracy occurs in both train-w2v and google-w2v); similarly, GloVe seems better for RNN.
  
|                          | CNN         |           |            |                |             |          |           |            |                |             |
|--------------------------|-------------|-----------|------------|----------------|-------------|----------|-----------|------------|----------------|-------------|
|                          | Freeze      |           |            |                |             | Finetune |           |            |                |             |
|                          | rand        | Train-w2v | Google-w2v | Stanford-glove | Train-glove | rand     | Train-w2v | Google-w2v | Stanford-glove | Train-glove |
| Initial Accuracy         | 0.13%       | 3.01%     | 6.38%      | 16.87%         | 9.86%       | 0.12%    | 13.09%    | **19.81%**     | 12.28%         | 1.43%       |
| Final Accuracy           | 59.76%      | 90.27%    | 90.04%     | 92.45%         | 91.48%      | 94.36%   | **94.87%**    | 94.41%     | 93.06%         | 93.72%      |
| (neg) Steps to Threshold (80%) | n/a         | -3        | -4         | **-2**             | -3          | -6       | **-2**        | -3         | -3             | -3          |
|                          |             |           |            |                |             |          |           |            |                |             |
| Jumpstart improve rate   | n/a         | 2.88%     | 6.25%      | 16.74%         | 9.73%       | n/a      | 12.97%    | **19.69%**     | 12.16%         | 1.31%       |
| Asymptotic improve rate  | n/a         | 30.51%    | 30.27%     | **32.69%**        | 31.71%      | n/a      | 0.50%     | 0.04%      | -1.30%         | -0.64%      |
|                          |             |           |            |                |             |          |           |            |                |             |
|--------------------------|-------------|-----------|------------|----------------|-------------|----------|-----------|------------|----------------|-------------|
|                          | **RNN_Vanilla** |           |            |                |             |          |           |            |                |             |
|                          | Freeze      |           |            |                |             | Finetune |           |            |                |             |
|                          | rand        | Train-w2v | Google-w2v | Stanford-glove | Train-glove | rand     | Train-w2v | Google-w2v | Stanford-glove | Train-glove |
| Initial Accuracy         | 11.79%      | 16.62%    | 16.14%     | 12.88%         | 13.14%      | 13.04%   | 10.77%    | 17.26%     | **19.41%**         | 14.43%      |
| Final Accuracy           | 60.02%      | 87.76%    | 87.02%     | 88.49%         | 90.50%      | 92.85%   | 93.22%    | **93.72%**     | 93.24%         | 92.33%      |
| (neg) Steps to Threshold (80%) | n/a         | -4        | -6         | -5             | **-3**          | -7       | -4        | -5         | -5             | **-3**          |
|                          |             |           |            |                |             |          |           |            |                |             |
| Jumpstart improve rate   | n/a         | 4.83%     | 4.35%      | 1.10%          | 1.35%       | n/a      | -2.28%    | 4.22%      | **6.37%**          | 1.38%       |
| Asymptotic improve rate  | n/a         | 27.74%    | 27.00%     | 28.47%         | **30.48%**      | n/a      | 0.37%     | 0.87%      | 0.39%          | -0.52%      |

2.We evaluate CNN finetune and freeze method by looking at statics of: Jumpstart, Asymptoic Performance, and Time (step) to threshold (we manually pick 80% accuracy as threshold)

  - Transfer when embedding layer is freezed (i.e., fixed after pre-train, not trainable)?
  
    - Train a GloVe model using SE data from scratch then transfer had the best result
    - There was no _negative_ transfer because the freeze_baseline (rand) was bad
    - also note that freezed pretrain w2v and glove achieved comparable results as finetuned rand
  
  ![Freeze](images/2018-06-20_freeze_stat_EmbCompare_batchsize128_evalevery10.png =463x491)

  - Transfer when embedding layer is finetuned (i.e., keep training after pre-train)?
  
    - Both Google pretrained w2v and self-pretrained w2v had good transfer, but w2v is only slightly better than GloVe
    - observe _negative_ transfer in both GloVe embedding (asymptotic performance worse than finetune_baseline (rand)), Stanford pretrain GloVe had worse negative transfer
    
  ![Finetune](images/2018-06-20_finetune_stat_EmbCompare_batchsize128_evalevery10.png =463x491) 
  ![Finetune: negative trasnfer](images/2018-06-20_finetune_neg_EmbCompare_batchsize128_evalevery10.png =160x120) 


