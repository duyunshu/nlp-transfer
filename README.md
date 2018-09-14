# nlp-transfer
Borealis AI Internship Project (Summer 2018)

This is Yunshu's Borealis AI internship project
Emperical study on transfer learning for text classification**

**CNN text classification is modified from
<https://github.com/dennybritz/cnn-text-classification-tf>
and
<https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras>**

**RNN text classification is modified from
<https://github.com/roomylee/rnn-text-classification-tf>**

**FastText text classification is modified from
<https://github.com/brightmart/text_classification>**

**Sentence similarity measurement is modified from
<https://github.com/nlptown/nlp-notebooks/blob/master/Simple%20Sentence%20Similarity.ipynb>**

**By: Yunshu Du

Sept, 2018**


## Requirements

- Python 3
- Tensorflow > 1.7
- Numpy

## Training

Train once (in default config):

```bash
python3 ./train.py
```

Train with all config combinations using CNN model:

[StackExchange Data](https://archive.org/details/stackexchange)

```bash
./run_se_cnn.sh
```

[SICK Data](http://clic.cimec.unitn.it/composes/sick.html)

```bash
./run_sick_cnn.sh
```

Can also run RNN and FastText in similar way (.sh scripts)

check FLAGS defined in train.py for detailed description of configurations

check [presentation slides](https://bitbucket.org/rbcmllab/nlp-transfer/src/master/Transfer%20in%20NLP%20(Yunshu%20internship).pdf) for experiment design details

## Sentence similarity

Under folder "sentence_similarity"

Compute similarity using [Universal Sentence Encoder](https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/2)

```bash
python3 USE.py
```

Compute similarity using [other baseline methods](http://nlp.town/blog/sentence-similarity/)

```bash
python3 sent_sim.py
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

