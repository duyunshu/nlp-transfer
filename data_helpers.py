
import re
import itertools
import os
import string
import requests

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
stops = set(stopwords.words("english"))


def pause():
    int(input('enter a num to cont...'))


def download_sick(f):
    response = requests.get(f).text
    lines = response.split("\n")[1:]
    lines = [l.split("\t") for l in lines if len(l) > 0]
    lines = [l for l in lines if len(l) == 5]
    df = pd.DataFrame(lines, columns=["idx", "sent_1", "sent_2", "sim", "label"])
    df['sim'] = pd.to_numeric(df['sim'])
    return df


def download_sick_all():
    """Download and load sick datasets"""
    sick_train = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_train.txt")
    sick_dev = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_trial.txt")
    sick_test = download_sick("https://raw.githubusercontent.com/alvations/stasis/master/SICK-data/SICK_test_annotated.txt")
    sick_all = sick_train.append(sick_dev).append(sick_test)
    num_train = sick_train.append(sick_dev).shape[0]
    # print(sick_dev.shape)
    # print(sick_test.shape)
    return sick_all, num_train


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"<p>", " ", string)
    return string.strip().lower()


def tag2index(tags, vocab):
    return [vocab.index(tag) for tag in tags]


def multihot(index, vocab_size):
    result = np.zeros(vocab_size)
    result[index] = 1
    return result


def create_multihot(y):
    vocab = list(set(np.concatenate(y, axis=0)))
    vocab_size = len(vocab)
    label_index = [tag2index(tags, vocab) for tags in y]
    return [multihot(index, vocab_size) for index in label_index]


def load_SE_data_and_labels(directory, multi_label=False):
    """
    Loads SE data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text = []
    y = []
    num_classes = 0
    for filename in os.listdir(directory):
        title = []
        tag = []
        if str(filename).endswith(".xml"):
            print("loading ", filename, "...")
            tree = ET.parse(os.path.join(str(directory), str(filename)))
            root = tree.getroot()
            for child in root:
                # only look at questions
                if child.attrib['PostTypeId'] == '1':
                    if multi_label:
                        tmp=re.split('[< >]', child.attrib['Tags'])
                        while '' in tmp:
                            tmp.remove('')
                        #tmp=' '.join(tmp)
                        tag.append(tmp)
                    else:
                        tag.append(num_classes)
                    # look at title for now
                    title.append(child.attrib['Title'].strip())

            num_classes += 1
        x_text += title
        y += tag

    # Split by words and clean
    x_text = [clean_str(sent) for sent in x_text]
    # Generate ont-hot labels if not multilabel
    if not multi_label:
        y = np.eye(num_classes)[y]
    # else get a vocab for y and build multi-one-hot
    else:
        y = create_multihot(y)
        y = np.asarray(y)

    return [x_text, y]


def load_SICK_data_and_labels(sick_all, sent_col='sent_1', sent_label='CONTRADICTION'):
    """
    Loads SICK data from df, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    x_text = sick_all[sent_col].values.tolist()
    y = sick_all['label'].values.tolist()
    y = [str(label).rstrip() for label in y]

    # convert label to 0, 1...etc (if more)
    # intuitively, sentences that are "contradict" *should not* transfer
    # sentences that are "entail" *should* transfer
    # use "neutral" as background
    # the task is classify "CONTRADICTION" (or "ENTAILMENT") vs. "NEUTRAL"
    # discard x!= sent_label
    if sent_label=="CONTRADICTION":
        discard_index = [i for i, x in enumerate(y) if x == "ENTAILMENT"]
        y = [label for index, label in enumerate(y) if index not in discard_index]
        x_text = [sent for index, sent in enumerate(x_text) if index not in discard_index]
    elif sent_label=="ENTAILMENT":
        discard_index = [i for i, x in enumerate(y) if x == "CONTRADICTION"]
        y = [label for index, label in enumerate(y) if index not in discard_index]
        x_text = [sent for index, sent in enumerate(x_text) if index not in discard_index]
    else:
        raise ValueError("sentence label %s not supported: CONTRADICTION or ENTAILMENT" %sent_label)
    print("training on %s, cur y label:"%sent_label)
    print(set(y))

    # check class balance
    classcount = Counter(y)
    print("before: ", classcount)
    num_duplicate = classcount['NEUTRAL']//classcount[sent_label]
    dup_index = [i for i, x in enumerate(y) if x == sent_label]
    y_duplicate = num_duplicate * [label for index, label in enumerate(y) if index in dup_index]
    x_duplicate = num_duplicate * [sent for index, sent in enumerate(x_text) if index in dup_index]
    y += y_duplicate
    x_text += x_duplicate
    print("after: ", Counter(y))
    assert int(len(y)) == int(len(x_text)), "num sentences does not match num labels!"

    # convert string labels to int
    # NEUTRAL is always 0, the other is always 1
    y = [0 if label == "NEUTRAL" else 1 for label in y ]
    num_classes=len(set(y))

    # clean
    x_text = [clean_str(sent) for sent in x_text]
    # Generate ont-hot labels
    y = np.eye(num_classes)[y]

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    # changed to only cound epoch as a step,
    # run enough epochs s.t. roughly all training data will be covered
    num_epochs = num_batches_per_epoch * 5
    num_batches_per_epoch = 1
    print("====trainin for %d epochs,"
          "each epoch has %d num batches,"
          "each batch is size %d====" %(num_epochs,
                                    num_batches_per_epoch,
                                    batch_size))
    print("====total steps: %d====" %(num_epochs * num_batches_per_epoch))

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
