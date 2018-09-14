from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import subprocess

import data_helpers


def pause():
    int(input("enter a num to cont..."))


def train_w2v(vocab_dict, text_index, emb_dim):
    # tokenized
    text = [[vocab_dict[w] for w in s] for s in text_index]

    # train a w2v model from scratch using StackExchange data
    model = Word2Vec(text, min_count=1, size=emb_dim)
    print("==========", model, "==========")

    # If we don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # save model
    model.save('./saved_model/model.bin')

    # get embedding weights
    embedding_weights = {key: model[word] if word in model
                                          else np.random.uniform(-0.01, 0.01, model.vector_size)
                         for key, word in vocab_dict.items()}
    # reshape as matrix
    embedding_weights = [v for _, v in embedding_weights.items()]
    embedding_weights = np.stack(embedding_weights, axis=0)

    return embedding_weights


def load_pretrain_w2v(vocab_dict, emb_dim, filename):
    # load Google pretrained w2v model
    model = KeyedVectors.load_word2vec_format(filename, binary=True)

    # get embedding weights
    embedding_weights = {key: model[word] if word in model
                                          else np.random.uniform(-0.01, 0.01, model.vector_size)
                         for key, word in vocab_dict.items()}
    # reshape as matrix
    embedding_weights = [v for _, v in embedding_weights.items()]
    embedding_weights = np.stack(embedding_weights, axis=0)

    return embedding_weights


def train_glove(vocab_dict, text_index, emb_dim):
    # tokenized
    text = [[vocab_dict[w] for w in s] for s in text_index]
    # flatten into whitespace seperated txt file
    text = np.concatenate(np.stack(text, axis=0))
    index = np.argwhere(text=='<UNK>')
    text = np.delete(text, index)

    # write to file, name as 'text8' to be consistant with glove folder
    with open("./glove/text8", "w") as outfile:
        for word in text:
            outfile.write(str(word)+' ')

    # train a glove model from scratch using StackExchange data
    # call the glove folder, ./demo.sh
    subprocess.call('./glove/demo.sh ' + str(emb_dim), shell=True)

    # retrive trained vectors.txt
    glove_input_file = './glove/vectors.txt'
    word2vec_output_file = glove_input_file + ".w2v"
    # convert glove to w2v format
    glove2word2vec(glove_input_file, word2vec_output_file)

    # load sekf-pretrained GloVe model
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # get embedding weights
    embedding_weights = {key: model[word] if word in model
                                          else np.random.uniform(-0.01, 0.01, model.vector_size)
                         for key, word in vocab_dict.items()}
    # reshape as matrix
    embedding_weights = [v for _, v in embedding_weights.items()]
    embedding_weights = np.stack(embedding_weights, axis=0)

    return embedding_weights


def load_pretrain_glove(vocab_dict, emb_dim, filename):
    # concate filename for corresponding embedding_dim
    filename += "glove.6B." + str(emb_dim) + "d.txt"
    glove_input_file = filename
    word2vec_output_file = filename + ".w2v"
    # convert glove to w2v format
    glove2word2vec(glove_input_file, word2vec_output_file)

    # load Stanford pretrained GloVe model
    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # get embedding weights
    embedding_weights = {key: model[word] if word in model
                                          else np.random.uniform(-0.01, 0.01, model.vector_size)
                         for key, word in vocab_dict.items()}
    # reshape as matrix
    embedding_weights = [v for _, v in embedding_weights.items()]
    embedding_weights = np.stack(embedding_weights, axis=0)

    return embedding_weights
