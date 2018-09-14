"""
This is Yunshu's Borealis AI internship project
Emperical study on transfer learning for text classification

CNN text classification is modified from
https://github.com/dennybritz/cnn-text-classification-tf
and
https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras

RNN text classification is modified from
https://github.com/roomylee/rnn-text-classification-tf

FastText text classification is modified from
https://github.com/brightmart/text_classification

Sentence similarity measurement is modified from
https://github.com/nlptown/nlp-notebooks/blob/master/Simple%20Sentence%20Similarity.ipynb

By: Yunshu Du
Sept, 2018
"""

import os
import time
import datetime
import csv
import gensim
import glob
import requests

import data_helpers
import word2vec

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec, KeyedVectors
from collections import defaultdict

from text_cnn import *


# Parameters
# ==================================================

# To train on SE (for text classification) or SICK (for sentence similarity)
tf.flags.DEFINE_boolean("train_sick", False, "train sentence similarity. When set to True, train SICK dataset and compare similarity instead of SE data. (default: False)")
tf.flags.DEFINE_string("sent_col", "sent_1", "Which sentence column to use for training: sent_1 or sent_2. (default: sent_1)")
tf.flags.DEFINE_string("sent_label", "CONTRADICTION", "train which sentence label as baseline: 'CONTRADICTION' or 'ENTAILMENT' (default: CONTRADICTION)")
tf.flags.DEFINE_boolean("transfer", False, "to perform transfer or not after training for a sentence (default: False)")

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_path", "./data/StackExchange/", "Data source location.")
tf.flags.DEFINE_string("google_w2v", "./saved_model/GoogleNews-vectors-negative300.bin", "pretrained w2v model on Google News")
tf.flags.DEFINE_string("glove_6b", "./saved_model/glove.6B/", "the smallest pretrained Stanford GloVe model")
tf.flags.DEFINE_string("glove_6b_w2v", "./saved_model/glove.6B/glove.6B.300d.txt.w2v", "the smallest pretrained Stanford GloVe model converted to w2v format")
tf.flags.DEFINE_string("glove_840b", "./saved_model/glove.840B/glove.840B.300d.txt", "the largest pretrained Stanford GloVe model")
tf.flags.DEFINE_string("glove_840b", "./saved_model/glove.840B/glove.840B.300d.txt.w2v", "the largest pretrained Stanford GloVe model converted to w2v format")
tf.flags.DEFINE_boolean("multi_label", False, "use single or multi-label. If single, predict only the high level StackExchange categories; if multi, predict tags inside one category (default: False)")

# Model Hyperparameters
tf.flags.DEFINE_integer("num_iter", 3, "number of iterations to train to calculate error bar (default: 3)")
tf.flags.DEFINE_string("model_type", "cnn", "choose which model to train on: fasttext (i.e., one-layer softmax), cnn, rnn (default: cnn)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# CNN
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# RNN
tf.flags.DEFINE_integer("hidden_size", 128, "RNN hidden layer size (Default: 128)")
tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell type: vanilla, lstm (Default: lstm)")
# Embedding mode
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300. Note to be able to use GloVe, this must be 300)")
tf.flags.DEFINE_string("emb_mode", "rand", "pick a embedding mode: rand, train-w2v, google-w2v, train-glove, stanford-glove (default: rand)")
tf.flags.DEFINE_boolean("finetune", True, "finetune (True) or freeze (False) embedding layers (default: True - finetune)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs (default: 40)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 10)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def pause():
    """
    For debug use (breakpoint)
    """
    int(input("enter a num to cont..."))


def preprocess(transfer=False, target_baseline=False):
    """
    Pre-process StackExchange or SICK data into:
    x_text: a list of sentenses
    y: one-hot representation of labels
    """
    # Load data
    if not FLAGS.train_sick:
        print("Loading SE data...")
        if not transfer:
            # this one reads source task data
            if not target_baseline:
                x_text, y = data_helpers.load_SE_data_and_labels(FLAGS.data_path+'source/',
                                                         multi_label=FLAGS.multi_label)
            # this one reads target task data and train a target baseline (without transfer)
            else:
                x_text, y = data_helpers.load_SE_data_and_labels(FLAGS.data_path+'target/',
                                                         multi_label=FLAGS.multi_label)
        # this one train target task with transfer
        else:
            x_text, y = data_helpers.load_SE_data_and_labels(FLAGS.data_path+'target/',
                                                     multi_label=FLAGS.multi_label)
    else:
        print("Loading SICK data...")
        sick_all, sick_num_train = data_helpers.download_sick_all()

        x_text, y = data_helpers.load_SICK_data_and_labels(sick_all,
                                                           sent_col=FLAGS.sent_col,
                                                           sent_label=FLAGS.sent_label)
    # Build vocabulary
    # this finds the longest sentence from sentence columns
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("max doc length: ", max_document_length)
    print("num classes: ", len(y[0]))
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)

    # this is for build vocabulary based on both columns
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # take a look at the vocabulary
    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
    # vocabulary = list(list(zip(*sorted_vocab))[0])
    # file = open("vocab.txt","w")
    # file.writelines('{}:{}\n'.format(v,k) for k, v in vocab_dict.items())
    # file.close()

    # sort vocabulary dictionary
    sorted_vocab_dict = {}
    for i in range(len(sorted_vocab)):
        sorted_vocab_dict[sorted_vocab[i][1]] = sorted_vocab[i][0]

    # Randomly shuffle data
    np.random.seed(int(round(time.time() * 1000)) % (2**32 - 1))
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("=====Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("=====Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, sorted_vocab_dict


def train_baseline(x_train, y_train, vocab_processor, pretrain_emb,
                   x_dev, y_dev, target_baseline=False):
    """
    Train a baseline model on source task: set "target_baseline=False"
    Train a baseline model on target task: set "target_baseline=True"
    """
    # construct output directory, tensorboard summaries and csvs
    if FLAGS.emb_mode != 'rand':
        assert int(len(vocab_processor.vocabulary_)) == int(pretrain_emb.shape[0]), "vocab length not equal to pretrain embedding row!"
        assert int(FLAGS.embedding_dim) == int(pretrain_emb.shape[1]), "pretrain embedding_dim not equal to embedding_dim!"

    if FLAGS.train_sick:
        datasetname = "SICK" + str(FLAGS.sent_col)+ "_" + str(FLAGS.sent_label) + "_"
    else:
        datasetname = "SE_"
    today = str(datetime.date.today())
    timestamp = datasetname + FLAGS.model_type + "_"
    if FLAGS.model_type == 'rnn':
        timestamp += FLAGS.cell_type + "_"
    timestamp += 'emb-'+FLAGS.emb_mode + "_"
    timestamp += 'finetune_' if FLAGS.finetune else 'freeze_'
    timestamp += 'batchsize' + str(FLAGS.batch_size) + "_"
    timestamp += "evalevery" + str(FLAGS.evaluate_every)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", today, timestamp))
    print("========Writing runs to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # write to csv
    csv_outdir = os.path.abspath(os.path.join(os.path.curdir,"runs", "results_csv", FLAGS.model_type))
    csv_filename = datasetname + str(FLAGS.model_type)
    if FLAGS.model_type == 'rnn':
        csv_filename += '_'+str(FLAGS.cell_type)
    csv_filename += '_'+str(FLAGS.emb_mode) + "_tune" + str(FLAGS.finetune)
    csv_filename += '_batchsize' + str(FLAGS.batch_size)
    csv_filename += "_evalevery" + str(FLAGS.evaluate_every)
    if not target_baseline:
        csv_filename_train = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_train_source.csv'))
        csv_filename_test = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_test_source.csv'))
    else:
        csv_filename_train = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_train_target.csv'))
        csv_filename_test = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_test_target.csv'))
    print("========Writing train csv to {}\n".format(csv_filename_train))
    print("========Writing test csv to {}\n".format(csv_filename_test))

    tf.reset_default_graph()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if FLAGS.model_type == 'cnn':
                print("=====Training in CNN=====")
                model = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    multi_label=FLAGS.multi_label,
                    l2_reg_lamb=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == 'rnn':
                print("=====Training in RNN=====")
                model = TextRNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    cell_type=FLAGS.cell_type,
                    hidden_size=FLAGS.hidden_size,
                    multi_label=FLAGS.multi_label,
                    l2_reg_lamb=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == 'fasttext':
                print("=====Training in fastText (avg-pooling)=====")
                model = fastText(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    multi_label=FLAGS.multi_label)
            else:
                raise ValueError("mode %s not supported. Valid mode: %s, %s" % (
                FLAGS.model_type, 'fasttext', 'cnn','rnn'))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar(str(FLAGS.emb_mode)+"_loss_"+str('finetune' if FLAGS.finetune else 'freeze'), model.loss)
            acc_summary = tf.summary.scalar(str(FLAGS.emb_mode)+"_acc_"+str('finetune' if FLAGS.finetune else 'freeze'), model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                # for metric: Update the running variables on new batch of samples
                _, step, summaries, loss, accuracy, pred = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy,
                     model.predictions], feed_dict)

                # Calculate the score on this batch
                precision_avg, recall_avg = 0., 0.
                if not FLAGS.multi_label:
                    y_true = np.argmax(y_batch, 1)
                    precision_avg = precision_score(y_true, pred, average='macro')
                    recall_avg = recall_score(y_true, pred, average='macro')
                else:
                    top_k = len(pred[0])
                    y_true = np.stack([arr.argsort()[-top_k:][::-1] for arr in y_batch])
                    for k in range(top_k):
                        precision_avg += precision_score(y_true[:, k], pred[:, k], average='macro')
                        recall_avg += recall_score(y_true[:, k], pred[:, k], average='macro')
                    precision_avg /= top_k
                    recall_avg /= top_k

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, "
                "precision {:g}, recall {:g}".format(time_str, step, loss,
                                                     accuracy, precision_avg, recall_avg))

                train_summary_writer.add_summary(summaries, global_step=step)

                mode = 'a' if os.path.exists(csv_filename_train) else 'w'
                if mode == 'w':
                    with open(csv_filename_train, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow(['step', 'accuracy', 'precision_avg','recall_avg'])
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                else:
                    with open(csv_filename_train, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])


            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on the entire dev set
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy, pred = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy,
                     model.predictions], feed_dict)

                # Calculate the score and confusion matrix on this batch
                precision_avg, recall_avg = 0., 0.
                if not FLAGS.multi_label:
                    y_true = np.argmax(y_batch, 1)
                    precision_avg = precision_score(y_true, pred, average='macro')
                    recall_avg = recall_score(y_true, pred, average='macro')
                else:
                    top_k = len(pred[0])
                    y_true = np.stack([arr.argsort()[-top_k:][::-1] for arr in y_batch])
                    for k in range(top_k):
                        precision_avg = precision_score(y_true[:, k], pred[:, k], average='macro')
                        recall_avg += recall_score(y_true[:, k], pred[:, k], average='macro')
                    precision_avg /= top_k
                    recall_avg /= top_k

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g},"
                "precision {:g}, recall {:g}".format(time_str, step, loss, accuracy,
                                                     precision_avg, recall_avg))
                if writer:
                    writer.add_summary(summaries, global_step=step)

                mode = 'a' if os.path.exists(csv_filename_test) else 'w'
                if mode == 'w':
                    with open(csv_filename_test, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow(['step', 'accuracy', 'precision_avg','recall_avg'])
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                else:
                    with open(csv_filename_test, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                return accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # 0-step eval
            print("\nEvaluation at step 0:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")

            moving_avg_test_acc = 0
            num_eval = 0
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    cur_test_acc = dev_step(x_dev, y_dev,
                                                    writer=dev_summary_writer)
                    moving_avg_test_acc += cur_test_acc
                    num_eval += 1
                    print("")
                    # save the current best model
                    if num_eval != 0 and moving_avg_test_acc / num_eval < cur_test_acc:
                        print("cur test acc:", cur_test_acc)
                        print("avg test acc: ", moving_avg_test_acc / num_eval)
                        path = saver.save(sess, checkpoint_prefix+'best', global_step=current_step)
                        print("Saved best model checkpoint to {}\n".format(path))

            path = saver.save(sess, checkpoint_prefix+'final', global_step=current_step)
            print("Saved final model checkpoint to {}\n".format(path))

            return csv_filename_train, csv_filename_test, checkpoint_dir


def train_transfer(x_train, y_train, vocab_processor, pretrain_emb, x_dev, y_dev,
                   source_ckpt, target_ckpt, pretrained_values=None):
    """
    Train a transfer model on target task: must pass "pretrained_values"
    Build model architecture using target task data,
    then load pre-trained model's weight value to it (instead of rand init)
    """
    # Output directory for models and summaries and csvs
    if FLAGS.emb_mode != 'rand':
        assert int(len(vocab_processor.vocabulary_)) == int(pretrain_emb.shape[0]), "vocab length not equal to pretrain embedding row!"
        assert int(FLAGS.embedding_dim) == int(pretrain_emb.shape[1]), "pretrain embedding col ot equal to embedding_dim!"

    if FLAGS.train_sick:
        datasetname = "SICK" + str(FLAGS.sent_col)+ "_" + str(FLAGS.sent_label) + "_"
    else:
        datasetname = "SE_"
    today = str(datetime.date.today())
    timestamp = datasetname + FLAGS.model_type + "_"
    if FLAGS.model_type == 'rnn':
        timestamp += FLAGS.cell_type + "_"
    timestamp += 'emb-'+FLAGS.emb_mode + "_"
    timestamp += 'finetune_' if FLAGS.finetune else 'freeze_'
    timestamp += 'batchsize' + str(FLAGS.batch_size) + "_"
    timestamp += "evalevery" + str(FLAGS.evaluate_every)
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", today, timestamp))
    print("========Writing runs to {}\n".format(out_dir))
    checkpoint_dir = target_ckpt
    checkpoint_prefix = os.path.join(checkpoint_dir, "modelbest")
    if not os.path.exists(checkpoint_dir):
        raise ValueError("new directory has not been created yet to save the transfer model!")

    # write to csv
    csv_outdir = os.path.abspath(os.path.join(os.path.curdir,"runs", "results_csv", FLAGS.model_type))
    csv_filename = datasetname + str(FLAGS.model_type)
    if FLAGS.model_type == 'rnn':
        csv_filename += '_'+str(FLAGS.cell_type)
    csv_filename += '_'+str(FLAGS.emb_mode) + "_tune" + str(FLAGS.finetune)
    csv_filename += '_batchsize' + str(FLAGS.batch_size)
    csv_filename += "_evalevery" + str(FLAGS.evaluate_every)
    csv_filename_train = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_train_transfer.csv'))
    csv_filename_test = os.path.abspath(os.path.join(csv_outdir, csv_filename+'_test_transfer.csv'))
    print("========Writing train csv to {}\n".format(csv_filename_train))
    print("========Writing test csv to {}\n".format(csv_filename_test))

    tf.reset_default_graph()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if FLAGS.model_type == 'cnn':
                print("=====Training in CNN=====")
                model = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    multi_label=FLAGS.multi_label,
                    l2_reg_lamb=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == 'rnn':
                print("=====Training in RNN=====")
                model = TextRNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    cell_type=FLAGS.cell_type,
                    hidden_size=FLAGS.hidden_size,
                    multi_label=FLAGS.multi_label,
                    l2_reg_lamb=FLAGS.l2_reg_lambda)
            elif FLAGS.model_type == 'fasttext':
                print("=====Training in fastText (avg-pooling)=====")
                model = fastText(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    pretrain_emb=pretrain_emb,
                    emb_mode=FLAGS.emb_mode,
                    finetune=FLAGS.finetune,
                    multi_label=FLAGS.multi_label)
            else:
                raise ValueError("mode %s not supported. Valid mode: %s, %s" % (
                FLAGS.model_type, 'fasttext', 'cnn','rnn'))
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar(str(FLAGS.emb_mode)+"_loss_"+str('finetune' if FLAGS.finetune else 'freeze'), model.loss)
            acc_summary = tf.summary.scalar(str(FLAGS.emb_mode)+"_acc_"+str('finetune' if FLAGS.finetune else 'freeze'), model.accuracy)

            # Train Summaries
            # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            graph = tf.get_default_graph()
            load_ops = []
            if pretrained_values != None:
                print("loading pretrained weight values")
                for key in pretrained_values:
                    print(key)
                    load_ops.append(tf.assign(graph.get_tensor_by_name(key),
                                    pretrained_values[key]))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                # for metric: Update the running variables on new batch of samples
                _, step, summaries, loss, accuracy, pred = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy,
                     model.predictions], feed_dict)

                # Calculate the score on this batch
                precision_avg, recall_avg = 0., 0.
                if not FLAGS.multi_label:
                    y_true = np.argmax(y_batch, 1)
                    precision_avg = precision_score(y_true, pred, average='macro')
                    recall_avg = recall_score(y_true, pred, average='macro')
                else:
                    top_k = len(pred[0])
                    y_true = np.stack([arr.argsort()[-top_k:][::-1] for arr in y_batch])
                    for k in range(top_k):
                        precision_avg += precision_score(y_true[:, k], pred[:, k], average='macro')
                        recall_avg += recall_score(y_true[:, k], pred[:, k], average='macro')
                    precision_avg /= top_k
                    recall_avg /= top_k

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, "
                "precision {:g}, recall {:g}".format(time_str, step, loss,
                                                     accuracy, precision_avg, recall_avg))

                train_summary_writer.add_summary(summaries, global_step=step)

                mode = 'a' if os.path.exists(csv_filename_train) else 'w'
                if mode == 'w':
                    with open(csv_filename_train, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow(['step', 'accuracy', 'precision_avg','recall_avg'])
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                else:
                    with open(csv_filename_train, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on the entire dev set
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.dropout_keep_prob: 1.0
                }

                step, summaries, loss, accuracy, pred = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy,
                     model.predictions], feed_dict)

                # Calculate the score and confusion matrix on this batch
                precision_avg, recall_avg = 0., 0.
                if not FLAGS.multi_label:
                    y_true = np.argmax(y_batch, 1)
                    precision_avg = precision_score(y_true, pred, average='macro')
                    recall_avg = recall_score(y_true, pred, average='macro')
                else:
                    top_k = len(pred[0])
                    y_true = np.stack([arr.argsort()[-top_k:][::-1] for arr in y_batch])
                    for k in range(top_k):
                        precision_avg = precision_score(y_true[:, k], pred[:, k], average='macro')
                        recall_avg += recall_score(y_true[:, k], pred[:, k], average='macro')
                    precision_avg /= top_k
                    recall_avg /= top_k

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g},"
                "precision {:g}, recall {:g}".format(time_str, step, loss, accuracy,
                                                     precision_avg, recall_avg))
                if writer:
                    writer.add_summary(summaries, global_step=step)

                mode = 'a' if os.path.exists(csv_filename_test) else 'w'
                if mode == 'w':
                    with open(csv_filename_test, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow(['step', 'accuracy', 'precision_avg','recall_avg'])
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                else:
                    with open(csv_filename_test, mode) as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',',
                                               quoting=csv.QUOTE_MINIMAL)
                        csvwriter.writerow([step, accuracy, precision_avg, recall_avg])
                return accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            if pretrained_values != None:
                sess.run([load_ops])

            # 0-step eval
            print("\nEvaluation at step 0:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")

            moving_avg_test_acc = 0
            num_eval = 0

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    cur_test_acc = dev_step(x_dev, y_dev,
                                            writer=dev_summary_writer)
                    moving_avg_test_acc += cur_test_acc
                    num_eval += 1
                    print("")
                    if num_eval != 0 and moving_avg_test_acc / num_eval < cur_test_acc:
                        print("cur test acc:", cur_test_acc)
                        print("avg test acc: ", moving_avg_test_acc / num_eval)
                        path = saver.save(sess, checkpoint_prefix+'best', global_step=current_step)
                        print("Saved best model checkpoint to {}\n".format(path))

            path = saver.save(sess, checkpoint_prefix+'final', global_step=current_step)
            print("Saved final model checkpoint to {}\n".format(path))

            return csv_filename_train, csv_filename_test, checkpoint_dir


def plot(filename, train_or_test):
    """
    Plot results for one run
    """
    acc = defaultdict(list)
    preci= defaultdict(list)
    recall = defaultdict(list)
    with open(filename,'r') as csvfile:
        data = csv.reader(csvfile)
        next(data, None)  # skip the headers
        for row in data:
            acc[int(row[0])].append(float(row[1]))
            preci[int(row[0])].append(float(row[2]))
            recall[int(row[0])].append(float(row[3]))
    # only plot acc for now, can add preci/recall later
    acc_mean = []
    acc_std = []
    preci_mean = []
    preci_std = []
    recall_mean = []
    recall_std = []
    keys = sorted(acc.keys())

    for key in keys:
        acc_mean.append(np.mean(np.array(acc[key])))
        acc_std.append(np.std(np.array(acc[key])))
        preci_mean.append(np.mean(np.array(preci[key])))
        preci_std.append(np.std(np.array(preci[key])))
        recall_mean.append(np.mean(np.array(recall[key])))
        recall_std.append(np.std(np.array(recall[key])))

    if train_or_test == "Test":
        plt.errorbar(x=keys, y=acc_mean, yerr=acc_std, fmt='-',
                     ecolor='black', color='red', barsabove=True, capsize=3, label="acc")
        plt.errorbar(x=keys, y=preci_mean, yerr=preci_std, fmt='-',
                     ecolor='black', color='blue', barsabove=True, capsize=3, label="precision")
        plt.errorbar(x=keys, y=recall_mean, yerr=recall_std, fmt='-',
                     ecolor='black', color='green', barsabove=True, capsize=3, label="recall")
    elif train_or_test == "Train":
        plt.plot(keys, acc_mean, label="acc", color="red")
        plt.plot(keys, preci_mean, label="precision", color="blue")
        plt.plot(keys, recall_mean, label="recall", color="green")
    else:
        raise ValueError("plot Train or Test only")

    plt.xlabel('step (eval every '+str(FLAGS.evaluate_every)+' steps, batch size '+str(FLAGS.batch_size))
    plt.ylabel(str(train_or_test) + ' value')
    if FLAGS.train_sick:
        title = 'SICK' + FLAGS.sent_col+ "_" + str(train_or_test) + '_'
        title += str(FLAGS.model_type) + '_' + str(FLAGS.emb_mode) + '_'
        title += 'finetune' + str(FLAGS.finetune) + '_'
        title += 'Multilabel' + str(FLAGS.multi_label)
    else:
        title = 'SE' + "_" + str(train_or_test)+'_'
        title += str(FLAGS.model_type) + '_' + str(FLAGS.emb_mode)+'_'
        title += 'finetune' + str(FLAGS.finetune) + '_'
        title += 'Multilabel' + str(FLAGS.multi_label)
    plt.title(title)
    plt.legend()
    plt.show()


def load_pretrain_emb(emb_mode, vocab_dict, x_train, x_dev):
    """
    Load embedding values
    """
    pretrain_emb = None
    if emb_mode == 'train-w2v':
        pretrain_emb = word2vec.train_w2v(vocab_dict, np.vstack((x_train, x_dev)), FLAGS.embedding_dim)
        print("========== self-pretrained w2v embedding dim: ", pretrain_emb.shape, "==========")
    elif emb_mode == 'google-w2v':
        pretrain_emb = word2vec.load_pretrain_w2v(vocab_dict, FLAGS.embedding_dim, FLAGS.google_w2v)
        print("========== google-pretrained w2v embedding dim: ", pretrain_emb.shape, "==========")
    elif emb_mode == 'train-glove':
        pretrain_emb = word2vec.train_glove(vocab_dict, np.vstack((x_train, x_dev)), FLAGS.embedding_dim)
        print("========== self-pretrained glove embedding dim: ", pretrain_emb.shape, "==========")
    elif emb_mode == 'stanford-glove':
        pretrain_emb = word2vec.load_pretrain_glove(vocab_dict, FLAGS.embedding_dim, FLAGS.glove_6b)
        print("========== stanford-pretrained glove embedding dim: ", pretrain_emb.shape, "==========")
    else:
        pretrain_emb = np.random.uniform(-0.01, 0.01, size=(len(vocab_dict), FLAGS.embedding_dim))
        print("==========using %s emb-mode, random initialization==========" %emb_mode)

    return pretrain_emb


def SICK_transfer_from_to(source_ckpt, target_ckpt):
    """
    Trigger transfer learning in SICK dataset between "sent_1" and "sent_2"
    not applicable to SE dataset
    This function train both target_baseline and target_transfer
    """
    # load pretrained weight as numpy matrixs
    tf.reset_default_graph()
    values = {}
    # get source model trainable variable values
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            models_meta = glob.glob(source_ckpt+"/modelbest*.meta")
            best_meta = max(models_meta)
            best_restore = best_meta.replace(".meta","")
            # load graph structure
            new_saver = tf.train.import_meta_graph(best_meta)
            # load values for parameters
            new_saver.restore(sess, best_restore)

            graph = tf.get_default_graph()
            trainable_vars = [v.name for v in tf.trainable_variables()]
            trainable_vars = [v for v in trainable_vars if 'output' not in v]
            trainable_vars = [v for v in trainable_vars if 'pretrain_emb_W' not in v]
            for var in trainable_vars:
                values[var] = sess.run(var)

    # load target data
    if 'sent_1' in target_ckpt:
        FLAGS.sent_col = 'sent_1'
    elif 'sent_2' in target_ckpt:
        FLAGS.sent_col = 'sent_2'
    else:
        raise ValueError("dataset path %s not supported. Valid mode: %s, %s" % (
        target_ckpt, 'sent_1', 'sent_2'))

    x_train, y_train, vocab_processor, x_dev, y_dev, vocab_dict = preprocess()

    for i in range(FLAGS.num_iter):
        # load pretrain emb
        pretrain_emb = load_pretrain_emb(FLAGS.emb_mode, vocab_dict,
                                         x_train, x_dev)
        # train with pretrained model
        trans_train_csv, trans_test_csv, _, = train_transfer(x_train, y_train,
                                                    vocab_processor,
                                                    pretrain_emb, x_dev, y_dev,
                                                    source_ckpt, target_ckpt,
                                                    pretrained_values=values)
        # train target from scratch without transfer
        target_train_csv, target_test_csv, _, = train_baseline(x_train, y_train,
                                                       vocab_processor,
                                                       pretrain_emb, x_dev, y_dev,
                                                       target_baseline=True)
    # plot text, train
    plot(trans_test_csv, "Test")
    plot(trans_train_csv, "Train")
    plot(target_test_csv, "Test")
    plot(target_train_csv, "Train")


def SE_transfer_from_to(source_ckpt, target_ckpt):
    """
    Trigger transfer learning in SE dataset
    not applicable to SICK dataset
    This function train both target_baseline and target_transfer
    """
    # load pretrained weight as numpy matrixs
    tf.reset_default_graph()
    values = {}
    # get source model trainable variable values
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            models_meta = glob.glob(source_ckpt+"/modelbest*.meta")
            best_meta = max(models_meta)
            best_restore = best_meta.replace(".meta","")
            # load graph structure
            new_saver = tf.train.import_meta_graph(best_meta)
            # load values for parameters
            new_saver.restore(sess, best_restore)

            graph = tf.get_default_graph()
            trainable_vars = [v.name for v in tf.trainable_variables()]
            trainable_vars = [v for v in trainable_vars if 'output' not in v]
            trainable_vars = [v for v in trainable_vars if 'pretrain_emb_W' not in v]
            for var in trainable_vars:
                values[var] = sess.run(var)

    (x_train, y_train, vocab_processor,
    x_dev, y_dev, vocab_dict) = preprocess(transfer=True)

    for i in range(FLAGS.num_iter):
        # load pretrain emb (same as training from scratch)
        pretrain_emb = load_pretrain_emb(FLAGS.emb_mode, vocab_dict,
                         x_train, x_dev)
        # train with pretrained model
        trans_train_csv, trans_test_csv, _, = train_transfer(x_train, y_train,
                                                       vocab_processor,
                                                       pretrain_emb, x_dev, y_dev,
                                                       source_ckpt, target_ckpt,
                                                       pretrained_values=values)
        # train target from scratch without transfer
        target_train_csv, target_test_csv, _, = train_baseline(x_train, y_train,
                                                       vocab_processor,
                                                       pretrain_emb, x_dev, y_dev,
                                                       target_baseline=True)
    # plot text, train
    plot(trans_test_csv, "Test")
    plot(trans_train_csv, "Train")
    plot(target_test_csv, "Test")
    plot(target_train_csv, "Train")


def main(argv=None):
    # load data (SE or SICK)
    x_train, y_train, vocab_processor, x_dev, y_dev, vocab_dict = preprocess()

    # Train text classification on SE data or
    # Train text classification on SICK data, to compare similarity
    for i in range(FLAGS.num_iter):
        pretrain_emb = load_pretrain_emb(FLAGS.emb_mode, vocab_dict,
                                         x_train, x_dev)
        (train_csv,
        test_csv,
        checkpoint_dir) = train_baseline(x_train, y_train, vocab_processor,
                                         pretrain_emb, x_dev, y_dev)
    # plot test/train
    plot(test_csv, "Test")
    plot(train_csv, "Train")

    # Launch transfer if transter=True
    if FLAGS.transfer and os.path.exists(checkpoint_dir):
        # if SICK data
        if FLAGS.train_sick:
            if 'sent_1' in checkpoint_dir:
                new_dir = checkpoint_dir.replace('sent_1', 'sent_2')
                new_dir += "_transfer_from_sent1"
            elif 'sent_2' in checkpoint_dir:
                new_dir = checkpoint_dir.replace('sent_2', 'sent_1')
                new_dir += "transfer_from_sent2"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            print("========Writing transfer runs to {}\n".format(new_dir))
            SICK_transfer_from_to(checkpoint_dir, new_dir)
        # if SE data
        else:
            new_dir = checkpoint_dir+"transfer"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            print("========Writing transfer runs to {}\n".format(new_dir))
            SE_transfer_from_to(checkpoint_dir, new_dir)


if __name__ == '__main__':
    tf.app.run()
