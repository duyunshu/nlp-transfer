import tensorflow as tf
import numpy as np


def pause():
    int(input('enter a num to cont...'))


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, pretrain_emb, emb_mode, finetune,
      filter_sizes, num_filters, multi_label, l2_reg_lamb=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.multi_label = multi_label
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.device('/gpu:0'):
            # Embedding layer
            with tf.name_scope("embedding"):
                # if emb_mode != "rand":
                W = tf.get_variable(shape=[vocab_size, embedding_size],
                                    initializer=tf.constant_initializer(pretrain_emb),
                                    trainable=finetune, name="pretrain_emb_W")
                self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            # three CNN+max-pool layers
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    name="output_W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                if not self.multi_label:
                    self.predictions = tf.argmax(self.scores, 1, name="predictions")
                else:
                    _, self.predictions = tf.nn.top_k(self.scores, k=5, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                if not self.multi_label:
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y, name="single_loss")
                    self.loss = tf.reduce_mean(losses, name="loss") + l2_reg_lamb * l2_loss
                else:
                    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y, name="multi_loss")
                    print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, num-classes).
                    losses = tf.reduce_sum(losses, axis=1) #shape=(?,). loss for all data in the batch
                    self.loss = tf.reduce_mean(losses, name='loss') #shape=().   average loss in the batch

            # Accuracy
            with tf.name_scope("accuracy"):
                if not self.multi_label:
                    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                else:
                    correct_predictions = tf.nn.in_top_k(self.scores,
                                                         tf.argmax(self.input_y, 1),
                                                         k=5)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                                   name="accuracy")


class TextRNN(object):
    """
    An RNN for text classification.
    Uses an embedding layer, followed by RNN cells and output layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, pretrain_emb, emb_mode, finetune,
      cell_type, hidden_size, multi_label, l2_reg_lamb=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.multi_label = multi_label
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # a vector of [batch_size, sequence_length]
        text_length = self._length(self.input_x)

        with tf.device('/gpu:0'):
            # Embedding layer
            with tf.name_scope("embedding"):
                # if emb_mode != "rand":
                W = tf.get_variable(shape=[vocab_size, embedding_size],
                                    initializer=tf.constant_initializer(pretrain_emb),
                                    trainable=finetune, name="pretrain_emb_W")
                self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

            # Create RNN layers
            with tf.name_scope("rnn"):
                cell = self._get_cell(hidden_size, cell_type)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                # if do not specify initial state, it resets to zero
                # just one layer of rnn with 100 hidden size
                # multi-layer rnn: MultiRnn
                # output shape [batch_size, sequence_length, hidden_size]
                all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.embedded_chars,
                                               sequence_length=text_length,
                                               dtype=tf.float32)

                # we take only the last cell of each sentence
                # as the input to fully connected layer,
                # so sequence_length is omitted (becomes 1)
                # h_output shape [batch_size, hidden_size]
                self.h_outputs = self.last_relevant(all_outputs, text_length)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[hidden_size, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_outputs, W, b, name="scores")
                if not self.multi_label:
                    self.predictions = tf.argmax(self.scores, 1, name="predictions")
                else:
                    _, self.predictions = tf.nn.top_k(self.scores, k=5, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                if not self.multi_label:
                    losses = tf.nn.softmax_cross_entropy_with_logits(
                             logits=self.scores, labels=self.input_y, name="single_loss")
                    self.loss = tf.reduce_mean(losses, name="loss") + l2_reg_lamb * l2_loss
                else:
                    losses = tf.nn.sigmoid_cross_entropy_with_logits(
                             logits=self.scores, labels=self.input_y, name="multi_loss")
                    print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, num-classes).
                    losses = tf.reduce_sum(losses, axis=1) #shape=(?,). loss for all data in the batch
                    self.loss = tf.reduce_mean(losses, name='loss') #shape=().   average loss in the batch

            # Accuracy
            with tf.name_scope("accuracy"):
                if not self.multi_label:
                    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                else:
                    correct_predictions = tf.nn.in_top_k(self.scores,
                                                         tf.argmax(self.input_y, 1),
                                                         k=5)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                                   name="accuracy")


    # Length of the sequence data
    def _length(self, seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def _get_cell(self, hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        if cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        # elif cell_type == "gru":
        #     return tf.nn.rnn_cell.GRUCell(hidden_size)
        raise ValueError("rnn_mode %s not supported. Valid mode: %s, %s" % (config.rnn_mode, 'vanilla', 'lstm'))

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is the 4th output of cell, so extract it.
    def last_relevant(self, seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)


class fastText(object):
    """
    fastText for text classification. Uses average-pooling for embedding
    Uses an embedding layer, a softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, pretrain_emb, emb_mode, finetune, multi_label):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        # set an unused parameter to cope with the framework when calling the train
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # multi_label
        self.multi_label = multi_label

        with tf.device('/gpu:0'):
            # Embedding layer
            with tf.name_scope("embedding"):
                # if emb_mode != "rand":
                W = tf.get_variable(shape=[vocab_size, embedding_size],
                                    initializer=tf.constant_initializer(pretrain_emb),
                                    trainable=finetune, name="pretrain_emb_W")
                embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
                self.embedded_chars = tf.reduce_mean(embedded_chars, axis=1)

            # 3.linear classifier layer
            with tf.name_scope("output"):
                W = tf.get_variable("W", shape=[embedding_size, num_classes])
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                #[None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,num_classes])+num_classes
                self.scores = tf.nn.xw_plus_b(self.embedded_chars, W, b, name="scores")
                if not self.multi_label:
                    self.predictions = tf.argmax(self.scores, 1, name="predictions")
                else:
                    _, self.predictions = tf.nn.top_k(self.scores, k=5, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                if not self.multi_label:
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                             labels=self.input_y, name="single_loss")
                    self.loss = tf.reduce_mean(losses, name='loss')
                else:
                    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                             labels=self.input_y, name="multi_loss")
                    print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, num-classes).
                    losses = tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
                    self.loss = tf.reduce_mean(losses, name='loss') #shape=().   average loss in the batch

            # Accuracy
            with tf.name_scope("accuracy"):
                if not self.multi_label:
                    correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                else:
                    correct_predictions = tf.nn.in_top_k(self.scores,
                                                         tf.argmax(self.input_y, 1),
                                                         k=5)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                                   name="accuracy")
