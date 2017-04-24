import tensorflow as tf
import numpy as np
from math import ceil
import sys
import os


class DMN(object):
    """docstring for DMN"""
    def __init__(self, config, word_vectors=[]):
        self.dim_proj = config['dim_proj']
        self.layers = config['layers']
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.n_words = config['n_words']
        self.learning_rate = config['learning_rate']
        self.num_classes = config['classes_num']
        self.word_vectors = word_vectors
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.config = config
        # self.str_summary_type = tf.placeholder(
        #     tf.string, name="str_summary_type")
        self.input_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_inp")
        self.output_keep_prob = tf.placeholder(
            tf.float32, name="keep_prob_out")
        self.sentence_len = config['sentence_len']
        self.max_gradient_norm = config["clip_threshold"]
        self.x = tf.placeholder(tf.int32, [None, self.sentence_len], name="x")
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")
        self.question = tf.placeholder(
            tf.int32, [None, self.sentence_len], name="question_")

        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        # values vital when using minibatches on dev set
        # really ugly workaround
        self.metrics_weight = tf.placeholder(tf.float32, name="metrics_weight")
        self.fixed_acc_value = tf.placeholder(tf.float32, name="f_acc_value")
        self.fixed_loss_value = tf.placeholder(tf.float32, name="f_loss_value")

        self.build_input()
        self.encoder()
        self.question_module()
        # self.attention()
        self.train()
        self.summarize()

    def build_input(self):
        """ declare graph """
        with tf.name_scope("word_embeddings"):
            """
            initialize embeddings currently able to use only one type of
            embeddings
            """
            index_ = 0
            self.w_embeddings = tf.get_variable(
                "W0_" + str(index_),
                shape=[self.n_words, self.dim_proj],
                trainable=self.config['train_embeddings'][index_],
                initializer=tf.constant_initializer(
                    np.array(self.word_vectors[index_]))
            )
            embedded_tokens = tf.nn.embedding_lookup(
                self.w_embeddings, self.x)
            self.rnn_input = embedded_tokens

            self.seq_lengths = self.get_seq_lenghts("facts", self.x)

    def encoder(self):
        with tf.name_scope("rnn_cell"):
            if self.config['GRU']:  # use GRU cell
                self.rnn_cell = tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.GRUCell(num_units=self.dim_proj),
                    input_keep_prob=self.input_keep_prob,
                    output_keep_prob=self.output_keep_prob
                )
        # create sequential rnn from single cells
        rnn_cell_seq = tf.contrib.rnn.MultiRNNCell(
            [self.rnn_cell] * self.layers, state_is_tuple=True)
        initial_state = rnn_cell_seq.zero_state(
            self.batch_size, tf.float32)

        if self.config['bidirectional']:
            self.dimensionality_mult = 2
            output, states = tf.nn.bidirectional_dynamic_rnn(
                inputs=self.rnn_input,
                cell_fw=rnn_cell_seq,
                cell_bw=rnn_cell_seq,
                initial_state_fw=initial_state,
                initial_state_bw=initial_state,
                sequence_length=self.seq_lengths
            )
            self.output = output

        else:
            self.dimensionality_mult = 1
            output, state = tf.nn.dynamic_rnn(
                rnn_cell_seq, self.rnn_input,
                initial_state=initial_state,
                sequence_length=self.seq_lengths
            )
            self.out_state = state
            self.output = output

    def question_module(self):
        """ declare question module """
        with tf.name_scope("question"):
            """ transform question to sequence of vectors
            """
            self.input_question = tf.nn.embedding_lookup(
                self.w_embeddings, self.question)
            self.seq_lengths_q = self.get_seq_lenghts(
                "questions", self.question)

            # create sequential rnn from single cells
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self.dim_proj),
                # input_keep_prob=self.input_keep_prob,
                # output_keep_prob=self.output_keep_prob
            )
            rnn_cell_seq = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell] * 1, state_is_tuple=True)
            initial_state = rnn_cell_seq.zero_state(
                self.batch_size, tf.float32)

            output, state = tf.nn.dynamic_rnn(
                rnn_cell_seq, self.input_question,
                initial_state=initial_state,
                sequence_length=self.seq_lengths_q
            )
            self.output_q = state

    def episodic_module(self):
        """takes the final state of question module
        and the sequence of outputs of our encoder
        and produces 'memories' using an rnn. 
        uses attention over the encoders' states
        """
        rnn_cell = ""  # TODO
        # initial state is the question vector
        initial_state = self.output_q
        rnn_cell_seq = tf.contrib.rnn.MultiRNNCell(
            [rnn_cell] * 1, state_is_tuple=True)

    def answer_module(self):
        """take the final state/episode of episodic module  and
        producess an answer (here a simplified version without an RNN
        to build the answer, using just a fc layer)"""
        with tf.name_scope("answer_module"):
            # initial state is the last memory
            # initial_state = self.last_memory
            shape = [int(self.last_memory.shape[1]), self.num_classes]
            W = tf.Variable(
                tf.truncated_normal(shape, stddev=0.01), name="W_answer",
            )
            b = tf.Variable(tf.constant(
                0.1, shape=[self.num_classes]),
                trainable=True, name="b_answer"
            )
            self.scores = tf.nn.xw_plus_b(self.last_memory, W, b)


    def attention(self):
        pass

    def memory(self):
        pass

    def train(self):
        """ calculate accuracies, train and predict """
        with tf.name_scope("predict"):
            self.predictions = tf.argmax(self.scores, 1)
            self.true_predictions = tf.argmax(self.y, 1)
            self.probs = tf.nn.softmax(self.scores)
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.y, name="losses")
            # self.total_loss = tf.reduce_sum(self.losses)
            self.mean_loss = tf.reduce_mean(self.losses)
        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(self.correct_predictions, "float"), name="accuracy")

        params = tf.trainable_variables()
        # if self.train_phase:
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.mean_loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(
            gradients, self.max_gradient_norm)

        # with tf.name_scope("grad_norms"):
        #     grad_summ = tf.scalar_summary("grad_norms", norm)

        self.update = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def summarize(self):
        self.mean_loss = self.metrics_weight * self.mean_loss + \
            self.fixed_loss_value
        self.accuracy = self.metrics_weight * self.accuracy + \
            self.fixed_acc_value
        loss_summary = tf.summary.scalar("loss", self.mean_loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)
        # Summaries
        self.summary_op = tf.summary.merge([loss_summary, acc_summary])

    def get_seq_lenghts(self, type_, input_):
        '''
        calculate actual length of each sentence -- known bug
        if a sentence ends with unknown tokens they are not considered
        in size, if they are at start or in between they are
        considered though
        '''
        with tf.name_scope("calc_sequences_length_" + type_):
            # this doesn't work due to zero padding and <UNK> being both zero
            # self.seq_lengths = tf.reduce_sum(tf.sign(self.x), 1)
            mask = tf.sign(input_)
            range_ = tf.range(
                start=1, limit=self.sentence_len + 1, dtype=tf.int32)
            mask = tf.multiply(mask, range_, name="mask")
            return tf.reduce_max(mask, axis=1)
