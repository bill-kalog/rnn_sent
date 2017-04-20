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

        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        # values vital when using minibatches on dev set
        # really ugly workaround
        self.metrics_weight = tf.placeholder(tf.float32, name="metrics_weight")
        self.fixed_acc_value = tf.placeholder(tf.float32, name="f_acc_value")
        self.fixed_loss_value = tf.placeholder(tf.float32, name="f_loss_value")

        self.make_graph(config)
        # self.attention()
        self.train()
        self.summarize(config)

    def make_graph():
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
                trainable=config['train_embeddings'][index_],
                initializer=tf.constant_initializer(
                    np.array(self.word_vectors[index_]))
            )
            embedded_tokens = tf.nn.embedding_lookup(
                self.w_embeddings, self.x)
            


    def memory():
        pass

    def train():
        pass

    def summarize():
        pass











