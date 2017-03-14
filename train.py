import time
import datetime
import os
import tensorflow as tf
# from conf import config
import numpy as np
from model import RNN
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import learn
import process_utils
import sys
import json


def init_vocabulary_processor(dx_train, dx_dev):
    """
    vocabulary processor
    implementation taken from 
    http://stackoverflow.com/questions/40661684/tensorflow-vocabularyprocessor#40741660
    """
    max_document_length = max([len(x.split(" ")) for x in dx_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    vocab_processor.fit(dx_train)
    x_train = np.array(list(vocab_processor.transform(dx_train)))
    x_dev = np.array(list(vocab_processor.transform(dx_dev)))

    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])

    return x_train, x_dev, vocab_dict, vocabulary


def set_train(sess, config, data, pretrained_embeddings=[]):

    dx_train, y_train, dx_dev, y_dev = data

    x_train, x_dev, vocab_dict, vocabulary = init_vocabulary_processor(
        dx_train, dx_dev)

    print("Vocabulary Size: {}".format(len(vocabulary)))
    print("Train/Dev split: {}/{},{}".format(
        len(y_train), len(y_dev), len(y_train) + len(y_dev)))

    # Build word embeddings
    config['n_words'] = len(vocabulary)
    config['sentence_len'] = x_train.shape[1]

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    config['out_dir'] = out_dir
    print("Writing to {}\n".format(out_dir))

    # network = RNN(config, sess, init_embd)
    network = RNN(config, sess)

    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(
        dev_summary_dir, sess.graph)

    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(
        train_summary_dir, sess.graph)

    tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())

    # train fucntion
    def train_step(x_batch, y_batch):
        print("batch lenght {}". format(len(x_batch)))
        feed_dict = {
            network.x: x_batch,
            network.y: y_batch,
            network.dropout_prob: config["dropout_rate"],
            # network.str_summary_type: "",
            network.input_keep_prob: config["keep_prob_inp"],
            network.output_keep_prob: config["keep_prob_out"],
            # network.train_phase: True
        }


        # _, step, summaries, loss, accuracy, word_embd, grad_summary = sess.run(
        #     [train_op, global_step, train_summary_op,
        #      network.loss, network.accuracy, network.word_embeddings,
        #      grad_summaries_merged],
        #     feed_dict)

        output_ = [network.update, network.global_step,
                   network.accuracy, network.mean_loss,
                   network.train_summary_op]
        _, current_step, accuracy, loss, net_sum = sess.run(output_, feed_dict)
        # if config['clipping_weights']:
        #     sess.run([weight_clipping])
        # cur_norm = sess.run([fc_layer_norm])
        train_summary_writer.add_summary(net_sum, current_step)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {}, acc {}".format(
            time_str, current_step, loss, accuracy))

        # train_summary_writer.add_summary(summaries, step)
        # grad_summaries_writer.add_summary(grad_summary, step)
        if current_step % config['evaluate_every'] == 0:
            pass
            # dev_step(x_dev, y_dev)

    # dev function
    def dev_step(x_batch, y_batch):
        feed_dict = {
            network.x: x_batch,
            network.y: y_batch,
            network.dropout_prob: 1.0,
            # network.str_summary_type: "",
            network.input_keep_prob: config["keep_prob_inp"],
            network.output_keep_prob: config["keep_prob_out"],
            # network.train_phase: False
        }
        # step, summaries, loss, accuracy = sess.run(
        #     [global_step, dev_summary_op, network.loss, network.accuracy],
        #     feed_dict)
        output_ = [network.global_step, network.accuracy, network.mean_loss]
        current_step, accuracy, loss = sess.run(output_, feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {}, acc {}, b_len {}".format(
            time_str, current_step, loss, accuracy, len(x_batch)))
        # if writer:
        #     writer.add_summary(summaries, step)

    # Generate batches
    print ("About to build batches for x:{} with number of words".format(
        len(x_train), config['n_words']))
    batches = process_utils.batch_iter(
        list(zip(x_train, y_train)), config['batch_size'], config['n_epochs'])

    conf_path = os.path.abspath(os.path.join(out_dir, "config.json"))
    json.dump(config, open(conf_path, 'w'), indent="\t")
    print("Saved configuration file at: {}".format(conf_path))

    print ("train loop starting for every batch")
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        # current_step = tf.train.global_step(sess, global_step)
        # if current_step % config['evaluate_every'] == 0:
        #     print("\nEvaluation:")
        #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
        #     print("")
        # if current_step % config['checkpoint_every'] == 0:
        #     path = saver.save(
        #         sess, checkpoint_prefix, global_step=current_step)
        #     print("Saved model checkpoint to {}\n".format(path))
