import time
import datetime
import os
import tensorflow as tf
# from conf import config
import numpy as np
from model import RNN
from model import RNN_Attention
from dmn import DMN
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import learn
import process_utils
import sys
import json
import operator
import pandas as pd
import re


def init_vocabulary_processor(dx_train, dx_dev):
    """
    vocabulary processor
    implementation taken from 
    http://stackoverflow.com/questions/40661684/tensorflow-vocabularyprocessor#40741660
    """
    max_document_length = max([len(x.split(" ")) for x in dx_train])
    if max_document_length > 100:
        max_document_length = 100
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length, tokenizer_fn=process_utils.tokenizer)
    vocab_processor.fit(dx_train)
    x_train = np.array(list(vocab_processor.transform(dx_train)))
    x_dev = np.array(list(vocab_processor.transform(dx_dev)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])

    return x_train, x_dev, vocab_dict, vocabulary, vocab_processor


def init_embeddings(config, pretrained_embeddings, vocabulary):
    init_embd = config['std_dev'] * np.random.randn(
        len(config['word_vector_type']) + 1,
        len(vocabulary), config['dim_proj']
    )
    # sample uniformaly in interval [-std, std] (b-a)*r + a
    # init_embd = 2 * config['std_dev'] * np.random.rand(
    #     len(config['word_vector_type']) + 1,
    #     len(vocabulary), config['dim_proj']
    # ) - config['std_dev']
    # sample uniformaly in interval [-1/(2d), 1/(2d)] (b-a)*r + a
    init_embd = 2 * 1 / (2 * config['dim_proj']) * np.random.rand(
        len(config['word_vector_type']) + 1,
        len(vocabulary), config['dim_proj']
    ) - 1 / (2 * config['dim_proj'])
    if pretrained_embeddings:
        for index_3d, stored_embedding in enumerate(pretrained_embeddings):
            counts = 0
            mappings = {}
            for index, entry in enumerate(vocabulary):
                if entry in stored_embedding.word_to_index:
                    vec_index = stored_embedding.word_to_index[entry]
                    mappings[vec_index] = index
                    counts += 1
                    init_embd[index_3d, index] = \
                        stored_embedding.vectors[vec_index]
            print (" Found {} words in pretrained vectors {} out of {}".format(
                counts, config['word_vector_type'][index_3d], len(vocabulary)))
            stored_embedding.set_mappings(mappings)
    return init_embd


def eval_model(sess, g, checkpoint_paths, data, config):

    def test_step(x_batch, y_batch):
        if config['split_dev']:  # will need to split dev set to smaller chunks
            mini_size = config['dev_minibatch']
            acc_sum = 0
            loss_sum = 0
            for i in range(0, len(x_batch), mini_size):
                if (i + mini_size < len(x_batch)):
                    mini_x_batch = x_batch[i:i + mini_size]
                    mini_y_batch = y_batch[i:i + mini_size]
                else:
                    mini_x_batch = x_batch[i:]
                    mini_y_batch = y_batch[i:]
                dropouts = [1.0, 1.0, 1.0]
                reg_metrics = [1, 0, 0]
                feed_dict = make_feed_dict(
                    mini_x_batch, mini_y_batch,
                    dropouts, reg_metrics, question)


                # output_ = [network.global_step, network.accuracy,
                #            network.mean_loss]
                # output_ = [network.accuracy]
                output_ = [graph_accuracy]

                net_accuracy = sess.run(output_, feed_dict)[0]
                acc_sum += len(mini_x_batch) * net_accuracy
                # loss_sum += len(mini_x_batch) * loss

            # loss = loss_sum / len(x_batch)
            loss = 0
            accuracy = acc_sum / len(x_batch)
            print ("           loss {} accuracy{}".format(loss, accuracy))
            dropouts = [1.0, 1.0, 1.0]
            reg_metrics = [0.0, accuracy, loss]
            feed_dict = make_feed_dict(
                mini_x_batch, mini_y_batch,
                dropouts, reg_metrics, question)
        else:  # use whole batch
            dropouts = [1.0, 1.0, 1.0]
            reg_metrics = [1, 0, 0]
            feed_dict = make_feed_dict(
                x_batch, y_batch,
                dropouts, reg_metrics, question)
        # output_ = [network.global_step, network.accuracy,
        #            network.mean_loss, network.summary_op]
        # output_ = [network.accuracy]
        output_ = [graph_accuracy]
        

        net_accuracy = sess.run(
            output_, feed_dict)[0]
        print("\nEvaluation dev set:")
        time_str = datetime.datetime.now().isoformat()
        print("{}: acc {}, b_len {}\n".format(
            time_str, accuracy, len(x_batch)))

    def make_feed_dict(
            x_batch, y_batch, dropouts, reg_metrics, question=None):
        """ build dictionary for feeding a network

        Args:
            x_batch : x values
            y_batch : y values
            dropouts : all dropouts hyperparameters in a list
                [dropout_prob, input_keep_prob, output_keep_prob]
            reg_metrics: list of weights used for normalizing dev set acc
            question: transformed string representing a question to be asked
                      at a Dynamc Memory Network
        """
        if graph_question is None:

            feed_dict = {
                x_values: x_batch,
                y_values: y_batch,
                dropout_prob: dropouts[0],
                input_keep_prob: dropouts[1],
                output_keep_prob: dropouts[2],
                batch_size: len(x_batch),
                metrics_weight: reg_metrics[0],
                fixed_acc_value: reg_metrics[1],
                fixed_loss_value: reg_metrics[2],
            }

        else:
            # print (question, question.shape)
            # have the exact same question for all batch
            mult_quest = np.reshape(np.repeat(
                question.T, len(x_batch), axis=0),
                [config['sentence_len'], len(x_batch)]).T

            feed_dict = {
                x_values: x_batch,
                y_values: y_batch,
                dropout_prob: dropouts[0],
                input_keep_prob: dropouts[1],
                output_keep_prob: dropouts[2],
                batch_size: len(x_batch),
                metrics_weight: reg_metrics[0],
                fixed_acc_value: reg_metrics[1],
                fixed_loss_value: reg_metrics[2],
                graph_question: mult_quest

            }

        return feed_dict

    # load data
    dx_train, y_train, dx_test, y_test = data
    vc_path = os.path.join(checkpoint_paths, "..", "vocabulary")
    vc_processor = learn.preprocessing.VocabularyProcessor.restore(vc_path)
    x_test = np.array(list(vc_processor.transform(dx_test)))
    x_test = np.array(list(vc_processor.transform(dx_train)))
    y_test = y_train

    string_question = ["what is the sentiment ?"]
    # string_question = ["Give some random question that makes no sense to work . . ."]
    question = np.array(list(vc_processor.transform(string_question)))
    saved_conf_path = os.path.join(checkpoint_paths, "..", "config.json")
    with open(saved_conf_path) as json_data:
        saved_conf = json.load(json_data)

    # retrieve configuration info
    config['n_words'] = saved_conf['n_words']
    config['sentence_len'] = saved_conf['sentence_len']
    use_dmn = saved_conf['dmn']

    # load model
    last_model = tf.train.latest_checkpoint(checkpoint_paths)
    print ("About to load: {}".format(last_model))
    saver = tf.train.import_meta_graph("{}.meta".format(last_model))
    saver.restore(sess, last_model)
    print ("Model loaded")


    # Get placeholders
    x_values = g.get_operation_by_name('x').outputs[0]
    y_values = g.get_operation_by_name('y').outputs[0]
    dropout_prob = g.get_operation_by_name("dropout_prob").outputs[0]
    input_keep_prob = g.get_operation_by_name("keep_prob_inp").outputs[0]
    output_keep_prob = g.get_operation_by_name("keep_prob_out").outputs[0]
    batch_size = g.get_operation_by_name("batch_size").outputs[0]
    metrics_weight = g.get_operation_by_name("metrics_weight").outputs[0]
    fixed_acc_value = g.get_operation_by_name("f_acc_value").outputs[0]
    fixed_loss_value = g.get_operation_by_name("f_loss_value").outputs[0]
    if use_dmn:  # if DMN we have a question placeholder too
        graph_question = g.get_operation_by_name("question_").outputs[0]
    else:
        graph_question = None

    graph_accuracy = g.get_operation_by_name("accuracy/accuracy").outputs[0]

    # print (x_values, y_values ,dropout_prob ,input_keep_prob ,output_keep_prob ,batch_size ,metrics_weight ,fixed_acc_value ,fixed_loss_value ,question )
    # print (graph_accuracy)


    
    # print (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    # print (tf.global_variables)
    # print ("TRAINABLE VARIABLES")
    # for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #     print (i.name)
    #     print (sess.run(i))
    #     # if (i.name == "answer_module/W_answer:0"):
    #     #     print (i)
    #     #     print (sess.run(i))
    #     #     a = i
    # print ("MODEL VARIABLES")
    # for i in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES):
    #     print (i.name)
    # print ("GLOABAL VARIABLES")

    # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     if (i.name == "attention_mechanism/rnn/multi_rnn_cell/cell_0/attention_based_gru_cell/gates/weights:0"):
    #         b = i
    #         print (sess.run(b))
    #     print (i.name)
        
    # print ("ACTIVATIONS")
    # for i in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
    #     print (i.name)
    # print (" __________  ")
    
    # print (a)
    # print (sess.run(a))
    # print (sess.run(b))
    # # all_vars = tf.get_collection('weights')
    # # for var in all_vars:
    # #     print (var)





    
    # build model graph
    # if config['dmn']:
    #     network = DMN(config)
    # elif config["use_attention"]:
    #     network = RNN_Attention(config)
    #     question = None
    # else:
    #     network = RNN(config)
    #     question = None

    # last_model = tf.train.latest_checkpoint(checkpoint_paths)
    # print ("About to load: {}".format(last_model))
    # saver = tf.train.import_meta_graph("{}.meta".format(last_model))
    # saver.restore(sess, last_model)
    # sess.run(tf.global_variables_initializer())
    # network.load(sess, checkpoint_paths)


    # print ("Model loaded")
    test_step(x_test, y_test)






def set_train(sess, config, data, pretrained_embeddings=[]):

    dx_train, y_train, dx_dev, y_dev = data

    x_train, x_dev, vocab_dict, vocabulary, vc_processsor = \
        init_vocabulary_processor(dx_train, dx_dev)

    print("Vocabulary Size: {}".format(len(vocabulary)))
    print("Train/Dev split: {}/{},{}".format(
        len(y_train), len(y_dev), len(y_train) + len(y_dev)))

    # Build word embeddings
    config['n_words'] = len(vocabulary)
    config['sentence_len'] = x_train.shape[1]

    word_embd_tensor = init_embeddings(
        config, pretrained_embeddings, vocabulary)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    config['out_dir'] = out_dir
    print("Writing to {}\n".format(out_dir))

    if config['dmn']:
        network = DMN(config, word_embd_tensor)
        string_question = ["what is the sentiment ?"]
        question = np.array(list(vc_processsor.transform(string_question)))
    elif config["use_attention"]:
        network = RNN_Attention(config, word_embd_tensor)
        question = None
    else:
        network = RNN(config, word_embd_tensor)
        question = None

    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(
        dev_summary_dir, sess.graph)

    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(
        train_summary_dir, sess.graph)

    # checkpoints
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_snap = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # Write vocabulary
    vc_processsor.save(os.path.join(out_dir, "vocabulary"))

    sess.run(tf.global_variables_initializer())

    # train fucntion
    def train_step(x_batch, y_batch, iter_):
        dropouts = [config["dropout_rate"], config["keep_prob_inp"], config["keep_prob_out"]]
        reg_metrics = [1, 0, 0]
        feed_dict = make_feed_dict(
            x_batch, y_batch,
            dropouts, reg_metrics, question)

        if (iter_ % 100 == 99):  # record full summaries:
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            output_ = [network.update, network.global_step,
                       network.accuracy, network.mean_loss,
                       network.summary_op]
            _, current_step, accuracy, loss, net_sum = sess.run(
                output_, feed_dict, options=run_options,
                run_metadata=run_metadata)

            train_summary_writer.add_run_metadata(
                run_metadata, 'step%d' % current_step)
        else:
            # a = []
            # shape_ = [4, 3]
            # seq_length = 5
            # batches = shape_[0]
            # for i in range(seq_length):
            #     arr_ = np.random.randn(shape_[0], shape_[1]) * i
            #     tens_ = tf.get_variable(
            #         name="a_name_" + str(i),
            #         shape=shape_,
            #         initializer=tf.constant_initializer(arr_)
            #     )
            #     a.append(tens_)
            # b = tf.stack(a)
            # shape_2 = [4, 5, 3]
            # res_b = tf.reshape(
            #     b, shape_2)

            # reshaped_tensor_2 = [b[:, i, :] for i in range(batches)]

            # sess.run(tf.global_variables_initializer())

            # # print debug info about outputs of rnn and attentio 
            # get_ = [network.output, network.out_state, network.seq_lengths , network.attention_scores]
            # outp_, out_st_, lengths_, att_scores_ = sess.run(get_, feed_dict)
            # print (outp_)
            # print (out_st_)
            # print (lengths_)

            # print (x_batch[0])
            # print (outp_[0][0])
            # print ("states ", out_st_[0])
            # print ("an output ", outp_[0][0])
            # print (" scores ", att_scores_[0])
            # print (" outputs shape ", outp_[0].shape, len(outp_))

            # sys.exit(0)

            
            output_ = [network.update, network.global_step,
                       network.accuracy, network.mean_loss,
                       network.summary_op]
            _, current_step, accuracy, loss, net_sum = sess.run(
                output_, feed_dict)

        if config['save_step'] == current_step:
            # save word embeddings
            emb_m = sess.run([network.w_embeddings], feed_dict)
            save_embedding(emb_m)
        # write train summary
        train_summary_writer.add_summary(net_sum, current_step)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {}, acc {}, b_len {}".format(
            time_str, current_step, loss, accuracy, len(x_batch)))

        # train_summary_writer.add_summary(summaries, step)
        # grad_summaries_writer.add_summary(grad_summary, step)
        if current_step % config['evaluate_every'] == 0:
            dev_step(x_dev, y_dev)

        if current_step in config['save_step_dev_info']:
            if config['classes_num'] == 2:  # plots work only for binary 
                save_dev_summary(
                    x_dev, y_dev, dx_dev,
                    "metrics_step_{}.pkl".format(current_step))
                save_dev_summary(
                    x_train, y_train, dx_train,
                    "metrics_train_step_{}.pkl".format(current_step))
            if config["use_attention"]:
                get_attention_weights(
                    x_dev, y_dev, dx_dev,
                    "attention_step_{}".format(current_step))
                get_attention_weights(
                    x_train, y_train, dx_train,
                    "attention_train_step_{}".format(current_step))
            # sys.exit(0)

    def dev_step(x_batch, y_batch):
        if config['split_dev']:  # will need to split dev set to smaller chunks
            mini_size = config['dev_minibatch']
            acc_sum = 0
            loss_sum = 0
            for i in range(0, len(x_batch), mini_size):
                if (i + mini_size < len(x_batch)):
                    mini_x_batch = x_batch[i:i + mini_size]
                    mini_y_batch = y_batch[i:i + mini_size]
                else:
                    mini_x_batch = x_batch[i:]
                    mini_y_batch = y_batch[i:]
                dropouts = [1.0, 1.0, 1.0]
                reg_metrics = [1, 0, 0]
                feed_dict = make_feed_dict(
                    mini_x_batch, mini_y_batch,
                    dropouts, reg_metrics, question)

                # ot_1 = [network.output, network.seq_lengths]
                # out, seq_lens = sess.run(ot_1, feed_dict)

                output_ = [network.global_step, network.accuracy,
                           network.mean_loss]
                current_step, accuracy, loss = sess.run(output_, feed_dict)
                acc_sum += len(mini_x_batch) * accuracy
                loss_sum += len(mini_x_batch) * loss

            loss = loss_sum / len(x_batch)
            accuracy = acc_sum / len(x_batch)
            print ("           loss {} accuracy{}".format(loss, accuracy))
            dropouts = [1.0, 1.0, 1.0]
            reg_metrics = [0.0, accuracy, loss]
            feed_dict = make_feed_dict(
                mini_x_batch, mini_y_batch,
                dropouts, reg_metrics, question)
        else:  # use whole batch
            dropouts = [1.0, 1.0, 1.0]
            reg_metrics = [1, 0, 0]
            feed_dict = make_feed_dict(
                x_batch, y_batch,
                dropouts, reg_metrics, question)

        output_ = [network.global_step, network.accuracy,
                   network.mean_loss, network.summary_op]
        current_step, accuracy, loss, net_sum = sess.run(
            output_, feed_dict)
        # save summary
        dev_summary_writer.add_summary(net_sum, current_step)

        print("\nEvaluation dev set:")
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {}, acc {}, b_len {}\n".format(
            time_str, current_step, loss, accuracy, len(x_batch)))
        # if writer:
        #     writer.add_summary(summaries, step)

    def make_feed_dict(
            x_batch, y_batch, dropouts, reg_metrics, question=None):
        """ build dictionary for feeding a network

        Args:
            x_batch : x values
            y_batch : y values
            dropouts : all dropouts hyperparameters in a list
                [dropout_prob, input_keep_prob, output_keep_prob]
            reg_metrics: list of weights used for normalizing dev set acc
            question: transformed string representing a question to be asked
                      at a Dynamc Memory Network
        """
        if question is None:
            feed_dict = {
                network.x: x_batch,
                network.y: y_batch,
                network.dropout_prob: dropouts[0],
                network.input_keep_prob: dropouts[1],
                network.output_keep_prob: dropouts[2],
                network.batch_size: len(x_batch),
                network.metrics_weight: reg_metrics[0],
                network.fixed_acc_value: reg_metrics[1],
                network.fixed_loss_value: reg_metrics[2]
            }
            # network.setBatchSize(len(x_batch))
        else:
            # print (question, question.shape)
            # have the exact same question for all batch
            mult_quest = np.reshape(np.repeat(
                question.T, len(x_batch), axis=0),
                [config['sentence_len'], len(x_batch)]).T

            # print ("mult quest: {}".format(mult_quest))
            # print (" shape {}".format(mult_quest.shape))
            feed_dict = {
                network.x: x_batch,
                network.y: y_batch,
                network.dropout_prob: dropouts[0],
                network.input_keep_prob: dropouts[1],
                network.output_keep_prob: dropouts[2],
                network.batch_size: len(x_batch),
                network.metrics_weight: reg_metrics[0],
                network.fixed_acc_value: reg_metrics[1],
                network.fixed_loss_value: reg_metrics[2],
                network.question: mult_quest
            }
        #     network.setBatchSize(len(x_batch))
        # print ("New batch size {}".format(network.batch_size))
        return feed_dict


    def get_attention_weights(x_batch, y_batch, x_strings_batch, filename):
        '''
        save attention weights for a batch of sentences
        '''
        path_ = os.path.join(out_dir, filename)
        scores_list = []
        seq_length_list = []
        if config['split_dev']:
            mini_size = config['dev_minibatch']
            for i in range(0, len(x_batch), mini_size):
                if (i + mini_size < len(x_batch)):
                    mini_x_batch = x_batch[i:i + mini_size]
                    mini_y_batch = y_batch[i:i + mini_size]
                else:
                    mini_x_batch = x_batch[i:]
                    mini_y_batch = y_batch[i:]
                dropouts = [1.0, 1.0, 1.0]
                reg_metrics = [1, 0, 0]
                feed_dict = make_feed_dict(
                    mini_x_batch, mini_y_batch,
                    dropouts, reg_metrics, question)

                # output_ = [network.predictions, network.true_predictions,
                #            network.probs, network.state_]
                if config["dmn"]:
                    num_episodes = config['episodes_num']
                    output_ = [network.all_attentions, network.seq_lengths]
                    scores, seq_lengths = sess.run(
                        output_, feed_dict)

                    sentences_in_batch = []
                    for i in range(len(mini_x_batch)):
                        sentence_scores = []
                        for k in range(config['sentence_len']):
                            attentions = [float(scores[j][i][k][0])
                                    for j in range(num_episodes)]
                            sentence_scores.append(attentions)
                        sentences_in_batch.append(sentence_scores)
                    
                    # print ("---------------------------")
                    # print (len(sentences_in_batch))
                    # print (sentences_in_batch[0])
                    # print (len(sentences_in_batch[0]))
                    # print (len(sentences_in_batch), len(sentences_in_batch[0]), len(sentences_in_batch[0][0]))
                    # print (scores_combo[1])
                    # print (scores[1])
                    # print (a, a.shape)
                    # for cc in range(100):
                    #     print(np.sum(scores[0][cc][:]), seq_lengths[cc])
                    scores_list += sentences_in_batch

                    # sys.exit()
                else:
                    output_ = [network.attention_scores, network.seq_lengths]
                    scores, seq_lengths = sess.run(
                        output_, feed_dict)
                    scores_list += scores.tolist()
                # print ("seqeunce len ------------", scores, scores.tolist())
                seq_length_list += seq_lengths.tolist()
            # print (len(scores_list))
            # print (len(seq_length_list))
            # sys.exit()

        else:
            print (
                "doesn't support having input as a single batch!! Set:"
                "config['split_dev'] to True ")
            sys.exit(1)
        # Build a dictionary to save at json format
        # adds some overhead
        dic_ = {}
        word_to_id = {}
        word_id = -1
        word_id_to_occ_num = {}  # number of occurences per word
        word_id_to_prob_sum = {}  # probability sum of words

        for i in range(len(x_strings_batch)):
            # uncomment to save sentence and attention seperately
            # dic_['sent_' + str(i)] = {
            #     "sentence": x_strings_batch[i], 'attention': scores_list[i]}

            # OR
            # save info in tuple pairs (attention_pro, word) and sum
            temp = []
            sum_ = 0
            # transform to text from vocab_processor
            reversed_text = list(vc_processsor.reverse([x_batch[i].tolist()]))
            rv_text_to_list = reversed_text[0].split()

            for index_ in range(seq_length_list[i]):
                word = rv_text_to_list[index_]
                if index_ >= len(scores_list[i]):
                    # sentence bigger than max length
                    break
                temp.append(
                    (word, scores_list[i][index_]))
                sum_ += np.sum(scores_list[i][index_])
                # store word probabilities
                if word not in word_to_id:
                    word_id += 1
                    word_to_id[word] = word_id
                word_id_to_occ_num[word_to_id[word]] = word_id_to_occ_num.get(
                    word_to_id[word], 0) + 1
                word_id_to_prob_sum[word_to_id[word]] = word_id_to_prob_sum.get(
                    word_to_id[word], 0.0) + np.sum(scores_list[i][index_])

            dic_['sent_id_' + str(i)] = {
                "mappings": temp,
                "prob_sum": sum_,
                "sent_length": seq_length_list[i],
                # "sent2num": x_batch[i].tolist(),
                "reversed": reversed_text,
                "sentence": x_strings_batch[i],
            }

        # sort dictionaries by word_id values
        # sort by values
        words = sorted(word_to_id.items(), key=operator.itemgetter(1))
        # sort by key
        occurences = sorted(
            word_id_to_occ_num.items(), key=operator.itemgetter(0))
        probabilities_ = sorted(
            word_id_to_prob_sum.items(), key=operator.itemgetter(0))
        words = np.asarray(words)
        d = {'id_': words[:, 1], 'word': words[:, 0],
             'occurences': np.asarray(occurences)[:, 1],
             'probabilities': np.asarray(probabilities_)[:, 1]}
        df = pd.DataFrame(data=d)
        # df['probabilities'] = df['probabilities'].astype(float)
        df['mean'] = df['probabilities'] / df['occurences']  # get mean prob
        df = df.sort_values(by='mean', ascending=False)
        # print (df)
        df.to_json(
            path_or_buf=path_ + "_vocab.json", orient='records')
        # df.to_json(
        #     path_or_buf=path_ + "_vocab.json", orient='records', lines=True)
        # json.dump(df, open(path_ + "_vocab.json", 'w'), indent="\t")

        json.dump(dic_, open(path_ + ".json", 'w'), indent="\t")
        print("Saved attention weights file at: {}".format(path_ + ".json"))

    def save_embedding(embd_matrix):
        '''
        save word embeddings in tf appropriate format
        '''
        summary_path = os.path.join(out_dir, 'summaries', 'embeddings')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        # store metadata
        metadata_path = os.path.join(
            summary_path, 'metadata.tsv')
        with open(metadata_path, 'w') as metadata_file:
            for row in vocabulary:
                metadata_file.write('{}\n'.format(row))

        embd_tensor = []

        writer = tf.summary.FileWriter(summary_path, sess.graph)
        configuration = projector.ProjectorConfig()
        for i_, sub_emb_tensor in enumerate(embd_matrix):
            w_var = tf.Variable(sub_emb_tensor, name='embd_' + str(i_))
            embd_tensor.append(w_var)
            sess.run(w_var.initializer)

            embedding = configuration.embeddings.add()
            embedding.tensor_name = w_var.name
            embedding.metadata_path = metadata_path
            projector.visualize_embeddings(
                writer, configuration)
        sess.run(embd_tensor)
        saver = tf.train.Saver(embd_tensor)
        saver.save(sess, os.path.join(
            summary_path, 'embedding_.ckpt'))

    def save_dev_summary(x_batch, y_batch, x_strings_batch, name_):
        '''
        save info for a batch in order to plot in
        bokeh later
        '''
        path_ = os.path.join(out_dir, name_)
        y_net = []
        prob_net = []
        layer = []
        true_labels = []
        if config['split_dev']:
            mini_size = config['dev_minibatch']
            for i in range(0, len(x_batch), mini_size):
                if (i + mini_size < len(x_batch)):
                    mini_x_batch = x_batch[i:i + mini_size]
                    mini_y_batch = y_batch[i:i + mini_size]
                else:
                    mini_x_batch = x_batch[i:]
                    mini_y_batch = y_batch[i:]
                dropouts = [1.0, 1.0, 1.0]
                reg_metrics = [1, 0, 0]
                feed_dict = make_feed_dict(
                    mini_x_batch, mini_y_batch,
                    dropouts, reg_metrics, question)

                output_ = [network.predictions, network.true_predictions,
                           network.probs, network.state_]
                predictions, true_pred, probs, fc_layer = sess.run(
                    output_, feed_dict)
                # print (predictions)
                # print (probs)
                # print (fc_layer)
                # print (fc_layer.shape)
                # print (true_pred)
                prob_net += probs.tolist()
                layer += fc_layer.tolist()
                y_net += predictions.tolist()
                true_labels += true_pred.tolist()

        else:
            print (
                "doesn't support having input as a single batch!! Set:"
                "config['split_dev'] to True ")
            sys.exit(1)

        # print (
        #     len(x_strings_batch), len(true_labels), len(y_net),
        #     len(prob_net), len(layer))
        process_utils.save_info(
            x_strings_batch, true_labels, y_net, prob_net, layer, path_)

    # Generate batches
    print ("About to build batches for x:{} with number of words".format(
        len(x_train), config['n_words']))
    batches = process_utils.batch_iter(
        list(zip(x_train, y_train)), config['batch_size'], config['n_epochs'])

    conf_path = os.path.abspath(os.path.join(out_dir, "config.json"))
    json.dump(config, open(conf_path, 'w'), indent="\t")
    print("Saved configuration file at: {}".format(conf_path))

    print ("train loop starting for every batch")
    for iter_, batch in enumerate(batches):
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, iter_)
        if iter_ % config['checkpoint_every'] == 0:
            # network.save(sess, checkpoint_prefix, iter_)
            s_path = save_snap.save(sess, checkpoint_prefix, global_step=iter_)
            print ("Saved model snapshot in {}\n".format(s_path))
