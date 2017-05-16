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


def eval_model(sess, g, checkpoint_paths, data, config):

    def test_step(x_batch, y_batch):
        if config['split_dev']:  # will need to split dev set to smaller chunks
            mini_size = config['dev_minibatch']
            acc_sum = 0
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

    def get_attention_weights(x_batch, y_batch, x_strings_batch, filename):
        '''
        save attention weights for a batch of sentences
        '''
        path_ = os.path.join(out_dir, filename)
        scores_list = []
        seq_length_list = []
        if saved_conf['split_dev']:
            mini_size = saved_conf['dev_minibatch']
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
                if saved_conf["dmn"]:
                    num_episodes = saved_conf['episodes_num']
                    output_ = [graph_all_attentions, graph_seq_lengths]
                    scores, seq_lengths = sess.run(
                        output_, feed_dict)
                    sentences_in_batch = []
                    for i in range(len(mini_x_batch)):
                        sentence_scores = []
                        for k in range(saved_conf['sentence_len']):
                            attentions = [float(scores[j][i][k])
                                          for j in range(num_episodes)]
                            sentence_scores.append(attentions)
                        sentences_in_batch.append(sentence_scores)

                    scores_list += sentences_in_batch

                    # sys.exit()
                else:
                    output_ = [graph_attention_scores, graph_seq_lengths]
                    scores, seq_lengths = sess.run(
                        output_, feed_dict)
                    print (len(scores.tolist()))
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
        print (len(scores_list))
        print (seq_length_list , len(seq_length_list))
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
            reversed_text = list(vc_processor.reverse([x_batch[i].tolist()]))
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

    def save_test_summary(x_batch, y_batch, x_strings_batch, name_):
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

                output_ = [graph_predictions, graph_true_predictions,
                           graph_probs, graph_state_]
                predictions, true_pred, probs, fc_layer = sess.run(
                    output_, feed_dict)

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

    # output directory for data
    timestamp = str(int(time.time()))
    out_dir = os.path.join(checkpoint_paths, "..", "evaluations", timestamp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # load data
    dx_train, y_train, dx_test, y_test = data
    vc_path = os.path.join(checkpoint_paths, "..", "vocabulary")
    vc_processor = learn.preprocessing.VocabularyProcessor.restore(vc_path)
    x_test = np.array(list(vc_processor.transform(dx_test)))
    # x_test = np.array(list(vc_processor.transform(dx_train)))
    # y_test = y_train

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
    # or get model from specific run
    num_ = 1500
    if num_ is not None:
        temp = last_model[:last_model.find("model-") + len("model-")]
        last_model = "{}{}".format(temp, num_)

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

    test_step(x_test, y_test)
    if saved_conf["use_attention"] or saved_conf["attention_GRU"] or use_dmn:
        # retrieve approriate operations_by_name
        if use_dmn:
            graph_all_attentions = g.get_operation_by_name(
                "episodic_module/all_attentions_transp").outputs[0]
            graph_seq_lengths = g.get_operation_by_name(
                "word_embeddings/calc_sequences_length_facts/Max").outputs[0]
        else:
            # **TODO**
            # graph_attention_scores = g.get_operation_by_name(
            #     "attention_fc_layer/while/Exit_2").outputs[0]
            # graph_seq_lengths = g.get_operation_by_name(
            #     "calc_sequences_length/Max").outputs[0]
            # plain lstm using attention GRU
            graph_attention_scores = g.get_operation_by_name(
                "attention_fc_layer/attention_calculation/Reshape").outputs[0]
            graph_seq_lengths = g.get_operation_by_name(
                "calc_sequences_length/Max").outputs[0]
        get_attention_weights(x_test, y_test, dx_test, "attention_test_")

    # get operations for prediction
    graph_predictions = g.get_operation_by_name(
        "predict/ArgMax").outputs[0]
    graph_true_predictions = g.get_operation_by_name(
        "predict/ArgMax_1").outputs[0]
    graph_probs = g.get_operation_by_name(
        "predict/Softmax").outputs[0]
    # TODO saveing for lstm without attentions
    if use_dmn:
        if saved_conf['episodes_num'] == 1:
            graph_state_ = g.get_operation_by_name(
                "episodic_module/memory_update/Relu").outputs[0]
        else:
            graph_state_ = g.get_operation_by_name(
                "episodic_module/memory_update_{}/Relu".format(
                    saved_conf['episodes_num'] - 1)).outputs[0]
    else:  # TODO for normal lstm
        # attention_fc_layer/Reshape_3
        if saved_conf['attention_GRU']:
            graph_state_ = g.get_operation_by_name(
                    "attention_fc_layer/attention_GRU/rnn/while/Exit_2").outputs[0]
        else:
            graph_state_ = g.get_operation_by_name(
                    "attention_fc_layer/Reshape_3").outputs[0]          
    save_test_summary(
        x_test, y_test, dx_test, 'metrics_test_{}.pkl'.format(
            last_model[last_model.find("model-") + len("model-"):]))





















