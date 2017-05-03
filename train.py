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

    tf.train.Saver(tf.global_variables())

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


            # get_ = [network.r, network.a_list, network.z_, network.seq_lengths]
            # r, a_list, out_st_, z_ = sess.run(
            #     get_, feed_dict)
            # print (r)
            # print (a_list)
            # print (z_)
            # sys.exit(0)

            # get_ = [network.update, network.r, network.a_list, network.z_, network.seq_lengths, network.unormalized_att_scores]
            # _, r, a_list,  z_, out_st_, un_scores = sess.run(
            #     get_, feed_dict)
            # print ("R values: {} R shape: {}".format(r, r.shape))
            # print ("a_list values: {} a_list shape: {}".format(a_list, a_list.shape))
            # print ("Z ", z_)
            # print (" unrome scores {}".format(un_scores[0]))
            # print ("sequence lengths: {}".format(out_st_))
            # print ("summation of Z {} shape {}".format(np.sum(z_), z_.shape))
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
        if current_step == config['save_step_dev_info']:
            save_dev_summary(x_dev, y_dev, dx_dev, "metrics.pkl")
            save_dev_summary(x_train, y_train, dx_train, "metrics_train.pkl")
            if config["use_attention"]:
                get_attention_weights(x_dev, y_dev, dx_dev, "attention")
                get_attention_weights(
                    x_train, y_train, dx_train, "attention_train")
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
                    output_ = [network.all_attentions, network.seq_lengths, network.a]
                    scores, seq_lengths, a = sess.run(
                        output_, feed_dict)
                    # print (scores)
                    # print (scores[0].shape)
                    # print (len(scores))
                    # print ("---------------------------")
                    # # print (scores[1])
                    # print (a, a.shape)
                    # for cc in range(100):
                    #     print(np.sum(scores[0][cc][:]), seq_lengths[cc])
                    # sys.exit()
                else:
                    output_ = [network.attention_scores, network.seq_lengths]
                    scores, seq_lengths = sess.run(
                        output_, feed_dict)
                scores_list += scores.tolist()
                # print ("seqeunce len ------------", scores, scores.tolist())
                seq_length_list += seq_lengths.tolist()

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
                sum_ += scores_list[i][index_]
                # store word probabilities
                if word not in word_to_id:
                    word_id += 1
                    word_to_id[word] = word_id
                word_id_to_occ_num[word_to_id[word]] = word_id_to_occ_num.get(
                    word_to_id[word], 0) + 1
                word_id_to_prob_sum[word_to_id[word]] = word_id_to_prob_sum.get(
                    word_to_id[word], 0.0) + float(scores_list[i][index_])

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
        # current_step = tf.train.global_step(sess, global_step)
        # if current_step % config['evaluate_every'] == 0:
        #     print("\nEvaluation:")
        #     dev_step(x_dev, y_dev, writer=dev_summary_writer)
        #     print("")
        # if current_step % config['checkpoint_every'] == 0:
        #     path = saver.save(
        #         sess, checkpoint_prefix, global_step=current_step)
        #     print("Saved model checkpoint to {}\n".format(path))
