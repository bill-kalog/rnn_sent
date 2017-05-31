import numpy as np
import os
import sys
import io
import pickle
import process_utils
import random
import re
from conf import config

'''
positive labels [0,1]
negative labels [1,0]
'''


class Dataset(object):
    """
    Class representing a single dataset
        self.dataset = raw data sentence/doc per datapoint
        self.tokenized = tokenized data (if available)
        self.labels = labels represented in a single vector
        self.labels_verbose = labels as a one-hot vector
    """
    def __init__(self, dataset_name='', preprocess=False, cv=10):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        if dataset_name.lower() == "sst":  # stanford sentiment treebank
            self.load_sst()
            return
        if dataset_name.lower() == "sst_phrase": # stanford sentiment treebank trained using all phrases as given in https://github.com/harvardnlp/sent-conv-torch
            self.load_sst_phrases()
            return
        elif dataset_name.lower() == "mr":  # movie reviews
            self.load_MR()
        elif dataset_name == "IMDB":  # IMDB dataset maas et al
            # http://ai.stanford.edu/~amaas/data/sentiment/
            self.load_IMDB()
        if preprocess:
            self.tokenized = [process_utils.clean_str(sentence)
                              for sentence in self.dataset]
        # assign each sentence to random fold
        random.seed(8)
        self.folds = [random.randint(0, cv - 1)
                      for _ in range(len(self.labels))]

    def load_sst(self):
        '''
        Stanford sentiment treebank dataset
        TODO: self.labels variable is currently empy for this
        dataset
        '''
        self.labels = np.array([])
        self.labels_verbose = None
        self.dataset = []

        parent_folder = os.path.join(config['dat_directory'],
                                     'stanfordSentimentTreebank')
        dictionary_path = os.path.join(parent_folder, 'dictionary.txt')
        dictionary = {}
        collisions = {}
        with io.open(
            dictionary_path, "r", encoding="utf-8", errors='ignore') as \
                dictionary_data:
            for line in dictionary_data:
                phrase, id_ = line.split('|')

                if phrase in dictionary:
                    if phrase in collisions:
                        print ("fsy: {}".format(phrase))
                        collisions[phrase] = collisions[phrase] + 1
                    else:
                        collisions[phrase] = 1

                dictionary[phrase] = id_.strip()
        print ("dictionary size:{}".format(len(dictionary)))

        sent_labels_path = os.path.join(parent_folder, 'sentiment_labels.txt')
        sentiment_labels = {}
        with io.open(
            sent_labels_path, "r", encoding="utf-8", errors='ignore') as \
                sentiment_data:
            next(sentiment_data)
            for line in sentiment_data:
                id_, sentiment_value = line.split("|")
                sentiment_labels[id_] = float(sentiment_value.strip())
        print ("sentiment labels size:{}".format(len(sentiment_labels)))

        # don't read this file(corrupted), read sostr.txt instead
        # data_sent_path = os.path.join(parent_folder, 'datasetSentences.txt')
        data_sent_path = os.path.join(parent_folder, 'SOStr.txt')
        dataset_sentences = {}
        with io.open(
            data_sent_path, "r", encoding="utf-8", errors='ignore') as \
                d_sentences:
            # next(d_sentences)
            for id_, line in enumerate(d_sentences):
                sentence = line.strip()
                sentence = re.sub(r"\|", " ", sentence)
                dataset_sentences[str(id_ + 1)] = sentence
        print ("number of sent:{}".format(len(dataset_sentences)))

        data_split_path = os.path.join(parent_folder, 'datasetSplit.txt')
        dataset_split = {}
        with io.open(
            data_split_path, "r", encoding="utf-8", errors="ignore") as \
                d_split:
            next(d_split)
            for line in d_split:
                id_, fold_ = line.split(",")
                dataset_split[id_] = int(fold_.strip())
        print ("split size:{}".format(len(dataset_split)))

        self.dataset = list(dataset_sentences.values())

        self.labels_verbose = None
        self.tokenized = []
        self.folds = []
        for sent_index in dataset_sentences:
            sentence = dataset_sentences[sent_index]
            dic_id = dictionary[sentence]
            sentiment = sentiment_labels[dic_id]
            s = process_utils.clean_str_sst(dataset_sentences[sent_index])

            if config['sst_finegrained']:
                temp = np.zeros((1, 5))
                if sentiment <= 0.2:
                    temp[0][0] = 1
                elif sentiment <= 0.4:
                    temp[0][1] = 1
                elif sentiment <= 0.6:
                    temp[0][2] = 1
                elif sentiment <= 0.8:
                    temp[0][3] = 1
                else:
                    temp[0][4] = 1
            else:
                temp = np.zeros((1, 2))
                if sentiment <= 0.4:
                    temp[0][0] = 1
                elif sentiment > 0.6:
                    temp[0][1] = 1
                else:  # sentiment > 0.4 and sentiment < 0.6:
                    continue
            self.tokenized.append(s)

            if self.labels_verbose is None:
                self.labels_verbose = temp
            else:
                self.labels_verbose = np.concatenate(
                    [self.labels_verbose, temp])
            self.folds.append(dataset_split[sent_index])

    def load_sst_phrases(self):
        '''
        load SST with phrases
        code is based and data where taken
        from https://github.com/harvardnlp/sent-conv-torch
        '''
        self.labels = np.array([])
        self.labels_verbose = None
        self.dataset = []
        parent_folder = os.path.join(config['dat_directory'],
                                     'sst_phrase')
        paths = {"SST1": ("stsa.fine.phrases.train",
                          "stsa.fine.test"
                          "stsa.fine.dev",
                          ),
                 "SST2": ("stsa.binary.phrases.train",
                          "stsa.binary.test",
                          "stsa.binary.dev",
                          )
                 }
        if config['sst_finegrained']:
            version_ = "SST1"
        else:
            version_ = "SST2"
        f_names = [os.path.join(parent_folder, file_)
                   for file_ in paths[version_]]
        self.dataset = []
        self.labels_verbose = None
        self.tokenized = []
        self.folds = []
        for num_, fold_ in enumerate(f_names):
            with io.open(
                    fold_, "r", encoding="utf-8", errors='ignore') as data_:
                for line in data_:
                    label, phrase_ = [int(line[0]), line[2:]]
                    self.dataset.append(phrase_)
                    phrase_ = process_utils.clean_str_sst(phrase_)
                    self.tokenized.append(phrase_)

                    if config["sst_finegrained"]:
                        temp = np.zeros((1, 5))
                    else:
                        temp = np.zeros((1, 2))
                    temp[0][label] = 1
                    if self.labels_verbose is None:
                        self.labels_verbose = temp
                    else:
                        self.labels_verbose = np.concatenate(
                            [self.labels_verbose, temp])
                    self.folds.append(num_ + 1)

    def load_IMDB(self):
        self.labels = np.array([])
        self.labels_verbose = None
        self.dataset = []
        # load training and test set
        paths = [os.path.join(config['dat_directory'], 'aclImdb', 'train'),
                 os.path.join(config['dat_directory'], 'aclImdb', 'test')]
        for path in paths:
            for top, direc, f in os.walk(path):
                if top[-3:] == "neg":  # reading negative reviews
                        buf = self.read_folder(top, f)
                        labels = np.ones(len(buf)) * -1
                        labels_v = np.zeros((len(buf), 2))
                        labels_v[:, 0] = 1

                        print ("loaded negative")
                elif top[-3:] == "pos":  # reading positive reviews
                    buf = self.read_folder(top, f)
                    labels = np.ones(len(buf))
                    labels_v = np.zeros((len(buf), 2))
                    labels_v[:, 1] = 1
                    print ("loaded positive")
                else:
                    continue
                self.dataset = self.dataset + buf
                self.labels = np.concatenate([self.labels, labels])
                if self.labels_verbose is None:
                    self.labels_verbose = labels_v
                else:
                    self.labels_verbose = np.concatenate(
                        [self.labels_verbose, labels_v])

    def read_folder(self, top, f):
        dataset = []
        for file in f:
            # print(top, file)
            loc = os.path.join(top, file)
            doc = io.open(loc, "r", encoding="utf-8", errors='replace')
            dataset.append(doc.read())  # .decode('utf-8', 'ignore'))
            doc.close()
        return dataset

    def load_MR(self):
        '''
        movie review dataset
        https://www.cs.cornell.edu/people/pabo/movie-review-data/
        http://www.cs.cornell.edu/home/llee/papers/pang-lee-stars.home.html
        '''
        parent_f = os.path.join(config['dat_directory'], 'MR',
                                'rt-polaritydata')
        path_pos = os.path.join(parent_f, 'rt-polarity.pos')
        path_neg = os.path.join(parent_f, 'rt-polarity.neg')

        # open files for reading
        fo = io.open(path_pos, "r", encoding="utf-8", errors='ignore')
        positive_reviews = fo.read().split("\n")[:-1]
        fo.close()
        fo = io.open(path_neg, "r", encoding="utf-8", errors='ignore')
        negative_reviews = fo.read().split("\n")[:-1]
        fo.close()
        # create dataset
        labels = np.ones(len(positive_reviews) * 2)
        labels[:len(negative_reviews)] = labels[:len(negative_reviews)] * -1
        labels_v = np.zeros((len(positive_reviews) * 2, 2))
        labels_v[:len(negative_reviews), 0] = 1  # verbose version of labels
        labels_v[len(negative_reviews):, 1] = 1
        self.labels = labels
        self.labels_verbose = labels_v
        self.dataset = negative_reviews + positive_reviews
        # print (len(positive_reviews) + len(negative_reviews))

    '''
    get the i-th(index) cross validation split
    format [train, labels_tr, dev, labels_dev]
    '''
    def cv_split(self, index=0, label_type='verbose'):
        train = []
        labels_tr = []
        dev = []
        labels_dev = []
        for i in range(len(self.tokenized)):
            if self.folds[i] == index:
                dev.append(self.tokenized[i])
                if label_type == 'verbose':
                    labels_dev.append(self.labels_verbose[i])
                else:
                    labels_dev.append(self.labels[i])
            else:
                train.append(self.tokenized[i])
                if label_type == 'verbose':
                    labels_tr.append(self.labels_verbose[i])
                else:
                    labels_tr.append(self.labels[i])
        return [train, labels_tr, dev, labels_dev]
        # return [(train, labels_tr), (dev, labels_dev)]

    def save_to_pickle(self, name):
        pickle.dump(self, open(name, "wb"))

    # def load_pickle(self, name):
    #         pass
