import numpy as np
import os
import sys
import io
import pickle
import pandas as pd
import csv
import struct


class WordVectors(object):
    """
    Class representing an instance of some word vector,
    assumes word vectors are unzipped and existing in some file
    given in pathfile
    several links to pretrained vectors are here
    https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models
    """
    def __init__(self, method, pathfile):
        super(WordVectors, self).__init__()
        self.method = method
        self.pathfile = pathfile
        if method.lower() == "glove":
            self.load_glove()
        if method.lower() == "w2v":
            self.load_w2v()
        if method.lower() == "fasttext":
            self.load_fastText()
        if method.lower() == "levy":
            self.load_levy()
        if method.lower() == "from_model":
            self.load_glove()

    def load_glove(self):
        '''
        load pretrained word vectors from
            https://github.com/stanfordnlp/GloVe
        '''
        # f = io.open(self.pathfile, "r", encoding="utf-8", errors='ignore')
        df = pd.read_csv(
            self.pathfile, sep='\n', delimiter=" ", header=None,
            quoting=csv.QUOTE_NONE, encoding='utf-8')
        self.dictionary = list(df.loc[:, 0])
        self.word_to_index = {
            word: i for i, word in enumerate(self.dictionary)}
        self.vectors = np.array(df.loc[:, 1:])

    def load_fastText(self):
        '''
        fastText pretrained vectors from
            https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md    
        '''
        df = pd.read_csv(
            self.pathfile, sep='\n', delimiter=' ', header=None,
            quoting=csv.QUOTE_NONE, encoding='utf-8', skiprows=1)
        self.dictionary = list(df.loc[:, 0])
        self.word_to_index = {
            word: i for i, word in enumerate(self.dictionary)}
        self.vectors = np.array(df.loc[:, 1:300])

    def load_w2v(self):
        '''
        load pretrained word2vec vectors
        follows : https://gist.github.com/ottokart/673d82402ad44e69df85
        '''
        vectors = []
        self.dictionary = []
        self.word_to_index = {}
        FLOAT_SIZE = 4  # 32bit float
        with io.open(self.pathfile, 'rb') as f:
            c = None

            # read the header
            header = ""
            while c != "\n":
                c = f.read(1).decode()
                header += c
            total_num_vectors, vector_len = (int(x) for x in header.split())
            print ("Number of vectors: {}".format(total_num_vectors))
            print ("Vector size: {}".format(vector_len))

            while len(vectors) < total_num_vectors:

                word = ""
                while True:
                    c = f.read(1).decode("utf-8", "ignore")
                    if c == " ":
                        break
                    word += c
                # print ("word:{}".format(word))
                self.dictionary.append(word)
                self.word_to_index[word] = len(vectors)

                binary_vector = f.read(FLOAT_SIZE * vector_len)
                vectors.append([struct.unpack_from(
                    'f', binary_vector, i)[0]
                    for i in range(0, len(binary_vector), FLOAT_SIZE)])

        self.vectors = np.array(vectors)

    def load_levy(self):
        '''
        use word vectors from (vector size 300)
        https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
        '''
        df = pd.read_csv(
            self.pathfile, sep="\n", delimiter=' ', header=None,
            quoting=csv.QUOTE_NONE, encoding='utf-8')
        self.dictionary = list(df.loc[:, 0])
        self.word_to_index = {
            word: i for i, word in enumerate(self.dictionary)}
        self.vectors = np.array(df.loc[:, 1:])

    def set_mappings(self, mappings):
        """
        dictionary mapping some/all wordVectors ids to some
        VocabularyProcessor ids { wV_1: mapping_1, wv_2: mapping_2 }
        """
        self.mappings = mappings
