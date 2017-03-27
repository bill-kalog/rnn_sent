import tensorflow as tf
from conf import config
from datasets import Dataset
from word_vectors import WordVectors
import train
import sys


dataset = Dataset("SST")
# dataset = Dataset("IMDB", preprocess=True)
# dataset = Dataset("MR", preprocess=True)


# load pretrained word vectors
pretrained_vectors = []
for index, type_ in enumerate(config['word_vector_type']):
    pretrained_vectors.append(WordVectors(
        type_, config["pretrained_vectors"][index]))
    print ("loaded vectors {}".format(config['word_vector_type'][index]))


# data = dataset.cv_split(index=5)

data = dataset.cv_split(index=2)
# # data = dataset.cv_split(index=5)


# dataset_1 = Dataset("MR", preprocess=True)
# sp_1 = dataset_1.cv_split(index=5)

# data = [dataset.tokenized[:25000], dataset.labels_verbose[:25000],
#         dataset.tokenized[25000:], dataset.labels_verbose[25000:]]

# data = [data[0] + sp_1[0], data[1] + sp_1[1], sp_1[2], sp_1[3]]


# data[2] = sp_1[2]
# data[3] = sp_1[3]

# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))

# data = dataset.cv_split(index=1)
# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))

# data = dataset.cv_split(index=3)
# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))


# init and run tf graph
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        train.set_train(
            sess, config, data,
            pretrained_embeddings=pretrained_vectors)
