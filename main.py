import tensorflow as tf
from conf import config
from datasets import Dataset
from word_vectors import WordVectors
import numpy as np
import train
import evaluate
import sys


dataset = Dataset("SST")
# dataset = Dataset("SST_phrase")
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

# SST splits (take only train and test)
# 1 = train
# 2 = test
# 3 = dev
# t1 = dataset.cv_split(index=2)
t1 = dataset.cv_split(index=3)
t2 = dataset.cv_split(index=1)
data = [t2[2], t2[3], t1[2], t1[3]]
# t_test_x = ["whether you 're moved", "whether you 're moved and love it" ,"whether you 're moved and love it , or bored", "whether you 're moved and love it , or bored or frustrated about the film", "whether you 're moved and love it , or bored or frustrated about the film , you 'll still feel something", "you 'll still feel something"]
# t_test_y = np.asarray([[1., 0.]] * 6)

# t_test_x = ["i like it", "i like it a lot !", "i hate it", "I hate it so much",
#             "the movie is good", "the movie is incredibly good", "good",
#             "not good", "bad", "not bad", "like", "n't like",
#             "i hate the movie though the plot is interesting",
#             "i like the movie though the plot is boring"]

# neg_ = [1., 0.]
# pos_ = [0., 1.]
# labels_ = [pos_, pos_, neg_, neg_, pos_, pos_, pos_, neg_,
#            neg_, pos_, pos_, neg_, neg_, pos_]
# t_test_y = np.asarray(labels_)
# data = [t2[2], t2[3], t_test_x, t_test_y]


# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))
# sys.exit()

# print (data[2][1])
# print (data[3][1])
# sys.exit()
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
g = tf.Graph()
with g.as_default():
    sess = tf.Session()
    with sess.as_default():
        if config['eval']:
            evaluate.eval_model(
                sess, g, config["load_last_checkpoint"], data, config,
            )
        else:
            train.set_train(
                sess, config, data,
                pretrained_embeddings=pretrained_vectors)
