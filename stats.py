import numpy as np
import sys
import matplotlib.pyplot as plt
import collections
import pandas as pd
import itertools
'''
plot range of values for the different classes
'''

path_ = 'metrics.pkl'
path_ = 'metrics_attS.pkl'
path_ = 'metrics_attGRU.pkl'
path_ = "./representations/DMN_2_f.pkl"
path_ = "./representations/RNN_GRU_binary.pkl"
path_ = "./representations/DMN_1_f.pkl"
path_ = "./representations/RNN_GRU_f.pkl"
df = pd.read_pickle(path_)
shape_ = df.shape


# all in single plot
fig = plt.figure()
axes = []
subplots_num = 4
mins = []

# for i in range(df.shape[0]):
#     sent_sub_phrase = "it 's somewhat clumsy and too lethargically paced"
#     cand_sent = df.get("x_dev")[i]
#     if sent_sub_phrase in cand_sent:
#         print ("was:{} pred:{}, sent: {}".format(df.get("y_dev")[i], df.get("y_net")[i], cand_sent))
# sys.exit()


# get all possible labelings (actual, predicted)
# combinations = list(itertools.product([0, 1], [0, 1]))
labels = {0: "negative", 1: "positive"}
labels = {0: "very negative", 1: "negative", 2: "neutral", 3: "positive",
          4: "very positive"}
combinations = list(
    itertools.product(
        range(len(labels)), range(len(labels))))
plot_counter = 0
for index_, label in enumerate(combinations):
    if not label[0] == label[1]:
        continue
    sentences = df[(df['y_dev'] == label[0]) & (df['y_net'] == label[1])]
    # if not sentences.size > 0:
    #     continue
    # get representations first as a list of lists
    representations = list(sentences.get("layer"))
    # then convert to numpy array
    print (len(representations))
    representations = np.asarray(representations)
    # get mean of each dimension of the representation
    mean_ = np.mean(representations, 0)
    # get min of each dimension
    min_ = np.min(representations, 0)
    # and max
    max_ = np.max(representations, 0)

    #  plot in a single plot
    # ax = fig.add_subplot(4, label[0] + 1, label[1] + 1)
    # ax = fig.add_subplot(4, 1, index_ + 1)
    ax = fig.add_subplot(5, 1, plot_counter + 1)
    plot_counter += 1
    axes.append(ax)

    plt.plot(mean_, "-o", label="mean")
    plt.plot(max_, "-o", label="max")
    plt.plot(min_, "-o", label="min")
    plt.legend()
    plt.title(
        "Sentence representations, predicted {} was {}".format(
            labels[label[1]], labels[label[0]]))

    # or seperately
    # plt.show()
# plt.show()

fig_2 = plt.figure()
plot_counter = 0
for index_, label in enumerate(combinations):
    if not label[0] == label[1]:
        continue
    sentences = df[(df['y_dev'] == label[0]) & (df['y_net'] == label[1])]
    # get representations first as a list of lists
    representations = list(sentences.get("layer"))
    # then convert to numpy array
    representations = np.asarray(representations)
    # get mean of each dimension of the representation
    mean_ = np.mean(representations, 0)
    # get min of each dimension
    min_ = np.min(representations, 0)
    # and max
    max_ = np.max(representations, 0)

    #  plot in a single plot
    # ax = fig.add_subplot(4, label[0] + 1, label[1] + 1)
    # ax = fig.add_subplot(4, 1, index_ + 1) # binary
    ax = fig.add_subplot(5, 1, plot_counter + 1)
    plot_counter += 1

    axes.append(ax)
    zeros = [i for i, value_ in enumerate(mean_) if value_ == 0]
    print (
        "Sentence representations, predicted {} was {}".format(
            labels[label[1]], labels[label[0]]))
    print (zeros)

    plt.plot(
        mean_, "-o", label="mean [pr:{}, act:{}]".format(
            labels[label[1]], labels[label[0]]))
    # plt.plot(max_, "-o", label="max")
    # plt.plot(min_, "-o", label="min")
    plt.legend()
    plt.title(
        "Sentence representations")



fig_2 = plt.figure()
plot_counter = 0
combinations = [(0, 0), (4, 4)]
for index_, label in enumerate(combinations):
    if not label[0] == label[1]:
        continue
    sentences = df[(df['y_dev'] == label[0]) & (df['y_net'] == label[1])]
    # get representations first as a list of lists
    representations = list(sentences.get("layer"))
    # then convert to numpy array
    representations = np.asarray(representations)
    # get mean of each dimension of the representation
    mean_ = np.mean(representations, 0)
    # get min of each dimension
    min_ = np.min(representations, 0)
    # and max
    max_ = np.max(representations, 0)

    #  plot in a single plot
    # ax = fig.add_subplot(4, label[0] + 1, label[1] + 1)
    # ax = fig.add_subplot(4, 1, index_ + 1) # binary
    ax = fig.add_subplot(5, 1, plot_counter + 1)
    plot_counter += 1

    axes.append(ax)
    zeros = [i for i, value_ in enumerate(mean_) if value_ == 0]
    print (
        "Sentence representations, predicted {} was {}".format(
            labels[label[1]], labels[label[0]]))
    print (zeros)

    plt.plot(
        mean_, "-o", label="mean [pr:{}, act:{}]".format(
            labels[label[1]], labels[label[0]]))
    # plt.plot(max_, "-o", label="max")
    # plt.plot(min_, "-o", label="min")
    plt.legend()
    plt.title(
        "Sentence representations")
plt.show()




fig_3 = plt.figure()
for index_, label in enumerate(combinations):
    sentences = df[(df['y_dev'] == label[0]) & (df['y_net'] == label[1])]
    if not label[0] == label[1]:
        continue
    # get representations first as a list of lists
    representations = list(sentences.get("layer"))
    # then convert to numpy array
    representations = np.asarray(representations)
    # get mean of each dimension of the representation
    mean_ = np.mean(representations, 0)
    # get min of each dimension
    min_ = np.min(representations, 0)
    # and max
    max_ = np.max(representations, 0)

    #  plot in a single plot
    # ax = fig.add_subplot(4, label[0] + 1, label[1] + 1)
    ax = fig.add_subplot(4, 1, index_ + 1)
    axes.append(ax)
    zeros = [i for i, value_ in enumerate(mean_) if value_ == 0]
    print (
        "Sentence representations, predicted {} was {}".format(
            labels[label[1]], labels[label[0]]))
    print (zeros)

    plt.plot(
        mean_, "-o", label="mean [pr:{}, act:{}]".format(
            labels[label[1]], labels[label[0]]))
    # plt.plot(max_, "-o", label="max")
    # plt.plot(min_, "-o", label="min")
    plt.legend()
    plt.title(
        "Sentence representations")

plt.show()
