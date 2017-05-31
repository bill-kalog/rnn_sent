import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import collections


# print accuracy per sentence length
files_ = ["attention_test_.json", "attention_test_1.json"]
files_ = ['bidGRU-attGRU.json', 'DMN-1.json', 'DMN-2.json']

legend = ['bidGRU-attGRU', 'DMN-1', 'DMN-2']
for index_, filename_ in enumerate(files_):
    with open(filename_) as json_data:
        att_dic = json.load(json_data)

    correct_per_length = {}
    wrong_per_length = {}
    for sentence in att_dic:
        # print (sentence)
        sent_len = att_dic[sentence]["sent_length"]
        if att_dic[sentence]["true_label"] == att_dic[sentence]["predicted_label"]:
            correct_per_length[sent_len] = correct_per_length.get(sent_len, 0) + 1
        else:
            wrong_per_length[sent_len] = wrong_per_length.get(sent_len, 0) + 1
    # print (wrong_per_length)

    acc_ = {}
    # [wrong, correct]
    # print (wrong_per_length)
    for length_ in wrong_per_length:
        acc_[length_] = [wrong_per_length[length_], 0]
    for length_ in correct_per_length:
        a = acc_.get(length_, 0)
        if type(a) is list:
            a = a[0]
        acc_[length_] = [a, correct_per_length[length_]]
    # print (acc_)
    od = collections.OrderedDict(sorted(acc_.items()))
    print (od)
    x = []
    y = []
    for k, val_ in od.items():
        print (val_)
        acc_calc = val_[1] / (val_[1] + val_[0])
        y.append(acc_calc)
        x.append(k)
    print (y)
    print (x)
    plt.plot(x, y, marker='o', label=legend[index_])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('accuracy')
plt.xlabel('sentence length')
plt.show()
sys.exit()

# print attention weights
# plot attention distribution of a classified sentence
# info from
# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor?noredirect=1&lq=1
filename_ = "attention_test_.json"
sentence_ids = ["sent_id_659", "sent_id_1652"]
sentence_ids = ["sent_id_983"]
sentence_ids = ["sent_id_1650", "sent_id_188", "sent_id_324", "sent_id_1458"] # RNN
sentence_ids = ["sent_id_389", "sent_id_1395", "sent_id_980", "sent_id_1735"] # 2 ep DMN
sentence_ids = ["sent_id_{}".format(i) for i in range(6)]

plain_rnn = True
with open(filename_) as json_data:
    att_dic = json.load(json_data)

for sent_id in sentence_ids:
    words = []
    attentions_list = []
    mappings = att_dic[sent_id]['mappings']
    # Gather all words and attentions of the sentence
    for element in mappings:
        word_, attentions = element
        words.append(word_)
        if plain_rnn:
            attentions_list.append([attentions])
        else:
            attentions_list.append(attentions)
    print (attentions_list)
    fig, ax = plt.subplots()
    # transpose output to be [episode, words]
    attentions_arr = np.asarray(attentions_list).T

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # turn off the frame
    # print (len(attentions_arr), attentions_arr)
    ax.set_frame_on(False)
    ax.set_yticks(np.arange(len(attentions_arr)) + 0.25, minor=False)
    ax.set_xticks(np.arange(len(words)) + 0.05, minor=False)
    if plain_rnn:
        y_label = ["attention".format(i + 1) for i in range(len(attentions_arr))]
    else:
        y_label = ["episode {}".format(i + 1) for i in range(len(attentions_arr))]
    ax.set_yticklabels(y_label, minor=False)
    ax.imshow(attentions_arr, cmap=plt.cm.Blues, interpolation='nearest')
    # rotate the labels
    plt.xticks(rotation=75)
    plt.yticks(rotation=45)

    # print (words)
    ax.set_xticklabels(words, minor=False)
    plt.show()
    # sys.exit()

