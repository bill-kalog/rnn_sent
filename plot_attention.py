import numpy as np
import sys
import json
import matplotlib.pyplot as plt

# plot attention distribution of a classified sentence
# info from
# http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor?noredirect=1&lq=1
filename_ = "attention_test_.json"
sentence_ids = ["sent_id_659", "sent_id_1652"]
sentence_ids = ["sent_id_983"]
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





















