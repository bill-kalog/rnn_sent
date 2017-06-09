from conf import config
from word_vectors import WordVectors
import numpy as np
import sys

import numpy
import codecs
import sys
import math
import os
from copy import deepcopy
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr


lp_map = {}
lp_map["english"] = u"en_"
lp_map["german"] = u"de_"
lp_map["italian"] = u"it_"
lp_map["russian"] = u"ru_"


def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator,
    which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def normalise_word_vectors(word_vectors, norm_=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm_
    return word_vectors


def evaluate_similarity(word_vectors, language="english", source="simlex"):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors. 
    The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt, 
    and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt 

    implementation taken from: https://github.com/nmrksic/eval-multilingual-simlex/blob/master/evaluate.py
    """
    data_dir = config['dat_directory']
    pair_list = []
    if source == "simlex":
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/simlex-" + language + ".txt", 'r', 'utf-8')
    else:
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/ws-353/wordsim353-" + source + ".txt", 'r', 'utf-8')

    total_num_pairs = 0
    line_number = 0
    for line in fread_simlex:

        if line_number > 0:
            total_num_pairs += 1
            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            # word_i = lp_map[language] + word_i
            # word_j = lp_map[language] + word_j
            # print (word_i, word_j)
            if (
                word_i in word_vectors.word_to_index and
                    word_j in word_vectors.word_to_index):
                pair_list.append(((word_i, word_j), score))
            else:
                pass
        line_number += 1

    pair_list.sort(key=lambda x: - x[1])
    # print (pair_list)
    print ("found {} pairs out of {}".format(len(pair_list), total_num_pairs))

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:
        (word_i, word_j) = x
        index_i = word_vectors.word_to_index[word_i]
        index_j = word_vectors.word_to_index[word_j]
        vector_i = word_vectors.vectors[index_i]
        vector_j = word_vectors.vectors[index_j]
        current_distance = distance(vector_i, vector_j)
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])
    # print (extracted_list)
    # sys.exit()
    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)

    return round(spearman_rho[0], 3), coverage


def main():
    """
    The user can provide the location of the config file as an argument. 
    If no location is specified, the default config file 
    (experiment_parameters.cfg) is used.
    """
    # load pretrained word vectors
    language = "english"
    pretrained_vectors = []
    for index, type_ in enumerate(config['word_vector_type']):
        pretrained_vectors.append(WordVectors(
            type_, config["pretrained_vectors"][index]))
        print ("loaded vectors {}".format(config['word_vector_type'][index]))

    # load from evaulation path
    type_ = "from_model"
    # words_path = "./runs/1496739336/evaluations/1496933648/words_embds.csv"
    words_path = "./runs/1496969351/best_snaps/../evaluations/1497016586/words_embds.csv"
    pretrained_vectors.append(WordVectors(
        type_, words_path))
    config['word_vector_type'].append(type_)



    for vec_num_, word_vectors in enumerate(pretrained_vectors):
        # print (word_vectors.dictionary)
        # print (word_vectors.vectors[0])

        print("\n============= Evaluating word vectors: {} for language: {}"
              " =============\n".format(
                  config['word_vector_type'][vec_num_], (language)))

        simlex_score, simlex_coverage = evaluate_similarity(
            word_vectors, language)
        print ("SimLex-999 score and coverage:", simlex_score, simlex_coverage)
        # sys.exit()

        # WordSim Validation scores:
        c1, cov1 = evaluate_similarity(
            word_vectors, language, source=language)
        c2, cov2 = evaluate_similarity(
            word_vectors, language, source=language + "-sim")
        c3, cov3 = evaluate_similarity(
            word_vectors, language, source=language + "-rel")
        print ("WordSim overall score and coverage:", c1, cov1)
        print ("WordSim Similarity score and coverage:", c2, cov2)
        print ("WordSim Relatedness score and coverage:", c3, cov3, "\n")


if __name__ == '__main__':
    main()
