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


def evaluate_similarity(
        word_vectors, language="english", source="simlex", dictionary_=None):
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
    elif source == "MEN":
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/MEN/MEN_dataset_natural_form_full", 'r', 'utf-8')
    elif source == "rw":
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/rw/rw.txt", 'r', 'utf-8')
    elif source == "SimVerb":
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/data/SimVerb-3500.txt", 'r', 'utf-8')
        relations = ["SYNONYMS", "ANTONYMS", "HYPER/HYPONYMS",
                     "COHYPONYMS", "NONE"]
        temp_fread_simlex = []
        if language in relations:  # used language as a relation type [bad:(]
            for line in fread_simlex:
                tokens = line.split("\t")
                # print ("{}--{}".format(tokens[4], language))
                # print ("{}--{}".format(tokens[4].strip(), language))
                # sys.exit()
                if tokens[4].strip() == language:
                    pair = "{}\t{}\t{}\t{}".format(
                        tokens[0], tokens[1], tokens[2], float(tokens[3]))
                    temp_fread_simlex.append(pair)
            fread_simlex = temp_fread_simlex
    else:
        fread_simlex = codecs.open(
            data_dir +
            "/evaluation/ws-353/wordsim353-" + source + ".txt", 'r', 'utf-8')

    total_num_pairs = 0
    line_number = 0
    for line in fread_simlex:

        if line_number > 0 or source in ["MEN", "rw", "simVerb"]:
            total_num_pairs += 1
            if source == "rw":
                tokens = line.split("\t")  # rw is tab delimited
            elif source == "SimVerb":
                tokens = line.split("\t")
                tokens[2] = tokens[3]
            else:
                tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            # word_i = lp_map[language] + word_i
            # word_j = lp_map[language] + word_j
            # print (word_i, word_j)
            if dictionary_ is not None:
                if (
                    word_i in dictionary_ and
                        word_j in dictionary_):
                    pair_list.append(((word_i, word_j), score))
            else:
                if (
                    word_i in word_vectors.word_to_index and
                        word_j in word_vectors.word_to_index):
                    pair_list.append(((word_i, word_j), score))
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
    # words_path = "./runs/1496969351/best_snaps/../evaluations/1497016586/words_embds.csv"
    # words_path = "./runs/1497028147/best_snaps/../evaluations/1497092785/words_embds.csv"
    words_path = "./runs/1497313304/best_snaps/../evaluations/1497438814/words_embds.csv"
    pretrained_vectors.append(WordVectors(
        type_, words_path))
    config['word_vector_type'].append(type_)
    flag_ = True



    for vec_num_, word_vectors in enumerate(pretrained_vectors):
        # print (word_vectors.dictionary)
        # print (word_vectors.vectors[0])
        # print (word_vectors.word_to_index["sermonize"], "seromnize")
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

        men_score, men_coverage = evaluate_similarity(
            word_vectors, language, source="MEN")
        print ("MEN score and coverage:", men_score, men_coverage, "\n")

        # SimVerb
        sim_relations = ["SYNONYMS", "ANTONYMS", "HYPER/HYPONYMS",
                         "COHYPONYMS", "NONE", "ALL"]
        for relation_ in sim_relations:
            sv_score, sv_coverage = evaluate_similarity(
                word_vectors, relation_, source="SimVerb")
            print (
                "SimVerb {} score and coverage: {} {}".format(
                    relation_, sv_score, sv_coverage))
        print ("\n")
        rw_score, rw_coverage = evaluate_similarity(
            word_vectors, language, source="rw")
        print ("RW score and coverage:", rw_score, rw_coverage)

    if flag_:
        print ("\n======== Evaluating only words from specified dictionary ========\n")
        for vec_num_, word_vectors in enumerate(pretrained_vectors[:-1]):
            dic_ = pretrained_vectors[-1].word_to_index
            simlex_score, simlex_coverage = evaluate_similarity(
                word_vectors, language, dictionary_=dic_)
            print ("SimLex-999 score and coverage:", simlex_score, simlex_coverage)
            # sys.exit()

            # WordSim Validation scores:
            c1, cov1 = evaluate_similarity(
                word_vectors, language, source=language,
                dictionary_=dic_)
            c2, cov2 = evaluate_similarity(
                word_vectors, language, source=language + "-sim",
                dictionary_=dic_)
            c3, cov3 = evaluate_similarity(
                word_vectors, language, source=language + "-rel",
                dictionary_=dic_)
            print ("WordSim overall score and coverage:", c1, cov1)
            print ("WordSim Similarity score and coverage:", c2, cov2)
            print ("WordSim Relatedness score and coverage:", c3, cov3, "\n")

            men_score, men_coverage = evaluate_similarity(
                word_vectors, language, source="MEN", dictionary_=dic_)
            print ("MEN score and coverage:", men_score, men_coverage, "\n")

            # SimVerb
            sim_relations = ["SYNONYMS", "ANTONYMS", "HYPER/HYPONYMS",
                             "COHYPONYMS", "NONE", "ALL"]
            for relation_ in sim_relations:
                sv_score, sv_coverage = evaluate_similarity(
                    word_vectors, relation_, source="SimVerb",
                    dictionary_=dic_)
                print (
                    "SimVerb {} score and coverage: {} {}".format(
                        relation_, sv_score, sv_coverage))
            print ("\n")
            rw_score, rw_coverage = evaluate_similarity(
                word_vectors, language, source="rw", dictionary_=dic_)
            print ("RW score and coverage:", rw_score, rw_coverage)

if __name__ == '__main__':
    main()
