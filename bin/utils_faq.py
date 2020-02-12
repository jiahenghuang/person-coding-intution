# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 2019/5/6
# @Author  : fanzfeng

import json
import pickle


def levenshtein_sim(sentence1, sentence2, sim=True):
    first, second = sentence1, sentence2
    sentence1_len, sentence2_len = len(first), len(second)
    maxlen = max(sentence1_len, sentence2_len)
    if sentence1_len > sentence2_len:
        first, second = second, first

    distances = range(len(first) + 1)# 短串+1
    for index2, char2 in enumerate(second):# 长字符串
        new_distances = [index2 + 1] #第几个字符串
        for index1, char1 in enumerate(first): # 短字符串
            if char1 == char2:
                new_distances.append(distances[index1]) #distances[ix]=ix
            else:
                min_ix = min((distances[index1], distances[index1+1], new_distances[-1]))
                new_distances.append(1+min_ix)
        distances = new_distances
    levenshtein = distances[-1]
    return float((maxlen - levenshtein) / maxlen) if sim else levenshtein


def save_json(data_series, save_file, complex_series=False):
    with open(save_file, "w", encoding="utf-8") as fp:
        if not complex_series:
            json.dump(data_series, fp, indent=4, ensure_ascii=False)
        else:
            for d in data_series:
                json.dump(d, fp, ensure_ascii=False)
                fp.write("\n")


def save_pkl(data, save_file):
    with open(save_file, "wr") as fp:
        pickle.dump(data, fp)
        print("Success save data to {}".format(save_file))


def load_pkl(save_file):
    with open(save_file, "rb") as fp:
        return pickle.load(fp)


def series_unique(series):
    new_series = []
    for s in series:
        if s not in new_series:
            new_series += [s]
    return new_series
