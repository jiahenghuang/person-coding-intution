# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 2019/3/30
# @Author  : fanzfeng

import os
import time
import json
import pickle
from tools.util_acsm import ACTree
from utils_faq import save_json, series_unique, load_pkl, save_pkl


class AsrCorrect(object):
    def __init__(self, dict_path, correct_json, acsm_pkl=None, refresh=False):
        t0 = time.time()
        assert os.path.exists(dict_path)
        self.correct_json = correct_json
        self.acsm_pkl = acsm_pkl
        if refresh:
            self.analyze(dict_path)
        elif os.path.exists(correct_json):
            with open(correct_json, "r", encoding="utf-8") as fp:
                self.correct_dict = json.load(fp)
            if isinstance(acsm_pkl, str) and os.path.exists(acsm_pkl):
                print("ASR acsm model exist, loading model...")
                self.model = load_pkl(acsm_pkl)
            else:
                self.voca = {k for k, v in self.correct_dict.items()}
                self.model = ACTree(self.voca)
                try:
                    save_pkl(self.model, acsm_pkl)
                    print("    success save acsm model")
                except Exception as e:
                    print("    fail save acsm model: %s", e)
        else:
            self.analyze(dict_path)
        print("  load done, use time %s s", time.time()-t0)

    def analyze(self, dict_path):
        res = self.dict_process(dict_path)
        if res is not None:
            self.voca, self.correct_dict = res
            save_json(self.correct_dict, save_file=self.correct_json)
            self.model = ACTree(self.voca)
            with open(self.acsm_pkl, 'wb') as fw:
                pickle.dump(self.model, fw)

    @staticmethod
    def dict_process(dict_path):
        words_dict = {}
        voca = []
        with open(dict_path, "r", encoding="utf-8") as fp:
            for d in fp.readlines():
                w_key, words = d.strip().split(":")
                words_list = series_unique(words.replace(" ", "").split(","))
                voca += words_list
                if w_key not in words_dict:
                    words_dict[w_key] = words_list
                else:
                    words_dict[w_key] += words_list
        voca_size = len(voca)
        if len(set(voca)) == voca_size:
            print("ASR vocabulary size: %s  unique: %s", voca_size, len(set(voca)))
        correct_dict = {}
        for w in words_dict:
            for s in words_dict[w]:
                if s not in correct_dict:
                    correct_dict[s] = w
                else:
                    if len(correct_dict[s]) < len(w):
                        correct_dict.update({s: w})
        return voca, correct_dict

    def text_correct(self, s):
        # t0 = time.time()
        res_search = self.model.search(s)
        # t1 = time.time()
        # print("search time: {}s".format(t1-t0))
        # res_ = self.model.easy_search(s)
        # t2 = time.time()
        # print("search time: {}s".format(t2 - t1))
        # print(res_)
        # print(res_search)
        if len(res_search) <= 0:
            return s
        else:
            res_sorted = sorted(res_search, key=lambda p: len(p[-1]), reverse=True)
            for r in res_sorted:
                s = s.replace(r[-1], self.correct_dict[r[-1]])
            return s


if __name__ == "__main__":
    from config import asr_faq_dict, asr_faq_json, asr_faq_pkl
    ac = AsrCorrect(dict_path=asr_faq_dict,
                    correct_json=asr_faq_json,
                    acsm_pkl=asr_faq_pkl)
    print(ac.text_correct("啊那我说你们是真爱是吧"))
