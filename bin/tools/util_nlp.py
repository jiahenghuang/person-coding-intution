# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 2019/4/25
# @Author  : fanzfeng

import os
import re
import jieba
import jieba.posseg as pseg
import pyltp
import platform


class ZhNlp(object):
    def __init__(self, config_lib="ltp", config_dict=None, config_stop=None, config_dir=None):
        if config_dir is None:
            self.config_dir = 'E:/Data/' if 'windows' in platform.architecture()[1].lower() else \
                '/users/fanzfeng/Data/'
        elif isinstance(config_dir, str) and os.path.exists(config_dir):
            self.config_dir = config_dir
        self.stop_config = False
        self.config_lib = config_lib
        if config_stop is not None and isinstance(config_stop, str) and os.path.exists(config_stop):
            self.stop_config = True
            with open(config_stop, "r", encoding="utf-8") as fp:
                self.stop_words = [k.strip() for k in fp.readlines()]
        if config_lib == "jieba":
            if config_dict is not None and isinstance(config_dict, str) and os.path.exists(config_dict):
                jieba.load_userdict(config_dict)
            self.seg = jieba.cut
            self.pos_seg = pseg.cut
        elif config_lib == "ltp":
            self.segmentor = pyltp.Segmentor()
            if config_dict is not None and isinstance(config_dict, str) and os.path.exists(config_dict):
                self.segmentor.load_with_lexicon(os.path.join(self.config_dir, "ltp_data_v3.4.0/cws.model"), config_dict)
            else:
                self.segmentor.load(os.path.join(self.config_dir, "ltp_data_v3.4.0/cws.model"))
            self.seg = self.segmentor.segment
            self.postagger = pyltp.Postagger()
            self.text_splitter = pyltp.SentenceSplitter.split
            self.postagger.load(os.path.join(self.config_dir, "ltp_data_v3.4.0/pos.model"))
            self.recognizer = pyltp.NamedEntityRecognizer()
            self.recognizer.load(self.config_dir + "ltp_data_v3.4.0/ner.model")

    def split_sentence(self, doc, delimiters=list("。？！")):
        if self.config_lib == "ltp":
            sents = self.text_splitter(doc)
            return list(sents)
        else:
            return re.split("|".join(delimiters), doc)

    def ltp_close(self):
        if self.config_lib == "ltp":
            self.segmentor.release()
            self.postagger.release()
            self.recognizer.release()

    def zh_seg(self, text_input, drop_stop=True, all_cut=False, output_postags=False, out_list=False):
        if isinstance(text_input, str):
            text_seq = [text_input]
        elif isinstance(text_input, (list, tuple)):
            text_seq = text_input
        grams_series = []
        if not output_postags:
            for x in text_seq:
                series_words = (self.seg(x, cut_all=all_cut) if self.config_lib == "jieba" else self.seg(x))
                if drop_stop and self.stop_config:
                    grams_series += [[w for w in series_words if w not in self.stop_words]]
                else:
                    grams_series += [list(series_words)]
            if not out_list:
                grams_series = [" ".join(s) for s in grams_series]
            return grams_series[0] if isinstance(text_input, str) else grams_series
        else:
            for s in text_seq:
                if self.config_lib == "ltp":
                    word_list = list(self.seg(s))
                    postags = self.postagger.postag(word_list)
                    ptag_list = list(postags)
                    if len(word_list) == len(ptag_list):
                        grams_series += [(word_list, ptag_list)]
                else:
                    seg_res = [(w, p) for w, p in self.pos_seg(s)]
                    grams_series += [([k[0] for k in seg_res], [k[1] for k in seg_res])]
            out_put = []
            if drop_stop and self.stop_config:
                for wlist, plist in grams_series:
                    w_list, p_list = [], []
                    for i in range(len(wlist)):
                        if wlist[i] not in self.stop_words:
                            w_list += [wlist[i]]
                            p_list += [plist[i]]
                    out_put += [(w_list, p_list)]
            else:
                for wlist, plist in grams_series:
                    w_list, p_list = [], []
                    for i in range(len(wlist)):
                        w_list += [wlist[i]]
                        p_list += [plist[i]]
                    out_put += [(w_list, p_list)]
            return out_put

    def zh_ner(self, text):
        if self.config_lib == "ltp":
            word_tag = self.zh_seg(text, drop_stop=False, output_postags=True)
            out_put = []
            for w, p in word_tag:
                netags = self.recognizer.recognize(w, p)
                out_put += [(w, list(netags))]
            return out_put

    def ner_format(self, text, ner_labels=['Ns', 'Ni'], ner_proun="D"):
        out_put = []
        if self.config_lib == "ltp":
            res_ner = self.zh_ner(text)
            for w_s, n_s in res_ner:
                for i in range(len(w_s)):
                    if sum(y in n_s[i] for y in ner_labels) > 0:
                        w_s[i] = ner_proun
                out_put += [w_s]
        return out_put[0] if isinstance(text, str) else out_put if isinstance(text, (list, tuple)) else None


if __name__ == "__main__":
    nlp = ZhNlp(config_dict="/Users/fanzfeng/project_code/feature-nlp/za_nlp/TextSim/config/jieba_dict",
                config_stop="/Users/fanzfeng/data/stop_words.txt")
    print(nlp.ner_format("不知道去北京还是上海"))
    print(nlp.zh_ner("瑞昌范镇天气"))
