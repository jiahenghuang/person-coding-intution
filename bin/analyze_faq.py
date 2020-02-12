# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 2019/4/9
# @Author  : fanzfeng

import jieba
import time
import utils
from tools.util_se import SearchEngine
from utils_faq import *
from analyze_asr_faq import AsrCorrect
from config import file_config_dict, pronouns_drop, spoken_words, stopwords_new, pronouns_config
from config import asr_faq_dict, asr_faq_json, asr_faq_pkl
from config import file_se_index, file_se_index_v1
# from tools.util_nlp import ZhNlp

ac = AsrCorrect(dict_path=asr_faq_dict, correct_json=asr_faq_json,
                acsm_pkl=None)
jieba.load_userdict(file_config_dict)
# nlp = ZhNlp(config_dict=file_config_dict)


class FaqSearch(object):
    def __init__(self, query_df=file_se_index, qid_col="qid", query_col="new_query", index_col="index_query",
                 rank_model="edit_distance", asr_correct=True, spoken_del=True, query_verify=False):
        self.input_data = query_df
        if self.input_data is not None:
            self.se = SearchEngine(query2rid_file=query_df, res_col=qid_col, query_col=query_col, index_col=index_col)
            self.res_set = self.se.res_set
            self.doc_format = "list"
            if rank_model == "edit_distance":
                self.rank_func = levenshtein_sim
                self.doc_format = "text"
            # elif rank_model == "wmd":
            #     word2vec = W2V(w2v_file="w2v_ch_fastText/wiki.zh.vec", binary=False)
            #     self.rank_func = word2vec.wmd_distance
            # elif rank_model == "tfidf":
            #     tf_idf = TfIdf(texts=[s.split() for s in query_df["new_query"].tolist()])
            #     self.rank_func = tf_idf.sim_rank
            self.rank_model = rank_model
            self.asr_correct = asr_correct
            self.spoken_del = spoken_del
            # self.query_verify = query_verify
            # if self.query_verify:
            #     self.query_gen = QueryIf(re_file=os.path.join(config_path, "kd_re.txt"))

    @staticmethod
    def query_process(x, out_list=False, drop_pronouns=True, text_correct=True, dup_window=2, redu_process=True,
                      loc_replace=False):
        '''
        :param x: 待处理的文本
        :param out_list: 是否输出词
        :param drop_pronouns: 是否去除代词
        :param text_correct: 是否进行文本纠错
        :param dup_window: 删除重复的窗口
        :param redu_process: 是否删除口语冗余
        :param loc_replace: 是否对地名格式化
        :return:
        '''
        # x = utils.fmt_pat_kd.sub('', x.strip())
        if len(x) < 1:
            return [] if out_list else ''
        for w in sorted(pronouns_drop, key=lambda pw: len(pw), reverse=True):
            x = x.replace(w, "")
        if redu_process:
            for s in spoken_words:
                x = x.replace(s, "")
        if text_correct:
            x = ac.text_correct(x)
        stopwords_config = stopwords_new
        if drop_pronouns:
            stopwords_config = stopwords_new + pronouns_config
        # if loc_replace:
        #     words = [w for w in nlp.ner_format(x, ner_labels=['Ns', 'Ni'], ner_proun="D") if w not in stopwords_config]
        # else:
        #     words = [w for w in jieba.cut(x) if w not in stopwords_config]
        words = [w for w in jieba.cut(x) if w not in stopwords_config]
        # if len(words) < 1:
        #     logger.warning("Query process no result: %s", x)
        new_words = []
        for j in range(len(words)):
            w = words[j]
            # if w not in words[max(0, j - dup_window):j]:
            if w == "D" or sum(w in w_bf for w_bf in words[max(0, j - dup_window):j]) <= 0:
                new_words.append(w)
        if out_list:
            return new_words
        else:
            return " ".join(new_words)

    @utils.debugger(prefix="FaqSearch")
    def rank_res(self, query_text, recall_num=50, min_score=0, mark=False, loc_replace=False):
        if self.input_data is not None:
            if not isinstance(query_text, str):
                return '', []
            if len(query_text) < 1:
                return query_text, []
            # if self.query_verify and not self.query_gen.recognize(query_text):
            #     return []
            text = self.query_process(query_text, out_list=True, drop_pronouns=False, text_correct=self.asr_correct,
                                      redu_process=self.spoken_del, loc_replace=loc_replace)
            if len(text) < 1:
                return text, []
            recall_res = self.se.query_search(text, res_num=recall_num, doc_format=self.doc_format)
            if self.rank_model == "edit_distance":
                text = "".join(text)
            if len(recall_res) < 1:
                return text, recall_res
            if self.rank_model == "tfidf":
                queries = [d['query_text'] for d in recall_res]
                sim_res = self.rank_func(text, queries, res_sort=False)
                for j in range(len(recall_res)):
                    recall_res[j]['rank_score'] = sim_res[j]
            else:
                for j in range(len(recall_res)):
                    recall_res[j]['rank_score'] = self.rank_func(text, recall_res[j]['query_text'])
            ranked_res = sorted(recall_res, key=lambda r: r['rank_score'], reverse=True)
            if mark:
                return ranked_res
            return text, [r for r in ranked_res if r['rank_score'] >= min_score]


faq_model_v0 = FaqSearch(asr_correct=True)
faq_model_v1 = FaqSearch(query_df=file_se_index_v1, asr_correct=True)
if __name__ == "__main__":
    print(faq_model_v0.rank_res("她个子高不高, 我想知道一下, 能告诉我么"))
    # print(faq.query_process("啊那我说你们是真爱是吧", drop_pronouns=False))
