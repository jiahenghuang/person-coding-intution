# -*- coding: utf-8 -*-
import time
import jieba
from collections import Counter
import pandas as pd
import config
import utils

class SearchEngine(object):
    '''
    实现kd功能
    '''
    def __init__(self):
        jieba.load_userdict(config.file_config_dict)
        with open(config.file_stop_dict, "r", encoding="utf-8") as fp:
            self.stops_list = [k.strip() for k in fp.readlines()]

        self.query2rid = pd.read_csv(config.query2rid_file, encoding="utf-8", sep="\t").set_index("question")["answer"].to_dict()
        self.rid2res = pd.read_excel(config.rid2res_file, sheet_name="sale_E").set_index("Id")["Answer"].to_dict()
        self._build_index()

    def text_process(self, sent, out_type="list", drop_stop=True, stop_limit=60):
        '''
        文本预处理，停词
        '''
        cut_res = list(jieba.cut(sent))
        if cut_res:
            if drop_stop:
                cut_res = [k for k in cut_res if k not in self.stops_list[0:stop_limit]]
            if out_type == "list":
                return cut_res
            return " ".join(cut_res)
        return [] if out_type == "list" else ""
    
    @utils.debugger(prefix='SearchEngine')
    def _build_index(self):
        self.key_dict = dict()
        self.ix2doc = dict()
        ix = 0
        for doc in self.query2rid:
            self.ix2doc[ix] = doc
            words = self.text_process(doc)
            for k in words:
                if k not in self.key_dict:
                    self.key_dict[k] = [ix]
                elif ix not in self.key_dict[k]:
                    self.key_dict[k] += [ix]
            ix += 1
    
    @utils.debugger(prefix='SearchEngine')
    def query_search(self, query_text, res_num=2):
        '''
        搜索结果
        '''
        res = []
        assert isinstance(query_text, (list, str))
        if isinstance(query_text, str):
            words = self.text_process(query_text)
        else:
            words = query_text

        query_len = len(words)
        if query_len > 0:
            doc_related = []
            for w in words:
                doc_related += self.key_dict.get(w, [])
            res_len = len(doc_related)
            if res_len > 0:
                doc_freq = Counter(doc_related)
                doc_res = doc_freq.most_common()[0:min(res_num, res_len)]
                for d in doc_res:
                    doc_text = self.ix2doc[d[0]]
                    res += [{"query_ix": d[0],
                             "query_text": doc_text,
                             "score": d[1]/query_len,
                             "answer": self.rid2res[self.query2rid[doc_text]]}]
        return res

search_engine = SearchEngine()
if __name__ == "__main__":
    print(search_engine.query_search("你们公司在哪里"))
