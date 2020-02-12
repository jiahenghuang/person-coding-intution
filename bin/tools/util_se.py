# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 2019/1/2
# @Author  : fanzfeng

from collections import Counter
import pandas as pd


class SearchEngine(object):
    '''
    实现简单的倒排索引
    1. query2rid_file接受已经完成预处理和切好词【空格分隔】的(query, rid)pair对，rid表示的是问题表示，对应唯一的answer
    2. query2rid_file可以为文件名，也可以为dataframe
    3. ix2doc为doc的索引，key_dict为word:list of doc_ix倒排索引
    4. 一个index对应多个query、多个answer，query与answer一一对应
    '''
    def __init__(self, query2rid_file, res_col="answer", query_col="question", index_col=None):
        if isinstance(query2rid_file, str):
            query2rid_data = pd.read_csv(query2rid_file, encoding="utf-8", sep=",")
        else:
            query2rid_data = query2rid_file.copy()
        self.ix2query = None
        self.res_set = set(query2rid_data[res_col])
        self.query2res = query2rid_data.set_index(query_col)[res_col].to_dict()
        self.index_data = {k: [v] for k, v in self.query2res.items()}
        if index_col is not None:
            self.index_data = query2rid_data.groupby(index_col).apply(lambda x: x[query_col].tolist()).to_dict()
        self.ix2doc = {}
        self.key_dict = {}
        self.index_col = index_col
        self.build_index()

    def build_index(self):
        '''
        新建倒排索引
        '''
        ix = 0
        for doc in self.index_data:
            self.ix2doc[ix] = doc
            words = doc.split()
            for k in words:
                if k not in self.key_dict:
                    self.key_dict[k] = [ix]
                elif ix not in self.key_dict[k]:
                    self.key_dict[k] += [ix]
            ix += 1

    def query_search(self, query_text, res_num=2, doc_format=["list", "text", "split_text"][-1]):
        '''
        搜索结果
        '''
        res = []
        assert isinstance(query_text, (list, tuple, str))
        if isinstance(query_text, str):
            words = query_text.split()
        else:
            words = query_text
        query_len = len(words)
        query_set = set(words)
        if query_len > 0:
            doc_related = []
            # weights = [1/query_len]*query_len
            for w in query_set:
                doc_related += self.key_dict.get(w, [])
            res_len = len(doc_related)
            if res_len > 0:
                doc_freq = Counter(doc_related)
                doc_res = doc_freq.most_common()[0:min(res_num, res_len)]
                for d in doc_res:
                    r = [self.ix2doc[d[0]]] if self.index_col is None else self.index_data[self.ix2doc[d[0]]]
                    for doc_text in r:
                        if doc_format == "list":
                            out_text = doc_text.split()
                        elif doc_format == "text":
                            out_text = doc_text.replace(" ", "")
                        # doc_set = set(doc_text.split())
                        # "score": len(doc_set & query_set) / len(doc_set | query_set)
                        res += [{"query_ix": d[0], "query_text": out_text,
                                 "qid": self.query2res[doc_text],
                                 "index_text": self.ix2doc[d[0]]}]
        return res
