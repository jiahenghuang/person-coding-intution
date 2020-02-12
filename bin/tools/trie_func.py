#coding: utf-8
import sys
import dawg
import os
import pickle

import utils
import config
from structs.match_obj import MatchObj
from tools.util_trie import Trie

class TrieDict(object):
    '''
    使用trie树封装的工具函数
    '''
    def __init__(self):
        self.dict_tries ={}
        self._load()
        
    def _load_trie_file(self):
        '''
        加载dict_trie
        '''
        with open(config.dict_trie_file,'rb') as fr:
            self.dict_tries = pickle.load(fr)

    def _make_trie(self):
        '''
        构建trie树
        '''        
        for dict_file, dict_name in config.dict_pairs:
            wordset = self._load_dict(dict_file)
            trie = Trie(wordset)
            self.dict_tries[dict_name] = trie
        
        #加载长文本纠错
        wordset, w2r_wordset_long = self._load_asr_dict(config.correct_dict_long)
        trie = Trie(wordset)
        self.dict_tries[config.asr_name_long] = trie

        #加载短文本纠错
        wordset, w2r_wordset_short = self._load_asr_dict(config.correct_dict_short)
        trie = Trie(wordset)
        self.dict_tries[config.asr_name_short] = trie

        self._save()
        self._save_asr(w2r_wordset_long, w2r_wordset_short)

    def _load(self):
        '''
        加载trie file,如果trie_file不是最新的就重新构建trie树并保存
        '''
        if os.path.exists(config.dict_trie_file):
            dict_trie_time = utils.get_last_modified(config.dict_trie_file)
        else:
            return self._make_trie()
                
        dict_files = [dict_file for (dict_file, dict_name) in config.dict_pairs]
        dict_files.extend([config.correct_dict_long, config.correct_dict_short])        
        dict_time = max(map(utils.get_last_modified, dict_files))

        if os.path.exists(config.dict_trie_file) and (dict_trie_time != None and dict_time != None and dict_trie_time > dict_time):
            self._load_trie_file()
        else:
            self._make_trie()
        print('load trie dict finished')
    
    def _save(self):
        '''
        将trie树保存到trie_file中
        '''
        if self.dict_tries:
            with open(config.dict_trie_file, 'wb') as fw:
                pickle.dump(self.dict_tries, fw)
        else:
            print('trie Null')
    
    def _save_asr(self, w2r_wordset_long, w2r_wordset_short):
        '''
        将trie树保存起来
        '''    
        if w2r_wordset_long:
            with open(config.asr_dict_file_long, 'wb') as fw:
                pickle.dump(w2r_wordset_long, fw)
        else:
            print('asr trie null')

        if w2r_wordset_short:
            with open(config.asr_dict_file_short, 'wb') as fw:
                pickle.dump(w2r_wordset_short, fw)
        else:
            print('asr trie null')


    def _load_dict(self, dict_file):
        '''
        载入词典
        '''
        wordset = []
        utils.exit_on_unexist(dict_file)

        for line in utils.safe_line_generator(dict_file) :
            line = line.strip()
            if line == '' :
                continue

            if line != '':
                wordset.append(line)
        return wordset

    def _load_asr_dict(self, dict_file):
        '''
        载入纠正字典
        '''
        wordset, w2r_wordset = [], {}
        utils.exit_on_unexist(dict_file)
        
        for line in utils.safe_line_generator(dict_file) :
            line = line.strip()
            if line == '' :
                continue

            if line != '':
                line = line.split(':')
                w_words = line[1].split(',')
                r_word = line[0]
                wordset.extend(w_words)
                for word in w_words:
                    w2r_wordset[word] = r_word
        return wordset, w2r_wordset

    # @utils.debugger(prefix='TrieDict')
    def search_all_by_name(self, line, dict_name, pos = 0):
        '''
        从给定的line寻找指定名称的token以mat形式展示
        '''
        if line.strip() == '' or len(line) <= pos:
            return None
        trie = self.dict_tries.get(dict_name, -1)
        mats = []
        if trie != -1:
            hits = trie.search_all(line)
            if hits:
                for hit in hits:
                    word = hit[0]
                    start_pos = hit[1]
                    end_pos = hit[2]

                    mat = MatchObj(dict_name, word, nature=None, line=line, start_pos=start_pos, end_pos=end_pos, ratio=0.0)
                    mat.ratio = 1.0 * len(word)/len(line)
                    mats.append(mat)
        return mats 

trie_dict = TrieDict()

if __name__ == '__main__' :
    mats = trie_dict.search_all_by_name(u'北京大学湖北日本美国','CITY')
    for mat in mats:
        print(mat)