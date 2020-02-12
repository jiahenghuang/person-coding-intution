#coding: utf-8
import sys
import re
import dawg
import utils
from structs.kv_obj import KVObj

class Trie(object):
    def __init__(self, tokens):
        self.trie = dawg.DAWG(tokens)

    def longest_prefix(self, line):
        '''
        返回最长前缀对应的token
        '''
        word = u''
        wsize = 0
        
        for w in self.trie.iterprefixes(line) :
            if len(w) > wsize : 
                wsize = len(w)
                word = w
        return word
    
    def prefix(self, line):
        '''
        返回最短匹配最长匹配等可能匹配到的所有词
        '''
        wsize = 0
        words = []
        for w in self.trie.iterprefixes(line) :
            words.append(w)
        return words

    def contain(self, tok):
        '''
        判断tok是否出现在trie树中
        '''
        return tok in self.trie
    
    def contain_tag(self, line) :
        '''
        判断一行中是否有出现在trie树中的词
        '''
        i = 0
        found = False
        while (i < len(tok) and not found) :
            for _ in self.trie.iterprefixes(tok[i:]) :
                found = True
                break
            i += 1
        return found
        
    def search_all(self, line):
        '''
        搜索出line中出现在trie树中的所有词
        '''
        hits = []
        i = 0
        while i < len(line) :
            w = self.longest_prefix(line[i:])
            if w != '':
                hits.append((w, i, i+len(w)))
                i += len(w)
            else :
                i += 1
        return hits

    def search_longest_mats(self, line):
        '''
        将line中信息抽取成mat
        '''
        max_len,word = 0, ''
        i = 0
        while i < len(line) :
            w = self.longest_prefix(line[i:])
            if w != '' and len(w) > max_len:
                word = w
                start_pos = i
                end_pos = i+len(w)
                i += len(w)
                max_len = len(w)
            else :
                i += 1
        
        if word:
            ratio = 1.0 * len(word)/len(line)
            mat = KVObj(group=None, rank=-1, type_name=None, vpat=None, word=word, start_pos=start_pos, end_pos=end_pos, ratio=ratio)
            return mat
        return []

    def search_all_words(self, line):
        '''
        抽取句子中所有出现在trie树中的词
        '''
        i = 0
        words = []
        while i < len(line) :
            w = self.prefix(line[i:])
            if w:
                w_len = max([len(xxx) for xxx in w])
                words.extend(w)
                i += w_len
            else :
                i += 1
        return words

    # @utils.debugger(prefix='Trie')
    def search_all_mats(self, line):
        '''
        将line中信息抽取成mat
        '''
        hits, obj = [], []
        i = 0
        while i < len(line) :
            w = self.longest_prefix(line[i:])
            if w != '':
                hits.append((w, i, i+len(w)))
                i += len(w)
            else :
                i += 1
        
        for hit in hits:
            word = hit[0]
            start_pos = hit[1]
            end_pos = hit[2]
            ratio = 1.0 * len(word)/len(line)
            mat = KVObj(group=None, rank=-1, type_name=None, vpat=None, word=word, start_pos=start_pos, end_pos=end_pos, ratio=ratio)
            obj.append(mat)
        return obj

if __name__ == '__main__' :
    trie = Trie([u'abc', u'哈哈'])
    print(u'ab' in trie.trie)
    print(u'哈哈' in trie.trie)
    print(trie.search_all(u'abc啊啊'))
    for mat in trie.search_all_mats(u'abc啊啊哈哈'):
        print(mat)
        