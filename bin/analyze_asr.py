#coding:utf-8
import pickle
from pypinyin import lazy_pinyin
import re

import constants
import config
import utils
from tools.trie_func import trie_dict

class AsrAnalyzer(object):
    '''
    语音转文本矫正功能
    '''
    def __init__(self):
        self._load()
        self.word_pat_long_yes, self.word_pat_long_no = self._build_pat(config.ambiguity_long)
        self.word_pat_short_yes, self.word_pat_short_no = self._build_pat(config.ambiguity_short)
        self.sp_pat=[re.compile('而且.*?(?!(什么))'),re.compile('而且.*什么')]

    def _build_pat(self, file_path):
        '''
        解析pat
        '''
        with open(file_path, "r", encoding="utf-8") as fr:
            pats = fr.readlines()
        ambi_pats = [pat.strip().split('=') for pat in pats if pat.startswith('$') and not pat.startswith('$$')]

        #解析出正则表达式
        word_pat_yes, word_pat_no = {}, {}
        for word, pat in ambi_pats:
            if 'yes' in word:
                word = word.replace('_yes', '').replace('$','')
                word_pat_yes[word] = re.compile(pat)
            else:
                word = word.replace('_no', '').replace('$','')
                word_pat_no[word] = re.compile(pat)
        return word_pat_yes, word_pat_no

    # @utils.debugger(prefix='AsrAnalyzer')
    def _load(self):
        '''
        加载asr_trie
        '''
        with open(config.asr_dict_file_long, 'rb') as fr:
            self.asr_dict_long = pickle.load(fr)

        with open(config.asr_dict_file_short, 'rb') as fr:
            self.asr_dict_short = pickle.load(fr)
        
    
    def _seg_sentence(self, word, line):
        '''
        将句子进行三三切分
        '''
        length = len(word)
        line_len = len(line)
        segged_words = []
        for i in range(line_len):
            start = i
            end = i + length
            tmp = line[start:end]

            if i > line_len - length:
                break
            segged_words.append(tmp)
        return segged_words

    def _correct(self, line):
        '''
        将编辑距离小于阈值的词进行替换
        '''
        flag = False
        for word in constants.CORRECT_WORDS:
            word_pinyin = ''.join(lazy_pinyin(word))
            segged_words = self._seg_sentence(word, line)
            for w in segged_words:
                w_pinyin = ''.join(lazy_pinyin(w))
                if utils.edit_distance(w_pinyin, word_pinyin) < constants.DISTANCE:
                    line = line.replace(w, constants.RIGHT_WORD)
                    flag = True
                    break
            if flag:
                break
        return line

    def _ambiguity_correct(self, line):
        '''
        修正歧义词
        '''
        #判断文本长度是否小于6如果小于6则不进行处理
        if len(line) <= 6:
            for word in self.word_pat_yes:
                line = line.replace(word, self.right_word)
            for word in self.word_pat_no:
                line = line.replace(word, self.right_word)
        return line        

    def _analyze_long(self, line):
        '''
        长文本解析
        '''
        mats = trie_dict.search_all_by_name(line, config.asr_name_long)
        end_pos, new_line = 0, []
        
        for mat in mats:
            correct_value = self.asr_dict_long.get(mat.value, None)
            if correct_value:
                word = line[end_pos:mat.end_pos]
                if constants.AMBIGUITY_WORDS_MORE.__contains__(mat.value):
                    if mat.value in self.word_pat_long_no: 
                        if not self.word_pat_long_no[mat.value].search(line):
                            continue
                    else:
                        if self.word_pat_long_yes[mat.value].search(line):
                            continue

                line_piece = word.replace(mat.value, correct_value)
                new_line.append(line_piece)
                end_pos = mat.end_pos
         
        if end_pos < len(line):
            line_piece = line[end_pos:]
            new_line.append(line_piece)
        return ''.join(new_line)

    def _analyze_short(self, line):
        '''
        短文本解析
        '''
        mats = trie_dict.search_all_by_name(line, config.asr_name_short)
        end_pos, new_line = 0, []
        
        for mat in mats:
            correct_value = self.asr_dict_short.get(mat.value, None)

            if correct_value:
                word = line[end_pos:mat.end_pos]
                if mat.value == '而且':
                    if self.sp_pat[0].search(line):
                        line_piece = word.replace(mat.value, '可以')
                        new_line.append(line_piece)
                        end_pos = mat.end_pos
                        continue
                    if self.sp_pat[1].search(line):
                        line_piece = word.replace(mat.value, '了解')
                        new_line.append(line_piece)
                        end_pos = mat.end_pos
                        continue

                if constants.AMBIGUITY_WORDS_LESS.__contains__(mat.value):
                    if mat.value in self.word_pat_short_no: 
                        if not self.word_pat_short_no[mat.value].search(line):
                            continue
                    else:
                        if self.word_pat_short_yes[mat.value].search(line):
                            continue

                line_piece = word.replace(mat.value, correct_value)
                new_line.append(line_piece)
                end_pos = mat.end_pos
         
        if end_pos < len(line):
            line_piece = line[end_pos:]
            new_line.append(line_piece)
        return ''.join(new_line)

    @utils.debugger(prefix='AsrAnalyzer')
    def analyze(self, line, long =True):
        '''
        矫正
        '''
        if not long:
            return self._analyze_short(line)
        return self._analyze_long(line)

asr_analyzer = AsrAnalyzer()
if __name__=='__main__':
    line = u'我现在时单人,你们是车贷网吗？'
    # line = '大声说哈哈哈不费电'
    # line = '而且啊啊啊什么'
    # line = '但是啊啊啊啊很不好意思'
    line = '没有听清楚啊'
    # import re
    # fmt_pat_cn = re.compile('[^一-龥]')
    # print(asr_analyzer.analyze(line))
    # with open('correct_corpus.txt') as fr:
    #     data = fr.readlines()
    if len(line)>6:
        flag = True
    else:
        flag = False

    new_line = asr_analyzer.analyze(line, long=flag)
    print(new_line)

    # with open('voice_corpus.txt', 'w') as fw:
    #     for line in data:
    #         new_line = asr_analyzer.analyze(line)
    #         new_line = fmt_pat_cn.sub('', new_line)
    #         fw.write(new_line+'\n')
