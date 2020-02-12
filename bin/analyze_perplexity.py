#coding:utf-8
import jieba
import pickle
from math import log

import utils
import config
logger = config.logger

class PerpAnalyzer(object):
    '''
    实现计算困惑度功能
    '''
    def __init__(self):
        self._load_data()

    def _bigram_words(self, line):
        '''
        生成词配对
        '''
        word_pairs = []
        n = len(line)
        for index in range(n):
            if index < n-1:
                word1 = line[index]
                word2 = line[index + 1] 
                word_pairs.append((word1, word2))
        return word_pairs

    # @utils.debugger(prefix='PerpAnalyzer')
    def _load_data(self):
        '''
        加载条件概率
        '''
        with open(config.condition_prob_file, 'rb') as fr:
            self.condition_prob = pickle.load(fr)

    # @utils.debugger(prefix='PerpAnalyzer')
    def calc_score(self, line):
        '''
        计算困惑度值
        '''
        new_line = utils.segger(line)
        prob = 0
        word_num = len(new_line)
        ugly_words = []
        if word_num > 1 :
            words = self._bigram_words(new_line)
            for w_w in words:
                if utils.recog_rule(w_w):
                    try:
                        w0 = w_w[0]
                        w1 = w_w[1]
                        prob += log(self.condition_prob[w0][w1])
                    except:
                        prob += -100
                        ugly_words.append((w0,w1))
        if word_num != 0:
            return (-1)*prob*1/word_num, ugly_words
        logger.warn('%s no make sense words' % line)
        return 'perplexity -inf'

perp_analyzer = PerpAnalyzer()
if __name__=='__main__':
    lines = ['哦不是不是',
'没有吧你好像打错了拜拜',
'不是不是不是',
'不是了',
'那个现在不需要了哈',
'不是',
'呃不是',
'打错',
'哦不需要好谢谢嗯',
'但是毛线啊你们那个都是假的',
'你打错了',
'啊不用不用谢谢啊',
'对啊我是无聊才',
'嗯现在不用了吧',
'不是不是',
'啊不是不是不是',
'你打错了吧',
'不是什么时候搞的',
'打错了吧',
'嗯不是不是',
'嗯不是了不是了',
'是什么搞的时候不',
'不谢谢无聊才毛线啊',
'打吧啊是哪有这么快',
'用的不是吗',
'当时是当时我是能直接身份',
'是针对当时不搞了',
'我没用过这玩的吗',
'对你这个是打我资料什么意思为什么来直接说',
'亲是不是',
'就单的嘛干嘛',
'我刚买是不是也',
'是不是信号不好',
'嗯忘记可干嘛',
'所以的都有马虎的被被子不是不是不是',
'对对对一般六点写代码吗',
'意思这是什么的来是不是',
'没有您是怎么了我现在忙',
'没有老沙的',
'没有诶都谈着呢',
'啊是打算啊你是不是莎莎',
'我是不是',
'是不是什么意思',
'哦是不是',
'不是哈您不是',
'该送了怎么了',
'啊是什么',
'怎么老是啊',
'我现在啊怎么了',
'意思是需要哈诶谢谢诶好嘞',
'那有什么事直接说吧好吧',
'那他是不是我门口',
'我说是怎么了',
'我现在单身就单身但是我的证件还没有到手',
'需要什么啊什么事',
'快单子了还没的单身',
'我没有听清你这是什么',
'啊是不是']
    # with open('/home/heng/projects/sentence_perplexity/corpus.txt') as fr:
    #     lines = fr.readlines()
     
    scores = {}

    for line in lines:
        score, ugly_words = perp_analyzer.calc_score(line)
        scores[line] = (score, ugly_words)
    result = sorted(scores.items(),key=lambda x:x[1][0])
    for a,b in result:
        print(a.strip(),scores[a])
    # import pdb;pdb.set_trace()
