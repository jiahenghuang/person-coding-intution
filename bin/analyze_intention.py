#coding:utf-8
import config
import lightgbm as lgb
import utils
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import jieba
import json

from analyze_asr import asr_analyzer

class IntentionAnalyzer(object):
    '''
    意图分类功能
    '''
    def __init__(self):
        self._load_model()
        jieba.load_userdict('data/words.dict')

    # @utils.debugger(prefix='IntentionAnalyzer')
    def _load_model(self):
        '''
        加载模型
        '''
        with open(config.vertorizer_file, 'rb') as fr:
            vectorizer = pickle.load(fr)
        
        with open(config.tfidftransformer_file, 'rb') as fr:
            self.tfidftransformer = pickle.load(fr)

        self.vectorizer = CountVectorizer(decode_error="replace", vocabulary=vectorizer)
        self.model = lgb.Booster(model_file=config.lightgbm_model_file)

    @utils.debugger(prefix='IntentionAnalyzer')
    def analyze(self, data):
        '''
        预测label
        '''
        data = utils.str2json(data)
        message = data.get('message',None)
        if not message:
            return
        new_message = asr_analyzer.analyze(message)
        segged_words = utils.segger(new_message)
        test_x = np.array([' '.join(segged_words)])

        test_tfidf = self.tfidftransformer.transform(self.vectorizer.transform(test_x))
        preds = self.model.predict(test_tfidf, num_iteration = self.model.best_iteration)
        result = int(np.argmax(preds))
        if result == 0:
            result = {'label':'yes', 'text':message,'yes':preds[0][0],'no':preds[0][1],'busy':preds[0][2]}
            return utils.to_json(result)
        elif result == 1:
            result = {'label':'no', 'text':message,'yes':preds[0][0],'no':preds[0][1],'busy':preds[0][2]}
            return utils.to_json(result)
        result = {'label':'busy', 'text':message,'yes':preds[0][0],'no':preds[0][1],'busy':preds[0][2]}
        return utils.to_json(result)

intention_analyzer = IntentionAnalyzer()
if __name__=='__main__':
    lines = [
'不是八八嗯。',
'不是刚刚有给我打电话，怎么又打你是哪哪边呢。',
'不是前两天是我朋友的手机，注册完的，我没有这个手机，是我的。',
'不是单是吗？。',
'不是单身。',
'不是单身了。',
'不是单身呀。',
'不是单身的。',
'不是单身的，我现在取老婆唻。',
'不是单身，一楼。',
'不是单身，就不会去注册了，是吧。',
'不是叫我打电话，我现在六对吧。',
'不是可以呀。',
'不是可以自由。',
'不是可以，只有我碰的时间。',
'不是可能弄错了吧。',
'不是可能搞错了吧。',
'不是吗。',
'不是吧',
'不是吧。',
'不是吧，挂。',
'不是吧，有的。',
'不是吧，省我现在找家吧。',
'不是吧，真有这么好吗？。',
'不是听到了吗？。',
'不是呀。',
'不是呀，什么？。',
'不是呀，找到了。',
'不是呃不，是不是。',
'不是呢。',
'不是呢，谢谢。',
'不是和住宿的。',
'不是哈，我看看能睡了吧，好像。',
'不是哦。',
'不是哪个地方呢？。',
'不是啊',
'不是啊。',
'不是啊，再见。',
'明天再打']
    pos = []
    neg = []
    busy = []

    pp = []
    nn = []
    bb = []
    with open('pos.txt') as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = asr_analyzer.analyze(line)
            tag = intention_analyzer.analyze(line)
            if tag != '肯定':
                pos.append(line+':'+tag)
            if tag == '肯定':
                pp.append(line)
            if tag == '否定':
                nn.append(line)
            if tag == '在忙':
                bb.append(line)
    

    with open('neg.txt') as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = asr_analyzer.analyze(line)
            tag = intention_analyzer.analyze(line)
            if tag != '否定':
                neg.append(line+':'+tag)
            
            if tag == '肯定':
                pp.append(line)
            if tag == '否定':
                nn.append(line)
            if tag == '在忙':
                bb.append(line)

    with open('busy.txt') as fr:
        lines = fr.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = asr_analyzer.analyze(line)
            tag = intention_analyzer.analyze(line)
            if tag != '在忙':
                busy.append(line+':'+tag)

            if tag == '肯定':
                pp.append(line)
            if tag == '否定':
                nn.append(line)
            if tag == '在忙':
                bb.append(line)
    import pdb;pdb.set_trace()