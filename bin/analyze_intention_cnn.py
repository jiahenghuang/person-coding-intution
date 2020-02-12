# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import os
import utils
import pandas as pd
import pickle
import logging

import config
from layer.textcnn import NN_config, CALC_config, TextCNNClassifier

class IntentionPredictor(object):
    '''
    使用textcnn做的预测接口
    '''

    def __init__(self):
        self.cnn_model, self.word2index_dict, self.label_list = {}, {}, {}

        for modelVersion in config.cnn_config:
            cnn_config = config.cnn_config.get(modelVersion)
            with open(cnn_config.get('word2index_path'), 'rb') as fr:
                word2index = pickle.load(fr)
            self.word2index_dict[modelVersion] = word2index
            vocab_size = len(word2index) + 2
            self.cnn_model[modelVersion] = self._load_data(cnn_config, vocab_size)
            self.label_list[modelVersion] = cnn_config.get('label_list')

    def _load_data(self, cnn_config, vocab_size):
        '''
        加载word2index和模型
        '''
        nn = NN_config(num_seqs       = cnn_config.get('num_seqs'),
                       num_classes    = cnn_config.get('num_classes'),
                       num_filters    = cnn_config.get('num_filters'),
                       filter_steps   = cnn_config.get('filter_steps'),
                       embedding_size = cnn_config.get('embedding_size'),
                       vocab_size     = vocab_size)

        calc = CALC_config(learning_rate = cnn_config.get('learning_rate'),
                           batch_size    = cnn_config.get('batch_size'),
                           num_epoches   = cnn_config.get('num_epoches'),
                           l2_ratio      = cnn_config.get('l2_ratio'))
    
        textcnn_model = TextCNNClassifier(nn, calc)
        textcnn_model.load_model(cnn_config.get('textcnn_path'))
        return textcnn_model

    @utils.debugger(prefix='IntentionPredictorByCNN')
    def analyze(self, line, modelVersion = ''):
        '''
        对句子意图进行预测
        '''
        textcnn_model = self.cnn_model.get(modelVersion)
        word2index = self.word2index_dict.get(modelVersion)

        words = utils.seg_sent(line)
        pad_sent = utils.pad_sent(word2index, words)

        result, score = textcnn_model.predict(np.array([pad_sent]))
        return {
                   "label": self.label_list.get(modelVersion)[result[0]],
                   "prob": round(score[0][0].tolist()[result[0]],3),
                   "responseType": 1
               }

predictor_intention_cnn = IntentionPredictor()
if __name__ == '__main__':
    import time
    from analyze_asr import asr_analyzer
    while True:
        # sent = '我不是单身不是'
        sent = input("please enter your sent: ")
        start = time.time()
        sent = utils.fmt_pat_not_cn.sub('', sent)
        sent = sent.strip()
        textLen = len(sent)
        if textLen > 6:
            sent = asr_analyzer.analyze(sent, long=True)
        else:
            sent = asr_analyzer.analyze(sent, long=False)
        sent = "".join(utils.getChineseChar(sent))
        result = predictor_intention_cnn.analyze(sent,modelVersion='M20190424B1')
        end = time.time()
        print('Running time: {} Seconds'.format(end - start))
        print(result)