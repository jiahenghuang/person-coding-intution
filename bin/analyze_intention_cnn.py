# coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import os
import utils
import pandas as pd
import pickle

import config
from layer.textcnn import NN_config, CALC_config, TextCNNClassifier


class IntentionPredictor(object):
    '''
    使用textcnn做的预测接口
    '''

    def __init__(self):
        with open(config.word2index_path, 'rb') as fr:
            self.word2index = pickle.load(fr)
        self.vocab_size = len(self.word2index) + 2
        self._load_data()

    def _load_data(self):
        '''
        加载word2index和模型
        '''
        nn = NN_config(num_seqs=config.num_seqs,
                       num_classes=config.num_classes,
                       num_filters=config.num_filters,
                       filter_steps=config.filter_steps,
                       embedding_size=config.embedding_size,
                       vocab_size=self.vocab_size)

        calc = CALC_config(learning_rate=config.learning_rate,
                           batch_size=config.batch_size,
                           num_epoches=config.num_epoches,
                           l2_ratio=config.l2_ratio)
        self.textcnn_model = TextCNNClassifier(nn, calc)
        self.textcnn_model.load_model(config.textcnn_path)
        # self.textcnn_model.load_model()

    @utils.debugger(prefix='IntentionPredictorByCNN')
    def analyze(self, line):
        '''
        对句子意图进行预测
        '''
        words = utils.seg_sent(line)
        pad_sent = utils.pad_sent(self.word2index, words)
        result, score = self.textcnn_model.predict(np.array([pad_sent]))

        return {
                   "label": config.label_list[result[0]],
                   "prob": round(score[0][0].tolist()[result[0]],3),
                   "responseType": 1
               }


predictor_intention_cnn = IntentionPredictor()
if __name__ == '__main__':
    import time
    while True:
        # sent = '我不是单身不是'
        sent = input("please enter your sent: ")
        start = time.time()
        result = predictor_intention_cnn.analyze(sent)
        end = time.time()
        print('Running time: {} Seconds'.format(end - start))
        print(result)