# -*- coding: utf-8 -*-
# @Time    : 2019-02-22 11:50
# @Author  : taotao.zhou@zhenai.com
# @File    : eval.py.py
# @Software: PyCharm
# @des     :意图识别性能测试

import os
import utils
import numpy as np
import pandas as pd

import logging

import utils
import config
from sklearn.metrics import classification_report,precision_score,accuracy_score,confusion_matrix

from analyze_asr import asr_analyzer
from analyze_pattern import pattern_analyzer
from analyze_intention_cnn import predictor_intention_cnn
from analyze_faq_pattern import faq_pattern
from analyze_chat import MainHandler
class EvalClass(object):
    """性能测试类
       包括关键词指标
       模型指标
       二者融合后的指标

    """
    def __init__(self):
        self.testDataPath = os.path.join(os.getcwd(), 'data/newdata/review/0228/test.csv')
        # self.trainDataPath = os.path.join(os.getcwd(), 'data/newdata/review/0228/train.csv')
        pass

    def _dealLabel(self,label):
        """6转为5

        Args:
            label: label值

        Returns:
            处理后的label值
        """
        if label == 6:
            label = 5
        elif label == 16:
            label = 1
        elif label == 26:
            label = 2
        elif label == 36:
            label = 3
        elif label == 46:
            label = 4
        elif label == 56:
            label = 5

        return label

    def loadTestData(self):
        """加载测试数据
        """
        """
            加载数据
            """
        # Load data from files
        test = pd.read_csv(self.testDataPath)

        predictors = ['text', 'label']

        test = test[predictors]
        test = test[test['label'].apply(lambda x: x in [1, 2, 3, 4, 5, 6,16,26,36,46,56])]  # 将label值为0,1,2的抽取出来

        test['hot_label'] = test['label'].apply(lambda x: self._dealLabel(x))
        test['hot_text'] = test['text'].apply(lambda x: utils.getChineseChar(x))
        test = test[test['hot_text'].apply(lambda x: x != "")]

        # train = pd.read_csv(self.trainDataPath)
        #
        # predictors = ['text', 'label']
        #
        # train = train[predictors]
        # train = train[train['label'].apply(lambda x: x in [1, 2, 3, 4, 5, 6])]  # 将label值为0,1,2的抽取出来
        #
        # train['hot_label'] = train['label'].apply(lambda x: self._dealLabel(x))
        # train['hot_text'] = train['text'].apply(lambda x: utils.getChineseChar(x))
        # train = train[train['hot_text'].apply(lambda x: x != "")]
        #
        # data = pd.concat([train, test])


        x_test = np.array(test['hot_text'])
        test_labels = np.array(test['hot_label'])
        return x_test, test_labels


    def confusion_matrix_print(self,labels_predict, labels_right,label):
        """
        混淆矩阵打印/保存
        Args:
            labels_predict:预测的label list
            labels_right: 正确的label list
            label:label类别list

        Returns:
            无
        """
        result = confusion_matrix(labels_right,labels_predict,label)
        # pd.crosstab(labels_right, labels_predict, rownames=["lable"], colnames=["predict"])
        # print(result)
        np.savetxt("data/confusion_matrix.txt", result, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
        return result

    def evalCompute(self, texts, labels_predict, labels_right, pattern=None, notes="model"):
        """指标评估，打印/保存文件

        Args:
            label_pred:预测的label list
            label_right:正确的label list

        Returns:
            无
        """
        acc_for_each_class = precision_score(labels_right, labels_predict, average=None)
        all_accuracy = accuracy_score(labels_right, labels_predict)
        target_names = ['单身', '否定', '拒绝', '在忙', '其他兜底']
        with open('data/result.txt', 'a') as fw:
            print(str("eval") + ':' + '\n' + classification_report(labels_right,
                                                                                              labels_predict,
                                                                                              target_names=target_names) + '\n' + 'acc_for_each_class: ' + ','.join(
                ['{:.3f}'.format(x) for x in acc_for_each_class]) + '\n' + 'total_acc: ' + '{:.3f}'.format(
                all_accuracy) + '\n')
            confusion_matrix_result = self.confusion_matrix_print(labels_right,labels_predict,label=[1, 2, 3 ,4, 5])
            print(confusion_matrix_result)
            fw.write(str("eval") + ':' + '\n' + classification_report(labels_right,
                                                                                                 labels_predict,
                                                                                                 target_names=target_names) + '\n' + 'acc_for_each_class: ' + ','.join(
                ['{:.3f}'.format(x) for x in acc_for_each_class]) + '\n' + 'total_acc: ' + '{:.3f}'.format(
                all_accuracy) + '\n')

        with open('data/error.csv', 'a') as fw:
            if pattern:
                fw.write(notes + ',' + '标注结果' + ',' + '预测结果' + ','+'关键词'+'\n')
                for i in range(len(texts)):
                    if labels_predict[i] != labels_right[i]:
                        fw.write(texts[i] + ',' + str(labels_right[i]) + ',' + str(labels_predict[i]) + ',' + str(pattern[i]) + '\n')
            else:
                fw.write(notes+','+'标注结果'+','+'预测结果'+'\n')
                for i in range(len(texts)):
                    if labels_predict[i] != labels_right[i] and (labels_right[i] == 5):
                        fw.write(texts[i]+','+str(labels_right[i])+','+str(labels_predict[i])+'\n')

        # fw_right = open('data/right.csv', 'w', encoding= 'utf-8-sig')
        # fw_error = open('data/error.csv', 'w', encoding= 'utf-8-sig')
        # fw_right.write('text,' + 'label' + '\n')
        # fw_error.write('text,' + 'label,' + 'predict' + '\n')
        # for i in range(len(texts)):
        #     if labels_predict[i] != labels_right[i] and labels_right[i] == 5:
        #         fw_error.write(texts[i] + ',' + str(labels_right[i]) + ',' + str(labels_predict[i]) + '\n')
        #     else:
        #         fw_right.write(texts[i] + ',' + str(labels_right[i]) + '\n')
        #
        # fw_right.close()
        # fw_error.close()

    def ruleEvalForRankACC(self):
        """ 测试集上计算关键词的每个rank的acc，作为置信度，在关键词和模型融合时使用

        Returns:
            无

        """
        input, label_right = self.loadTestData()
        label_pred = []
        texts = []
        rank_list = []
        text_type_list = []
        for text in input:
            text = asr_analyzer.analyze(text)
            texts.append(text)
            textLen = len(text)
            if textLen > 6:
                tmp_result = pattern_analyzer.parse(text, long=True)
                text_type = 'long'
            else:
                tmp_result = pattern_analyzer.parse(text, long=False)
                text_type = 'short'
            if tmp_result:
                pred_label = tmp_result.group
                rank = tmp_result.rank
            else:
                pred_label = 5
                rank = 20

            if pred_label == "yes":
                pred_label = 1
            elif pred_label == "no":
                pred_label = 2
            elif pred_label == "refuse":
                pred_label = 3
            elif pred_label == "busy":
                pred_label = 4
            label_pred.append(pred_label)
            rank_list.append(rank)
            text_type_list.append(text_type)

        ret = {}
        ret['long'] = {}
        ret['short'] = {}
        for i in range(len(label_right)):
            if str(rank_list[i]) not in ret[text_type_list[i]]:
                ret[text_type_list[i]][str(rank_list[i])] = {}
                ret[text_type_list[i]][str(rank_list[i])]['right'] = 0
                ret[text_type_list[i]][str(rank_list[i])]['error'] = 0
            if label_right[i] == label_pred[i]:
                ret[text_type_list[i]][str(rank_list[i])]['right'] += 1
            else:
                ret[text_type_list[i]][str(rank_list[i])]['error'] += 1

        print(ret)


        evalRet = {}
        for key1 in ret:
            evalRet[key1] = {}
            for key2 in ret[key1]:
                evalRet[key1][key2] = round(1.0 * ret[key1][key2]['right']/(ret[key1][key2]['right']+ret[key1][key2]['error']), 3)

        çountRet = {}
        for key1 in ret:
            çountRet[key1] = {}
            for key2 in ret[key1]:
                çountRet[key1][key2] = str(ret[key1][key2]['right'])+'/'+str((ret[key1][key2]['right']+ret[key1][key2]['error']))
        # print(sorted(evalRet,key=lambda student: student.age))

        totalRet = {}
        totalRet["count"] = {}
        totalRet["acc"] = {}
        # sortRet = {}
        # totalRet["acc"]['long'] = sorted(evalRet['long'].items(), key=lambda x: int(x[0]), reverse=False)
        # totalRet["acc"]['short'] = sorted(evalRet['short'].items(), key=lambda x: int(x[0]), reverse=False)
        # print(sortRet)
        # totalRet["count"]['long'] = sorted(çountRet['long'].items(), key=lambda x: int(x[0]), reverse=False)
        # totalRet["count"]['short'] = sorted(çountRet['short'].items(), key=lambda x: int(x[0]), reverse=False)

        totalRet["count"] = çountRet
        totalRet["acc"] = evalRet
        print(totalRet)

    def ruleEval(self):
        """
        关键词意图分类效果评估
        """

        input ,label_right = self.loadTestData()
        label_pred = []
        texts = []
        pattern = []
        for text in input:
            text = asr_analyzer.analyze(text)
            texts.append(text)
            textLen = len(text)
            if textLen > 6:
                tmp_result = pattern_analyzer.parse(text, long=True)
            else:
                tmp_result = pattern_analyzer.parse(text, long=False)

            if tmp_result:
                pred_label = tmp_result.group
                pattern.append(tmp_result.word)
            else:
                pred_label = 5
                pattern.append(" ")

            if pred_label == "yes":
                pred_label = 1
            elif pred_label == "no":
                pred_label = 2
            elif pred_label == "refuse":
                pred_label = 3
            elif pred_label == "busy":
                pred_label = 4
            label_pred.append(pred_label)
        self.evalCompute(texts, label_pred, label_right, pattern=pattern, notes="关键词结果")

    def modelEval(self):
        """
        模型意图分类效果评估
        """
        input, label_right = self.loadTestData()
        label_pred = []
        texts = []
        for text in input:
            text = asr_analyzer.analyze(text)
            textLen = len(text)
            text = "".join(utils.seg_sent(utils.getChineseChar(text)))
            texts.append(text)
            result = predictor_intention_cnn.analyze(text)
            if result["prob"] >= config.model_threshold:
                pred_label = result["label"]
            else:
                pred_label = 5

            if pred_label == "yes":
                pred_label = 1
            elif pred_label == "no":
                pred_label = 2
            elif pred_label == "refuse":
                pred_label = 3
            elif pred_label == "busy":
                pred_label = 4
            elif pred_label == "":
                pred_label = 5
            label_pred.append(pred_label)
        self.evalCompute(texts, label_pred, label_right, notes="model result")

    def fuseEval(self):
        """
        关键词和模型融合效果评估
        """

        input, label_right = self.loadTestData()
        label_pred = []
        texts = []
        for text in input:
            text = asr_analyzer.analyze(text)
            textLen = len(text)
            texts.append(text)
            # rule
            if textLen > 6:
                ruleResult = pattern_analyzer.parse(text, long=True)
            else:
                ruleResult = pattern_analyzer.parse(text, long=False)
            #model

            # 提取中文，去停用词
            text = "".join(utils.seg_sent(utils.getChineseChar(text)))

            # 文本过短，不用模型预测
            if len(text) <= 2:
                modelResult = {
                    'label': '',  # 意图
                    'prob': 0.0,  # 置信度
                    'responseType': 1  # 1表示模型输出的结果
                }
            else:
                modelResult = predictor_intention_cnn.analyze(text)

            ret = utils.fuseModel(ruleResult=ruleResult, modelResult=modelResult, textLen=textLen)
            pred_label = ret["label"]
            if pred_label == "yes":
                pred_label = 1
            elif pred_label == "no":
                pred_label = 2
            elif pred_label == "refuse":
                pred_label = 3
            elif pred_label == "busy":
                pred_label = 4
            elif pred_label == "":
                pred_label = 5
            label_pred.append(pred_label)
        self.evalCompute(texts, label_pred, label_right, notes="fuse result")

if __name__ == "__main__":
    obj = EvalClass()
    # obj.ruleEval()
    # obj.modelEval()
    # obj.fuseEval()
    obj.ruleEvalForRankACC()