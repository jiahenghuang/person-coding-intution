#coding: utf-8
import os
import sys
import re
import io
import string
import time
import jieba
import datetime
import logging
import chardet
from functools import wraps
import numpy as np
import pandas as pd
import json

from tflearn.data_utils import pad_sequences
import config

PUNCTS_EN = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
PUNCTS_CN = '！“”￥…‘’（），–—－。、：；《》？【】·√◆⾃●'
PUNCTS_SPACE = ' '
PUNCTS_ALL = PUNCTS_EN + PUNCTS_CN + PUNCTS_SPACE + u'\t\r\n'

punct_set = set(string.punctuation)
space_set = set(' \t\r\n')

fmt_pat = re.compile('[^a-zA-Z0-9一-龥]+')
fmt_pat_start = re.compile('^[^a-zA-Z0-9一-龥]+')
fmt_pat_end = re.compile('[^a-zA-Z0-9一-龥]+$')

div = lambda x, y: 0 if y==0 else x*1.0/y
fmt_pat_cn = re.compile('[一-龥]')
fmt_pat_not_cn = re.compile('[^一-龥]')
fmt_pat_kd = re.compile('[^(一-龥|，|？|！|。)]')
max_sentence_length=30
PAD_ID, UNK_ID = 0, 1

fmt_pat = re.compile('[^一-龥]+')

def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1,len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i     
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i-1] == word2[j-1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i-1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]

def to_json(obj):
    '''
    将数据转换成json格式
    '''
    return json.dumps(obj, ensure_ascii=False)

def str2json(json_str):
    '''
    将string转化为json格式
    '''
    try:
        return json.loads(json_str)
    except:
        return None

def transform_multilabel_as_multihot(label):
    """
    将label转换为one-hot
    """
    result=[0,0,0]
    result[label] = 1
    return result

def pad_sent(word2index, words):
    """
    padding
    """
    words_id_list=[word2index.get(x,UNK_ID) for x in words if x.strip()]
    sentences = pad_sequences([words_id_list], maxlen=max_sentence_length, value=0.) 
    return sentences[0].tolist()

def get_word_list(word_embedding_path):
    '''
    获取word列表
    '''
    with open(word_embedding_path) as fr:
        lines_wv = fr.readlines()
    word_list= [line.strip().split('\t')[0] for line in lines_wv]
    return word_list

def write_word(word_list, vocab_path):
    '''
    将word写出到文件
    '''
    count = 2
    with open(vocab_path, 'w') as fw:
        word2index={}
        for word in word_list:
            word2index[word]=count
            fw.write(word+"\n")
            count += 1
    return word2index

def seg_sent(string):
    """
    切词
    """
    # segged_words = list(jieba.cut(string))
    segged_words = list(string)
    new_sent = []
    for word in segged_words:
        if not fmt_pat.match(string) and not isHasStopWord(word):
            new_sent.append(word)
    if new_sent:
        return new_sent
    return []


def loadStopWord(stopWordPath="data/newdata/stopwords.txt"):
    """加载提供词表

    Returns:
        返回停用词列表
    """
    txt = open(stopWordPath, "r", encoding="UTF-8")
    txt_list = []
    for line in txt.readlines():
        line = line.strip()  # 去掉每行头尾空白
        txt_list.append(line)
    return txt_list

stopwordList = loadStopWord()

def isHasStopWord(_word):
    """判断当前词是否是停用词

    Args:
        _sent: 输入词、字

    Returns:
        是否是停用词、字
    """
    if _word in stopwordList:
        return True
    else:
        return False

def load_data_and_labels(train_file, test_file, word2index):
    """
    加载数据
    """
    # Load data from files
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    predictors = ['keyword','label']

    train = train[predictors]
    train = train[train['label'].apply(lambda x:x in [0, 1, 2])]  #将label值为0,1,2的抽取出来
    train['words'] = train['keyword'].apply(lambda x:seg_sent(x))   #切词
    train = train[train['words'].apply(lambda x:x != [])]  #去除切词后移除标点符号的数据
    train['pad_words'] = train['words'].apply(lambda x:pad_sent(word2index, x))

    test = test[predictors]
    test = test[test['label'].apply(lambda x:x in [0, 1, 2])]  #将label值为0,1,2的抽取出来
    test['words'] = test['keyword'].apply(lambda x:seg_sent(x))   #切词
    test = test[test['words'].apply(lambda x:x != [])]  #去除切词后移除标点符号的数据
    test['pad_words'] = test['words'].apply(lambda x:pad_sent(word2index, x))

    train['hot_label'] = train['label'].apply(lambda x:transform_multilabel_as_multihot(x)) 
    test['hot_label'] = test['label'].apply(lambda x:transform_multilabel_as_multihot(x))

    # Split by words
    x_train = np.array(train['pad_words'].tolist())
    x_test = np.array(test['pad_words'].tolist())

    train_labels = np.array(train['hot_label'].tolist())
    test_labels = np.array(test['hot_label'].tolist())
    return x_train, train_labels, x_test, test_labels

def recog_rule(words):
    word1 = words[0]
    word2 = words[1]
    if fmt_pat_cn.match(word1) and fmt_pat_cn.match(word2):
        return True
    return False

def safe_str(x, enc='utf8', lower=True):
    '''
    utf-8解码
    '''
    x = '' if x is None else x
    x = x.decode(enc, 'ignore')
    x = x.lower() if lower else x
    return x
        
def segger(line):
    '''
    分词
    '''
    return list(jieba.cut(line))

def encode(x):
    '''
    utf-8编码
    '''
    return '' if x is None else x.encode('utf8', 'ignore')

def get_enc(data):
    '''
    获取数据的编码方式
    '''
    try :
        result = chardet.detect(data)
        enc = result['encoding']
    except :
        enc = 'utf8'    
    return enc
    
def safe_line_generator(fname, enc='utf8', lower=True, strip=0):
    '''
    产生器：从文件中读取line，同时进行进行utf8解码
    '''
    lno = 0
    for line in io.open(fname, encoding=enc) :
        line = line.lower() if lower else line
        if lno == 0 and len(line) > 0 and line[0] == u'\ufeff' :
            line = line[1:]
        lno += 1
        
        yield line.strip() if strip else line

def fmt_line(line):
    '''
    格式化line，仅保留中英文数字
    '''
    return fmt_pat.sub('', line)

def fmt_line_strip(line):
    '''
    格式化line，仅保留以中英文数字开头，以中英文数字结尾的中间文本，中间文本可以包含非中英文数字的文本
    '''
    line = fmt_pat_start.sub('', line)
    line = fmt_pat_end.sub('', line)
    return line
    
def strip_puncts(x):
    '''
    去除所有的标点符号
    '''
    return x.strip(PUNCTS_ALL)

def re_compile(x):
    '''
    正则表达式的compile
    '''
    rep = None
    try :
        rep = re.compile(x)
    except :
        raise        
    return rep

def exit_on_unexist(fname):
    '''
    判断文件存在不存在
    '''
    if not os.path.exists(fname) :
        sys.exit(-1)

def get_last_modified(fname):
    '''
    获取文件的最新更新时间
    '''
    if not os.path.exists(fname) :
        return None
    
    statbuf = os.stat(fname)
    return datetime.datetime.fromtimestamp(statbuf.st_mtime)

def is_chinese(ch):
    '''
    判断一个unicode是否为汉字
    '''
    if ch >= u'\u4e00' and ch <= u'\u9fa5':
        return True
    else:
        return False

def is_number(ch):
    '''
    判断一个unicode是否为数字
    '''
    if ch >= u'\u0030' and ch <= u'\u0039':
        return True
    else:
        return False

def is_alpha(ch):
    '''
    判断一个unicode是否为英文字母
    '''
    if (ch >= u'\u0041' and ch <= u'\u005a') or (ch >= u'\u0061' and ch <= u'\u007a'):
        return True
    else:
        return False

def is_punct_en(ch):
    '''
    判断是否为英文标点
    '''
    if ch in punct_set:
        return True
    return False

def is_space(ch):
    '''
    判断是否为空白字符
    '''
    if ch in space_set:
        return True
    return False

def is_punct_zh(ch):
    '''
    判断是否为中文标点。一个unicode既非数字、英文字母、英文标点，又非汉字，则为中文本标点
    '''
    if not (is_chinese(ch) or is_number(ch) or is_alpha(ch) or is_punct_en(ch) or is_space(ch)):
        return True
    else:
        return False

pattern = "[\u4e00-\u9fa5]+"
regex = re.compile(pattern)
def getChineseChar(sent):
    """

    Args:
        sent: 需要处理的句子

    Returns:
        处理好的句子

    """
    results = "".join(regex.findall(str(sent)))
    return results

func2t = {}
def debugger(logger=config.logger, level=logging.INFO, prefix='', details=0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global func2t
            #logger.log(level, 'begin: (%s.%s)' % (prefix, func.__name__,))
            t_start = time.time()
            res = func(*args, **kwargs)
            cost = time.time() - t_start
            func_str = '%s.%s'%(prefix, func.__name__)
            cost_str = '<%.5f> secs'%cost if cost > 1.0 else '<%.3f> ms'%(cost*1000)
            logger.log(level, 'finished: (%s), cost: %s' % (func_str, cost_str))
            
            func2t[func_str] = cost*1000
            if details :
                cost_list = filter(lambda x: x[1] > 0, func2t.iteritems())
                cost_list.sort(key=lambda x: x[1], reverse=True)
                cost_str = '|'.join(map(lambda x: '%s:%d ms'%x, cost_list))
                logger.log(level, 'finished: (%s), cost: %s' % (func_str, cost_str))
                func2t = {}
                
            return res
        return wrapper
    return decorator

def fuseModel(ruleResult,modelResult, textLen=7, requestType=0,keywordVersion='',modelVersion=''):
    """规则和模型融合

    融合规则：
         1. 规则和模型都没有结果，返回兜底
         2. 规则和模型都有结果，二者选取置信度高的结果
         3. 只有规则结果，采用规则
         4. 只有模型结果，采用模型

    Args:
        ruleReuslt: 规则结果
         {
            "group":'yes',
            "prob":0.9,#置信度
            "word":"%单身%",
            "rank":"level1"
         }
        modelResult: 模型结果
        {
            'label': 'no', #意图
            'prob': 0.9975164, #置信度
            'responseType': 1 #1表示模型输出的结果
        }
    Returns:
        二者融合结果
        resutl={
            "label":"yes",
            "prob":0.9,
            "responseType":1,
            "word":"%sss%",#非必须，默认""
            "rank":3#非必须，默认-1
        }
    """
    ret = {
        "label": "",
        "prob": -1,
        "responseType": 0,
        "word": "",  # 非必须，默认""
        "rank": -1, # 非必须，默认-1
        "errorCode":0
    }
    try:
        if requestType == 3 and modelResult:
            try:
                ret["label"] = modelResult["label"]
                ret["prob"] = modelResult["prob"]
                ret["responseType"] = 3
            except Exception as e:
                print(e)
                ret["errorCode"] = 1
            return ret
        elif requestType == 0 and ruleResult:
            try:
                ret["label"] = ruleResult.group
                ret["responseType"] = 0
                ret["word"] = ruleResult.word
                ret["rank"] = ruleResult.rank 
            except Exception as e:
                print(e)
                ret["errorCode"] = 1
            return ret
        elif requestType == 1:
            # 特殊处理 prob
            if ruleResult:
                if not hasattr(ruleResult, "prob"):
                    if textLen>6:
                        if keywordVersion == 'K20190424B1':
                            prob_dict = [1.0,0.883,0.973,0.557,0.0,0.865,0.0,0.653,0.571]
                        else:
                            prob_dict = [1.0, 1.0, 0.922, 0.962, 0.996, 0.5, 0.886, 1.0, 0.0, 0.667,0.833, 0.455]
                    else:
                        if keywordVersion == 'K20190424B1':
                            prob_dict = [0.953,0.986,0.923,0.963,0.0,0.952,0.4]
                        else:
                            prob_dict = [0.905, 1.0, 0.962, 0.987, 0.976, 1.0, 0.714, 0.923, 0.933, 0.0, 0.078]
                    rank = int(ruleResult.rank)    
                    ruleResult.prob = prob_dict[rank-1]

            model_threshold = config.cnn_config.get(modelVersion).get('model_threshold')
            if ruleResult is None and modelResult["prob"] < model_threshold:
                # 规则和模型都没有结果，返回兜底
                ret["label"] = ""
            elif ruleResult is not None and modelResult["prob"] < model_threshold:
                #只有规则结果，采用规则
                ret["label"] = ruleResult.group
                ret["prob"] = ruleResult.prob
                ret["responseType"] = 0
                ret["word"] = ruleResult.word
                ret["rank"] = ruleResult.rank
            elif ruleResult is None and modelResult["prob"] >= model_threshold:
                # 只有模型结果，采用模型
                ret["label"] = modelResult["label"]
                ret["prob"] = modelResult["prob"]
                ret["responseType"] = 1
            elif ruleResult is not None and modelResult["prob"] >= model_threshold:
                # 规则和模型都有结果，二者选取置信度高的结果
                if ruleResult.prob > modelResult["prob"]:
                    ret["label"] = ruleResult.group
                    ret["prob"] = ruleResult.prob
                    ret["responseType"] = 0
                    ret["word"] = ruleResult.word
                    ret["rank"] = ruleResult.rank
                else:
                    ret["label"] = modelResult["label"]
                    ret["prob"] = modelResult["prob"]
                    ret["responseType"] = 1
                    ret["word"] = ruleResult.word
                    ret["rank"] = ruleResult.rank
    except Exception as e:
        print(e)
        ret["errorCode"] = 1
    return ret

def props(obj):
    """转换obj ->>> dict

    Args:
        obj: class obj
    Returns:
        dict
    """

    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value):
            pr[name] = value
    return pr

if __name__ == '__main__' :
    pass