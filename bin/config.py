#coding:utf-8
import os
import sys
import datetime
import platform
import logging

path_root = os.getcwd()

#返回label的映射
label_map = {
'TB4':'TB2',
'TP4':'TP2'
}

#纠错模块
ambiguity_short = os.path.join(path_root, 'templates/ambiguity_less6.pat')
ambiguity_long = os.path.join(path_root, 'templates/ambiguity_more6.pat')

dict_trie_file = os.path.join(path_root, 'data/dict_tries.trie')  #保存非纠正错别字所对应的字典树
asr_dict_file_short = os.path.join(path_root, 'data/asr_dict_short.dict')  #保存纠正错别字所对应字典树
asr_dict_file_long = os.path.join(path_root, 'data/asr_dict_long.dict')  #保存纠正错别字所对应字典树

correct_dict_long = os.path.join(path_root, 'templates/correct_more6.dict')  #错别字字典
correct_dict_short = os.path.join(path_root, 'templates/correct_less6.dict')  #错别字字典

asr_name_long = 'ASR_LONG'  #ASR变量名
asr_name_short = 'ASR_SHORT'

dict_pairs = [(os.path.join(path_root,'data/city.dict'), 'CITY'),                # 城市名字典
              (os.path.join(path_root,'data/keyword.dict'), 'KEYWORDS')]         # 关键词字典

vertorizer_file = os.path.join(path_root, 'models/lightgbm/vectorizer_0.pkl')
tfidftransformer_file = os.path.join(path_root, 'models/lightgbm/tfidftransformer_0.pkl')
lightgbm_model_file = os.path.join(path_root, 'models/lightgbm/lightgbm_model_0.model')

# condition_prob_file = os.path.join(path_root, 'data/condition_prob.prob')
condition_prob_file = os.path.join(path_root, 'data/condition_prob.pkl')
module_name = 'chat-bot'

#textcnn参数
num_seqs = 30
num_classes = 5
num_filters = 128
filter_steps = [1,2,3,4,5]
embedding_size = 200
learning_rate = 0.001
batch_size    = 128
num_epoches   = 12
l2_ratio      = 0.01

model_threshold = 0.4

label_list = ["yes", "no", "refuse", "busy", ""]
textcnn_path = os.path.join(path_root, "models/textcnn/model-1320")
word2index_path = os.path.join(path_root, 'data/newdata/review/char2index.pkl')
stopWrodPath = os.path.join(path_root, 'data/newdata/stopwords.txt')

#kd路径
query2rid_file = os.path.join(path_root, "data_config/qa_2_baidu_E_config")
rid2res_file = os.path.join(path_root, "data_config/doc.xls")
file_config_dict = os.path.join(path_root, "data_config/jieba_dict")
file_stop_dict = os.path.join(path_root, "data_config/jieba_stopwords")


# if not os.path.exists(os.path.join('/dockerLog','log')):
if not os.path.exists(os.path.join(path_root,'log')):
    os.mkdir('log')
    # os.mkdir(os.path.join('/dockerLog','log'))

# path_log = os.path.join('/dockerLog', 'log')
path_log = os.path.join(path_root, 'log')
path_log_f = os.path.join(path_log, module_name + '.log')

#pattern file path
# pat_fname_more6 = os.path.join(path_root,'templates/intention_more6.pat')
# pat_fname_less6 = os.path.join(path_root,'templates/intention_less6.pat')

#自动获取keyword
keyword_folders = os.listdir(os.path.join(path_root,'templates/keyword'))
pat_fnames = {}
for folder in keyword_folders:
    file_names = sorted(os.listdir(os.path.join(path_root+'/templates/keyword', folder)))
    file_names = [os.path.join(path_root+'/templates/keyword/'+folder,file) for file in file_names]
    pat_fnames[folder] = file_names

kd1_pat_file = os.path.join(path_root,'templates/kd1.pattern')
kd2_pat_file = os.path.join(path_root,'templates/kd2.pattern')

class LevelFilter(logging.Filter):
    '''
    日志分级
    '''
    def __init__(self, min_level, max_level):
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, rec):
        if rec.levelno >= self.min_level and rec.levelno <= self.max_level:
            return 1
        return 0

def _get_log_file(name):
    log_file = os.path.join(path_log, name + '.log')
    date = str(datetime.datetime.now())[:10]
    no = 0
    while True:
        no += 1
        fname = '%s.%s.%d' % (log_file, date, no)
        if not os.path.exists(fname):
            break

    #fname = '%s.%s.%d' % (log_file, date, no)
    fname = '%s.%s' % (log_file, date)
    return fname


def _get_logger(name):
    formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] '
                                      '[%(filename)s:%(lineno)d] [%(message)s]')

    log_file = _get_log_file(name)

    # 正常日志
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.addFilter(LevelFilter(logging.DEBUG, logging.INFO))

    # 异常日志
    log_file_wf = log_file + '.wf'
    handler_wf = logging.FileHandler(log_file_wf)
    handler_wf.setFormatter(formatter)
    handler_wf.setLevel(logging.WARNING)
    
    # 控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(handler_wf)
    logger.addHandler(console)

    return logger

logger = _get_logger(module_name)
logger.info('configurations setup ok')