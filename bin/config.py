# coding:utf-8
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

# 纠错模块
asr_faq_dict = os.path.join(path_root, "data_config/correct.dict") #保存原始错词映射
asr_faq_json = os.path.join(path_root, "data_config/correct_dict.json") #保存处理后的错词映射
asr_faq_pkl = os.path.join(path_root, "data_config/acsm.pkl") #保存用于文本纠错的AC自动机

ambiguity_short = os.path.join(path_root, 'templates/ambiguity_less6.pat')
ambiguity_long = os.path.join(path_root, 'templates/ambiguity_more6.pat')

dict_trie_file = os.path.join(path_root, 'data/dict_tries.trie')  #保存非纠正错别字所对应的字典树
asr_dict_file_short = os.path.join(path_root, 'data/asr_dict_short.dict')  #保存纠正错别字所对应字典树
asr_dict_file_long = os.path.join(path_root, 'data/asr_dict_long.dict')  #保存纠正错别字所对应字典树

correct_dict_long = os.path.join(path_root, 'templates/correct_more6.dict')  #错别字字典
correct_dict_short = os.path.join(path_root, 'templates/correct_less6.dict')  #错别字字典

asr_name_long = 'ASR_LONG'  #ASR变量名
asr_name_short = 'ASR_SHORT'

dict_pairs = [(os.path.join(path_root, 'data/city.dict'), 'CITY'),                # 城市名字典
              (os.path.join(path_root, 'data/keyword.dict'), 'KEYWORDS')]         # 关键词字典

vertorizer_file = os.path.join(path_root, 'models/lightgbm/vectorizer_0.pkl')
tfidftransformer_file = os.path.join(path_root, 'models/lightgbm/tfidftransformer_0.pkl')
lightgbm_model_file = os.path.join(path_root, 'models/lightgbm/lightgbm_model_0.model')

# condition_prob_file = os.path.join(path_root, 'data/condition_prob.prob')
condition_prob_file = os.path.join(path_root, 'data/condition_prob.pkl')
module_name = 'chat-bot'

#textcnn参数
cnn_config = {
    'M20190402A2':
        {
        'num_seqs' : 30,
        'num_classes' : 5,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", "refuse", "busy", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20190402A2/model-1320"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20190402A2/char2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
        },
    'M20190424B1':
    {
        'num_seqs' : 30,
        'num_classes' : 3,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20190424B1/model-552"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20190424B1/char2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    # 'M20190620B2':
    # {
    #     'num_seqs' : 30,
    #     'num_classes' : 3,
    #     'num_filters' : 128,
    #     'filter_steps' : [1,2,3,4,5],
    #     'embedding_size' : 200,
    #     'learning_rate' : 0.001,
    #     'batch_size'    : 128,
    #     'num_epoches'   : 12,
    #     'l2_ratio'      : 0.01,
    #     'model_threshold' : 0.4,
    #     'label_list' : ["yes", "no", ""],
    #     'textcnn_path' : os.path.join(path_root, "models/textcnn/M20190620B2/model-1044"),
    #     'word2index_path' : os.path.join(path_root, 'models/textcnn/M20190620B2/word2index.pkl'),
    #     'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    # },
    'M20190620J1':
    {
        'num_seqs' : 30,
        'num_classes' : 3,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20190620J1/model-528"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20190620J1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    'M20190723L1':
    {
        'num_seqs' : 30,
        'num_classes' : 3,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20190723L1/model-480"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20190723L1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    'M20191025B1':
    {
        'num_seqs' : 30,
        'num_classes' : 5,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", "refuse", "busy", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20191025B1/model-1044"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20191025B1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    'M20191025J1':
    {
        'num_seqs' : 30,
        'num_classes' : 5,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", "refuse", "busy", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20191025J1/model-528"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20191025J1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    'M20191025L1':
    {
        'num_seqs' : 30,
        'num_classes' : 5,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.01,
        'model_threshold' : 0.4,
        'label_list' : ["yes", "no", "refuse", "busy", ""],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20191025L1/model-480"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20191025L1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    },
    'M20200102R1':
    {
        'num_seqs' : 50,
        'num_classes' : 5,
        'num_filters' : 128,
        'filter_steps' : [1,2,3,4,5],
        'embedding_size' : 200,
        'learning_rate' : 0.001,
        'batch_size'    : 128,
        'num_epoches'   : 12,
        'l2_ratio'      : 0.1,
        'model_threshold' : 0.4,
        'label_list' : ["", "yes", "no", "refuse", "busy"],
        'textcnn_path' : os.path.join(path_root, "models/textcnn/M20200102R1/model-600"),
        'word2index_path' : os.path.join(path_root, 'models/textcnn/M20200102R1/word2index.pkl'),
        'stopWordPath' : os.path.join(path_root, 'data/newdata/stopwords.txt')
    }
}

# 主流程意图关键词模块配置
# pattern file path
# pat_fname_more6 = os.path.join(path_root,'templates/intention_more6.pat')
# pat_fname_less6 = os.path.join(path_root,'templates/intention_less6.pat')

keyword_folders = os.listdir(os.path.join(path_root, 'templates/keyword'))
pat_fnames = {}
for folder in keyword_folders:
    file_names = sorted(os.listdir(os.path.join(path_root+'/templates/keyword', folder)))
    file_names = [os.path.join(path_root+'/templates/keyword/'+folder,file) for file in file_names]
    pat_fnames[folder] = file_names


# 日志控制
# if not os.path.exists(os.path.join('/dockerLog','log')):
if not os.path.exists(os.path.join(path_root,'log')):
    os.mkdir('log')
    # os.mkdir(os.path.join('/dockerLog','log'))

# path_log = os.path.join('/dockerLog', 'log')
path_log = os.path.join(path_root, 'log')
path_log_f = os.path.join(path_log, module_name + '.log')


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
    handler = logging.FileHandler(log_file, 'w', 'utf-8')
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
# logger = logging
# logger.basicConfig(level=logging.INFO)
logger.info('configurations setup ok')


# kd模块路径及文本预处理所需配置
file_query_rules = os.path.join(path_root, 'templates/kd_re.txt')
kd_pat_path = os.path.join(path_root, 'templates')
kd_pat_files = [os.path.join(kd_pat_path, f) for f in os.listdir(kd_pat_path) if ".pattern" in f]
# kd1_pat_file = os.path.join(path_root, 'templates/kd1.pattern')
# kd2_pat_file = os.path.join(path_root, 'templates/kd2.pattern')
# kd3_pat_file = os.path.join(path_root, 'templates/kd3.pattern')


share_path = "/sha"
kd_faq_path = os.path.join(path_root, "data_config")

query2rid_file = os.path.join(kd_faq_path, "qa_2_baidu_E_config")
rid2res_file = os.path.join(kd_faq_path, "doc.xls")
file_config_dict = os.path.join(kd_faq_path, "jieba_dict")
file_stop_dict = os.path.join(kd_faq_path, "jieba_stopwords")
file_spoken_words = os.path.join(kd_faq_path, "slu_words_del")

index_filename = "index_v3.0"
local_se_index = os.path.join(share_path, index_filename)
if not os.path.exists(local_se_index):
    file_se_index = os.path.join(kd_faq_path, index_filename)
else:
    file_se_index = local_se_index
file_se_index_v1 = "data_config/index_v4.0"
logger.info("SearchEngine index path: {}".format(file_se_index))
first_kd = ['Tkd{}'.format(i) for i in (1, 3, 8, 12, 15, 16, 17, 19, 21, 22, 23, 24, 25, 27, 28)]
del_kd = [2, 4, 5, 6, 7, 9]
kd_version = {"kd1": 20, "kd2": 21, "kd3": 28}

file_nlu_dict = os.path.join(share_path, "nlu_dict")
if os.path.exists(file_nlu_dict):
    vital_words = [w.strip() for w in open(file_nlu_dict, "r", encoding="utf-8").readlines()]
    logger.info("Spoken vocabulary path: {}".format(file_nlu_dict))
else:
    vital_words = "现在 在 哪里 那里 哪块 哪边 那边 哪儿 那儿 哪 地方 谁 哪个 那个 怎么 干嘛 什么 你 她 他 它 " \
              "啥 有 有没有 没有 没 不要 要 联系 要求 一样 时候 什么样 到 打 拨打 情况 介绍 看见 留意 看到 认识 说明 上面".split()

pronouns_drop = ["珍爱网", "珍爱", "你们", "你", "您们", "您"][2:]
pronouns_config = "我 她 他 它 他们 她们 它们".split()
stopwords_init = [w.strip() for w in open(file_stop_dict, "r", encoding="utf-8").readlines()]
stopwords_new = [w for w in stopwords_init if w not in vital_words]

spoken_words = [w.strip() for w in open(file_spoken_words, "r", encoding="utf-8").readlines()]
