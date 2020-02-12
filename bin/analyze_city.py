#coding:utf-8
from tools.trie_func import trie_dict
import utils

class CityAnalyzer(object):
    '''
    抽取对话中的地理位置信息
    '''
    def __init__(self):
        pass

    @utils.debugger(prefix='CityAnalyzer')
    def analyze(self, line):
        '''
        抽取地理位置
        '''
        mats = trie_dict.search_all_by_name(line, 'CITY')
        return mats
    
if __name__ == '__main__':    
    line = u'我现在在北京市呢，下个月回无锡'
    mats = CityAnalyzer().analyze(line)
    for mat in mats:
        print(mat)
    # files = ['data/extract/10.19_11.19_doudi.txt',
    #         'data/extract/11.19_12.19_doudi.txt',
    #         'data/extract/10.19_11.19.txt',
    #         'data/extract/11.19_12.19.txt']
    # for file in files:
    #     with open(file) as fr:
    #         content = fr.readlines()
    #     lines = []
    #     for line in content:
    #         mats = CityAnalyzer().analyze(line)
    #         if mats:
    #             lines.append(line)
    #     with open(file+'_city', 'w') as fw:
    #         for i in lines:
    #             fw.write(i)