#coding:utf-8

class KVObj(object):
    '''
    代表某一个kname下的一条正则
    '''
    def __init__(self, group='', rank=-1, type_name='', vpat='', word='', start_pos= -1, end_pos = -1, ratio = 0.0):
        self.group = group  # 肯定、否定、在忙、拒绝
        self.rank = rank  # 排行，1,2,3,4...代表匹配顺序
        self.type = type_name  # 代表是正则表达式还是字典, regu_expr or dict
        self.vpat = vpat  #表达式的值
        self.word = word
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.ratio = ratio

    def __str__(self):
        return 'group=<%s>,rank=<%d>,type=<%s>,vpat=<%s>,word=<%s>,start_pos=<%d>,end_pos=<%d>,ratio=<%.3f>'%(self.group, self.rank, self.type, self.vpat, self.word, self.start_pos, self.end_pos, self.ratio)