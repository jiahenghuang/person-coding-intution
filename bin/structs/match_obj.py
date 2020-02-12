#coding: utf-8

class MatchObj(object) :
    '''
    定义匹配出来的文本的属性
    '''
    def __init__(self, kname, value, nature=None, line=None, start_pos=-1, end_pos=-1, ratio = 0, mat=None):
        self.kname = kname  # 字典名，如city
        self.value = value  # value
        self.nature = nature  # 肯定、否定、中性等
        self.line = line   #从哪段话中匹配到的文本 
        self.start_pos = start_pos  # 在line中的起始位置
        self.end_pos = end_pos      # 在line中的结束位置
        self.ratio = ratio  #匹配到的文本占line的一个比率
        
    def __str__(self):
        '''
        打印匹配出来的文本
        '''
        x = 'kname=<%s> value=<%s> nature=<%s> line=<%s> start_pos=<%d> end_pos=<%d> ratio=<%.2f>'% \
            (self.kname, self.value, self.nature, self.line, self.start_pos, self.end_pos, self.ratio)
        return x
        
if __name__ == '__main__' :
    mr = MatchObj('city', u'北京', None, u'北京大学', 0, 2, 0.5)
    print('%s'%mr)
