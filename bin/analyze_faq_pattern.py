#coding:utf-8
import config
import re
import utils

class FaqPattern(object):
    '''
    使用正则表达式解析pattern
    '''
    def __init__(self):
        self._parse_from_file_kd1()
        self._parse_from_file_kd2()
    
    def _parse_from_file_kd1(self):
        with open(config.kd1_pat_file) as fr:
            data = fr.readlines()
        data = [line.strip() for line in data]
        self.pats_kd1 = []
        for line in data:
            line = line.split('=')
            kd = line[0]
            pat = line[1]
            pat = self._build_re_kd1(pat)
            self.pats_kd1.append([pat,kd])

    # @utils.debugger(prefix="FaqPattern")
    def _build_re_kd1(self, pat):
        '''
        编译正则表达式
        '''
        pats = pat.split(',')
        pats = [re.compile(inner_pat.replace('%','.{0,100}').replace('#','\\b')) for inner_pat in pats]
        return pats

    def _parse_from_file_kd2(self):
        self.pats_kd2 = []
        with open(config.kd2_pat_file, encoding="utf-8") as fp:
            for d in fp.readlines():
                d = d.strip().replace(" ", "").split('-->')
                kd, pat = d[0], d[1]
                pat = self._build_re_kd2(pat)
                self.pats_kd2.append([pat, kd])

    def _build_re_kd2(self, pat):
        '''
        编译正则表达式
        '''
        pats = pat.split(',')
        pats = [re.compile(inner_pat.replace('%', '.{0,100}').replace('#', '\\b').replace('_', '.{0,1}').
                           replace('=', '.{0,4}').replace('&', '.{0,6}')) for inner_pat in pats]
        return pats

    @utils.debugger(prefix="FaqPattern")
    def analyze(self, line, questionId = ''):
        '''
        解析句子中出现的kd词汇
        '''
        if questionId == 'KD1':
            for pat, kd in self.pats_kd1:
                for inner_pat in pat:
                    result = inner_pat.search(line)
                    if result:
                        return kd, result.group() 

        if questionId == 'KD2':
            for pat, kd in self.pats_kd2:
                for inner_pat in pat:
                    result = inner_pat.search(line)
                    if result:
                        return kd, result.group()
        return '', ''

faq_pattern = FaqPattern()
if __name__ == '__main__':
    line = '珍爱网是干嘛的'
    # line = '我的对象在哪里？'
    # line = '我想找一个靠谱的对象，你们是什么公司，有没有线下门店'
    # line = '靠谱的啊啊啊啊对象'
    line = '怎么联系你们'
    # line = '啥是相亲网站'
    line = '什么公司'
    print(faq_pattern.analyze(line,questionId='KD1'))