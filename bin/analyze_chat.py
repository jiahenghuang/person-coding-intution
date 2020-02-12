#-*- encoding:utf-8 -*-
import logging
import concurrent
import tornado.web
import tornado.ioloop
import tornado.options
from tornado import gen
import tornado.websocket
from tornado.options import define, options
import time
import logging

import utils
import config
from analyze_asr import asr_analyzer
from analyze_pattern import pattern_analyzer
from analyze_intention_cnn import predictor_intention_cnn
from analyze_faq_pattern import faq_pattern

define("port", default=8700, help="run on the given port", type=int)
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=500)
# executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)

class Interface(object):
    '''
    返回接口结果
    '''
    def __init__(self, requestId='', label='T0',prob=-1,keyword='',rank=-1,textLen=-1,text='',contextId='',serviceCode='',responseType=0,reqType=-1,
                status='',describe='',startStamp=-1,kdKeyword='',kdAnswer='',kdResponseType=-1,kdProb=-1,errorCode=0,requestType=0, questionId='',
                keywordVersion=''):
        self.requestId = requestId
        self.label = label
        self.prob = prob
        self.keyword = keyword
        self.rank = rank
        self.textLen = textLen
        self.text = text
        self.contextId = contextId
        self.serviceCode = serviceCode
        self.responseType = responseType
        self.reqType = reqType
        self.status = status
        self.describe = describe
        self.startStamp = startStamp
        self.kdKeyword = kdKeyword
        self.kdAnswer = kdAnswer
        self.kdResponseType = kdResponseType
        self.kdProb = kdProb
        self.errorCode = errorCode
        self.requestType = requestType
        self.questionId = questionId
        self.keywordVersion = keywordVersion

        if self.label == 'yes':
            self.label = 'T%s1' % self.status
        elif self.label == 'no':
            self.label = 'T%s2' % self.status
        elif self.label == 'refuse':
            self.label = 'T%s3' % self.status
        elif self.label == 'busy':
            self.label = 'T%s4' % self.status
        elif self.label == '':
            self.label = 'T0'

        if config.label_map.get(self.label, None):
            self.label = config.label_map[self.label]

        if self.requestType == 1:
            if 0 == self.responseType:
                self.responseType = 11
            elif 1 == self.responseType:
                self.responseType = 10

    def __call__(self):
        result = {'requestId':self.requestId,'label':self.label,'prob':self.prob, 'keyword':self.keyword,'rank':self.rank,'textLen':self.textLen,
        'text':self.text, 'contextId':self.contextId,'serviceCode':self.serviceCode,'responseType':self.responseType,
        'reqType':self.reqType,'status':self.status,'describe':self.describe,'startStamp':self.startStamp,'kdKeyword':self.kdKeyword,
        'kdAnswer':self.kdAnswer,'kdResponseType':self.kdResponseType,'kdProb':self.kdProb,'errorCode':self.errorCode,'requestType':self.requestType}
        return utils.to_json(result)

    def __str__(self):
        x = 'requestId=<%s> label=<%s> prob=<%.3f> keyword=<%s> rank=<%d> textLen=<%d> text=<%s> contextId=<%s> serviceCode=<%s> \
            responseType=<%d> reqType=<%d> status=<%s> describe=<%s> startStamp=<%d> kdKeyword=<%s> kdAnswer=<%s> \
            kdResponseType=<%d> kdProb=<%.3f> errorCode=<%d> requestType=<%d>' % (self.requestId, self.label, self.prob, self.keyword, self.rank, self.textLen, self.text,
            self.contextId,self.serviceCode,self.responseType,self.reqType,self.status,self.describe,self.startStamp,
            self.kdKeyword,self.kdAnswer,self.kdResponseType,self.kdProb, self.errorCode,self.requestType)
        return x



class Application(tornado.web.Application):
    def __init__(self):
        handlers = [(r"/", MainHandler)]
        settings = dict(debug=True)
        tornado.web.Application.__init__(self, handlers, **settings)

class MainHandler(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        logging.info("A client connected.")

    def on_close(self):
        logging.info("A client disconnected")

    @utils.debugger(prefix='analyze_chat')
    def analyze(self, data):
        data = utils.str2json(data)
        if not isinstance(data, dict):
            logging.info('请输入正确格式请求')
            result = Interface(errorCode=1)
            return result()

        message = data.get('message',None)
        requestId = data.get('requestId', None)
        contextId = data.get('contextId', None)
        serviceCode = data.get('serviceCode', None)
        reqType = data.get('reqType', None)
        status = data.get('status', None)
        describe = data.get('describe', None)
        startStamp = data.get('startStamp', None)
        requestType = data.get('requestType', 0)
        questionId = data.get('questionId', None)
        keywordVersion = data.get('keywordVersion', None)
        
        if not (requestId != None and message != None and contextId != None and serviceCode != None and reqType != None and status != None and describe != None and startStamp != None and questionId != None and keywordVersion != None):
            result = Interface(errorCode=1,text=message,requestId=requestId,contextId=contextId,serviceCode=serviceCode,reqType=reqType,status=status,describe=describe,startStamp=startStamp)
            return result()

        text = utils.fmt_pat_not_cn.sub('',message)
        text = text.strip()
        textLen = len(text)

        if not text:
            result = Interface(text=message,contextId=contextId,serviceCode=serviceCode,reqType=reqType,
                            status=status,describe=describe,startStamp=startStamp,requestId=requestId)
            return result()

        kd, kd_keyword = faq_pattern.analyze(text, questionId = questionId)
        
        if textLen > 6:
            text = asr_analyzer.analyze(text, long=True)
            tmp_result = pattern_analyzer.parse(text, keywordVersion = keywordVersion, long=True)
        else:
            text = asr_analyzer.analyze(text, long=False)
            tmp_result = pattern_analyzer.parse(text, keywordVersion = keywordVersion, long=False)
        """
           关键词输出格式
           tmp_result = {
                "group":'yes',
                "prob":0.9,#置信度
                "word":"%单身%",
                "rank":"level1"
           }
        
        """
        rule_result = tmp_result
        logging.info('rule Response %s' % rule_result)

        #提取中文，去停用词
        text = "".join(utils.getChineseChar(text))
        # 文本过短，不用模型预测label
        if len(text) <= 2 or 0 == requestType:
            model_result = {
                'label': '',  # 意图
                'prob': 0.0,  # 置信度
                'responseType': 1  # 1表示模型输出的结果
            }
        else:
            model_result = predictor_intention_cnn.analyze(text)

        logging.info('model response:%s' % model_result)
        """
            模型输出格式
            model_result= = {
                'label': 'no', #意图
                'prob': 0.9975164, #置信度
                'responseType': 1 #1表示模型输出的结果
            }
        """

        fuse_result = utils.fuseModel(rule_result, model_result, textLen, requestType)

        """
            二者融合输出格式
            resutl={
                "label":"yes",
                "prob":0.9,
                "responseType":1,
                "word":"%sss%",#非必须，默认""
                "rank":3#非必须，默认-1
            }
        """
        result = Interface(label=fuse_result["label"], keyword=fuse_result["word"], rank=fuse_result["rank"],
                           prob=fuse_result["prob"],errorCode=fuse_result["errorCode"],
                           responseType=fuse_result["responseType"], textLen=textLen, text=message, contextId=contextId,
                           requestId=requestId, serviceCode=serviceCode, reqType=reqType, status=status,
                           describe=describe, startStamp=startStamp, kdAnswer=kd, kdKeyword=kd_keyword, requestType=requestType)
        return result()
        """
        if tmp_result:
            mat = tmp_result
            label = mat.group
            keyword = mat.word
            rank = mat.rank
            result=Interface(label=label,keyword=keyword,rank=rank,textLen=textLen,text=message,contextId=contextId,requestId =requestId,
                            serviceCode=serviceCode,reqType=reqType,status=status,describe=describe,startStamp=startStamp,kdAnswer=kd,kdKeyword=kd_keyword)
            return result()
        
        result = Interface(label='',textLen=textLen,text=message,contextId=contextId,serviceCode=serviceCode,reqType=reqType,
                            status=status,describe=describe,startStamp=startStamp,requestId=requestId,kdAnswer=kd,kdKeyword=kd_keyword)
        return result()
        
        """


    # @utils.debugger(prefix='MainHandler')
    @gen.coroutine
    def on_message(self, message):
        # result = yield executor.submit(intention_analyzer.analyze, message)
        logging.info('Request:%s' % message)
        result = self.analyze(message)
        # result = yield executor.submit(intention_analyzer.analyze, message)
        logging.info('Response:%s' % result)
        self.write_message(result)

def main():
    '''
    主函数
    '''
    tornado.options.parse_command_line()
    app = Application()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
