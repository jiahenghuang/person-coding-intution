# -*- encoding:utf-8 -*-
import logging
import tornado.web
import tornado.ioloop
import tornado.options
from tornado import gen
import tornado.websocket
from tornado.options import define, options

import utils
import config
import json
import time
from analyze_asr import asr_analyzer
from analyze_pattern import pattern_analyzer
from analyze_intention_cnn import predictor_intention_cnn
from analyze_faq_pattern import faq_pattern
try:
    from mysql_cache import myquery
except Exception as e:
    logging.error('!!! mysql获取数据失败: {}'.format(e))

define("port", default=8703, help="run on the given port", type=int)
# executor = concurrent.futures.ThreadPoolExecutor(max_workers=500)
# executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)


class Interface(object):
    '''
    返回接口结果
    '''
    def __init__(self, res_json):
        self.use_time = res_json.get('time', -1)
        self.requestId = res_json.get('requestId', '')
        self.label = res_json.get('label', '')
        self.prob = res_json.get('prob', -1)
        self.keyword = res_json.get('keyword', '')
        self.rank = res_json.get('rank', -1)
        self.textLen = res_json.get('textLen', -1)
        self.text = res_json.get('text', '')
        self.asrText = res_json.get('asrText', '')
        self.contextId = res_json.get('contextId', '')
        self.serviceCode = res_json.get('serviceCode', '')
        self.responseType = res_json.get('responseType', 0)
        self.reqType = res_json.get('reqType', '')
        self.status = res_json.get('status', '')
        self.describe = res_json.get('describe', '')
        self.startStamp = res_json.get('startStamp', -1)
        self.kdKeyword = res_json.get('kdKeyword', '')
        self.kdAnswer = res_json.get('kdAnswer', '')
        self.ifFaq = res_json.get('if_faq', 0)
        self.kdResponseType = res_json.get('kdResponseType', -1)
        self.kdProb = res_json.get('kdProb', -1)
        self.errorCode = res_json.get('errorCode', 0)
        self.errorMsg = res_json.get('errorMsg', '')
        self.requestType = res_json.get('requestType', 0)
        self.questionId = res_json.get('questionId', '')
        self.keywordVersion = res_json.get('keywordVersion', '')
        self.modelVersion = res_json.get('modelVersion', '')

        self.reqLabel = res_json.get('reqLabel', '')
        self.reqKeyword = res_json.get('reqKeyword', '')
        self.reqKdAnswer = res_json.get('reqKdAnswer', '')
        self.reqKdKeyword = res_json.get('reqKdKeyword', '')

        if self.label == 'yes':
            self.label = 'T%s1' % self.status
        elif self.label == 'no':
            self.label = 'T%s2' % self.status
        elif self.label == 'refuse':
            self.label = 'T%s3' % self.status
        elif self.label == 'busy':
            self.label = 'T%s4' % self.status
        elif self.label == '':
            if '0' in self.reqType.split(','):
                self.label = 'T0'
        if config.label_map.get(self.label, None):
            self.label = config.label_map[self.label]

        # if self.requestType == 1:
        #     if 0 == self.responseType:
        #         self.responseType = 11
        #     elif 1 == self.responseType:
        #         self.responseType = 10
        # elif self.requestType == 3:
        #     self.responseType = 3

    def __call__(self):
        result = {'requestId':self.requestId,'label':self.label,'prob':self.prob, 'keyword':self.keyword,'rank':self.rank,
        'textLen':self.textLen,'text':self.text, 'asrText':self.asrText,'contextId':self.contextId,'serviceCode':self.serviceCode,
        'responseType':self.responseType,'reqType':self.reqType,'status':self.status,'describe':self.describe,
        'startStamp':self.startStamp,'kdKeyword':self.kdKeyword,'kdAnswer':self.kdAnswer,'ifFaq':self.ifFaq,
        'kdResponseType':self.kdResponseType,'kdProb':self.kdProb,'errorCode':self.errorCode,'requestType':self.requestType,
        'modelVersion':self.modelVersion, 'errorMsg': self.errorMsg, 'questionId': self.questionId,
        'keywordVersion': self.keywordVersion, 'use_time': self.use_time, 'reqLabel': self.reqLabel,
        'reqKeyword': self.reqKeyword, 'reqKdAnswer': self.reqKdAnswer, 'reqKdKeyword': self.reqKdKeyword}
        return utils.to_json(result)

    def __str__(self):
        x = 'requestId=<%s> label=<%s> prob=<%.3f> keyword=<%s> rank=<%d> textLen=<%d> text=<%s> asrText=<%s> ' \
            'contextId=<%s> serviceCode=<%s> responseType=<%d> reqType=<%s> status=<%s> describe=<%s> ' \
            'startStamp=<%d> kdKeyword=<%s> kdAnswer=<%s> ifFaq=<%s> kdResponseType=<%d> kdProb=<%.3f> ' \
            'errorCode=<%d> requestType=<%d> modelVersion=<%s> errorMsg=<%s> questionId=<%s> keywordVersion=<%s> ' \
            'use_time=<%d> reqLabel=<%s> reqKeyword=<%s> reqKdAnswer=<%s> reqKdKeyword=<%s>' % (self.requestId,
                self.label, self.prob, self.keyword, self.rank, self.textLen, self.text, self.asrText, self.contextId,
                self.serviceCode, self.responseType, self.reqType, self.status, self.describe, self.startStamp,
                self.kdKeyword, self.kdAnswer, self.ifFaq, self.kdResponseType, self.kdProb, self.errorCode,
                self.requestType, self.modelVersion, self.errorMsg, self.questionId, self.keywordVersion, self.use_time,
                self.reqLabel, self.reqKeyword, self.reqKdAnswer, self.reqKdKeyword)
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

    @staticmethod
    def nlp_bot(nlp_type, message=None, version_json=None):
        if message is None and version_json is None:
            return {'errorCode': 1, 'errorMsg': '参数message错误'}
        elif message is None and isinstance(version_json, dict):
            message = version_json.get('message', None)
        if not isinstance(message, str):
            return {'errorCode': 1, 'errorMsg': '参数message错误'}
        if len(message) < 1:
            return {'errorMsg': '成功'}
        if nlp_type == -1:
            text = utils.fmt_pat_not_cn.sub('', message)
            text = text.strip()
            if not text:
                return {'errorMsg': '成功'}
            textLen = len(text)
            if textLen > 6:
                text = asr_analyzer.analyze(text, long=True)
            else:
                text = asr_analyzer.analyze(text, long=False)
            return {'textLen': textLen, 'asrText': text, 'errorMsg': '成功'}
        elif nlp_type == 0 and isinstance(version_json, dict):
            if 'modelVersion' not in version_json and 'keywordVersion' not in version_json and 'modelId' not in version_json:
                return {'errorCode': 1, 'errorMsg': '参数modelVersion、keywordVersion和modelId错误'}

            modelId = version_json.get('modelId', None)
            keywordVersion = version_json.get('keywordVersion', None)
            modelVersion = version_json.get('modelVersion', None)
            requestType = version_json.get('requestType', None)
            reqLabel = version_json.get('label', None)
            reqKeyword = version_json.get('keyword', None)

            rule_result, model_result = None, None
            textLen = len(message)
            label_list = config.cnn_config[version_json['modelVersion']]['label_list']
            label_index = [idx + 1 for idx in range(len(label_list) - 1)]

            if reqLabel is None or reqLabel == "":
                label = None
            else:
                label = reqLabel

            if reqKeyword is None or reqKeyword == "":
                reqKeyword = None
            else:
                pass
            
            if label is None:
                label_num = label
            else:
                label_num = int(list(label)[-1])

            if modelId is None or modelId == "":
                modelId = None
            else:
                pass
            logging.info("reqLabel: {}, reqKeyword: {}, modelId: {}, label_num: {}, label_index: {}".format(
                label, reqKeyword, modelId, label_num, label_index))

            if label is not None and reqKeyword is not None and modelId is not None:
                logging.info('Keyword result taken {}: new scene'.format(label))
                return {'label': version_json["label"], 'keyword': version_json["keyword"],
                        'rank': -1, 'prob': -1, 'errorCode': 0,
                        'responseType': 4, 'errorMsg': '成功'}
            elif modelId is None and reqKeyword is not None and label_num not in label_index and label_num != 8:
                logging.info('Keyword result taken {}: old scene but out of index'.format(label))
                return {'label': version_json["label"], 'keyword': version_json["keyword"],
                        'rank': -1, 'prob': -1, 'errorCode': 0,
                        'responseType': 4, 'errorMsg': '成功'}
            else:
                if requestType in [0, 1]:
                    if keywordVersion is None or keywordVersion == "":
                        return {'errorCode': 1, 'errorMsg': '参数requestType与keywordVersion错误'}
                    if textLen > 6:
                        tmp_result = pattern_analyzer.parse(message, keywordVersion=keywordVersion, long=True)
                    else:
                        tmp_result = pattern_analyzer.parse(message, keywordVersion=keywordVersion, long=False)
                    rule_result = tmp_result
                    logging.info('rule Response %s' % rule_result)

                if requestType in [1, 3]:
                    if modelVersion is None or modelVersion == "":
                        return {'errorCode': 1, 'errorMsg': '参数requstType与ModelVersion错误'}
                    # 提取中文，去停用词
                    text = "".join(utils.getChineseChar(message))
                    # 文本过短，不用模型预测label
                    if 3 == requestType:
                        model_result = predictor_intention_cnn.analyze(text, modelVersion=modelVersion)
                    elif len(text) <= 2 or 0 == requestType:
                        model_result = {
                            'label': '',  # 意图
                            'prob': 0.0,  # 置信度
                            'responseType': 1  # 1表示模型输出的结果
                        }
                    else:
                        model_result = predictor_intention_cnn.analyze(text, modelVersion=modelVersion)

                    logging.info('model response:%s' % model_result)
                fuse_result = utils.fuseModel(rule_result, model_result, textLen, requestType,
                                              keywordVersion=keywordVersion, modelVersion=modelVersion)

                return {'label': fuse_result["label"], 'keyword': fuse_result["word"], 'rank': fuse_result["rank"],
                        'prob': fuse_result["prob"], 'errorCode': fuse_result["errorCode"],
                        'responseType': fuse_result["responseType"], 'errorMsg': '成功'}
        elif nlp_type == 1:
            questionId = version_json.get('questionId', None)
            if questionId is None or questionId == "":
                return {'errorCode': 1, 'errorMsg': '参数questionId错误'}
            kd_cond = version_json.get('kd_limit', None)
            kd, kd_keyword, score = faq_pattern.analyze(utils.fmt_pat_kd.sub('', message), questionId=questionId,
                                                        context_id=version_json.get('status', ''),
                                                        sys_answer=version_json.get('kdAnswer', None),
                                                        kd_limit=kd_cond)
            if_faq = 1 * (score >= 0)
            return {'kdAnswer': kd, 'if_faq': if_faq, 'kdProb': round(score, 3), 'errorMsg': '成功'}
        return {'errorCode': 1, 'errorMsg': '待确定错误'}

    @staticmethod
    def param_format(x, v_default=''):
        if x is None:
            return v_default
        return str(x)

    @utils.debugger(prefix='analyze_chat')
    def analyze(self, data):
        time_use = []
        base_info = ["requestId", "contextId"]
        time_use.append(time.time())
        res = dict()
        data = utils.str2json(data)
        if not isinstance(data, dict):
            logging.info('请输入正确格式请求')
            res.update({'errorCode': 1, 'errorMsg': '请求数据格式错误'})
            result = Interface(res_json=res)
            return result()
        for c in base_info:
            if c in data:
                res.update({c: data.get(c)})
        message = data.get('message', None)
        if message is None:
            res.update({'errorCode': 1, 'errorMsg': '缺少有效的message文本信息'})
            result = Interface(res_json=res)
            return result()
        serviceCode = self.param_format(data.get('serviceCode', ''), v_default='')
        status = self.param_format(data.get('status', ''), v_default='')
        modelId = self.param_format(data.get('modelId', ''), v_default='')
        res.update({'serviceCode': serviceCode, 'status': status})
        data.update({'status': status})
        time_use.append(time.time())
        try_cnt = 0
        su = 0
        while su <= 0 and try_cnt <= 3:
            try:
                query_data = myquery()
                su += 1
            except Exception as e:
                logging.warning("{}th get data from mysql failed {}".format(try_cnt, e))
                try_cnt += 1
        if su <= 0:
            res.update({'errorCode': 1, 'errorMsg': 'mysql连接失败,无法获取数据'})
            result = Interface(res_json=res)
            return result()
        # else:
        #     logging.info("kd of our system config: {}".format(query_data["kd"]))
        time_use.append(time.time())
        if modelId != '' and "__" not in modelId and modelId != "null":
            df_m = query_data["ApiNlpModel"]
            params_list = df_m[df_m.index.isin(modelId.split(','))]["params"].tolist()
            if len(params_list) < 1:
                logging.warning("ModelId input error: {}".format(modelId))
                res.update({'errorCode': 1, 'errorMsg': '通用模型id错误'})
                result = Interface(res_json=res)
                return result()
            else:
                logging.info("load formal api model...")
                for p in params_list:
                    data.update({'modelId': modelId})
                    data.update(json.loads(p))
        elif serviceCode == "" or serviceCode == "null":
            res.update({'errorCode': 1, 'errorMsg': 'serviceCode输入无效'})
            result = Interface(res_json=res)
            return result()
        else:
            df_1, df_2 = query_data["ScNlpModel"], query_data["StatusNlpModel"]
            try:
                params_list = [df_1.loc[serviceCode, "params"]]
            except:
                log_df1 = df_1.sort_values("updateTime").iloc[-1].tolist()
                logging.warning("ScNlpModel table query error with last record: {}".format(log_df1))
                params_list = []
            kj = "{}_{}".format(serviceCode, status)
            try:
                params_list.append(df_2.loc[kj, "params"])
            except:
                log_df2 = df_2.sort_values("updateTime").iloc[-1].tolist()
                logging.warning("StatusNlpModel table query error with last record: {}".format(log_df2))
            if len(params_list) < 1:
                res.update({'errorCode': 1, 'errorMsg': '测试模型配置表查询结果为空'})
                result = Interface(res_json=res)
                return result()
            else:
                logging.info("load nlp test model...")
                for p in params_list:
                    data.update(json.loads(p))
        time_use.append(time.time())
        tmp_param = [-1]
        if 'questionId' in data:
            try:
                data["kd_limit"] = query_data["kd"].loc[serviceCode]
            except:
                logging.warning("Failed to get kd limit of {}".format(serviceCode))
            tmp_param += [1]
        if 'nlpType' in data:
            data['requestType'] = data['nlpType']
            tmp_param += [0]
        res.update({'text': message, 'reqType': ",".join([str(p) for p in tmp_param])})
        if 'reqKdAnswer' in data:
            res.update({'kdAnswer': data['reqKdAnswer']})
        elif 'kdAnswer' in data:
            res.update({'reqKdAnswer': data['kdAnswer']})
            if 'kdKeyword' in data:
                res.update({'reqKdKeyword': data['kdKeyword']})
        params_req = ['requestId', 'contextId', 'describe', 'startStamp', 'questionId', 'requestType', 'keywordVersion',
                      'modelVersion', 'kdAnswer', 'label', 'keyword', 'reqKdAnswer', 'reqKdKeyword']
        res.update({p: data[p] for p in params_req if p in data})
        time_use.append(time.time())
        for req_id in tmp_param:
            update_res = self.nlp_bot(nlp_type=req_id, message=res.get('asrText', None), version_json=data)
            if res.get('errorMsg', '') not in ['成功', '']:
                update_res['errorMsg'] = res.get('errorMsg', '') + " && " + update_res['errorMsg']
            res.update(update_res)
        time_use.append(time.time())
        if time_use[-1] - time_use[0] >= 0.25:
            logging.error("!!!timeout use, please check...")
            logging.info("use time: {}".format([time_use[j] - time_use[j - 1] for j in range(1, len(time_use))]))
        res.update({"time": round(time_use[-1] - time_use[0], 5)})
        result = Interface(res_json=res)
        return result()

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
