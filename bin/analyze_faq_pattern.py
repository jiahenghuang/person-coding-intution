# coding:utf-8
import config
import re
import utils
from config import logger
from analyze_faq import faq_model_v0, faq_model_v1


with open(config.file_query_rules, 'r', encoding='utf-8') as fr:
    questions = fr.readlines()
questions = [line.strip().replace('XX', '.*') for line in questions]
re_questions = [re.compile(pat) for pat in questions]


def recognize(text):
    for pat in re_questions:
        if pat.search(text):
            return True
    return False


def qid_int(x):
    try:
        return int(x.replace("Tkd", ""))
    except:
        return 0


def faq_output(line, context_id='', sys_res=None, kd_ver=None, easy=True, recall_n=5):
    if kd_ver is not None:
        assert kd_ver in config.kd_version
    faq_model = faq_model_v1
    if kd_ver == "kd1":
        faq_model = faq_model_v0
    faq_max = max(qid_int(x) for x in faq_model.res_set)
    _, faq_res = faq_model.rank_res(line)
    if len(faq_res) < 1:
        logger.info("Res FAQ return none.")
        return '', 0
    max_prob = faq_res[0]['rank_score']
    if easy:
        return '' if max_prob <= 0.3 else faq_res[0]['qid'], max_prob
    # 融合策略
    if sys_res is not None and qid_int(sys_res) > faq_max:
        logger.info("Res question not in your FAQ Data.")
        return sys_res, -1
    qid_v = config.kd_version.get(kd_ver)
    logger.info("...Last kd of model: {}".format(qid_v))
    if max_prob <= 0.3:
        logger.info("Res return prob le 0.3")
        return '', max_prob
    elif max_prob <= 0.6:
        logger.info("Res return prob 0.3~0.6")
        return sys_res, max_prob
    logger.info("Res check faq return and return")
    j = 0
    if isinstance(context_id, str) and 'A' in context_id:
        rs = faq_res[0]
        for r in faq_res:
            j += 1
            if j <= recall_n and r['qid'] in config.first_kd and qid_int(r['qid']) <= qid_v:
                rs = r
                break
        return rs['qid'], rs['rank_score']
    else:
        for r in faq_res:
            j += 1
            if j <= recall_n and qid_int(r['qid']) <= qid_v:
                return r['qid'], r['rank_score']
    return sys_res if sys_res is not None else '', -1


class FaqPattern(object):
    def __init__(self):
        self._parse_from_file_kd()

    def _build_re_kd(self, pat):
        if isinstance(pat, str):
            pats = pat.split(',')
        else:
            pats = pat
        pats = [re.compile(inner_pat.replace('%', '.{0,100}').replace('#', '\\b').replace('_', '.{0,1}').
                           replace('=', '.{0,4}').replace('&', '.{0,6}')) for inner_pat in pats]
        return pats

    def _parse_from_file_kd(self):
        self.pats_kd_dict = dict()
        for f in config.kd_pat_files:
            kd_version = f.split('/')[-1].replace('.pattern', '')
            self.pats_kd_dict[kd_version] = []
            logger.info("parse kd re file{}".format(f))
            with open(f, "r", encoding="utf-8") as fp:
                for d in fp.readlines():
                    d = d.strip().replace(" ", "").split('-->')
                    kd, pat = d[0], d[1]
                    pat_origin = sorted(pat.split(','), key=lambda s: len(s), reverse=True)
                    pat_reg = self._build_re_kd(pat_origin)
                    self.pats_kd_dict[kd_version].append([pat_reg, pat_origin, kd])

    @staticmethod
    def re_match(str_, pats):
        for pat, reg, kd in pats:
            for i in range(len(pat)):
                result = pat[i].search(str_)
                if result:
                    return kd, reg[i]
        return None

    @utils.debugger(prefix="FaqPattern")
    def analyze(self, line, questionId='', context_id='A', sys_answer=None, sys_keyword=None, kd_limit=None):
        if isinstance(questionId, str):
            questionId = questionId.lower()
        if kd_limit:
            kd_limit = [q.lower() for q in kd_limit]
        logger.info("analyse {} with model [{}]...".format(line, questionId))
        if '-' not in questionId and questionId in self.pats_kd_dict:
            kd_ver = questionId.lower()
            if kd_ver in self.pats_kd_dict:
                r = self.re_match(line, self.pats_kd_dict[kd_ver])
                if r is not None:
                    return r[0], r[1], -1
                logger.info("no keyword match...")

        elif re.match("^kd[0-9]-faq\_?$", questionId):
            cond, mt = questionId.split("-")
            qid, keyword = sys_answer, sys_keyword
            if sys_answer is None:
                qid, keyword, _ = self.analyze(line, questionId=cond)
                logger.info("kd re local parse return: {} {}".format(qid, keyword))
            else:
                logger.info("kd re system parse return: {}".format(qid))
            if qid != '' and qid is not None:
                is_easy = (mt[-1] == '_')
                faq_r = faq_output(line, context_id=context_id, sys_res=qid, kd_ver=cond, easy=is_easy)
                if kd_limit:
                    if faq_r[0].lower() not in kd_limit:
                        logger.info("model res {} not in sys kd".format(faq_r[0]))
                        return qid, keyword, -1
                return faq_r[0], keyword, faq_r[1]

        elif re.match("^kd[0-9]-if_query-faq$", questionId):
            qid, _, _ = self.analyze(line, questionId=questionId.replace("-if_query-faq", ""))
            if qid == '' or qid is None:
                if recognize(line):
                    faq_r = faq_output(line, easy=True)
                    return faq_r[0], '', faq_r[1]

        elif questionId == 'if_query-faq':
            if recognize(line):
                faq_r = faq_output(line, easy=True)
                return faq_r[0], '', faq_r[1]

        elif questionId == "faq":
            faq_r = faq_output(line, easy=True)
            return faq_r[0], '', faq_r[1]

        else:
            logger.info("! ! !{} maybe no module match".format(questionId))
        return '', '', -1


faq_pattern = FaqPattern()
if __name__ == '__main__':
    # line = '呃，对要不您加我微信，可以吗？'
    # line = '我的对象在哪里？'
    # line = '我想找一个靠谱的对象，你们是什么公司，有没有线下门店'
    # line = '靠谱的啊啊啊啊对象'
    # line = '怎么联系你们'
    # line = '啥是相亲网站'
    # line = '什么公司'
    # print(faq_pattern.analyze(line, questionId='KD2'))
    # print(faq_pattern.analyze(line, questionId='KD2-FAQ'))
    # import pandas as pd
    # df = pd.read_csv("testdata/kd_test_data", sep="\t", encoding="utf-8")
    # res = [faq_pattern.analyze(r, questionId='KD2-FAQ') for r in df["keyword"].tolist()]
    # df["if"] = [x[-1] for x in res]
    # df["y_api"] = ["ook" if len(x[0]) < 1 else x[0] for x in res]
    # df_ = df[df["if"]]
    # print("api -->", "local analyze")
    # # print("total conflict rate: ", (df["y_api"] != df["y_model"]).mean())
    # print("query conflict rate: ", (df_["y_api"] != df_["y_model"]).mean())
    print(faq_pattern.analyze(line="你，你公司在哪里啊", questionId='KD3-FAQ', context_id='A'))
