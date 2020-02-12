# coding:utf-8
import os
import re
import utils
from analyze_asr import asr_analyzer
import time
import sys

from structs.match_obj import MatchObj
from structs.kv_obj import KVObj
from tools.util_trie import Trie
import config

logger = config.logger


class PatternAnalyzer(object):
    '''
    解析正则表达式模板和字典库
    '''

    def __init__(self):
        # 从正则表达式模板中读取正则表达式和字典
        self._parse_all_files()

        # kvpats时解析出来正则表达式，kvdicts是解析出来的关键词，kv_contain是匹配不匹配的正则表达式
        self.parsed_pats = {}
        for key, pats in self.all_pats.items():
            long_pats = pats[0]
            short_pats = pats[1]

            contain_vpats = long_pats[0]
            vpats = long_pats[1]
            dict_pats = long_pats[2]
            kv_contain_long, kvpats_long, kvdicts_long = self._make_vpats(contain_vpats, vpats, dict_pats, long=True)

            contain_vpats = short_pats[0]
            vpats = short_pats[1]
            dict_pats = short_pats[2]
            kv_contain_short, kvpats_short, kvdicts_short = self._make_vpats(contain_vpats, vpats, dict_pats,
                                                                             long=False)
            # 编译长度大于6和小于6的正则表达式和trie树
            self.parsed_pats[key] = [(kv_contain_long, kvpats_long, kvdicts_long),
                                     (kv_contain_short, kvpats_short, kvdicts_short)]

        for key, value in self.parsed_pats.items():
            long = value[0]
            short = value[1]
            self._build(kv_contain=long[0], kvpats=long[1], kvdicts=long[2])
            self._build(kv_contain=short[0], kvpats=short[1], kvdicts=short[2])
            self.parsed_pats[key] = (self._rank_group(kvpats=long[1], kvdicts=long[2], kv_contain=long[0]),
                                     self._rank_group(kvpats=short[1], kvdicts=short[2], kv_contain=short[0]))

    def _parse_all_files(self):
        '''
        解析所有的pat模板
        '''
        self.all_pats = {}
        for key, fnames in config.pat_fnames.items():
            pat_fname_less = fnames[0]
            pat_fname_more = fnames[1]

            contain_vpats_long, vpats_long, dict_pats_long = self._parse_from_file(pat_fname_more)
            contain_vpats_short, vpats_short, dict_pats_short = self._parse_from_file(pat_fname_less)
            self.all_pats[key] = [(contain_vpats_long, vpats_long, dict_pats_long),
                                  (contain_vpats_short, vpats_short, dict_pats_short)]

    def _parse_from_file(self, pat_fname):
        '''
        用于正则表达式的获取
        '''
        if not os.path.exists(pat_fname):
            logger.fatal('value pattern file <%s> does not exist' % (pat_fname))
            sys.exit(-1)

        with open(pat_fname) as fr:
            pats = fr.readlines()

        pats = [pat.strip() for pat in pats]
        contain_vpats = [pat for pat in pats if pat.startswith('$') and 'contain' in pat]  # 包含contain的正则表达式
        vpats = [pat for pat in pats if
                 pat.startswith('$') and 'str' not in pat and 'contain' not in pat]  # 正则表达式非contain的
        dict_pats = [pat for pat in pats if pat.startswith('$') and 'str' in pat]
        return contain_vpats, vpats, dict_pats

    def _make_vpats(self, contain_vpats, vpats, dict_pats, long=True):
        '''
        用于模板中字典的获取
        '''
        kvpats, kvdicts, kv_contain = [], [], []

        # 解析包含不包含
        group, rank, type_name = '', '', ''
        for contain_pat in contain_vpats:
            contain_pat = contain_pat.split('=')
            kname = contain_pat[0]
            pats = contain_pat[1].split(',')

            info = kname.split('_')
            group = info[0].replace('$', '')
            rank = int(info[-1].replace('rank', ''))

            type_name = 're_expr'

            for pat in pats:
                norm_pat, wrong_pat = self._match_not_match(pat)
                if norm_pat and wrong_pat:
                    contain_pat = KVObj(group=group, rank=rank, type_name=type_name, vpat=(norm_pat, wrong_pat))
                    kv_contain.append(contain_pat)

        # 解析正则表达式
        vpats_num = len(vpats)
        group, rank, type_name = '', '', ''
        for i, vpat in enumerate(vpats):
            vpat = vpat.split('=')
            kname = vpat[0]
            pats = '='.join(vpat[1:]).split(',')  # yes_rank1

            info = kname.split('_')
            if info[0].replace('$', '') != group:
                if group:
                    kv_pat = KVObj(group=group, rank=rank, type_name=type_name, vpat=inner_pats)
                    kvpats.append(kv_pat)
                group = info[0].replace('$', '')
                rank = int(info[-1].replace('rank', ''))
                type_name = 'regu_expr'
                inner_pats = pats
            else:
                inner_pats.extend(pats)
            if i == vpats_num - 1:
                kv_pat = KVObj(group=group, rank=rank, type_name=type_name, vpat=inner_pats)
                kvpats.append(kv_pat)

        # 解析关键词词典
        group, rank, type_name = '', '', ''
        for vdict in dict_pats:
            vdict = vdict.split('=')
            kname = vdict[0]
            words = vdict[1].replace('[', '').replace(']', '')
            words = words.split(',')

            info = kname.split('_')
            group = info[0].replace('$', '')
            rank = int(info[-1].replace('rank', ''))
            type_name = 'dict'
            kv_dict = KVObj(group=group, rank=rank, type_name=type_name, vpat=words)
            kvdicts.append(kv_dict)
        kvdicts = self._combine_pats(kvdicts)
        return kv_contain, kvpats, kvdicts

    def _combine_pats(self, kvdicts):
        '''
        合并kvdicts，当group、rank都一致的时候
        '''
        kvdicts_tmp, new_kvdicts = {}, []
        group, rank = '', ''

        for kvdict in kvdicts:
            group = kvdict.group
            rank = kvdict.rank
            if (group, rank) not in kvdicts_tmp:
                kvdicts_tmp[(group, rank)] = [kvdict]
            else:
                kvdicts_tmp[(group, rank)].append(kvdict)

        for kvdict in kvdicts_tmp.values():
            tmp = kvdict[0]
            for item in kvdict[1:]:
                tmp.vpat.extend(item.vpat)
            new_kvdicts.append(tmp)
        return new_kvdicts

    # @utils.debugger(prefix='PatternParser')
    def _build(self, kvpats=None, kv_contain=None, kvdicts=None):
        '''
        编译正则表达式，将字典trie树化
        '''
        for mat in kvpats:
            all_pats = []
            for pat in mat.vpat:
                # if pat == '关你=事':
                #     import pdb;pdb.set_trace()
                all_pats.extend(self._parse_pat(pat))
            # 对正则表达式按照长度排序
            all_pats = sorted(all_pats, key=lambda x: len(utils.fmt_pat_not_cn.sub('', x)), reverse=True)
            mat.vpat = [(re.compile(pat), pat) for pat in all_pats]

        all_pats = []
        for mat in kv_contain:
            pat = mat.vpat
            norm_pat = pat[0]
            wrong_pat = pat[1]
            norm_pat = self._parse_pat(norm_pat)
            wrong_pat = self._parse_pat(wrong_pat)
            all_pats.append((re.compile(norm_pat[0]), re.compile(wrong_pat[0])))
            mat.vpat = all_pats

        for mat in kvdicts:
            # 对关键词按照长度进行排序
            mat.vpat = Trie(mat.vpat)

    def _rank_group(self, kvpats=None, kvdicts=None, kv_contain=None):
        '''
        根据rank将kvpats和kvdicts分组
        '''
        sorted_group = []  # 排序后的group
        pat_top_rank = max([mat.rank for mat in kvpats])
        dict_top_rank = max([mat.rank for mat in kvdicts])
        if kv_contain:
            contain_top_rank = max([mat.rank for mat in kv_contain])
        else:
            contain_top_rank = 0

        top_rank = max(pat_top_rank, contain_top_rank, dict_top_rank)
        for i in range(1, top_rank + 1):
            tmp_group = []
            kv_contain = list(filter(lambda mat: mat.rank == i, kv_contain))
            kv_pats = list(filter(lambda mat: mat.rank == i, kvpats))
            kv_dict = list(filter(lambda mat: mat.rank == i, kvdicts))

            tmp_group.append(kv_contain)
            tmp_group.append(kv_pats)
            tmp_group.append(kv_dict)
            sorted_group.append(tmp_group)
        return sorted_group

    def _bigram_combine(self, list1, list2):
        '''
        将list1和list2中所有元素相互组合
        '''
        new_list = []
        for i in list1:
            for j in list2:
                element = i + j
                new_list.append(element)
        return new_list

    def _match_not_match(self, pat):
        '''
        实现匹配不匹配功能,目前仅仅实现只出现一次<>的模板
        '''
        if '<' in pat and '>' in pat:
            start = pat.find('<')
            end = pat.find('>')
            norm_pat = pat[:start]
            wrong_pat = pat[start + 1:end]
            return norm_pat, wrong_pat
        return None, None

    def _parse_pat(self, pat):
        '''
        转化为正则表达式

        [_]代表0-1个字
        [=]代表0-4个字
        [&]代表0-6个字
        [%]代表0-999个字
        <= 6的匹配一个规则，>6的匹配一个规则
        [<>]代表前面匹配后面不匹配
        '''
        # pat = pat.replace('__', '.{0,2}')
        pat = pat.replace('_', '.{0,1}')
        pat = pat.replace('=', '.{0,4}')
        pat = pat.replace('&', '.{0,6}')
        pat = pat.replace('%', '.{0,999}')
        # pat = pat.replace('#', '^')
        ##(|)组合的解析
        sub_string = []
        if '(' in pat:
            flag = True
            line = pat
            while flag:
                start = line.find('(')
                string = line[:start]
                if string:
                    sub_string.append(string)
                end = line.find(')')
                sub_string.append(line[start + 1:end])
                line = line[end + 1:]
                if line.find('(') == -1:
                    if line:
                        sub_string.append(line)
                    flag = False
            if sub_string:
                pat = [string.split('|') for string in sub_string]

            for i, string in enumerate(pat):
                if i > 0:
                    new_list = self._bigram_combine(pat[i - 1], string)
                    pat[i] = new_list
            pat = new_list
        # ##a匹配b不匹配
        pat = [pat] if isinstance(pat, str) else pat
        return pat

    @utils.debugger(prefix='PatternParserByRule')
    def parse(self, line, keywordVersion='', long=True):
        '''
        返回匹配到的结果
        '''
        match_mat, re_mat, dict_mat = None, None, None
        if long:
            if keywordVersion:
                sorted_group = self.parsed_pats[keywordVersion][0]
            else:
                logger.fatal('<%s> and <%s> does not exist' % (questionId, keywordVersion))
                sys.exit(-1)
        else:
            if keywordVersion:
                sorted_group = self.parsed_pats[keywordVersion][1]
            else:
                logger.fatal('<%s> and <%s> does not exist' % (questionId, keywordVersion))
                sys.exit(-1)

        for rank_group in sorted_group:
            match_group = rank_group[0]  # 包含不包含group
            re_group = rank_group[1]  # 每个group中有两种匹配方法，正则表达式group
            dict_group = rank_group[2]  # trie group

            # 先匹配，匹配不匹配。<>内是不匹配的内容，<>外时匹配的内容
            for mat in match_group:
                pats = mat.vpat
                for pat in pats:
                    match_pat = pat[0]
                    not_match_pat = pat[1]
                    res_match = match_pat.search(line)
                    res_not_match = not_match_pat.search(line)

                    if res_match and not res_not_match:
                        pos = res_match.span()
                        if pos:
                            start_pos = pos[0]
                            end_pos = pos[1]
                            ratio = 1.0 * (end_pos - start_pos) / len(line)
                            word = res_match.group()
                            match_mat = KVObj(group=mat.group, rank=mat.rank, type_name=mat.type, word=word,
                                              start_pos=start_pos, end_pos=end_pos, ratio=ratio)
                            break
                    if match_mat:
                        break

            # 第二步，匹配正则表达式中内容
            for mat in re_group:
                pats = mat.vpat
                for pat in pats:
                    compiled_pat, origin_pat = pat[0], pat[1]
                    res_mat = compiled_pat.search(line)
                    if res_mat:
                        pos = res_mat.span()
                        if pos:
                            start_pos = pos[0]
                            end_pos = pos[1]
                            ratio = 1.0 * (end_pos - start_pos) / len(line)
                            re_mat = KVObj(group=mat.group, rank=mat.rank, type_name=mat.type,
                                           word=origin_pat, start_pos=start_pos, end_pos=end_pos, ratio=ratio)
                            break
                if re_mat:
                    break

            # 第三步，匹配字典中出现内容
            for mat in dict_group:
                trie = mat.vpat
                dict_mat = trie.search_longest_mats(line)
                if dict_mat:
                    dict_mat.group = mat.group
                    dict_mat.rank = mat.rank
                    dict_mat.type = mat.type
                    break

            if re_mat and dict_mat:
                if re_mat.ratio > dict_mat.ratio:
                    return re_mat
                else:
                    return dict_mat
            if re_mat:
                return re_mat
            if dict_mat:
                return dict_mat
        return None


pattern_analyzer = PatternAnalyzer()
if __name__ == '__main__':
    # line = '不单身注册干什么,哈哈哈哈'
    # line = '肯定是弄错了'
    # line = '我是随便注册的'
    # line='我是替我弟弟'
    # obj = pattern_parser.parse(line)
    # line='还没对象'
    # import json
    # with open('data/corpus.txt') as fr:
    #     data = fr.readlines()
    # results = []
    # for line in data:
    #     send_info = {"requestId":"123456","message":"%s" % line,"contextId":"haha","serviceCode":"hehe","reqType":1,"status":"678","describe":"asdfasd","startStamp":1}
    #     result = pattern_analyzer.analyze(json.dumps(send_info))
    #     print(result)
    #     result = json.loads(result)
    #     print(result.get('label'))
    # import json
    # data = {"contextId":"1rVfd19012419043510232824","message":"受的","reqType":0,
    # "requestId":"1rVfd190124190435102328241548327889175","serviceCode":"10000","startStamp":1548327889175,"status":"A"}
    # result = pattern_analyzer.analyze(json.dumps(data))
    # print(result)
    # print(type(result))
    import pandas as pd
    data = pd.read_csv('testdata/A话术是否单身问题测试数据.csv')
    texts = data['keyword'].values
    for text in texts:
        textLen = len(text)
        if textLen > 6:
            flag = True
        else:
            flag = False
        print(text)
        text = asr_analyzer.analyze(text, long=flag)
        print(text)
        result = pattern_analyzer.parse(text, long=flag, keywordVersion='V20190402')
        print(result)
        print('----------')