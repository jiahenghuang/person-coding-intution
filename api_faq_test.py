# -*- coding: utf-8 -*-

import json
import asyncio
import websockets
import pandas as pd

# WS_URL = "ws://127.0.0.1:8703"
WS_URL = "ws://10.51.6.13:8090"

df = pd.read_csv("bin/testdata/kd_test_data_V1.0", sep="\t", encoding="utf-8")
print("test data num: ", len(df))


async def api_test(text_seq, kd_param, status_seq=None):
    async with websockets.connect(WS_URL) as ws:
        result = []
        if status_seq is None:
            status_seq = ["" for _ in text_seq]
        for j in range(len(text_seq)):
            send_info = {"message": text_seq[j],
                         "requestId": "123456",
                         "contextId": "haha",
                         "serviceCode": "3000008",
                         "reqType": "-1,1",
                         "status": status_seq[j],
                         "describe": "asdfasd",
                         "startStamp": 1,
                         'questionId': kd_param}
            await ws.send(json.dumps(send_info))
            res_str = await ws.recv()
            res_dict = json.loads(res_str)
            try:
                result += [(res_dict['kdAnswer'], res_dict["kdKeyword"], res_dict["kdProb"], res_dict['errorMsg'],
                            res_dict['use_time'])]
            except Exception as e:
                print("err: ", e)
                print(res_dict)
                return None
            # print("offline {} online {}".format(df.loc[j, "y_model"], greeting["kdKeyword"]))
    return result


res = asyncio.get_event_loop().run_until_complete(api_test(text_seq=df["keyword"].tolist(),
                                                           status_seq=df["status"].tolist(),
                                                           kd_param="KD3-FAQ"))
# res = asyncio.get_event_loop().run_until_complete(api_test(text_seq=["微软亚洲研究院首席研发经理邹欣在工作之余，出版了几本书，其中《编程之美》、《构建之法》在程序员界颇具名气。他还是在微博社交网络平台拥有30余万粉丝的大V"]*10000,
#                                                            status_seq=["A"]*10000,
#                                                            kd_param="KD3-FAQ"))
df["y_api"] = ["ook" if len(x[0]) < 1 else x[0] for x in res]
df["info"] = [x[-2] for x in res]
print(df["info"].value_counts())
ts_list = [x[-1] for x in res if x[-1] > 0]
print("use time cnt {} avg {} ge0.25 rate {}".format(len(ts_list), sum(ts_list)/len(ts_list), sum(x >= 0.25 for x in ts_list)/len(ts_list)))
print("api -->", WS_URL)
print("query conflict rate: ", (df["y_api"] != df["y_model"]).mean())

# @asyncio.coroutine
# def api_test(text_seq, kd_param="KD2-FAQ"):
#     ws = yield from websockets.connect(WS_URL)
#     result = []
#     for text in text_seq:
#         send_info = {"message": text,
#                      "requestId": "123456",
#                      "contextId": "haha",
#                      "serviceCode": "hehe",
#                      "reqType": 1,
#                          "status": "678",
#                          "describe": "asdfasd",
#                          "startStamp": 1,
#                          'questionId': kd_param,
#                          'keywordVersion': "V20190424B1"}
#         yield from ws.send(json.dumps(send_info))
#         res_str = yield from ws.recv()
#         res_dict = json.loads(res_str)
#         try:
#             result += [(res_dict['kdAnswer'], res_dict["kdKeyword"], res_dict["ifFaq"])]
#         except Exception as e:
#             print("err: ", e)
#             print(res_dict)
#             return None
#         # print("offline {} online {}".format(df.loc[j, "y_model"], greeting["kdKeyword"]))
#     return result
