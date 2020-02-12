#coding:utf-8
# 'http://www.blue-zero.com/WebSocket/'
import websocket
import time
import threading
import json
import multiprocessing
from threadpool import ThreadPool, makeRequests
import random

#修改成自己的websocket地址
# WS_URL = "ws://10.1.29.167:8080"
WS_URL = "ws://127.0.0.1:8703"
# WS_URL = "ws://10.51.6.13:8080"
#定义进程数
processes=5
#定义线程数（每个文件可能限制1024个，可以修改fs.file等参数）
thread_num=100

def load_data():
    with open('bin/data/corpus.txt', "r", encoding="utf-8") as fr:
        corpus = fr.readlines()
    return [line.strip() for line in corpus]

corpus = load_data()
n = len(corpus)

def on_message(ws, message):
    pass
    # print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def random_select():
    index=random.randint(0,n)
    return corpus[index]

def on_open(ws):
    def send_trhead():
        #设置你websocket的内容
        send_info = {"message": "%s" % random_select(),
                     "contextId": "haha",
                     "serviceCode": "hehe",
                     "reqType": 1,
                     "status": "678",
                     "describe": "asdfasd",
                     "startStamp": 1}
        while True:
            ws.send(json.dumps(send_info))
    t = threading.Thread(target=send_trhead)
    t.start()

def on_start(num):
    # time.sleep(num%20)
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(WS_URL, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

def thread_web_socket():
    #线程池
    pool = ThreadPool(thread_num)
    num = list()
    #设置开启线程的数量
    for ir in range(thread_num):
        num.append(ir)
    requests = makeRequests(on_start, num)
    [pool.putRequest(req) for req in requests]
    pool.wait()


if __name__ == "__main__":
    #进程池
    pool = multiprocessing.Pool(processes=processes)
    #设置开启进程的数量
    for i in range(processes):
        pool.apply_async(thread_web_socket)
    pool.close()
pool.join()
