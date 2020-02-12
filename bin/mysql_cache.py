# -*- coding: utf-8 -*-

import os, time
import pymysql
import socket
import pandas as pd
import netifaces as ni
from config import logger


def get_ip_address():
    try:
        return ni.ifaddresses('eth0')[ni.AF_INET][0]['addr']
    except:
        try:
            return ni.ifaddresses('en0')[ni.AF_INET][0]['addr']
        except Exception as e:
            return e


def sql_connect():
    if env_ip == "10.11.80.90":
        cnn_id = "10.11.4.135"
        user = 'telephone_control'
        passwd = 'V8Gq45#3jE'
        db = "zhenai_call_control"
    else:
        cnn_id = "10.51.6.17"
        user = 'mysql_change'
        passwd = '1U9pNgY8d0'
        db = "zhenai_call_control_inner"
    logger.info("connecting ip {}...".format(cnn_id))
    return pymysql.connect(host=cnn_id, user=user, passwd=passwd, db=db, autocommit=True,
                           use_unicode=True, charset="utf8")


def load_data(sql_cnn):
    data = dict()
    kd_sql = '''select distinct serviceCode, questionId
	            from Kd 
                where serviceCode in (select serviceCode 
                                      from Config 
                                      where `status`=1)'''
    for t in ["ScNlpModel", "StatusNlpModel", "ApiNlpModel"]:
        df = pd.read_sql("select * from {}".format(t), sql_cnn)
        if "status" in df.columns:
            df["ix"] = df.apply(lambda r: str(r["serviceCode"])+"_"+str(r["status"]), axis=1)
        elif "serviceCode" in df.columns:
            df["ix"] = df["serviceCode"]
        else:
            df["ix"] = df["id"]
        df["mid"] = df["ix"]
        data[t] = df.set_index("ix")[["mid", "params", "updateTime"]]
    data["kd"] = pd.read_sql(kd_sql, sql_cnn).groupby("serviceCode").apply(lambda se: se["questionId"].tolist())
    return data


def timedCacheDecorator(func):
    def wrap(*args, **kwargs):

        key = str(args)+str(kwargs)
        global lastTime
        if key not in cache or time.time() - lastTime > cache_time:
            lastTime = time.time()
            cache[key] = func(*args, **kwargs)

        return cache[key]

    return wrap


@timedCacheDecorator
def myquery():
    global cnn
    logger.info("pull data from mysql...")
    try:
        return load_data(cnn)
    except:
        logger.warning("! ! !connect of mysql expire")
        cnn = sql_connect()
        return load_data(cnn)


cur_ip = get_ip_address()
# logger.info("machine ip: {}".format(cur_ip))
try:
    cur_name = socket.gethostname()
    # logger.info("machine name: {}".format(cur_name))
except:
    pass
env_ip = os.getenv('HOST_IP', '-')
# logger.info("HOST_IP: {}".format(env_ip))

cache = {}
lastTime = time.time()
cnn = sql_connect()
cache_time = 10*60
# query_data = myquery()


if __name__ == "__main__":
    cnt = 0
    while cnt < 1000:
        print(cnt, " : ", len(myquery()))
        cnt += 1
        time.sleep(15)
