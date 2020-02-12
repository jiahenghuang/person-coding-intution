本项目实现电话机器人！

#电话聊天机器人思路：
1. 文本进来先做纠错
2. 根据传入参数决定调用 kd 或者意图识别
  - reqType=1: FAQ 模型
  - reqType=0: 意图识别模型
     - 包括关键词识别意图和textcnn分类模型

#文件目录结构

├── bin
│   ├── analyze_asr.py              #是asr语音转文本接口
│   ├── analyze_chat.py             #是主调用的py文件，里面调用了其他接口
│   ├── analyze_city.py             #是抽取城市的接口
│   ├── analyze_faq_pattern.py      #是使用正则表达式模板做问答
│   ├── analyze_faq.py              
│   ├── analyze_intention_cnn.py    #使用cnn做意图分类
│   ├── analyze_intention.py        #使用tfidf+lightgbm做意图分类
│   ├── analyze_pattern.py
│   ├── analyze_perplexity.py       #是计算句子困惑度
│   ├── config.py
│   ├── constants.py
│   ├── data                         #放代码里用到的各种数据
│   │   ├── asr_dict.dict
│   │   ├── city.dict
│   │   ├── city_result.doudi
│   │   ├── city_result.txt
│   │   ├── condition_prob.prob
│   │   ├── corpus.txt
│   │   ├── correct.dict
│   │   ├── dict_tries.trie
│   │   ├── keyword.dict
│   │   ├── keyword.doudi
│   │   ├── newdata
│   │   │   ├── char2index.pkl
│   │   │   ├── index2char.pkl
│   │   │   ├── liming.csv
│   │   │   ├── review
│   │   │   │   ├── 0228
│   │   │   │   │   ├── test.csv
│   │   │   │   │   └── train.csv
│   │   │   │   ├── char2index.pkl
│   │   │   │   ├── index2char.pkl
│   │   │   │   ├── test_long.txt
│   │   │   │   ├── test_short.txt
│   │   │   │   ├── test.txt
│   │   │   │   ├── train_long.txt
│   │   │   │   ├── train_short.txt
│   │   │   │   └── train.txt
│   │   │   ├── stopwords.txt
│   │   │   ├── test_long.txt
│   │   │   ├── test_short2.txt
│   │   │   ├── test_short.txt
│   │   │   ├── test.txt
│   │   │   ├── train_long.txt
│   │   │   ├── train_short2.txt
│   │   │   ├── train_short.txt
│   │   │   ├── train.txt
│   │   │   └── word2index.pkl
│   │   ├── voice_recog.txt
│   │   ├── word2index.pkl
│   │   └── words.dict
│   ├── data_config                     #存放kd文本匹配用到的各种数据
│   │   ├── doc.xls
│   │   ├── jieba_dict
│   │   ├── jieba_stopwords
│   │   └── qa_2_baidu_E_config
│   ├── eval.py
│   ├── __init__.py
│   ├── layer
│   │   ├── __init__.py
│   │   └── textcnn.py 
│   ├── models                          #存放离线模型文件
│   │   ├── lightgbm
│   │   │   ├── lightgbm_model_0.model
│   │   │   ├── lightgbm_model_1.model
│   │   │   ├── lightgbm_model_2.model
│   │   │   ├── lightgbm_model_3.model
│   │   │   ├── lightgbm_model_4.model
│   │   │   ├── tfidftransformer_0.pkl
│   │   │   ├── tfidftransformer_1.pkl
│   │   │   ├── tfidftransformer_2.pkl
│   │   │   ├── tfidftransformer_3.pkl
│   │   │   ├── tfidftransformer_4.pkl
│   │   │   ├── vectorizer_0.pkl
│   │   │   ├── vectorizer_1.pkl
│   │   │   ├── vectorizer_2.pkl
│   │   │   ├── vectorizer_3.pkl
│   │   │   └── vectorizer_4.pkl
│   │   └── textcnn
│   │       ├── checkpoint
│   │       ├── model-1000.data-00000-of-00001
│   │       ├── model-1000.index
│   │       ├── model-1000.meta
│   │       ├── model-1100.data-00000-of-00001
│   │       ├── model-1100.index
│   │       ├── model-1100.meta
│   │       ├── model-1200.data-00000-of-00001
│   │       ├── model-1200.index
│   │       ├── model-1200.meta
│   │       ├── model-1300.data-00000-of-00001
│   │       ├── model-1300.index
│   │       ├── model-1300.meta
│   │       ├── model-1320.data-00000-of-00001
│   │       ├── model-1320.index
│   │       ├── model-1320.meta
│   │       ├── model-1524.data-00000-of-00001
│   │       └── model-1524.meta
│   ├── setup.sh
│   ├── structs
│   │   ├── __init__.py
│   │   ├── knames.py
│   │   ├── kv_obj.py
│   │   └── match_obj.py
│   ├── templates                                    #保存模板信息，如少于6个字的关键词模板和多于6个字的关键词模板，kd.pattern
│   │   ├── ambiguity.pat
│   │   ├── index.html
│   │   ├── __init__.py
│   │   ├── intention_less6.pat
│   │   ├── intention_more6.pat
│   │   └── kd.pattern
│   ├── test.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── trie_func.py
│   │   └── util_trie.py
│   └── utils.py
├── ChatBotManager
├── Dockerfile
├── docker_log.sh
├── Documents
│   ├── 意图识别融合说明文档.md
│   └── 目录说明文档.md
├── README.md
├── requirements.txt
└── test_chat_bot.py             #测试聊天机器人并发和延迟的代码


#docker日志设置上限
#vim /etc/docker/daemon.json

```

{
"registry-mirrors":["http://f613ce8f.m.daocloud.io"],
"log-driver":"json-file",
"log-opts":{"max-size":"500m","max-file":"3"}
}
```

#docker build方法
  - 重启docker服务  sudo service docker restart
  - 关闭docker服务  service docker stop
  - 开启docker服务  service docker start

#build镜像
  - docker build -t chat-bot:v1.x .
  - 进入 ./chat-bot/目录，sudo docker build .

#运行镜像
  - sudo docker run -p 127.0.0.1:8080:8080 chat-bot:v1.x
#开机自启动服务
  - sudo docker run -p 127.0.0.1:8700:8700 --restart=always chat-bot:v1.2
  - sudo docker run -d -p 10.1.71.252:8700:8700 --restart=always chat-bot:v1.2

#停止镜像运行
  - sudo docker stop imageid

#删除所有容器
  - sudo docker rm $(sudo docker ps -a -q)

#NLP服务测试样例
##测试网址
  - http://www.blue-zero.com/WebSocket/
## 应用层封装
1. 参数: serviceCode+status+modelId+message
2. 样例
```
{"status":"A","message":"是的","serviceCode": "3000008","modelId":""}
```

## 老版本接口测试用例
```
{"contextId":"3D5n41907101435079295353","describe":"","kdAnswer":"","keywordVersion":"","message":"是的","questionId":"","reqType":"-1,0,1","requestId":"3D5n419071014350792953531562740511364","requestType":0,"serviceCode":"32","startStamp":1562740511364,"status":"A"}
```
## ASR 纠错
```
{"message":"我不但是", "reqType":"-1"}
```
## 意图识别
### 测试单关键词结果
```
{"requestId":"123456","message":"没有时间","contextId":"haha","serviceCode":"hehe","reqType":"-1,0","status":"A","describe":"asdfasd","startStamp":1,"requestType":0,"keywordVersion":"K20190424B1","modelVersion":""}
```
### 测试融合结果
```
{"requestId":"123456","message":"没有时间","contextId":"haha","serviceCode":"hehe","reqType":"-1,0","status":"A","describe":"asdfasd","startStamp":1,"requestType":1,"keywordVersion":"K20190424B1","modelVersion":"M20190424B1"}
```

### 测试单模型结果
```
{"requestId":"123456","message":"没有时间","contextId":"haha","serviceCode":"hehe","reqType":"-1,0","status":"A","describe":"asdfasd","startStamp":1,"requestType":3,"keywordVersion":"","modelVersion":"M20190424B1"}
```

## kd 测试数据
```
{"requestId":"123456","message":"你哪里啊","contextId":"haha","serviceCode":"hehe","reqType":"1",
 "status":"A","describe":"asdfasd","startStamp":1,"requestType":1,"questionId":"KD1-FAQ", "kdAnswer": "Tkd11"}
```

## 同时解析 kd 与 kd 结果
```
{"message":"没有时间","reqType":"-1,0,1","status":"A",
"describe":"asdfasd","startStamp":1,"requestType":1,"keywordVersion":"K20190424B1","modelVersion":"M20190424B1",
"questionId":"KD1-FAQ", "kdAnswer": ""}
```

#查看日志
  - sudo docker logs --since 30m container-id
  - sudo docker logs container-id

#
  - systemctl daemon-reload
  - systemctl restart docker

#自动部署
  - sudo vi /etc/profile
  - sudo docker run container-id
