#!/bin/sh

sudo docker build -t chat-bot:v5.0 .
# sudo docker build -t nginx-server:v5.0 /home/heng/projects/nginx-server/020_总结文档/001_nginx负载均衡/

sudo docker save -o ../chat-bot:v5.0.tar chat-bot:v5.0
# sudo docker save -o ../nginx-server:v5.0.tar nginx-server:v5.0

sudo chmod 777 ../chat-bot:v5.0.tar
# sudo chmod 777 ../nginx-server:v5.0.tar