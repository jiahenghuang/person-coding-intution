#!/bin/sh

scp heng@10.1.29.167:/home/heng/projects/chat-bot:v5.0.tar .
scp heng@10.1.29.167:/home/heng/projects/nginx-server:v5.0.tar .

sudo docker load -i chat-bot:v5.0.tar
sudo docker load -i nginx-server:v5.0.tar

sudo docker stop $(sudo docker ps -a -q)
sudo docker rm $(sudo docker ps -a -q)

sudo docker run -d -v /home/yw/bot_config:/share -p 10.1.29.167:8800:8703 --restart=always chat-bot:v5.0
sudo docker run -d -v /home/yw/bot_config:/share -p 10.1.29.167:8801:8703 --restart=always chat-bot:v5.0
sudo docker run -d -p 10.1.29.167:8080:80 --restart=always nginx-server:v5.0