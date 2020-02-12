#!/usr/bin/env bash
#@Author  : fanzfeng
# add by fanzfeng 201905
cid=$(sudo docker ps | grep chat-bot | awk '{print $1}')
sudo docker stop $cid
sudo docker rm $cid
sudo docker rmi -f chat-bot:v5.0
echo "clear containers and images of chat-bot okay"
#sudo docker load -i chat-bot:v5.0.tar
#sudo docker run --env HOST_IP=10.51.6.13 -d -v /home/yw/bot_config:/share -p 10.11.80.90:8800:8703 --restart=always chat-bot:v5.0
#sudo docker run --env HOST_IP=10.51.6.13 -d -v /home/yw/bot_config:/share -p 10.11.80.90:8801:8703 --restart=always chat-bot:v5.0
