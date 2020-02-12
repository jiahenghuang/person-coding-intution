#!/usr/bin/env bash

docker build -t chat-bot:v5.0 .
docker save -o ../chat-bot:v5.0.tar chat-bot:v5.0
