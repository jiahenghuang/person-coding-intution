FROM debian
MAINTAINER www.zhenai.com
ENV LANG C.UTF-8

# Copy code
RUN mkdir -p /src
WORKDIR /src

RUN /bin/cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo 'Asia/Shanghai' > /etc/timezone
RUN apt-get update && apt-get install -y unzip libbz2-dev
# Install python3
RUN apt-get install -y python3
RUN apt-get install -y python3-dev

# Install pip
RUN apt-get install -y wget vim
RUN apt-get install -y build-essential gcc g++

RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3 /tmp/get-pip.py
RUN pip3 install --upgrade pip

# Install app dependencies
COPY requirements.txt /src/
RUN pip3 install -r requirements.txt

# Bundle app source
COPY . /src/
WORKDIR /src/bin
# CMD ["sh","setup.sh"]
CMD ["python3", "analyze_chat.py"]