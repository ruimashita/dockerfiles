FROM buildpack-deps:trusty

MAINTAINER takuya.wakisaka@moldweorp.com

# faster apt source
# RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt trusty main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-updates main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-backports main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-security main restricted universe multiverse" > /etc/apt/sources.list

# intall python 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python \
    python-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/
