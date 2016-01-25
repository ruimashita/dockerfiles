FROM ruimashita/numpy

MAINTAINER takuya.wakisaka@moldweorp.com

# install scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/ \
    && pip install scipy
