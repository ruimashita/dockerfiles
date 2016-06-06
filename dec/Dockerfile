FROM ruimashita/caffe-gpu:rc3

MAINTAINER takuya.wakisaka@moldweorp.com

WORKDIR /opt/dec

RUN cd /opt && git clone https://github.com/piiswrong/dec.git

RUN apt-get update && apt-get install -y python-opencv bc

RUN pip install scikit-learn

RUN cd /opt/dec/caffe && \

    cp Makefile.config.example Makefile.config && \
    sed -i -e 's|/usr/lib/python2.7/dist-packages/numpy/core/include|/usr/local/lib/python2.7/dist-packages/numpy/core/include|' Makefile.config && \

    # Build Caffe core
    make -j"$(nproc)" all && \

    # Install python requirements
    for req in $(cat python/requirements.txt); do pip install --no-cache-dir $req; done && \

    # Build Caffe python
    make -j"$(nproc)" pycaffe 

RUN cd /opt/dec/caffe && \
    # test
    make -j"$(nproc)" test


RUN cd mnist && ./get_data.sh
# RUN cd dec && python make_mnist_data.py


CMD cd dec && python make_mnist_data.py && python dec.py mnist
