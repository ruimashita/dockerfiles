FROM ruimashita/cudnn-scikit-learn-image

MAINTAINER takuya.wakisaka@moldweorp.com

ENV PYTHONPATH /opt/caffe/python
ENV PATH $PATH:/opt/caffe/build/tools
ENV CAFFE_VERSION=master
WORKDIR /opt/caffe

RUN apt-get update && apt-get install -y --no-install-recommends \
  # for caffe
  bc \
  libprotobuf-dev \
  libleveldb-dev \
  libsnappy-dev \
  libopencv-dev \
  libhdf5-serial-dev \
  protobuf-compiler \
  libatlas-base-dev \  
  libgflags-dev \ 
  libgoogle-glog-dev \
  liblmdb-dev \
  libboost-all-dev \ 

  # for /opt/caffe/python/draw_net.py
  python-pydot \

  # for /opt/caffe/scripts/download_model_from_gist.sh
  unzip \

  # for /opt/caffe/tools/extra/parse_log.sh
  bsdmainutils \

  && apt-get clean \
  && rm -rf /var/lib/apt/lists/


RUN cd /opt && git clone https://github.com/pmgysel/caffe.git && cd caffe && git checkout $CAFFE_VERSION 

RUN cp Makefile.config.example Makefile.config && \
    echo "DEBUG := 1"  >> Makefile.config && \
    # echo "USE_CUDNN := 1" >> Makefile.config && \
    # echo "WITH_PYTHON_LAYER := 1" >> Makefile.config && \
    sed -i -e 's|/usr/lib/python2.7/dist-packages/numpy/core/include|/usr/local/lib/python2.7/dist-packages/numpy/core/include|' Makefile.config && \

    # Build Caffe core
    make -j"$(nproc)" all && \

    # Install python requirements
    for req in $(cat python/requirements.txt); do pip install --no-cache-dir $req; done && \

    # Build Caffe python
    make -j"$(nproc)" pycaffe && \

    # test
    make -j"$(nproc)" test

    # run test
    # make -j"$(nproc)" runtest && \

    # pytest
    # make -j"$(nproc)" pytest


RUN cd /opt && git clone https://github.com/DeepScale/SqueezeNet.git

RUN cp /opt/SqueezeNet/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel /opt/caffe/models/SqueezeNet/


# download CIFAR-10
RUN ./data/cifar10/get_cifar10.sh

# create examples/cifar10_train_lmdb and examples/cifar10_test_lmdb
RUN ./examples/cifar10/create_cifar10.sh

# setup flower photos
RUN wget http://download.tensorflow.org/example_images/flower_photos.tgz
RUN tar xzf flower_photos.tgz && \  
    rm flower_photos.tgz && \
    mv flower_photos /opt/caffe/data/

COPY flower_photos/val.txt /opt/caffe/data/flower_photos/
COPY flower_photos/train.txt /opt/caffe/data/flower_photos/
COPY flower_photos/create_lmdb.sh /opt/caffe/data/flower_photos/
RUN bash data/flower_photos/create_lmdb.sh


COPY quantized.prototxt /opt/caffe/models/SqueezeNet/RistrettoDemo/
COPY solver_finetune.prototxt /opt/caffe/models/SqueezeNet/RistrettoDemo/
COPY train_val.prototxt /opt/caffe/models/SqueezeNet/
RUN cat ./examples/ristretto/00_quantize_squeezenet.sh


# for bug "libdc1394 error: Failed to initialize libdc1394"
# RUN ln /dev/null /dev/raw1394
