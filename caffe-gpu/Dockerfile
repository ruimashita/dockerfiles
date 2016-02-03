FROM ruimashita/cuda-scipy

MAINTAINER takuya.wakisaka@moldweorp.com

ENV PYTHONPATH /opt/caffe/python
ENV PATH $PATH:/opt/caffe/.build_release/tools
ENV CAFFE_VERSION=master

RUN apt-get update && apt-get install -y --no-install-recommends \
  # for caffe
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

  && apt-get clean \
  && rm -rf /var/lib/apt/lists/

###########
# caffe
###########

RUN cd /opt && git clone https://github.com/BVLC/caffe.git && cd caffe &&  git checkout $CAFFE_VERSION

WORKDIR /opt/caffe

# Build Caffe core
RUN cp Makefile.config.example Makefile.config && \
  make -j"$(nproc)" all

# Install python deps
RUN for req in $(cat python/requirements.txt); do pip install $req; done

# Build Caffe python
RUN make -j"$(nproc)" pycaffe

# test + run tests
RUN make -j"$(nproc)" test
# RUN cd /opt/caffe && make runtest

# for bug "libdc1394 error: Failed to initialize libdc1394"
# RUN ln /dev/null /dev/raw1394
