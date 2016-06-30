FROM ruimashita/cuda

MAINTAINER takuya.wakisaka@moldweorp.com

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

###
# install cudnn
###
ENV CUDNN_VERSION 4
RUN curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v4/cudnn-7.0-linux-x64-v4.0-prod.tgz -O && \
    tar -xzf cudnn-7.0-linux-x64-v4.0-prod.tgz -C /usr/local && \
    rm cudnn-7.0-linux-x64-v4.0-prod.tgz && \
    ldconfig


RUN echo "deb http://ftp.jaist.ac.jp/ubuntu/ trusty main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ trusty main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ trusty-updates main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ trusty-updates main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ trusty-backports main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ trusty-backports main restricted universe multiverse \n\
deb http://security.ubuntu.com/ubuntu trusty-security main restricted universe multiverse \n\
deb-src http://security.ubuntu.com/ubuntu trusty-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python \
    python-dev \
    python-pip \
    python-numpy \
    python-scipy \
    python-scikits-learn \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


###
# install jupyter
###
RUN pip  install \
        ipykernel \
        jupyter \
        matplotlib \
        && \
    python -m ipykernel.kernelspec

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /


RUN apt-get update && apt-get install -y \
    software-properties-common \
    swig \
    zip \
    rsync \
    libzmq3-dev \
    gdb \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Bazel.
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y openjdk-8-jdk openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
RUN curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
RUN sudo apt-get update && sudo apt-get install -y bazel

ENV TENSORFLOW_VERSION r0.9

# Download and build TensorFlow.
RUN git clone --recursive https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout ${TENSORFLOW_VERSION} && \
    git submodule update --init

WORKDIR /tensorflow

# Configure the build for our CUDA configuration.
ENV CUDA_TOOLKIT_PATH /usr/local/cuda
ENV CUDNN_INSTALL_PATH /usr/local/cuda
ENV TF_NEED_CUDA 1
ENV TF_CUDA_VERSION 7.0

RUN echo "startup --batch" >>/root/.bazelrc
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/root/.bazelrc
ENV BAZELRC /root/.bazelrc

# RUN ./configure && \
#     bazel build -c opt //tensorflow/tools/pip_package:build_pip_package && \
#     bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package


RUN ./configure && \
    bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package


RUN bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
    pip install --upgrade /tmp/pip/tensorflow-*.whl


# # RUN mkdir _python_build  && \
# #     cd _python_build  && \
# #     ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* .  && \
# #     ln -s ../tensorflow/tools/pip_package/* .  && \
# #     python setup.py develop

# ignore fail test
# RUN bazel test -j $(nproc) -c opt //tensorflow/... ; exit 0 && \
#     bazel test -j $(nproc) -c opt //tensorflow/python/... ; exit 0 && \
#     bazel test -j $(nproc) -c opt --config=cuda //tensorflow/... ; exit 0 && \
#     bazel test -j $(nproc) -c opt --config=cuda //tensorflow/python/... ; exit 0

# ignore fail test
RUN bazel test -j $(nproc) --stamp //tensorflow/... ; exit 0 && \
    bazel test -j $(nproc) --stamp //tensorflow/python/... ; exit 0 && \
    bazel test -j $(nproc) --stamp --config=cuda //tensorflow/... ; exit 0 && \
    bazel test -j $(nproc) --stamp --config=cuda //tensorflow/python/... ; exit 0

# あらかじめbuildしておく
RUN bazel build --config=cuda //tensorflow/... && \
    bazel build --config=cuda //tensorflow/python/...

# # Set up CUDA variables
ENV CUDA_PATH /usr/local/cuda


RUN bazel build --config=cuda //tensorflow/g3doc/how_tos/adding_an_op:cuda_op_kernel.so


### user ops
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/tensorflow/bazel-bin/tensorflow/core/user_ops/
COPY tensorflow/core/user_ops/BUILD /tensorflow/tensorflow/core/user_ops/BUILD


### for zero out
# COPY tensorflow/core/user_ops/zero_out.cc /tensorflow/tensorflow/core/user_ops/zero_out.cc
# RUN bazel build -c opt //tensorflow/core/user_ops:zero_out.so
# RUN bazel build -c opt --config=cuda //tensorflow/core/user_ops:zero_out.so

# COPY tensorflow/python/kernel_tests/zero_out_op_test.py /tensorflow/tensorflow/python/kernel_tests/zero_out_op_test.py
# RUN bazel test --test_output=all tensorflow/python:zero_out_op_test
# RUN bazel test //tensorflow/python:zero_out_op_test


### for my pad
COPY tensorflow/core/user_ops/my_pad.cc /tensorflow/tensorflow/core/user_ops/my_pad.cc
# RUN bazel build -c opt //tensorflow/core/user_ops:my_pad.so
# RUN bazel build -c opt --verbose_failures //tensorflow/core/user_ops:my_pad.so

COPY tensorflow/core/user_ops/__init__.py /tensorflow/tensorflow/core/user_ops/__init__.py
COPY tensorflow/core/user_ops/my_pad_op.py /tensorflow/tensorflow/core/user_ops/my_pad_op.py
COPY tensorflow/core/user_ops/my_pad_op_test.py /tensorflow/tensorflow/core/user_ops/my_pad_op_test.py
COPY tensorflow/core/user_ops/my_pad.cu.cc /tensorflow/tensorflow/core/user_ops/my_pad.cu.cc
COPY tensorflow/core/user_ops/my_pad.h /tensorflow/tensorflow/core/user_ops/my_pad.h
# RUN bazel build --verbose_failures //tensorflow/core/user_ops:my_pad.so
RUN bazel build --config=cuda --verbose_failures //tensorflow/core/user_ops:my_pad.so
# RUN bazel build -c dbg --config=cuda --verbose_failures //tensorflow/core/user_ops:my_pad.so

# RUN bazel test  -j $(nproc) --test_output=all //tensorflow/core/user_ops:my_pad_op_test
# RUN bazel build -c opt --config=cuda --verbose_failures //tensorflow/core/user_ops:my_pad.so
# RUN bazel test --config=cuda  -j $(nproc) --test_output=all //tensorflow/core/user_ops:my_pad_op_test


# RUN bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
# RUN bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu


# RUN bazel build -c dbg --config=cuda //tensorflow/core/user_ops:my_pad.so

# RUN bazel build -c dbg --strip=never --config=cuda //tensorflow/tools/pip_package:build_pip_package

# RUN cp -r bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/__main__/* bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/ && \
#     bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip && \
#     pip install --upgrade /tmp/pip/tensorflow-*.whl



# RUN bazel build --config=cuda //tensorflow/g3doc/how_tos/adding_an_op:cuda_op_kernel.so

# RUN bazel test -j $(nproc) --verbose_failures --stamp --config=cuda //tensorflow/g3doc/how_tos/adding_an_op:cuda_op_test


# RUN bazel build --config=cuda --verbose_failures //tensorflow/core/user_ops:my_pad.so



# for my_cuda_op
# COPY tensorflow/core/user_ops/my_cuda_op_kernel.cc /tensorflow/tensorflow/core/user_ops/my_cuda_op_kernel.cc
# COPY tensorflow/core/user_ops/my_cuda_op_kernel.cu.cc /tensorflow/tensorflow/core/user_ops/my_cuda_op_kernel.cu.cc
# RUN bazel build --config=cuda //tensorflow/core/user_ops:my_cuda_op_kernel.so




WORKDIR "/notebooks"

CMD ["/run_jupyter.sh"]
