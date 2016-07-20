FROM buildpack-deps:trusty

MAINTAINER takuya.wakisaka@moldweorp.com

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_RUNFILE_URL=http://developer.download.nvidia.com/compute/cuda/7.0/Prod/local_installers/cuda_7.0.28_linux.run


# faster apt source
# RUN echo "deb mirror://mirrors.ubuntu.com/mirrors.txt trusty main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-updates main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-backports main restricted universe multiverse \n\
# deb mirror://mirrors.ubuntu.com/mirrors.txt trusty-security main restricted universe multiverse" > /etc/apt/sources.list

# intall python 
RUN apt-get update && apt-get install -y --no-install-recommends \
    axel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

RUN cd /tmp && \
    # Download run file
    axel $CUDA_RUNFILE_URL && \
    # Make the run file executable and extract
    chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
    # Install toolkit (silent)
    ./cuda-linux64-rel-*.run -noprompt && \
    # Clean up
    rm -rf *

# Install nvidia drivers (silent, no kernel)
RUN cd /tmp && \
    axel http://us.download.nvidia.com/XFree86/Linux-x86_64/352.93/NVIDIA-Linux-x86_64-352.93.run  && \
    chmod +x  ./NVIDIA-Linux-x86_64-*.run && \
    ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
    rm -rf *
