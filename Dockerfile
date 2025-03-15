FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

ENV DEEPSTREAM_VERSION=6.3
ENV CUDA_VER=12.1
ENV NVDS_VERSION=6.3

# Run pre-require
RUN python3 --version # 3.10.12
RUN apt update
RUN apt install --fix-broken -y
RUN apt -y install python3-gi python3-gst-1.0 python-gi-dev git meson \
    python3 python3-pip cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin libgstreamer1.0-dev libtool m4 autoconf automake libgirepository1.0-dev libcairo2-dev
RUN pip3 install --upgrade pip

# Install libs python
RUN pip install pyds 

# Install libs
RUN apt-get update
RUN apt-get install autoconf automake libtool make g++ unzip -y
RUN mkdir /deepstream && cd /deepstream

# Install custom plugin
COPY gst-plugins /deepstream/gst-plugins
COPY includes /deepstream/includes
COPY libs /deepstream/libs
COPY install_opencv_cuda.sh /deepstream/install_opencv_cuda.sh
COPY setting_enviroment.sh /deepstream/setting_enviroment.sh

WORKDIR /deepstream

# Install opencv
RUN bash install_opencv_cuda.sh

#Setting environment
RUN bash setting_enviroment.sh /deepstream/ 



