FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel

ENV DEEPSTREAM_VERSION=6.3
ENV CUDA_VER=12.1
ENV NVDS_VERSION=6.3

# Run pre-require
RUN pip3 install --upgrade pip

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



