# can be run like:
# docker build -t go-faiss .
# docker run -it --rm go-faiss bash

FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    vim \
    sudo \
    curl \
    wget \
    gnupg2 \
    gnupg-agent \
    iputils-ping \
    apt-transport-https \
    ca-certificates \
    lsb-release

RUN apt-get install -y build-essential libblas-dev liblapack-dev swig && apt-get update

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install --upgrade pip

RUN apt-get install -y git

RUN git clone https://github.com/DataIntelligenceCrew/go-faiss.git
RUN git clone https://github.com/facebookresearch/faiss.git

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.20.4/cmake-3.20.4-linux-x86_64.sh --output /opt/cmake-3.20.4-linux-x86_64.sh && chmod +x /opt/cmake-3.20.4-linux-x86_64.sh
RUN cd /opt && yes | sudo bash /opt/cmake-3.20.4-linux-x86_64.sh
RUN sudo ln -s /opt/cmake-3.20.4-linux-x86_64/bin/* /usr/local/bin/


RUN \
  curl -L https://golang.org/dl/go1.16.5.linux-amd64.tar.gz --output /tmp/go1.16.5.linux-amd64.tar.gz \
  && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go1.16.5.linux-amd64.tar.gz

ENV PATH=$PATH:/usr/local/go/bin

RUN cd /faiss \
    && cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON \
    && make -C build -j2 faiss \
    && make -C build install \
    && cp ./build/c_api/libfaiss_c.so /usr/lib/

RUN cd /faiss/c_api && cmake -B build . && cd build && make
