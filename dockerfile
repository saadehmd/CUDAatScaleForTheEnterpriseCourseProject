FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libfreeimage3 libfreeimage-dev cmake build-essential git wget && rm -rf /var/lib/apt/lists/*
WORKDIR /project
RUN git clone https://github.com/NVIDIA/cuda-samples.git /project/cuda-samples