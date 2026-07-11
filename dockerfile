FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libfreeimage3 libfreeimage-dev cmake build-essential git wget && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libopencv-core-dev libopencv-imgcodecs-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /project
