#!/bin/bash
# build.sh - 构建 FunASR Docker 镜像

# 镜像名称和标签
IMAGE_NAME="funasr-service"
IMAGE_TAG="v1"

# 构建镜像
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
