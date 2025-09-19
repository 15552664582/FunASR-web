# 1基础镜像：已经包含 PyTorch + CUDA 11.8 + cuDNN 9
FROM hub-bigdata.17usoft.com/bigdata/pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# 设置容器工作目录
WORKDIR /funasr

# 安装系统依赖
RUN \
    apt update && \
    apt install -y libsndfile1 && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 到容器
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制应用代码
COPY main.py .
