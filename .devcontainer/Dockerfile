FROM python:3.10

# タイムゾーンを設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2 \
    libffi-dev \
    libssl-dev \
    curl \
    zlib1g-dev \
    libjpeg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# MT5をインストールするために必要なライブラリ（Windows DLLとの互換性のため）
RUN apt-get update && apt-get install -y \
    wine64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
RUN pip install --upgrade pip setuptools wheel 

WORKDIR /workspace

# OpenGL support for MetaTrader5 visualization (optional)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*