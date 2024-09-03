FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    wget \
    git \
    python3-venv \
    curl \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*


RUN wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tar.xz \
    && tar xvf Python-3.11.4.tar.xz \
    && cd Python-3.11.4 \
    && ./configure --prefix=/usr/local \
    && make -j $(nproc) \
    && make altinstall


RUN ln -s /usr/local/bin/python3.11 /usr/bin/python3.11 \
    && ln -s /usr/local/bin/pip3.11 /usr/bin/pip3.11

RUN ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/pip3.11 /usr/bin/pip


RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"


RUN pip install --upgrade pip \
    && pip install git+https://github.com/joanag05/CofactorPredictionTool.git \
    && pip install Flask

RUN curl -fsSL https://get.nextflow.io | bash \
    && mv nextflow /usr/local/bin




COPY ./nextflow /home

EXPOSE 80

CMD ["python", "/home/worker.py"]

