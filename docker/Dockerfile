ARG PYTORCH="1.7.0"
ARG CUDA="11.0"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# install kits19 starter code dependencies
RUN git clone https://github.com/neheller/kits19.git /tmp/kits19
RUN pip install -r /tmp/kits19/requirements.txt && \
    rm -rf /tmp/kits19

ENV PYTHONPATH $PYTHONPATH:/work:/data/kits19

WORKDIR /work