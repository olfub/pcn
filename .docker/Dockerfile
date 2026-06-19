# Dockerfile for dev container
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin

RUN apt-get update

RUN apt install -y zsh

RUN apt install -y git

RUN pip install matplotlib pgmpy rtpt wandb

# Add the following lines to install packages in the specified directory
COPY ./models/simple-einet /workspaces/pcn/models/simple-einet
WORKDIR "/workspaces/pcn/models/simple-einet"
RUN pip install -e .

RUN pip install --upgrade numpy scipy

WORKDIR "/workspaces/pcn"