FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update -qq && \
    apt-get install -y zip git git-lfs vim libgtk2.0-dev ffmpeg libsm6 libxext6 && \
    rm -rf /var/cache/apk/*

COPY requirements.txt /workspace

# Activate conda environment and install packages
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN pip --no-cache-dir install -r /workspace/requirements.txt

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /workspaces/pi3detr