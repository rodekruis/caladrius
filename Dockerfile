# Use the official PyTorch runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /app
WORKDIR /workspace

# Add the data directory to the workspace
VOLUME ["/workspace/data"]

# Copy the current directory contents into the container at /app
COPY . /workspace

ENV HOME="/home"
ENV PATH="$HOME/conda/bin:$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH"

# Install any needed packages
RUN apt-get update &&\
    apt-get install -y --no-install-recommends curl vim less

ENV HOME="/root"

# Install conda
ENV PATH="$HOME/conda/bin:$PATH"
RUN mkdir $HOME/.conda &&\
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > $HOME/miniconda.sh &&\
    chmod 0755 $HOME/miniconda.sh &&\
    /bin/bash $HOME/miniconda.sh -b -p $HOME/conda &&\
    rm $HOME/miniconda.sh &&\
    $HOME/conda/bin/conda update -n base -c defaults conda

# Install NodeJS and Yarn
ENV PATH="$HOME/.yarn/bin:$HOME/.config/yarn/global/node_modules/.bin:$PATH"
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends nodejs &&\
    rm -rf /var/lib/apt/lists/* &&\
    curl -o- -L https://yarnpkg.com/install.sh | bash

# Install Caladrius
RUN /bin/bash caladrius_install.sh &&\
    echo "conda activate caladriusenv" >> ~/.bashrc

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME CALADRIUS
