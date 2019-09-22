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
    apt-get install -y --no-install-recommends curl vim less &&\
    mkdir $HOME/.conda &&\
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > $HOME/miniconda.sh &&\
    chmod 0755 $HOME/miniconda.sh &&\
    $HOME/miniconda.sh -b -p $HOME/conda &&\
    rm $HOME/miniconda.sh &&\
    conda update -n base -c defaults conda &&\
    curl -sL https://deb.nodesource.com/setup_10.x | bash &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends nodejs &&\
    rm -rf /var/lib/apt/lists/* &&\
    curl -o- -L https://yarnpkg.com/install.sh | bash &&\
    /bin/bash caladrius_install.sh &&\
    echo "source activate caladriusenv" > ~/.bashrc

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME CALADRIUS
