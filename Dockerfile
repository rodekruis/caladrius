# Use the official PyTorch runtime as a parent image
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

# Set the working directory to /app
WORKDIR /workspace

# Copy the current directory contents into the container at /app
COPY . /workspace

# Install any needed packages
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash &&\
	apt-get install -y vim less nodejs &&\
	curl -o- -L https://yarnpkg.com/install.sh | bash &&\
	pip install --upgrade pip &&\
	pip install --trusted-host pypi.python.org -r requirements.txt &&\
	$HOME/.yarn/bin/yarn install

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME CALADRIUS
