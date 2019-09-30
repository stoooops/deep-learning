FROM nvcr.io/nvidia/tensorflow:19.08-py3

# Create workspace
ARG WORKSPACE=/workspace/deep-learning
RUN mkdir -p ${WORKSPACE}

# NVIDIA docker is missing some cv2 requirements
# Solution from https://github.com/NVIDIA/nvidia-docker/issues/864
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]

# Helpful command-line libraries
RUN ["apt-get", "install", "-y", "jq"]

# Upgrade underlying pip
RUN pip install --upgrade pip setuptools

# Add requirements.txt before rest of repo for caching (doesn't seem to be working well - TODO revisit)
ADD requirements.txt ${WORKSPACE}
WORKDIR ${WORKSPACE}
RUN pip install -r requirements.txt

# Add repo
#ADD . ${WORKSPACE}

ENV KERAS_MASKRCNN_DIR /workspace/keras-maskrcnn
# RUN pip install /workspace/keras-maskrcnn

# Jupyter
RUN jupyter notebook --generate-config
# Allow LAN access for Jupyter Notebooks
RUN sed -i "s/#c.NotebookApp.ip = 'localhost'/c.NotebookApp.ip = '*'/" ~/.jupyter/jupyter_notebook_config.py
# no password
RUN sed -i "s/#c.NotebookApp.token = '<generated>'/c.NotebookApp.token = ''/" ~/.jupyter/jupyter_notebook_config.py
RUN echo >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py




