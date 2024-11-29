# PyTorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get clean
RUN apt update
# RUN apt-get install -y libmpich-dev
RUN conda install mpi4py

# Set work directory
RUN mkdir -p /home/se_task
WORKDIR /home/se_task

# Clone git repository
RUN apt-get install -y git
RUN git clone https://github.com/niravshah249/se_task_private.git

# Keep container alive
CMD ["sleep", "infinity"]
