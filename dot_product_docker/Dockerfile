# Pull dolfinx 0.7.3 docker image (for MPI support)
FROM dolfinx/dolfinx:v0.7.3

ENV DOLFINX_DIR=/usr/local/dolfinx-real

# Set work directory
RUN mkdir -p /home/se_task
WORKDIR /home/se_task

# Clone git repository
RUN apt-get install -y git
RUN git clone https://github.com/niravshah249/se_task_private.git

# Install clang-tidy
RUN pip3 install clang-tidy

# Keep container alive
CMD ["sleep", "infinity"]
