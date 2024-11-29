# Pull dolfinx 0.7.3 docker image
FROM dolfinx/dolfinx:v0.7.3

ENV DOLFINX_DIR=/usr/local/dolfinx-real

# Set work directory
RUN mkdir -p /home/se_task
WORKDIR /home/se_task

# Clone git repository
RUN apt-get install -y git
RUN git clone https://github.com/niravshah249/se_task_private.git


# Keep container alive
CMD ["sleep", "infinity"]
