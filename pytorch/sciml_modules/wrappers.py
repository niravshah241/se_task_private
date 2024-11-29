import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401


def model_synchronise(model, verbose=False):
    """Model synchronisation for same parameter
    in different processes

    Args:
        model (torch.nn Module): neural net
        verbose (bool, optional): Information print. Defaults to False.
    """
    for param in model.parameters():
        dist.barrier()
        if verbose is True:
            print(f"Params before synchronisation: {param.data}")
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()
        if verbose is True:
            print(f"Params after synchronisation: {param.data}")


def init_cpu_process_group(comm):
    """Initiliase CPU process group

    Args:
        comm: MPI Sub-communicator
    """
    os.environ['MASTER_ADDR'] = 'localhost'

    if comm.rank == 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        s.close()
    else:
        free_port = None

    free_port = comm.bcast(free_port, root=0)
    comm.Barrier()

    os.environ['MASTER_PORT'] = str(free_port)

    dist.init_process_group("gloo", rank=comm.rank,
                            world_size=comm.size)


def init_gpu_process_group(comm):
    """Initiliase GPU process group

    Args:
        comm: MPI Sub-communicator
    """
    os.environ["MASTER_ADDR"] = "localhost"

    if comm.rank == 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        free_port = s.getsockname()[1]
        s.close()
    else:
        free_port = None

    free_port = comm.bcast(free_port, root=0)
    comm.Barrier()

    os.environ['MASTER_PORT'] = str(free_port)
    dist.init_process_group("nccl", rank=comm.rank,
                            world_size=comm.size)
