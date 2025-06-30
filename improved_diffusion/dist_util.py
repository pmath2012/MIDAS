"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
from mpi4py import MPI
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3

def print_distributed_environment():
    """
    Print key environment variables related to distributed setup to confirm torchrun is working properly.
    """
    rank = os.environ.get('RANK', 'Not Set')
    world_size = os.environ.get('WORLD_SIZE', 'Not Set')
    local_rank = os.environ.get('LOCAL_RANK', 'Not Set')
    master_addr = os.environ.get('MASTER_ADDR', 'Not Set')
    master_port = os.environ.get('MASTER_PORT', 'Not Set')

    print(f"Distributed Environment Variables:")
    print(f"RANK: {rank}")
    print(f"WORLD_SIZE: {world_size}")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {master_port}")

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    os.environ["RANK"] = str(MPI.COMM_WORLD.Get_rank())
    os.environ["WORLD_SIZE"] = str(MPI.COMM_WORLD.Get_size())

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostname()
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["MASTER_PORT"] = "29500"  # Default port
    dist.init_process_group(backend=backend, init_method="env://")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{get_rank()}")
    return th.device("cpu")

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across ranks.
    """
    if get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = th.load(f, **kwargs)
        
        # Adjust for models saved with DistributedDataParallel
        new_data = {}
        for key, value in data.items():
            if key.startswith('module.'):
                new_data[key[7:]] = value
            else:
                new_data[key] = value
        data = new_data
    else:
        data = None
    
    # Broadcast data from rank 0 to all other ranks
    if dist.is_initialized():
        data_list = [data]
        dist.broadcast_object_list(data_list, src=0)
        data = data_list[0]
        
    return data

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
