import time
import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI

from sciml_modules.partitioned_dataset import CustomPartitionedDataset
from sciml_modules.neural_net import Net
from sciml_modules.train_validate_distributed \
    import train_nn, valid_nn
from sciml_modules.wrappers \
    import init_cpu_process_group, \
    init_gpu_process_group


torch.manual_seed(74)
np.random.seed(74)

# Data type
dtype = torch.float64

num_samples = 1000  # Number of samples
sample_dim = 3  # Input dimensions
label_dim = 5  # Output dimensions
hidden_dim = 4  # Hidden layer dimensions / Number of neurons

# World communicator
world_comm = MPI.COMM_WORLD

# Splitting communicators
procs0 = world_comm.group.Incl([0, 1])
procs0_comm = world_comm.Create_group(procs0)

procs1 = world_comm.group.Incl([2, 3])
procs1_comm = world_comm.Create_group(procs1)

procs2 = world_comm.group.Incl([4, 5])
procs2_comm = world_comm.Create_group(procs2)

procs3 = world_comm.group.Incl([6, 7])
procs3_comm = world_comm.Create_group(procs3)

# List of sub-communicators for neural network trainig.
# Each sub-communicator traing one neural network.
comm_list = [procs0_comm, procs1_comm,
             procs2_comm, procs3_comm]

num_train_data = int(0.8 * num_samples)  # 80% data for training
num_val_data = num_samples - num_train_data  # Remaining data for validation
batch_size_train = 200  # Training data batch size
batch_size_val = num_val_data  # Validation data batch size


for j in range(len(comm_list)):
    if comm_list[j] != MPI.COMM_NULL:
        # Select device and Data type
        if torch.cuda.is_available():
            device = f"cuda:{comm_list[j].rank}"
            init_gpu_process_group(comm_list[j])
        else:
            device = "cpu"
            init_cpu_process_group(comm_list[j])

        # Indices for dataset partitioning
        indices_train = np.arange(comm_list[j].rank,
                                  num_train_data, comm_list[j].size)
        indices_val = np.arange(comm_list[j].rank,
                                num_val_data, comm_list[j].size)

# Create shared memory for sharing datasets
nbytes = num_samples * sample_dim * MPI.DOUBLE.Get_size()
win0 = MPI.Win.Allocate_shared(
    nbytes, MPI.DOUBLE.Get_size(), comm=MPI.COMM_WORLD)
buf0, itemsize = win0.Shared_query(0)
x = np.ndarray(buffer=buf0, dtype="d", shape=(num_samples, sample_dim))

nbytes = label_dim * sample_dim * MPI.DOUBLE.Get_size()
win1 = MPI.Win.Allocate_shared(
    nbytes, MPI.DOUBLE.Get_size(), comm=MPI.COMM_WORLD)
buf1, itemsize = win1.Shared_query(0)
w1 = np.ndarray(buffer=buf1, dtype="d", shape=(label_dim, sample_dim))

nbytes = label_dim * 1 * MPI.DOUBLE.Get_size()
win2 = MPI.Win.Allocate_shared(
    nbytes, MPI.DOUBLE.Get_size(), comm=MPI.COMM_WORLD)
buf2, itemsize = win2.Shared_query(0)
b1 = np.ndarray(buffer=buf2, dtype="d", shape=(label_dim, 1))

nbytes = num_samples * label_dim * MPI.DOUBLE.Get_size()
win3 = MPI.Win.Allocate_shared(
    nbytes, MPI.DOUBLE.Get_size(), comm=MPI.COMM_WORLD)
buf3, itemsize = win3.Shared_query(0)
y = np.ndarray(buffer=buf3, dtype="d", shape=(num_samples, label_dim))

# Input parameters
if world_comm.rank == 0:
    x[:] = np.random.rand(num_samples, sample_dim) * 2. - 1.

    # True Weight and Bias parameter
    w1[:] = np.random.rand(label_dim, sample_dim)
    b1[:] = np.random.rand(label_dim, 1) * 0.1

    # Output values
    y[:] = np.sin(np.matmul(w1, x.T) + b1).T

    print(f"Input - Shape: {x.shape}, Data type: {x.dtype}")
    print(f"Output - Shape: {y.shape}, Data type: {y.dtype}")
    print(f"Input in NumPy: \n {x}")

world_comm.barrier()

# Conversion to torch tensors
x_torch = torch.from_numpy(x).to(dtype=dtype).to(device=device)
y_torch = torch.from_numpy(y).to(dtype=dtype).to(device=device)
w1_torch = torch.from_numpy(w1).to(dtype=dtype).to(device=device)
b1_torch = torch.from_numpy(b1).to(dtype=dtype).to(device=device)

print(f"Inputs: x \n {x}, \n y \n {y}")
print(
    f"""World rank: {world_comm.rank},
     \n Indices (train): {indices_train},
     \n Indices (val): {indices_val}""")

# Artificial Neural Network
net = Net(sample_dim, list(), label_dim, torch.sin,
          dtype=dtype, device=device)

for param in net.parameters():
    # dist.barrier()
    print(
        f"""World rank: {world_comm.rank},
         Params before synchronisation: {param.data}""")
    dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    param.data /= dist.get_world_size()
    print(
        f"""World rank: {world_comm.rank},
         Params after synchronisation: {param.data}""")

for param in net.parameters():
    print(
        f"""NN parameter Shape: {param.shape},
        Device {param.device}, Data type: {param.dtype}""")

# Training partitioned dataset and dataloader
customDataset_train = CustomPartitionedDataset(
    x_torch[:num_train_data, :], y_torch[:num_train_data, :], indices_train)
customDataLoader_train = torch.utils.data.DataLoader(
    customDataset_train, batch_size=batch_size_train, shuffle=False)

print(f"Number of training samples: {len(customDataLoader_train.dataset)}")
print(f"Number of training batches: {len(customDataLoader_train)}")

for batch, (input_batch, output_batch) in enumerate(customDataLoader_train):
    print(
        f"""Training: Batch {batch+1},
         Batch size (Input, Output)
         ({input_batch.shape, output_batch.shape})""")

# Validation partitioned dataset and dataloader
customDataset_val = CustomPartitionedDataset(
    x_torch[num_train_data:, :], y_torch[num_train_data:, :], indices_val)
customDataLoader_val = torch.utils.data.DataLoader(
    customDataset_val, batch_size=batch_size_val, shuffle=False)

print(f"Number of validation samples: {len(customDataLoader_val.dataset)}")
print(f"Number of validation batches: {len(customDataLoader_val)}")

for batch, (input_batch, output_batch) in enumerate(customDataLoader_val):
    print(
        f"""Validation: Batch {batch+1},
         Batch size (Input, Output)
          ({input_batch.shape, output_batch.shape})""")

max_epochs = 20000  # Maximum number of epochs
# Optimiser with step size = lr (learning rate)
optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

# MSE loss
loss_fn = torch.nn.MSELoss(reduction="sum")

train_loss_list = list()
val_loss_list = list()

# Start of the neural network training
start_epoch = 0
start_time = time.time()
for epochs in range(start_epoch, max_epochs):
    current_training_loss = train_nn(customDataLoader_train,
                                     net, loss_fn, optimiser)
    train_loss_list.append(current_training_loss)
    current_validation_loss = valid_nn(customDataLoader_val,
                                       net, loss_fn)
    val_loss_list.append(current_validation_loss)
    if epochs > 0 and current_validation_loss > min_validation_loss:  # noqa: F821, E501
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss = min(val_loss_list)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"True w1: \n {w1}")
print(f"True b1: \n {b1}")

print(f"ANN w1: \n {list(net.parameters())[0]}")
print(f"ANN b1: \n {list(net.parameters())[1]}")

world_comm.barrier()

# Testing phase
if world_comm.rank == 0:

    num_test_samples = 25
    x_np = np.random.rand(num_test_samples, sample_dim) * 2. - 1.
    x_test = (torch.from_numpy(x_np)).to(device=device).to(dtype=dtype)

    y_np = (np.sin(np.matmul(w1, x_np.T) +
            b1).T)

    y_pred = net(x_test)
    y_pred = y_pred.detach().cpu().numpy()

    print(f"\n True: \n {y_np[:10, :]}")
    print(f"\n Prediction: \n {y_pred[:10, :]}")
    print(
        f"""\n Absolute error between True
            and prediction:
            \n {abs(y_np[:10, :] - y_pred[:10, :])}""")
