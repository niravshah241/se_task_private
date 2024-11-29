import time
import numpy as np
import torch

from sciml_modules.dataset import CustomDataset
from sciml_modules.neural_net import Net
from sciml_modules.train_validate \
    import train_nn, valid_nn

torch.manual_seed(74)
np.random.seed(74)

# Select device and Data type
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

dtype = torch.float64

num_samples = 1000  # Number of samples
sample_dim = 3  # Input dimensions
label_dim = 5  # Output dimensions
hidden_dim = 4  # Hidden layer dimensions / Number of neurons

# Input parameters
x = torch.from_numpy(np.random.rand(num_samples, sample_dim))
x = x.to(device=device).to(dtype=dtype) * 2. - 1.

# True Weight and Bias parameters
w1 = torch.from_numpy(np.random.rand(label_dim, sample_dim))
w1 = w1.to(device=device).to(dtype=dtype)
b1 = torch.from_numpy(np.random.rand(label_dim, 1))
b1 = b1.to(device=device).to(dtype=dtype) * 0.1

# Output values
y = torch.sin(torch.matmul(w1, x.T) + b1).T

print(f"Input - Shape: {x.shape}, Device: {x.device}, Data type: {x.dtype}")
print(f"Output - Shape: {y.shape}, Device {y.device}, Data type: {y.dtype}")
if x.device == "cpu":
    print(f"Input in NumPy: \n {x.numpy()}")
else:
    print(f"Input in NumPy: \n {x.cpu().numpy()}")

# Initialise neural network
net = Net(sample_dim, list(), label_dim, torch.sin,
          dtype=dtype, device=device)


for param in net.parameters():
    print(
        f"""NN parameter Shape: {param.shape},
         Device {param.device}, Data type: {param.dtype}""")

num_train_data = int(0.8 * x.shape[0])  # 80% data for training
batch_size_train = 400  # Training data batch size

# Training dataset and dataloader
customDataset_train = CustomDataset(
    x[:num_train_data, :], y[:num_train_data, :])
customDataLoader_train = torch.utils.data.DataLoader(
    customDataset_train, batch_size=batch_size_train, shuffle=False)

print(f"Number of training samples: {len(customDataLoader_train.dataset)}")
print(f"Number of training batches: {len(customDataLoader_train)}")

for batch, (input_batch, output_batch) in enumerate(customDataLoader_train):
    print(
        f"""Training: Batch {batch+1},
         Batch size (Input, Output)
          ({input_batch.shape, output_batch.shape})""")

num_val_data = x.shape[0] - num_train_data  # ~20% data for validation
batch_size_val = num_val_data  # ~20% data batch size

# Validation dataset and dataloader
customDataset_val = CustomDataset(x[num_train_data:, :], y[num_train_data:, :])
customDataLoader_val = torch.utils.data.DataLoader(
    customDataset_val, batch_size=batch_size_val, shuffle=False)

print(f"Number of validation samples: {len(customDataLoader_val.dataset)}")
print(f"Number of validation batches: {len(customDataLoader_val)}")

for batch, (input_batch, output_batch) in enumerate(customDataLoader_val):
    print(f"""Validation: Batch {batch+1},
           Batch size (Input, Output)
           ({input_batch.shape, output_batch.shape})""")

max_epochs = 20000  # Maximum number of epochs
# Optimiser with step size = lr (learning rate)
optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

# MSE loss
loss_fn = torch.nn.MSELoss(reduction="sum")

train_loss_list = list()
val_loss_list = list()

# Start of neural network training
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


# Testing phase on new data
num_test_samples = 25
x_np = np.random.rand(num_test_samples, sample_dim) * 2. - 1.
x_test = (torch.from_numpy(x_np)).to(device=device).to(dtype=dtype)

y_test = (torch.sin(torch.matmul(w1, x_test.T) +
          b1).T).to(device=device).to(dtype=dtype)

y_np = y_test.cpu().numpy()
y_pred = net(x_test)
y_pred = y_pred.detach().cpu().numpy()

print(f"\n True: \n {y_np[:10, :]}")
print(f"\n Prediction: \n {y_pred[:10, :]}")
print(
    f"""\n Absolute error between True and prediction:
        \n {abs(y_np[:10, :] - y_pred[:10, :])}""")
