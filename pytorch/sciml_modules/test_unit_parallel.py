import unittest
from neural_net import Net
from partitioned_dataset import CustomPartitionedDataset
from train_validate_distributed import train_nn, valid_nn
from wrappers import init_cpu_process_group
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from mpi4py import MPI
import numpy as np


class TestSciml(unittest.TestCase):
    """Unit test for SciML serial
    """
    def test_partioned_dataset(self):
        """Test data set partition
        i.e. if number of partition < total no. of data
        """
        world_comm = MPI.COMM_WORLD
        input_data = torch.rand(22, 7)
        output_data = torch.rand(22, 1)
        indices = np.arange(world_comm.rank, 22, world_comm.size)
        customDataset = CustomPartitionedDataset(input_data, output_data,
                                                 indices)
        input_data_transformed = customDataset.some_reverse_transformation(
            customDataset.some_transformation(input_data))
        assert indices.shape[0] < 22
        torch.testing.assert_close(input_data_transformed, input_data)

    def test_train_val_distributed(self):
        """training validation test
        i.e. tests whether training results
        in reduction of loss function value
        """
        world_comm = MPI.COMM_WORLD
        init_cpu_process_group(world_comm)
        torch.manual_seed(60)
        input_data_torch = torch.rand(100, 5)
        output_data_torch = torch.sin(torch.rand(100, 2)) * 0.5 + 0.5
        indices_train = np.arange(world_comm.rank, 75, world_comm.size)
        indices_val = np.arange(world_comm.rank, 25, world_comm.size)

        trainDataset = CustomPartitionedDataset(input_data_torch[:75, :],
                                                output_data_torch[:75, :],
                                                indices_train)
        train_dataloader = DataLoader(trainDataset, batch_size=20)

        validDataset = CustomPartitionedDataset(input_data_torch[75:, :],
                                                output_data_torch[75:, :],
                                                indices_val)
        valid_dataloader = DataLoader(validDataset, batch_size=25)

        net = Net(5, list(), 2, torch.sigmoid,
                  device="cpu", dtype=torch.float32)

        for param in net.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

        training_loss_list = list()
        validation_loss_list = list()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(),
                                     lr=1e-5)
        for epoch in range(5):
            training_loss_list.append(
                train_nn(train_dataloader, net, loss_fn,
                         optimiser, report=False))
            validation_loss_list.append(
                valid_nn(valid_dataloader, net,
                         loss_fn, report=True))

        for i in range(1, len(validation_loss_list)):
            assert validation_loss_list[i] <= validation_loss_list[i - 1]


if __name__ == "__main__":
    unittest.main()
