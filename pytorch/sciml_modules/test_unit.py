import unittest
from dataset import CustomDataset
from neural_net import Net
from train_validate import train_nn, valid_nn
import torch
from torch.utils.data import DataLoader


class TestSciml(unittest.TestCase):
    """Unit test for SciML serial
    """
    def test_dataset(self):
        """Test data set
        i.e. if transformation and reverse transformation
        results in original dataset
        """
        input_data = torch.rand(5, 7)
        output_data = torch.rand(5, 1)
        customDataset = CustomDataset(input_data, output_data)
        input_data_transformed = customDataset.some_reverse_transformation(
            customDataset.some_transformation(input_data))
        torch.testing.assert_close(input_data_transformed, input_data)

    def test_neuralnet(self):
        """Test neural network
        i.e. Check whether number of neural netwrok parameters
        are as intended
        """
        net = Net(5, list(), 2, torch.sigmoid,
                  device="cpu", dtype=torch.float32)
        num_params = sum(params.numel()
                         for params in net.parameters())
        assert num_params == 5 * 2 + 2
        net = Net(5, [8, 4], 2, torch.sigmoid,
                  device="cpu", dtype=torch.float32)
        num_params = sum(params.numel()
                         for params in net.parameters())
        assert num_params == (8 * 5 + 8) \
            + (4 * 8 + 4) + (2 * 4 + 2)
        # num_params = weight params + bias params

    def test_train_val(self):
        """Training validation test
        i.e. tests whether training results
        in reduction of loss function value
        """
        torch.manual_seed(60)
        input_data_torch = torch.rand(100, 5)
        output_data_torch = torch.sin(torch.rand(100, 2)) * 0.5 + 0.5

        trainDataset = CustomDataset(input_data_torch[:75, :],
                                     output_data_torch[:75, :])
        train_dataloader = DataLoader(trainDataset, batch_size=60)

        validDataset = CustomDataset(input_data_torch[75:, :],
                                     output_data_torch[75:, :])
        valid_dataloader = DataLoader(validDataset, batch_size=25)

        net = Net(5, list(), 2, torch.sigmoid,
                  device="cpu", dtype=torch.float32)
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
