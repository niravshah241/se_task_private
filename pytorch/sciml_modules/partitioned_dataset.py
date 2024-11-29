try:
    from .dataset import CustomDataset
except ImportError:
    from dataset import CustomDataset


class CustomPartitionedDataset(CustomDataset):
    """ Custom partitioned dataset creation
    for neural network training

    Args:
        torch (Tensor): Input data
        torch (Tensor): Output data
        torch or Numpy (array): Indices for partition
    """
    def __init__(self, input_data, output_data, indices):
        super().__init__(input_data[indices, :],
                         output_data[indices, :])

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        return self.input_data[index, :], self.output_data[index, :]
