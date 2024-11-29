import torch


class CustomDataset(torch.utils.data.Dataset):
    """ Custom dataset creation for neural network training

    Args:
        torch (Tensor): Input data
        torch (Tensor): Output data
    """
    def __init__(self, input_data, output_data):
        """__init__ method

        Args:
            input_data (torch Tensor): Input data (Features)
            output_data (torch Tensor): Output data (Label)
        """
        self.input_data = self.some_transformation(input_data)
        self.output_data = self.some_transformation(output_data)

    def __len__(self):
        """Get number of data points

        Returns:
            int: Number of data points
        """
        return self.input_data.shape[0]

    def __getitem__(self, index):
        """Get tensor at a given index

        Args:
            index (Integer): Index of data point

        Returns:
            _type_: _description_
        """
        return self.input_data[index, :], self.output_data[index, :]

    def some_transformation(self, input_data):
        """Some transformaion before passing data to Neural network

        Args:
            input_data (torch Tensor): Input tensor

        Returns:
            torch Tensor: Transformed tensor
        """
        return input_data

    def some_reverse_transformation(self, input_data):
        """Revert the transformation

        Args:
            input_data (torch Tensor): Transformed tensor

        Returns:
            torch Tensor: Original tensor
        """
        return input_data
