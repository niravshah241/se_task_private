import torch

# Artificial Neural Network


class Net(torch.nn.Module):
    """Initialise a neural network
    """
    def __init__(self, dim_in, dim_hidden_layers, dim_out,
                 activation_function, dtype=torch.float64,
                 device="cuda:0"):
        """__init__ method

        Args:
            dim_in (integer): Number of features in input
            dim_hidden_layers (list of integers): Neurons in hidden layer
            dim_out (integer): Number of labels in output
            activation_function (torch function): Activaion function
            dtype (Data type, optional): Precision. Defaults to torch.float64.
            device (Device, optional): Defaults to "cuda:0".
        """
        super().__init__()
        linear_layers = torch.nn.ModuleList()
        ann_dims = torch.nn.ModuleList()
        ann_dims = dim_hidden_layers
        ann_dims.insert(0, dim_in)
        ann_dims.insert(len(ann_dims), dim_out)
        del dim_hidden_layers
        for i in range(len(ann_dims) - 1):
            linear_layers.append(
                torch.nn.Linear(ann_dims[i], ann_dims[i+1])
            ).to(device=device).to(dtype=dtype)
        self.linear_layers = linear_layers
        self.activation_function = activation_function

    def forward(self, result):
        """Forward pass

        Args:
            result (torch Tensor): Input data

        Returns:
            torch Tensor: Output after neural network inference
        """
        # Return the result of forward pass
        linear_layers = self.linear_layers
        for i in range(len(linear_layers)):
            result = \
                self.activation_function(self.linear_layers[i](result))
        return result
