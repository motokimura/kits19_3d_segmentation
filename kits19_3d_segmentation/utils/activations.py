from torch.nn import Sigmoid, Softmax


def get_activation(activation):
    """Get an activation module.

    Args:
        activation (str): name of the activation to get.

    Raises:
        ValueError:

    Returns:
        torch.nn.Module: activation module.
    """
    if activation == 'softmax':
        return Softmax(dim=1)
    elif activation == 'sigmoid':
        return Sigmoid()
    else:
        raise ValueError()
