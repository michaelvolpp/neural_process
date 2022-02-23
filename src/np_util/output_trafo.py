import torch

softplus_stiffness = 1.0

def output_trafo(output, lower_bound):
    return lower_bound + torch.nn.Softplus(beta=softplus_stiffness)(output)

def inv_output_trafo(x, lower_bound):
    assert (x > lower_bound).all(), "x has to lie in the image of output_trafo!"
    k = softplus_stiffness
    x = x - lower_bound
    return x + 1 / k * torch.log(-torch.expm1(-k * x))