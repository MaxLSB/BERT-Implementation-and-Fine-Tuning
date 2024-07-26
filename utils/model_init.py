import torch

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.zeros_(m.bias.data) 