import torch

def initialize_weights(m):
    if isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
        if m.padding_idx is not None:
            m.weight.data[m.padding_idx].zero_()
    elif hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.zeros_(m.bias.data)