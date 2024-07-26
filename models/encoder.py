import torch

# feed forward size should be : 4*hidden_size
# From 'Attention is all you need' article
# BERT uses GELU activation function instead of ReLU
# BERT uses dropout of 0.1 on all layers

class Encoderlayer(torch.nn.Module):
    def __init__(self, d_model, nheads, d_feed_forward, dropout):
        super().__init__()
        
        self.layernorm = torch.nn.ModuleList([torch.nn.LayerNorm(d_model), torch.nn.LayerNorm(d_model)])
        self.multi_head_attention = torch.nn.MultiheadAttention(d_model, nheads)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_feed_forward),
            torch.nn.GELU(),
            torch.nn.Linear(d_feed_forward, d_model)
        )
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, input, mask):
        output = self.dropout(self.multi_head_attention(input, input, input, mask, need_weights=False))
        output = self.layernorm[0](output + input)
        input = output
        output = self.dropout(self.feed_forward(output))
        output = self.layernorm[1](output + input)
        return output
        
        
    