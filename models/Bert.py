import torch
from models.encoder import Encoderlayer
from utils.preProcessing import InputEmbedding

class BERT(torch.nn.Module):
    def __init__(self, vocab_size, d_model, nheads, dropout, n_encoder_layers, seq_len, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nheads = nheads
        self.dropout = dropout
        self.n_encoder_layers = n_encoder_layers
        self.d_feed_forward = d_model*4
        self.device = device
        
        self.Encoder = torch.nn.ModuleList([Encoderlayer(self.d_model, self.nheads, self.d_feed_forward, self.dropout) for _ in range(self.n_encoder_layers)])
        self.embedding = InputEmbedding(self.vocab_size, self.d_model, seq_len, self.dropout, device)
        
    def forward(self, sequence, segment_label):
        # sequence shape is (batch_size, seq_len=max_len=64)
        mask = (sequence > 0) # mask is True for all elements of the sequence that are not padding
        print(mask.shape)
        sequence = self.embedding(sequence, segment_label)
        for encoderlayer in self.Encoder:
            sequence = encoderlayer(sequence, mask)
        return sequence
    
# Reminder : The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood.
class NSP(torch.nn.Module):
    def __init__(self, d_model=768):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        
    def forward(self, input):
        return self.softmax(self.linear(input))

class MLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model=768):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        
    def forward(self, input):
        return self.softmax(self.linear(input))

class BERT_NSP_MLM(torch.nn.Module):
    def __init__(self, bert: BERT, vocab_size, d_model):
        super().__init__()
        
        self.bert = bert
        self.MLM = MLM(vocab_size, d_model)
        self.NSP = NSP(d_model)
        
    def forward(self, sequence, segment_label):
        output = self.bert(sequence, segment_label)
        return self.NSP(output), self.MLM(output)
