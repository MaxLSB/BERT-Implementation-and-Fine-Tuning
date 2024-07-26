import random
from torch.utils.data import Dataset
import torch
import math

class BERTDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.vocab_size = self.tokenizer.vocab_size
       
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
       
        sentenceA, sentenceB, is_next = self.get_label_element(item)
        
        # We mask the tokens in the sentence and get the corresponding mask_ids
        masked_sentenceA, mask_idsA = self.get_masked(sentenceA)
        masked_sentenceB, mask_idsB = self.get_masked(sentenceB)
        
        # We concatenata the two sentences for each embedding (we don't take into account the [CLS] )
        token_embeddings = (masked_sentenceA + masked_sentenceB[1:])[:self.seq_length]
        mask_ids = (mask_idsA + mask_idsB[1:])[:self.seq_length]
        segment_embeddings = ([1]*len(masked_sentenceA) + [2]*len(masked_sentenceB[1:]))[:self.seq_length]
        
        # We pad the embeddings to the max length if needed otherwise padding = [] and nothing changes
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_length - len(token_embeddings))]
        token_embeddings.extend(padding)
        mask_ids.extend(padding)
        segment_embeddings.extend([0]*len(padding))
        
        output = {"token_embeddings" : token_embeddings,
                  "segment_embeddings" : segment_embeddings,
                  "mask_ids" : mask_ids,
                  "is_next" : is_next}
        # We convert the values to torch tensors
        return {key: torch.tensor(value) for key, value in output.items()}
        
    def get_masked(self, sentence):
        # tokens is the corresponding tokenized sentence
        tokens = self.tokenizer(sentence, max_length=self.seq_length, truncation=True)['input_ids']
        mask_ids = [0]*len(tokens)
        # We don't want to mask the [CLS] and [SEP] tokens so we donc take in count the first and last token
        for i in range(1,len(tokens)-1):
            # Masking 15% of the tokens
            if random.random() < 0.15:
                value = random.random()
                if value < 0.8:
                    # 80% of the time, replace with [MASK]
                    mask_ids[i] = tokens[i]
                    tokens[i] = self.tokenizer.vocab['[MASK]']
                elif value < 0.9:
                    # 10% of the time, replace with random token
                    mask_ids[i] = tokens[i]
                    tokens[i] = random.randint(4, self.vocab_size - 1)
                
        return tokens, mask_ids
                    
    def get_label_element(self, index):
        sentenceA, sentenceB = self.dataset[index][0], self.dataset[index][1]
        
        # 50% of the time, we return unrelated sentences
        if random.random() > 0.5:
            sentenceB = self.get_random_element()
            return sentenceA, sentenceB, 0
        
        return sentenceA, sentenceB, 1
            
    def get_random_element(self):
        index = random.randint(0, len(self.dataset)-1)
        return self.dataset[index][1]


class PositionalEmbedding(torch.nn.Module):
    # d_model is the hidden size for BERT : 768 (from the article)
    def __init__(self, d_model, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # pe is the positional embedding of size 64x768
        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.requires_grad = False
        
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/self.d_model)))
        
        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        return self.pe
    
    
# Initialize the embedding vectors for all the tokens of the vocabulary
class InputEmbedding(torch.nn.Module):
    # embed_size is the hidden size for BERT : 768 (from the article)
    def __init__(self, vocab_size, embed_size, seq_len, dropout, device):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.device = device
        
        # Note : self.token(token_index) returns the embedding vector of the token_index
        
        # Initially all the embedding vectors weights are random and will be updated during the training process
        # padding_ids to designate the padding tokens which will remain zeros
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, sequence, segment_label):
        # self.token(sequence) creates a tensor that contains the embedding vectors for all the tokens in the input sequence.
        # self.token gathers the embedding vectors of all the vocabulary tokens
        # self.segment gathers the embedding vectors of the 3 possible segment labels
        # self.position gathers the positional embedding vectors of the 64 possible positions
        embedded_token = self.token(sequence).to(self.device)
        embedded_seg = self.segment(segment_label).to(self.device)
        embbeded_pos = self.position(sequence).to(self.device)
        
        # The input embedding is the sum of the token embedding, the positional embedding and the segment embedding
        x = embedded_token + embbeded_pos + embedded_seg
        # We apply dropout to deactivate some embedding features within the embedding vector for each token.
        return self.dropout(x)
