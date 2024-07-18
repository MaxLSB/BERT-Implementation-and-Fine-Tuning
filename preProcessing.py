from utils.tokenizer_loader import load_tokenizer
from utils.data_loader import load_data

import random
from torch.utils.data import Dataset

class BERTPreProcessing(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.vocab_size = len(self.tokenizer.vocab)
       
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
        
        return output
        
    def get_masked(self, sentence):
        # tokens is the corresponding tokenized sentence
        tokens = self.tokenizer(sentence, max_length=self.seq_length, truncation=True)['input_ids']
        mask_ids = [0]*len(tokens)
        # We don't take into account the [CLS] and [SEP] tokens
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


# dataset = load_data('data/processed_data.json')
# tokenizer = load_tokenizer('tokenizers/')
# Dataset = BERTPreProcessing(dataset, tokenizer)
# print(Dataset.__getitem__(0))
