import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.data_loader import load_data
from utils.preProcessing import BERTDataset, BERT_NSP_MLM
from utils.tokenizer_loader import load_tokenizer
from models.bert import BERT

class BERTTraining:
    def __init__(self, model, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), log_freq=10):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.log_freq = log_freq

        # ignore_index=0: ignore the padding token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)
        
        # print("Number of Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def train(self, epoch, data_loader):
        avg_loss = 0
        total_correct_NSP = 0
        total_element_NSP = 0
        # total_correct_MLM = 0
        # total_element_MLM = 0
        
        data_iter = enumerate(data_loader)
        
        for i, data in data_iter:
            
            NSP_output, MLM_output = self.model(data["token_embedding"], data["segment_embedding"])

            NSP_loss = self.criterion(NSP_output, data["is_next"])
            # (batch_size,vocab_size,seq_len) 
            MLM_loss = self.criterion(MLM_output.transpose(1,2), torch.tensor(data["mask_ids"]))
    
            loss = NSP_loss + MLM_loss
            avg_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                
            correct_NSP = NSP_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            total_correct_NSP += correct_NSP
            total_element_NSP += data["is_next"].nelement()
            
            # correct = MLM_output.argmax(dim=-1).eq(torch.tensor(data["mask_ids"])).sum().item()
            # total_correct_MLM += correct
            # total_element_MLM += torch.tensor(data["mask_ids"]).nelement()
            
            if i % self.log_freq == 0:
                print(f"Epoch {epoch} | AvgLoss {avg_loss /(i+1)} | Accuracy {total_correct_NSP/total_element_NSP * 100} | Loss {loss.item()}")
        
        print(f"Epoch {epoch} | AvgLoss {avg_loss / len(data_loader)} | Accuracy {total_correct_NSP/total_element_NSP * 100}")
        

def main(epochs, batch_size, d_model, n_heads):
    
    dataset = load_data()
    tokenizer = load_tokenizer
    vocab_size = tokenizer.vocab_size
    
    train_data = BERTDataset(dataset, seq_len=64, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    model = BERT(vocab_size, d_model, n_heads, dropout, n_encoder_layers)
    bert = BERT_NSP_MLM(model, vocab_size, d_model)
    trainer = BERTTraining(bert, train_data, None, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), warmup_steps=10000, log_freq=10)
    

    for epoch in range(epochs):
        trainer.train(epoch, train_loader)
    
    print('Training completed')
                
        
if __name__ == "__main__":
    
    epochs = 10
    batch_size = 32
    d_model = 768
    n_heads = 12
    dropout = 0.1
    n_encoder_layers = 12
    
    main(epochs, batch_size, d_model, n_heads, dropout, n_encoder_layers)