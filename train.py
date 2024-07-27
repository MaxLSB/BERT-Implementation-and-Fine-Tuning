import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.data_loader import load_data
from utils.preProcessing import BERTDataset
from utils.tokenizer_loader import load_tokenizer
from models.bert import BERT, BERT_NSP_MLM
from utils.model_init import initialize_weights


class BERTTraining:
    def __init__(self, model, lr, weight_decay, betas, log_freq, device):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.log_freq = log_freq
        self.device = device

        # ignore_index=0: ignore 'non masked' and padded tokens during loss calculation
        self.criterionNSP = torch.nn.NLLLoss()
        self.criterionMLM = torch.nn.NLLLoss(ignore_index=0)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)
        
        print("Number of Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def train(self, epoch, data_loader):
        avg_loss = 0
        total_correct_NSP = 0
        total_element_NSP = 0
        total_correct_MLM = 0
        total_element_MLM = 0
        
        data_iter = enumerate(data_loader)
        
        for i, data in data_iter:
            
            token_embedding = data["token_embeddings"].to(self.device)
            segment_embedding = data["segment_embeddings"].to(self.device)
            is_next = data["is_next"].to(self.device)
            mask_ids = data["mask_ids"].to(self.device)

            NSP_output, MLM_output = self.model(token_embedding, segment_embedding)
            
            NSP_loss = self.criterionNSP(NSP_output, is_next)
            
            # (batch_size,vocab_size,seq_len) 
            MLM_loss = self.criterionMLM(MLM_output.transpose(1,2), mask_ids).float().mean()
            loss = NSP_loss + MLM_loss
            avg_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   
                
            correct_NSP = (NSP_output.argmax(dim=-1)).eq(is_next).sum().item()
            total_correct_NSP += correct_NSP
            total_element_NSP += is_next.nelement()
            
            MLM_pred = MLM_output.argmax(dim=-1)

    
            mask = mask_ids != 0
            correct_MLM = (MLM_pred[mask] == mask_ids[mask]).sum().item()
            total_correct_MLM += correct_MLM
            total_element_MLM += mask.sum().item()
            
            if i % self.log_freq == 0:
                print(f"AvgLoss: {avg_loss /(i+1)} | AccuracyNSP: {total_correct_NSP/total_element_NSP * 100} | AccuracyMLM: {total_correct_MLM/total_element_MLM * 100} | Loss: {loss.item()}")
        
        print(f"Epoch {epoch} | Total-Loss: {avg_loss / len(data_loader)} | AccuracyNSP: {total_correct_NSP/total_element_NSP * 100} | AccuracyMLM: {total_correct_MLM/total_element_MLM * 100}")
        

def main(epochs, batch_size, d_model, n_heads, dropout, n_encoder_layers, seq_len, device, path_data, path_tokenizer):
    
    dataset = load_data(path_data)
    tokenizer = load_tokenizer(path_tokenizer)
    vocab_size = tokenizer.vocab_size
    
    train_data = BERTDataset(dataset, tokenizer, seq_len)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    model = BERT(vocab_size, d_model, n_heads, dropout, n_encoder_layers, seq_len, device).to(device)
    bert = BERT_NSP_MLM(model, vocab_size, d_model).to(device)
    # bert.apply(initialize_weights)
    trainer = BERTTraining(bert, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), log_freq=500 , device=device)
    

    for epoch in range(epochs):
        trainer.train(epoch, train_loader)
    
    torch.save(bert.state_dict(), 'trained_model/bert.pth')
    print('Training completed')
                
    
if __name__ == "__main__":
    
    # BERT-Tiny
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 30
    batch_size = 32
    d_model = 128
    n_heads = 2
    dropout = 0.1
    n_encoder_layers = 4
    seq_len = 64
    path_data = "data/processed_data.json"
    path_tokenizer = "tokenizers/vocab.txt"
    
    main(epochs, batch_size, d_model, n_heads, dropout, n_encoder_layers, seq_len, device, path_data, path_tokenizer)