import torch

class EmotionClassifier(torch.nn.Module):
    def __init__(self, bert_model, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert_model = bert_model
        self.linear = torch.nn.Linear(768, num_classes)
    
    def forward(self, inputs, attention_mask):
        outputs = self.bert_model(inputs, attention_mask)
        # We only need the pooled output of the [CLS] token
        outputs = self.linear(outputs.pooler_output)
        return outputs