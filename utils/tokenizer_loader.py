from transformers import BertTokenizer

# Load tokenizer we have trained from path
def load_tokenizer(path):
    
    return BertTokenizer.from_pretrained(path, local_files_only=True)