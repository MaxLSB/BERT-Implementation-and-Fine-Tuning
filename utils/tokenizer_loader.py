from transformers import BertTokenizer


def load_tokenizer(path):
    
    return BertTokenizer.from_pretrained(path, local_files_only=True)