from pandas import read_csv
from tokenizers import BertWordPieceTokenizer
from utils.data_loader import load_data
import tqdm

def create_tokenizer():
    return BertWordPieceTokenizer(handle_chinese_chars=False, strip_accents=False)

def batch_iterator(dataset, batch_size=10000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def train_tokenizer(dataset, tokenizer):
    for batch in tqdm.tqdm(batch_iterator(dataset)):
        tokenizer.train_from_iterator(batch)

def save_tokenizer(tokenizer, save_path):
    tokenizer.save_model(save_path)

def main():
    
    dataset = load_data('./data/processed_data.json')

    # Create and train tokenizer
    tokenizer = create_tokenizer()
    train_tokenizer(dataset, tokenizer)
    
    # Save the trained tokenizer
    save_tokenizer(tokenizer, './tokenizers')

if __name__ == "__main__":
    main()
