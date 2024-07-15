import data_corpus as data_corpus
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

dataset = data_corpus.create_dataset()

text_batch = []
n_files = 0
for sentence in dataset:
    text_batch.append(sentence[0])
    if len(text_batch) == 10000:
        with open(f'./util/tokenizer_batch/batch_{n_files}.txt', 'w') as file:
            file.write('\n'.join(text_batch))
        n_files += 1
        text_batch = []
        
paths = [str(x) for x in Path('./util/tokenizer_batch').glob("*.txt")]     

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train( 
    files=paths,
    vocab_size=30_000, 
    min_frequency=5,
    limit_alphabet=1000, 
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )