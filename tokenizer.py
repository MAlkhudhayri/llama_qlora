from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from datasets import load_from_disk
tokenizer = AutoTokenizer.from_pretrained("/hdd4/zoo/llama2/llama2-7b-hf")

dataset = load_from_disk('/hdd3/mohammed/llama_qlora/llama_format_feb27.dat')
df = pd.DataFrame(dataset)

print(df.isna().sum())
print(df)
column=df['output']

tokenized = [tokenizer.tokenize(s) for s in column]
unique_tokens = set(token for sublist in tokenized for token in sublist)

vocab_size = 32000  
one_hot_vector = np.zeros(vocab_size, dtype=int)
for token in unique_tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id < vocab_size:
        one_hot_vector[token_id] = 1

max_tokens = max(len(t) for t in tokenized)
print("One hot vectro:, ", one_hot_vector)
print("Length, ",len(one_hot_vector))
print("Number of 1's, ", sum(one_hot_vector))
print("Max number of tokens in a string:", max_tokens)

