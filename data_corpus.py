import ast
import pandas as pd
import json

# In the paper, combined_length = 512 here we take 128 because we have less data

max_length = 64
movies_lines = "./data/movie_lines.txt"
movies_conversations = "./data/movie_conversations.txt"


with open(movies_lines, 'r') as file:
    lines = file.readlines()

with open(movies_conversations, 'r') as file:
    conversations = file.readlines()

# We create a dictionary with the line ids to the actual text
dict = {} 
for line in lines:
    line = line.split(" +++$+++ ")
    dict[line[0]] = line[-1]
dataset = []

# We iterate over the conversations and extract the text thanks to the ids to create the dataset
for conversation in conversations:
    ids = ast.literal_eval(conversation.split(" +++$+++ ")[-1])
    length = len(ids)
    for i in range(length):
        if i+1 == length:
            break
        # BERT takes 2 sentences as input for NSP so we make pairs
        # We take the successive pairs of sentences with a max length of 64 and a combined max length of 128
        sentenceA = dict[ids[i]].strip().split()[:max_length]
        sentenceB = dict[ids[i+1]].strip().split()[:max_length]
        dataset.append([' '.join(sentenceA), ' '.join(sentenceB)])

# We save the dataset as a json file
with open('./data/processed_data.json', 'w') as f:
        json.dump(dataset, f)


        
        
