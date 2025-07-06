import pandas as pd
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer
import torch
from config import Config

data_file_path = './kagglehub/datasets/razanaqvi14/real-and-fake-news/versions/1'
TRAIN_OUTPUT_PATH = "./data/train_corpus.txt"
TEST_OUTPUT_PATH = "./data/test_corpus.txt"

def cleansing_text(text):
        pattern = r'(^[A-Z/]+\s+)?\(Reuters\)\s* - '
        return re.sub(pattern, '', text)

def preprocess_data_for_tokenizer():
    sample_df = pd.read_csv(data_file_path+"/True.csv")

    politicsNews = sample_df[sample_df['subject']=='politicsNews']
    politicsNews_train, politicsNews_test = train_test_split(politicsNews, test_size=0.05, random_state=42)
    worldnews = sample_df[sample_df['subject']=='worldnews']
    worldnews_train, worldnews_test = train_test_split(worldnews, test_size=0.05, random_state=42)

    train_df = pd.concat([politicsNews_train, worldnews_train], ignore_index=True)
    test_df = pd.concat([politicsNews_test, worldnews_test], ignore_index=True)
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")

    os.makedirs('data', exist_ok=True)
    with open(TRAIN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            text = str(cleansing_text(row["text"])).strip()
            title = row["title"]
            combined = f"<s> {text} <sep> {title} </s>\n"
            f.write(combined)
        
    with open(TEST_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            text = str(cleansing_text(row["text"])).strip()
            title = row["title"]
            combined = f"<s> {text} <sep> {title} </s>\n"
            f.write(combined)


def train_tokenizer():
    tokenizer = ByteLevelBPETokenizer()

    SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    tokenizer.train(
        files=[TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH],
        vocab_size=Config.VOCAB_SIZE,
        min_frequency=Config.MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS
    )

    save_path = "tokenizer"
    os.makedirs(save_path, exist_ok=True)

    tokenizer.save_model(save_path)


def encode_data(target):
    tokenizer = ByteLevelBPETokenizer("./tokenizer/vocab.json", "./tokenizer/merges.txt")

    if target == 'train':
        target_data_path = TRAIN_OUTPUT_PATH
        target_save_path = "./data/encoded_train_data.pkl"
    else:
        target_data_path = TEST_OUTPUT_PATH
        target_save_path = "./data/encoded_test_data.pkl"

    all_token_ids = []

    with open(target_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line).ids
            all_token_ids.extend(ids + [tokenizer.token_to_id("</s>")])  # add end-of-sequence marker
            
    # Save Encoded IDs for Training
    with open(target_save_path, "wb") as f:
        pickle.dump(all_token_ids, f)


def text2embedding(input):
    tokenizer = ByteLevelBPETokenizer("./tokenizer/vocab.json", "./tokenizer/merges.txt")
    ids = tokenizer.encode(input).ids
    return torch.tensor([ids], dtype=torch.long).to(Config.device)

def embedding2text(ids):
    tokenizer = ByteLevelBPETokenizer("./tokenizer/vocab.json", "./tokenizer/merges.txt")
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze().tolist()
    return tokenizer.decode(ids)