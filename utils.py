import torch
import pickle
from config import Config
from tokenizers import ByteLevelBPETokenizer

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return {"X": x, "y": y}

def load_training_data():
    with open("./data/encoded_train_data.pkl", "rb") as f:
        all_ids = pickle.load(f)

    train_data = torch.tensor(all_ids, dtype=torch.long)
    return train_data

def load_test_data():
    with open("./data/encoded_test_data.pkl", "rb") as f:
        all_ids = pickle.load(f)

    test_data = torch.tensor(all_ids, dtype=torch.long)
    return test_data
    

def load_dataset():
    train_ids = load_training_data()
    test_ids = load_test_data()

    train_dataset = GPTDataset(train_ids, Config.block_size)
    test_dataset = GPTDataset(test_ids, Config.block_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.batch_size)

    return train_loader, test_loader

def text2embedding(input):
    tokenizer = ByteLevelBPETokenizer("./tokenizer/vocab.json", "./tokenizer/merges.txt")
    ids = tokenizer.encode(input).ids
    return torch.tensor([ids], dtype=torch.long).to(Config.device)

def embedding2text(ids):
    tokenizer = ByteLevelBPETokenizer("./tokenizer/vocab.json", "./tokenizer/merges.txt")
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze().tolist()
    return tokenizer.decode(ids)

def generate_text(
        model: torch.nn.Module, 
        max_tokens: int,
        input_tensor
    ):
    model = model.to(Config.device)

    output_ids = model.generate(input_tensor.to(Config.device), max_tokens=max_tokens)[0]
    output = embedding2text(output_ids)

    return output

def load_best_model(model, checkpoint_dir):
    """
    Loads the model checkpoint with the lowest validation loss.
    """
    print("Searching for best saved model...")
    best_model = None
    best_loss = float('inf')

    for filename in os.listdir(checkpoint_dir):
        try:
            loss_str = filename.split("_")[-1].split(".pt")[0]
            current_loss = float(loss_str)
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = filename
        except Exception as e:
            print(f"Skipping file {filename}: {e}")

    if best_model:
        path = os.path.join(checkpoint_dir, best_model)
        print(f"Loading model from: {path}")
        model.load_state_dict(torch.load(path, map_location=Config.device))
        model.to(Config.device)
    else:
        print("No valid checkpoint found.")

    return model