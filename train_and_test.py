import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Union, Tuple
from model import GPT
from config import Config
from tqdm import tqdm

def train(
        train_loader, 
        model
    ) -> Dict[str, Union[torch.tensor, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.initial_lr)
    
    # lr scheduler
    lambda_func = lambda epoch: max(0.99 ** epoch, Config.min_lr / Config.initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    model = model.to(Config.device)

    start = time.time()
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, sample in tqdm(enumerate(train_loader)):
        X = sample["X"].to(Config.device)
        y = sample["y"].to(Config.device)
        
        logits = model(X)
        loss = Config.criterion(logits, y.view(-1,))
        losses[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    time_elapsed = time.time() - start
    train_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return train_info


def eval(
        test_loader, 
        model
    ) -> Dict[str, Union[torch.tensor, float]]:

    start = time.time()
    model.eval()
    losses = torch.zeros(len(test_loader))
    with torch.inference_mode():
        for i, sample in enumerate(test_loader):
            X = sample["X"].to(Config.device)
            y = sample["y"].to(Config.device)

            logits = model(X)
            loss = Config.criterion(logits, y.view(-1,))
            losses[i] = loss.item()
    time_elapsed = time.time() - start
    test_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return test_info