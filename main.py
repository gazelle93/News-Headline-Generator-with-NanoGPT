import os
import torch

import data_preprocessing
import utils
import train_and_test

from model import GPT
from config import Config

def main():
    checkpoint_dir = "./results"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # DATA PREPROCESSING (Uncomment if needed)
    # print("Preprocessing data for tokenizer...")
    # data_preprocessing.preprocess_data_for_tokenizer()
    #
    # print("Training tokenizer...")
    # data_preprocessing.train_tokenizer()
    #
    # print("Encoding training data...")
    # data_preprocessing.encode_data('train')
    #
    # print("Encoding test data...")
    # data_preprocessing.encode_data('test')

    # Add Condition of parameters
    model = GPT()

    # Load or train
    load_existing = True
    if load_existing:
        print("Loading the saved model...")
        model = utils.load_best_model(model, checkpoint_dir)

    else:
        # LOAD DATA
        print("Loading datasets...")
        train_loader, test_loader = utils.load_dataset()
        print(f"Training batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")

        # Training loop
        epochs = 3
        best_val_loss = float('inf')

        print("Starting training...")
        for epoch in range(epochs):
            train_info = train_and_test.train(train_loader, model)
            print(f"Epoch {epoch + 1}: Train loss {train_info['loss']:.2f}, time {train_info['time']:.2f}s")

            test_info = train_and_test.eval(test_loader, model)
            print(f"Epoch {epoch + 1}: Test loss {test_info['loss']:.2f}, time {test_info['time']:.2f}s")

            # Save best checkpoint
            if test_info['loss'] < best_val_loss:
                best_val_loss = test_info['loss']
                save_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_loss_{best_val_loss:.2f}.pt")
                torch.save(model.state_dict(), save_path)
                print(f"Saved improved model to {save_path}")

        # GENERATE A TITLE
        print("Generating title from given news text...")
        prompt = (
            "<s> The United States and China will work together to get nuclear-armed North Korea "
            "take “a different course”, U.S. Secretary of State Rex Tillerson said on Saturday, "
            "softening previous criticism of Beijing after talks with his Chinese counterpart. "
            "China has been irritated at being repeatedly told by Washington to rein in North Korea’s "
            "surging nuclear and ballistic missile programmes, one of a series of hurdles in ties between "
            "the world’s two largest economies. <sep>"
        )

        input_tensor = data_preprocessing.text2embedding(prompt)
        generated = utils.generate_text(model, 128, input_tensor)

        # Extract and print result
        news_body = prompt[prompt.find('<s>') + len('<s>'):prompt.find('<sep>')].strip()
        generated_title = generated[generated.find('<sep>') + len('<sep>'):generated.find('</s>')].strip()

        print("\n========== GENERATED EXAMPLE ==========")
        print(f"Input News Body:\n{news_body}\n")
        print(f"Generated Title:\n{generated_title}")
        print("=======================================")


if __name__ == "__main__":
    main()