# News-Headline-Generator-with-NanoGPT

This project is a simplified GPT-based model for generating **news article titles** from given news text bodies. It is inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).

## Objective

Given a body of news text, the model generates a relevant title. For example:

> **Input News Body:**
>
> The United States and China will work together to get nuclear-armed North Korea take “a different course”, U.S. Secretary of State Rex Tillerson said on Saturday, softening previous criticism of Beijing after talks with his Chinese counterpart. ... (full news text here)

> **Generated Title:**
>
> Trump trade adviser says China will pay more on North Korea's threat ahead of Trump’s visit

---

## Project Structure
```bash
.
├── data/                         # Encoded and preprocessed data
├── kagglehub/                    # Original raw dataset from Kaggle
├── results/                      # Saved model checkpoints
├── tokenizer/                    # Trained tokenizer files
├── config.py                     # Configuration settings
├── data_preprocessing.py         # Data cleaning, tokenizer training, encoding
├── main.py                       # Entry point for training, evaluation, generation
├── model.py                      # GPT model implementation
├── train_and_test.py             # Training and evaluation loops
├── utils.py                      # Helper functions (loading data, text generation)
└── requirements.txt              # Python dependencies
```


---

## Features

- Trains a simple GPT-style Transformer on paired news text and titles.
- Custom tokenizer (ByteLevel BPE).
- Supports checkpoint saving and resuming.
- Optional Mixture-of-Experts (MoE) feedforward layer for increased model capacity and conditional computation.
- Command-line overrides for configuration.
- Generates titles given new news text bodies.

---

## Setup

1. Clone the repository.

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset (e.g., the Kaggle CSV of real and fake news) in the appropriate location as defined in your preprocessing.

---

## Workflow

### 1. Data Preprocessing

Before training, you need to:

- **Clean and save raw text** for tokenizer training:

    Uncomment in `main.py`:

    ```python
    data_preprocessing.preprocess_data_for_tokenizer()
    ```

- **Train the tokenizer**:

    ```python
    data_preprocessing.train_tokenizer()
    ```

- **Encode the training and test sets**:

    ```python
    data_preprocessing.encode_data('train')
    data_preprocessing.encode_data('test')
    ```

These steps produce encoded datasets in text files (IDs) for model training.

---

### 2. Training

After preprocessing, train the model:

- In `main.py`, set:

    ```python
    load_existing = False
    ```

- Run:

    ```bash
    python main.py
    ```

Model checkpoints will be saved in the `./results/` directory with filenames showing epoch and loss.

---

### 3. Resuming / Loading Saved Model

To use the best checkpoint automatically:

- In `main.py`, set:

    ```python
    load_existing = True
    ```

- Run:

    ```bash
    python main.py
    ```

The code will find and load the model with the lowest loss in the results folder.

---

### 4. Generating Titles

After loading or training, the script will generate a title for a given news body.

Example from running:
```bash
========== GENERATED EXAMPLE ==========

Input News Body:
<s> The given news body input <sep>

Generated Title:
The generated title based on the given news body input.
```

## Mixture-of-Experts (MoE) Support
This project optionally supports Mixture-of-Experts (MoE) layers in the Transformer feedforward blocks. When enabled:
- Each feedforward layer contains multiple experts (MLPs).
- A learned gating network selects the top-k experts for each token.
- Enables conditional computation, scaling model capacity efficiently.

Enable MoE via command-line:
```bash
python main.py --isMoe --num_experts 8 --top_k 2
```

---

## Configuration

You can modify `config.py` for:
| Parameter        | Default (from config.py) | Description                              |
| ---------------- | ------------------------ | ---------------------------------------- |
| --vocab\_size    | 5000                     | Vocabulary size for tokenizer and model  |
| --min\_frequency | 2                        | Minimum frequency for BPE tokenizer      |
| --block\_size    | 256                      | Context window length                    |
| --emb\_dim       | 512                      | Embedding dimension                      |
| --num\_heads     | 4                        | Number of attention heads                |
| --num\_layers    | 1                        | Number of transformer blocks             |
| --dropout        | 0.1                      | Dropout rate                             |
| --dim\_expansion | 4                        | Feedforward dimension expansion ratio    |
| --bias           | False                    | Include bias in Linear layers            |
| --isMoe          | False                    | Use Mixture-of-Experts feedforward layer |
| --num\_experts   | 4                        | Number of experts in MoE                 |
| --top\_k         | 2                        | Top-k experts selected in MoE gating     |
| --initial\_lr    | 3e-4                     | Initial learning rate                    |
| --min\_lr        | 1e-4                     | Minimum learning rate                    |
| --batch\_size    | 450                      | Batch size for training                  |

Example:

```python
class Config:
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    VOCAB_SIZE = 5_000
    MIN_FREQUENCY = 2


    vocab_size = VOCAB_SIZE
    block_size = 256
    emb_dim = 512
    num_heads = 4
    num_layers = 1
    dropout = 0.1
    dim_expansion = 4
    bias = False

    isMoe = False
    num_experts = 4
    top_k = 2

    initial_lr = 3e-4
    min_lr = 1e-4

    batch_size = 450
```

## Credits
- Model architecture inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
- News data sample from [Kaggle - Real and Fake News Dataset](https://www.kaggle.com/datasets/razanaqvi14/real-and-fake-news).
