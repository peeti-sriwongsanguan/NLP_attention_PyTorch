import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from package.data_processing import load_and_preprocess_data, balance_dataset, preprocess_text
from package.model import build_model, tokenize_and_pad
from pytorch_model import timing_decorator



if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)


def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch[0], batch[1]
        predictions = model(text.to(device), text_lengths).squeeze(1)
        loss = criterion(predictions, batch[2].to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch[0], batch[1]
            predictions = model(text.to(device), text_lengths).squeeze(1)
            loss = criterion(predictions, batch[2].to(device))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

@timing_decorator
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataframe_path = os.path.join(script_dir, 'data', 'reviews_Automotive_5.json.gz')

    if not os.path.exists(dataframe_path):
        print(f"Error: File not found at {dataframe_path}")
        return

    review_df = load_and_preprocess_data(dataframe_path)

    # Balance dataset
    sample_df = balance_dataset(review_df)
    sample_df = preprocess_text(sample_df)

    # Tokenize and build vocabulary
    global tokenizer
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(sample_df['reviewText']), specials=['<unk>', '<pad>'])
    vocab.set_default_index(vocab['<unk>'])

    # Tokenize and pad sequences
    max_length = 100  # You might want to adjust this based on your data
    texts, lengths = tokenize_and_pad(sample_df['reviewText'], tokenizer, max_length)
    texts = [vocab(t) for t in texts]
    X = torch.LongTensor(texts)
    lengths = torch.LongTensor(lengths)
    y = torch.FloatTensor(sample_df['positive'].values)

    # Split data
    X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
        X, y, lengths, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_data = TensorDataset(X_train, lengths_train, y_train)
    test_data = TensorDataset(X_test, lengths_test, y_test)
    train_iterator = DataLoader(train_data, batch_size=32, shuffle=True)
    test_iterator = DataLoader(test_data, batch_size=32)

    # Build model
    model = build_model(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    # Train model
    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion)
        test_loss = evaluate(model, test_iterator, criterion)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Test Loss: {test_loss:.3f}')

    # Evaluate model
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f}')


if __name__ == "__main__":
    main()