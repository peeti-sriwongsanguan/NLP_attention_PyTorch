import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from package.data_processing import load_and_preprocess_data, balance_dataset, preprocess_text
from package.pytorch_model import build_model, tokenize_and_pad
from package.time_record import timing_decorator

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

@timing_decorator
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch[0].to(device), batch[1].to(device)
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch[2].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

@timing_decorator
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch[0].to(device), batch[1].to(device)
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch[2].to(device))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

@timing_decorator
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataframe_path = os.path.join(script_dir, 'data', 'reviews_Automotive_5.json.gz')

    if not os.path.exists(dataframe_path):
        print(f"Error: File not found at {dataframe_path}")
        return

    review_df = load_and_preprocess_data(dataframe_path)
    sample_df = balance_dataset(review_df)
    sample_df = preprocess_text(sample_df)

    # Create vocabulary
    all_words = set(' '.join(sample_df['reviewText']).split())
    word_to_idx = {word: i + 2 for i, word in enumerate(all_words)}
    word_to_idx['<pad>'] = 0
    word_to_idx['<unk>'] = 1

    max_length = 500  # Increased max_length
    texts, lengths = tokenize_and_pad(sample_df['reviewText'], word_to_idx, max_length)

    # Remove sequences with length 0
    non_zero_indices = [i for i, length in enumerate(lengths) if length > 0]
    texts = [texts[i] for i in non_zero_indices]
    lengths = [lengths[i] for i in non_zero_indices]
    y = sample_df['positive'].iloc[non_zero_indices].values

    X = torch.LongTensor(texts)
    lengths = torch.LongTensor(lengths)
    y = torch.FloatTensor(y)

    X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
        X, y, lengths, test_size=0.2, random_state=42)

    train_data = TensorDataset(X_train, lengths_train, y_train)
    test_data = TensorDataset(X_test, lengths_test, y_test)
    train_iterator = DataLoader(train_data, batch_size=32, shuffle=True)
    test_iterator = DataLoader(test_data, batch_size=32)

    model = build_model(len(word_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    n_epochs = 10
    clip = 1.0  # Gradient clipping value

    for epoch in range(n_epochs):
        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        test_loss = evaluate(model, test_iterator, criterion)
        scheduler.step(test_loss)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Test Loss: {test_loss:.3f}')

    test_loss = evaluate(model, test_iterator, criterion)
    print(f'Final Test Loss: {test_loss:.3f}')


if __name__ == "__main__":
    main()