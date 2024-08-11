import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        # Sort the sequences by length in descending order
        text_lengths, sort_idx = text_lengths.sort(descending=True)
        embedded = embedded[sort_idx]

        # Pack the sorted sequences
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # Unsort the sequences
        _, unsort_idx = sort_idx.sort()
        hidden = hidden[unsort_idx]

        return self.fc(hidden)


def build_model(vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True,
                dropout=0.5):
    return LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)


def tokenize_and_pad(texts, word_to_idx, max_length):
    tokenized = [[word_to_idx.get(word, word_to_idx['<unk>']) for word in text.split()] for text in texts]
    lengths = [min(len(t), max_length) for t in tokenized]
    padded = [t[:max_length] + [word_to_idx['<pad>']] * (max_length - len(t)) for t in tokenized]
    return padded, lengths