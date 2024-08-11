# import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
def tokenize_and_pad(texts, max_features=2000):
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences), tokenizer

def build_model(max_features, embed_dim, lstm_out, input_length):
    model = Sequential([
        Embedding(max_features, embed_dim, input_length=input_length),
        SpatialDropout1D(0.2),
        LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model