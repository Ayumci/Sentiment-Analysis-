import comet_ml
from comet_ml import Experiment
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
experiment = comet_ml.Experiment(
    api_key="Btyq9WqX6zq5ERVHuOMKKsopK",
    project_name="testing"
)
TRAIN_DIR = "train.csv"
TEST_DIR = "test.csv"
train_df = pd.read_csv(TRAIN_DIR)
test_df = pd.read_csv(TEST_DIR)
train_df.head(5)
labelencode = LabelEncoder()
train_df['sentiment_encoded'] = labelencode.fit_transform(train_df['sentiment'])
X = train_df['text']
y = to_categorical(train_df['sentiment_encoded'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
words = 10000  
max_sequence_length = 200  
tokenizer = Tokenizer(num_words=words)
tokenizer.fit_on_texts(X_train)
X_test_data = test_df['text']
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)
embedding_dim = 100  
model = Sequential()
model.add(Embedding(input_dim=words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax')) 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True)
epochs = 10
batch_size = 512
history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, )


