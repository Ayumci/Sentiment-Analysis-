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

# Data Preprocessing
le = LabelEncoder()
train_df['sentiment_encoded'] = le.fit_transform(train_df['sentiment'])
X = train_df['text']
y = to_categorical(train_df['sentiment_encoded'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and Padding
max_words = 10000  # Limit the vocabulary size
max_sequence_length = 200  # Limit the sequence length
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_test_data = test_df['text']
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Build the LSTM Model
embedding_dim = 100  # Dimensionality of word embeddings
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(2, activation='softmax'))  # 2 output classes (binary classification)

# Compile the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
plot_model(model, show_shapes=True)


# Train the Model
epochs = 100
batch_size = 512
history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, )


# Make Predictions
predictions = model.predict(X_test_pad)

# Convert the model's predictions back to sentiment labels
predicted_sentiments = [le.classes_[np.argmax(prediction)] for prediction in predictions]

# Add the predicted sentiments to the test DataFrame
test_df['predicted_sentiment'] = predicted_sentiments

# Display or save the results
print(test_df[['text', 'predicted_sentiment']])

test_df.to_csv("predicted.csv")


