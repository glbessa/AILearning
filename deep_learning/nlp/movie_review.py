import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

import os
import numpy as np
import pandas as pd

VOCAB_SIZE = 88584
MAX_LEN = 250
BATCH_SIZE = 64
EPOCHS = 10
VAL_SPLIT = 0.2
LOSS = 'binary_crossentropy'
OPTIMIZER = 'rmsprop'
METRICS=['acc']

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#print(train_data[0])
#print(train_labels)

# padding or cutting data
train_data = sequence.pad_sequences(train_data, MAX_LEN)
test_data = sequence.pad_sequences(test_data, MAX_LEN)

#print(train_data[0])

# creating model
# here we're trying to predict if it was a good or a bad review
model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE, 32),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

#model.summary()

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=VAL_SPLIT)

# saving history
df = pd.DataFrame(history)
df.to_csv('./stats.csv')

results = model.evaluate(test_data, test_labels)
print(results)

# saving model
model.save('./model.keras')

# making predictions
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAX_LEN)[0]

reverse_word_index = {value: key for key, value in word_index.items()}

def decode_integers(integers):
    pad = 0
    text = ""
    for num in integers:
        if num != pad:
            text += reverse_word_index[num] + " "

    return text[:-1]

def predict(text, model):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred)
    return result[0]

positive_review = ''
predict(positive_review, model)

negative_review = ''
predict(negative_review, model)

# this has to be tested yet