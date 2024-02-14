import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.kears.processing import sequence

import os
import numpy as np
import pandas as pd

filepath = tf.keras.utils.get_file('shakespeare.txt', '')

text = ''
with open(filepath, 'rb') as reader:
    text = reader.read().decode(encoding='utf-8')

print(text[:250])

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass

    return ''.join(idx2char[ints])

text_as_int = text_to_int(text)

print(f'Text: {text[:13]}')
print(f'Encoded: {text_to_int(text[:13])}')

SEQ_LEN = 100
EXEMPLES_PER_EPOCH = len(text) // (SEQ_LEN + 1)

char_dataset = tf.data.Dataset.from_slices(text_as_int)

sequences = char_dataset.batch(SEQ_LEN + 1, drop_remainder = True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
    print(f'Exemples')

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10_000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, batch_input_shape=[BATCH_SIZE, None]),
    layers.LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    layers.Dense(VOCAB_SIZE)
])

#print(model.summary())

def loss(labels, logits):
    return tf.keras.losses.sparce_categorical_crossentropy(labels, logits, from_logits=True)