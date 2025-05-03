# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from keras.preprocessing import sequence

import os
import numpy as np
from tensorflow.python.keras.saving.saved_model.serialized_attributes import recurrent
from tensorflow.python.keras.saving.saved_model_experimental import sequential

VOCAB_SIZE = 3;
MAXLEN = 1;
BATCH_SIZE = 32;
text = "RPS";
dev_oppo_hist = ["R","P","S","R","P","S","R","P","S"]
vocab = dev_oppo_hist;

# print(keras.__version__)
def player(prev_play, opponent_history=[]):
    print("Playing {} against {}".format(prev_play, opponent_history))
    opponent_history.append(prev_play)

    guess = random.choice(["R", "P", "S"])
    if len(opponent_history) < 100:
        return guess
    else:

        return opponent_history[0]#begin model training, testing, predicting



# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)
vocab_as_int = text_to_int(vocab)
print("Text:", vocab)
print("Encoded:", text_to_int(vocab))

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print("int-to-text result: ", int_to_text(vocab_as_int))
seq_length=1
char_dataset = tf.data.Dataset.from_tensor_slices(vocab_as_int)
BATCH_SIZE = 1;
VOCAB_SIZE=len(vocab);
EMBEDDING_DIM=256
RNN_UNITS=128;
BUFFER_SIZE=1000;
sequences = char_dataset.batch(BATCH_SIZE);
print(sequences)
# print(sequences[0:3])

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[2:]
    return input_text, target_text

# dataset = sequences.map(split_input_target)
# for x, y in dataset.take(5):
#   print("\n\nEXAMPLE\n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\nOUTPUT")
#   print(int_to_text(y))
#   print(dataset)
#
# data = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE);
# for x, y in data.take(5):
#   print("\n\nEXAMPLE\n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\nOUTPUT")
#   print(int_to_text(y))
#   print(data)

# data = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE);

# guess = player("R", ["P"])
# print(guess)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model_b = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_size),
        # ( ) (vocab_size, embedding_dim,
        #                            batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model_b

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

