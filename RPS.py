# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import random
import tensorflow as tf


#import keras.api._v2.keras as keras
# K = tensorflow._tf_uses_legacy_keras.backend
# KL = tensorflow._tf_uses_legacy_keras.layers
# Lambda, Input, Flatten = KL.Lambda, KL.Input, KL.Flatten
# Model = tensorflow._tf_uses_legacy_keras.Model
# from K.models import Sequential
from tensorflow import python
from tensorflow import zeros_initializer
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import initializers

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
        #retrain model every 100 plays?
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
BATCH_SIZE = 2;
VOCAB_SIZE=len(vocab);
EMBEDDING_DIM=256
RNN_UNITS=128;
BUFFER_SIZE=1000;
sequences = char_dataset.batch(BATCH_SIZE);
print("len of sequences: ",len(sequences))
# print(sequences[0:3])

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[2:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
# for x, y in dataset.take(5):
#   print("\n\nEXAMPLE\n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\nOUTPUT")
#   print(int_to_text(y))
#   print(dataset)
#
data = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE);
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
###Build Model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model_b = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_shape=(batch_size,)),
        # ( ) (vocab_size, embedding_dim,
        #                            batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=False,#stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size ),
    ])
    return model_b

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

###Create Loss Function
##See Prediction
for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape
  print(target_example_batch.shape, "# (batch_size, sequence_length, vocab_size)")


pred = example_batch_predictions[0]
print(len(pred))
print(pred)

# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabillity)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars  # and this is what the model predicted for training sequence 1
print("Predicted chars: ", predicted_chars)

##Define Loss Function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

###Compile Model
model.compile(optimizer='adam', loss=loss)

###Create Checkpoint
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "/", "ckpt_{epoch}.weights.h5")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

###Training
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

###Loading Model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))

###'Generating Text'(Make RPS choice!!!)
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))