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
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import initializers


import os
import numpy as np
from tensorflow import raw_ops
from tensorflow.python.keras.saving.saved_model.serialized_attributes import recurrent
from tensorflow.python.keras.saving.saved_model_experimental import sequential

#VOCAB_SIZE = 3;
#text = "RPS";
dev_oppo_hist = ["R","P","S","R","P","S","R","P","S","R","P","S"]
vocab = sorted(set(dev_oppo_hist));
VOCAB_SIZE = len(vocab);

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
char2idx = {u:i for i, u in enumerate(dev_oppo_hist)}
idx2char = np.array(dev_oppo_hist)

def text_to_int(txt):
  return np.array([char2idx[c] for c in txt])


text_as_int = text_to_int(dev_oppo_hist)
dev_oppo_hist_as_int = text_to_int(dev_oppo_hist)
print("Text:", text_as_int)
print("Encoded:", text_to_int(dev_oppo_hist))

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

print("int-to-text result: ", int_to_text(dev_oppo_hist_as_int))
seq_length=3
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
BATCH_SIZE = 3;
EMBEDDING_DIM=128
RNN_UNITS=128;
BUFFER_SIZE=500;
data = char_dataset.batch(seq_length+1, drop_remainder=True)
# print("len of sequences: ",len(sequences))
# print(sequences[0:3])

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[2:]
    return input_text, target_text

# dataset = sequences.map(split_input_target)
# for batch in dataset:
#     print(batch)
# for x, y in dataset.take(5):
#   print("\n\nEXAMPLE\n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\nOUTPUT")
#   print(int_to_text(y))
#   print(dataset)
# #
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
###Build Model
def build_model(vocab_size, rnn_units, batch_size):
    inputShape = tf.keras.Input(shape=(vocab_size, batch_size))
    model_b = tf.keras.Sequential()
    model_b.add(inputShape)
    model_b.add(tf.keras.layers.LSTM(rnn_units, activation="relu", return_sequences=True))
    model_b.add(tf.keras.layers.LSTM(rnn_units, activation="relu",return_sequences=True))
    model_b.add(tf.keras.layers.LSTM(int(rnn_units/4), activation="relu", return_sequences=False))
    # model_b.add(tf.keras.layers.Dense(1))
    # model_b.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'Precision', 'Recall'])
    # model_b.fit(tf.expand_dims(data, 0), batch_size=batch_size, epochs=1)
    return model_b

model = build_model(VOCAB_SIZE, RNN_UNITS, BATCH_SIZE)
model.summary()

###Create Loss Function
##See Prediction
# predData = data.unbatch()
# data = predData.batch(BATCH_SIZE)
for pd in data:
    print("Tensor Data===>:",pd,"<=====|\n")
for input_example_batch, target_example_batch in data.take(1):
    print("input: ",input_example_batch, "\n")
    print("target: ", target_example_batch)
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
print(checkpoint_prefix)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

###Training
history = model.fit(data, epochs=10, callbacks=[checkpoint_callback])

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