import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load data
data_path = "/Users/thear/Documents/Code/sarcasm/" \
    "Sarcasm_Headlines_Dataset.json" 
with open(data_path, 'r') as f:
    
    sentences = []
    labels = []
    urls = []
    
    for line in f:
        item = json.loads(line)
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

# Split data into test/train
X_train, X_test, y_train, y_test = train_test_split(
    sentences,
    labels, 
    test_size = 0.2)

# Preprocess text
OOV_TOKEN = "<OOV>"
VOCAB_SIZE = 10000
tokenizer = Tokenizer(
    num_words = VOCAB_SIZE,
    oov_token = OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index

# Build a small transformer network.
vocab_size = 100
network = nlp.networks.BertEncoder(
    vocab_size=vocab_size, 
    # The number of TransformerEncoderBlock layers
    num_layers=3)

MAX_LENGTH = 100
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(
    train_sequences,
    padding = 'post',
    maxlen = MAX_LENGTH,
    truncating = 'post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(
    test_sequences,
    padding = 'post',
    maxlen = MAX_LENGTH,
    truncating = 'post')
print()
print(train_padded[0])
print(train_padded.shape)

# Convert data to Numpy arrays for input to Tensorflow
train_padded = np.array(train_padded)
y_train = np.array(y_train)
test_padded = np.array(test_padded)
y_test = np.array(y_test)


# Create model
EMBEDDING_DIM = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        VOCAB_SIZE, 
        EMBEDDING_DIM, 
        input_length = MAX_LENGTH,
        mask_zero = True),
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.RNN(cell, kwargs)
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        16, 
        activation = 'tanh',
        kernel_regularizer = tf.keras.regularizers.L2(0.18))),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(
        16, 
        activation = 'relu',
        kernel_regularizer = tf.keras.regularizers.L1L2(
            l1 = 0.18, 
            l2 = 0.18)),
    tf.keras.layers.Dense(
        1, 
        activation = 'sigmoid')])

# Add EarlyStopping for effiency
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_accuracy', 
    mode = 'min',
    verbose = 1,
    patience = 20
    )

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])

# Print model summary
model.summary()

# Train model
N_EPOCHS = 30
N_BATCH = 200
history = model.fit(
    train_padded, 
    y_train, 
    epochs = N_EPOCHS, 
    validation_data = (test_padded, y_test),
    validation_split = 0.2,
    callbacks = early_stop,
    batch_size = N_BATCH,
    shuffle = True,
    verbose = 2)

# Plot model performance
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_' + string])
  plt.title("Sarcasm Classifier " + string)
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_' + string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Test model with some new sentences
new_sentence = [input("\n Check your sentence for sarcasm: ")]
print("")
# new_sentence = ["Are you really going to wear that?", 
#             "You are my oldest sister"]
sequences = tokenizer.texts_to_sequences(new_sentence)
new_padded = pad_sequences(
    sequences, 
    maxlen = MAX_LENGTH, 
    padding = 'post', 
    truncating = 'post')

if model.predict(new_padded) >= 0.5:
    print('\n That sounds like sarcasm')
elif model.predict(new_padded) < 0.5:
    print('\n That sounds sincere')

# print(model.predict(new_padded))

