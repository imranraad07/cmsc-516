# /**
#  * @author Mia Mohammad Imran
#  */

import pathlib
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

batch_size = 64
buf_size = 2000
learning_rate = 1e-3
SEED = 1731

path = 'dataset/final_data_set_2.csv'
df = pd.read_csv(path)
data = df[['tweet', 'class']]

x_train, x_test, y_train, y_test = train_test_split(data['tweet'], data['class'], test_size=0.25, random_state=SEED)

neg, pos = np.bincount(df['class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))

num_words = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
max_len1 = max([len(el) for el in x_train])
max_len2 = max([len(el) for el in x_test])
max_len = max(max_len1, max_len2)
print("max_len: ", max_len)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding='post')
print(x_train, x_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
print(y_train.shape, y_test.shape)

path = str(pathlib.Path().absolute()) + "/" + path
print(path)
corpus_file = datapath(path)
vec_size = 100
model = FastText(size=vec_size)
model.build_vocab(corpus_file=corpus_file)
model.train(
    corpus_file=corpus_file,
    epochs=10,
    total_examples=model.corpus_count,
    total_words=model.corpus_total_words,
    model='CBOW',
    min_count=3,
    word_ngrams=1,
)
vocab_size = len(tokenizer.word_index)
print(vocab_size, num_words)
embedding_matrix = np.zeros((num_words, vec_size))
print(embedding_matrix.shape)
for i, word in enumerate(tokenizer.word_index):
    if i == num_words: break
    embedding_matrix[i] = model[word]

num_words = num_words
embedding_dim = vec_size

rate_drop_lstm = 0.15
rate_drop_dense = 0.15

if model:
    del model


def build_model(num_words, embedding_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    return model


model = build_model(num_words, embedding_dim)
model.summary()

optimizer = tf.optimizers.Adam(learning_rate)

time0 = time()
epochs = 10

from sklearn.metrics import classification_report

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(name='BC_loss'),
    optimizer=optimizer,
    metrics=tfa.metrics.F1Score(2),
)

history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

print(path)
print("Training Time (in minutes) =", (time() - time0) / 60)