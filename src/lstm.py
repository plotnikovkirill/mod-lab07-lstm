import numpy as np
import tensorflow as tf

from keras.src.models import Sequential
from keras.src.layers import Dense, Activation
from keras.src.layers import LSTM

from keras.src.optimizers import RMSprop

from keras.src.callbacks import LambdaCallback
from keras.src.callbacks import ModelCheckpoint
from keras.src.callbacks import ReduceLROnPlateau
from keras._tf_keras.keras.utils import to_categorical
import random
import sys

text = open('input.txt', encoding='utf-8').read().lower()
words = text.split()

vocabulary = sorted(list(set(words)))
word_to_index = {word: i for i, word in enumerate(vocabulary)}
index_to_word = {i: word for i, word in enumerate(vocabulary)}

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_index[word]] = 1
    y[i, word_to_index[next_words[i]]] = 1
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


model.fit(X, y, batch_size=128, epochs=50)


def generate_text(length, diversity):
    start_index = random.randint(0, len(words) - max_length - 1)
    sentence_words = words[start_index: start_index + max_length]
    generated = sentence_words.copy()

    for _ in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, word in enumerate(sentence_words):
            x_pred[0, t, word_to_index[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_word = index_to_word[next_index]

        generated.append(next_word)
        sentence_words = sentence_words[1:] + [next_word]

    return ' '.join(generated)


txt = generate_text(1500, 0.2)
with open('../result/gen.txt', 'w', encoding='utf-8') as fre:
    fre.write(txt)
print(txt)