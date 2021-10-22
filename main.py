import nltk
nltk.download('punkt')
import numpy
# import tflearn

import tensorflow as tf
from tensorflow.python.framework import ops

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import random
import json

with open('intents.json') as file:
  data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
  for pattern in intent["patterns"]:
    wrds = nltk.word_tokenize(pattern)
    words.extend(wrds)
    docs_x.append(pattern)
    docs_y.append(intent["tag"])

  if intent["tag"] not in labels:
    labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

# bag of words

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

ops.reset_default_graph()

tf.random.set_seed(42)

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=[None, len(training[0])]),
  tf.keras.layers.Dense(8, name="hidden_layer"),
  tf.keras.layers.Dense(8, name="hidden_layer2"), 
  tf.keras.layers.Dense(len(output[0]), name="output_layer", activation="softmax")                             
])
# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)

# model = tflearn.DNN(net)
model.compile(loss="mae",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.Accuracy()])


model.fit(training, output, epochs=1000, batch_size=8)
model.save("model.h5")

model.predict("hello")