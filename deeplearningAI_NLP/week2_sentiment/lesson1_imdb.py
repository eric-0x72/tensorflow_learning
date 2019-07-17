# NOTE: PLEASE MAKE SURE YOU ARE RUNNING THIS IN A PYTHON3 ENVIRONMENT

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# !pip install -q tensorflow-datasets
# print(tf.__version__)

tf.enable_eager_execution()

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
print(imdb)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())

# training_labels_final = np.array(training_labels)
# testing_labels_final = np.array(testing_labels)
