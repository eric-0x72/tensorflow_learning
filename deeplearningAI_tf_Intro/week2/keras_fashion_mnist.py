# coursera: intro to tensorflow: week 2
import tensorflow as tf

print(tf.__version__)

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# input shape 28*28 = 784

# normalise
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#
# model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
#                                     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(256, activation=tf.nn.relu),
#                                     tf.keras.layers.Dense(5, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=10)

print('evaluation: ', model.evaluate(test_images, test_labels))


classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
