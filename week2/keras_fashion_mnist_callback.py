import tensorflow as tf

print(tf.__version__)


class myCallback1(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


class myCallback2(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):  # only check if epoch ends
        if logs.get('acc') > 0.6:
            # print(logs) {'loss': 0.47559950805107754, 'acc': 0.82945}
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


# instance
callbacks = myCallback2()
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])  # add callbacks
